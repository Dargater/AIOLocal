"""Self-install of the external pieces AIO Local uses.

Everything this module downloads is unpacked below the app's own data directory::

    <data>/tools/ovms        OpenVINO Model Server
    <data>/tools/node        portable Node.js (REA / npx)
    <data>/tools/opencode    portable OpenCode server
    <data>/tools/gimp        private GIMP 3 installation
    <data>/tools/python      private Python 3.12 (GIMP plugin worker)
    <data>/tools/gimp-plugin Intel OpenVINO GIMP plugin + weights
    <data>/gimp/3.0          GIMP personal folder, plug-ins and settings

Nothing is written to the user's own OpenCode, GIMP or Python folders. The manager
runs every launch, reports what is missing and can install/repair components in the
background while publishing progress to the UI.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import urlsplit

from .config import Paths, SettingsStore

OVMS_VERSION = "2026.4.1"
NODE_VERSION = "24.22.2"
OPENCODE_VERSION = "2.2.5"
GIMP_VERSION = "3.0.6"
PYTHON_VERSION = "3.12.10"
GIMP_PLUGIN_VERSION = "3.3.0"

OVMS_URL = (
    "https://github.com/openvinotoolkit/model_server/releases/download/"
    f"v{OVMS_VERSION}/ovms_windows_{OVMS_VERSION}_python_on.zip"
)
OVMS_SHA_URL = OVMS_URL + ".sha256"
NODE_URL = f"https://nodejs.org/dist/v{NODE_VERSION}/node-v{NODE_VERSION}-win-x64.zip"
NODE_SHA_URL = f"https://nodejs.org/dist/v{NODE_VERSION}/SHASUMS256.txt"
OPENCODE_URL = (
    "https://github.com/anomalyco/opencode/releases/download/"
    f"v{OPENCODE_VERSION}/opencode-windows-x64.zip"
)
GIMP_URL = f"https://download.gimp.org/gimp/v3.0/windows/gimp-{GIMP_VERSION}-setup-1.exe"
GIMP_SHA_URL = "https://download.gimp.org/gimp/v3.0/windows/SHA256SUMS"
PYTHON_URL = (
    f"https://www.python.org/ftp/python/{PYTHON_VERSION}/"
    f"python-{PYTHON_VERSION}-embed-amd64.zip"
)
GIMP_PLUGIN_URL = (
    "https://github.com/intel/openvino-ai-plugins-gimp/archive/refs/tags/"
    f"{GIMP_PLUGIN_VERSION}.zip"
)
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"

CHUNK = 512 * 1024


def _human_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _safe_extract(zip_path: Path, target: Path) -> None:
    """Extract a zip without allowing entries to escape ``target``."""
    target = target.resolve()
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            name = info.filename.replace("\\", "/")
            if not name or name.endswith("/"):
                continue
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe path in archive: {info.filename}")
            destination = (target / Path(*pure.parts)).resolve()
            if target != destination and target not in destination.parents:
                raise ValueError(f"unsafe path in archive: {info.filename}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, open(destination, "wb") as out:
                shutil.copyfileobj(source, out, length=1024 * 1024)


def _find_exe(root: Path, names: tuple[str, ...], depth: int = 4) -> Path | None:
    if not root.is_dir():
        return None
    root_depth = len(root.parts)
    try:
        for current, dirs, files in os.walk(root):
            if len(Path(current).parts) - root_depth > depth:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d not in ("__pycache__",)]
            for name in names:
                candidate = Path(current) / name
                if candidate.is_file():
                    return candidate
    except OSError:
        return None
    return None


def _run_version(exe: Path, args: list[str], timeout: float = 8.0) -> str:
    try:
        proc = subprocess.run(
            [str(exe), *args],
            capture_output=True,
            timeout=timeout,
            creationflags=0x08000000 if os.name == "nt" else 0,
        )
        text = (proc.stdout or proc.stderr).decode("utf-8", "replace").strip()
        return text.splitlines()[0][:120] if text else ""
    except (OSError, subprocess.SubprocessError):
        return ""


class RequirementsManager:
    def __init__(self, paths: Paths, settings_store: SettingsStore, publish=None) -> None:
        self.paths = paths
        self.settings_store = settings_store
        self._publish = publish
        self._lock = threading.RLock()
        self._check_lock = threading.Lock()
        self._log_lock = threading.Lock()
        self._threads: dict[str, threading.Thread] = {}
        self._all_thread: threading.Thread | None = None
        self._progress: dict[str, dict] = {}
        self._errors: dict[str, str] = {}
        self._last_check = 0.0
        self._status_cache: dict[str, dict] = {}
        self._status: dict[str, dict] = {}
        self._setup_log = paths.log_dir / "setup.log"
        self._setup_log.parent.mkdir(parents=True, exist_ok=True)
        # Detection is lazy: spawning `--version` for GIMP/OpenCode/OVMS at import
        # time would delay the window. The Setup view asks for a check explicitly.

    # ------------------------------------------------------------------ paths
    @property
    def downloads(self) -> Path:
        path = self.paths.tools_dir / "downloads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def gimp_personal(self) -> Path:
        path = self.paths.data_dir / "gimp" / "3.0"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def gimp_plugin_models(self) -> Path:
        path = self.paths.data_dir / "gimp" / "openvino-models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def component_log(self, component_id: str) -> Path:
        return self.paths.log_dir / f"setup-{component_id}.log"

    # ------------------------------------------------------------------ logging
    def _log(self, component_id: str, message: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{component_id}] {message.rstrip()}\n"
        with self._log_lock:
            for path in (self._setup_log, self.component_log(component_id)):
                try:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    with open(path, "a", encoding="utf-8") as handle:
                        handle.write(line)
                except OSError:
                    pass

    def _publish_progress(self, component_id: str, **extra) -> None:
        if not self._publish:
            return
        try:
            payload = {"component": component_id, **self._progress.get(component_id, {}), **extra}
            self._publish("requirement", payload)
        except Exception:
            pass

    def _set_progress(
        self,
        component_id: str,
        *,
        state: str | None = None,
        percent: float | None = None,
        downloaded: int | None = None,
        total: int | None = None,
        message: str | None = None,
        phase: str | None = None,
    ) -> None:
        with self._lock:
            current = self._progress.setdefault(component_id, {})
            if state is not None:
                current["state"] = state
            if percent is not None:
                current["percent"] = max(0.0, min(100.0, round(float(percent), 1)))
            if downloaded is not None:
                current["downloaded"] = int(downloaded)
            if total is not None:
                current["total"] = int(total)
            if message is not None:
                current["message"] = message
            if phase is not None:
                current["phase"] = phase
            current["updated_at"] = int(time.time() * 1000)
            payload = dict(current)
        self._publish_progress(component_id, **payload)

    # ------------------------------------------------------------------ download
    def _download(
        self,
        component_id: str,
        url: str,
        destination: Path,
        *,
        expected_sha: str | None = None,
        label: str = "",
    ) -> Path:
        import httpx

        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.stat().st_size > 0 and expected_sha:
            if self._sha256(destination).lower() == expected_sha.lower():
                self._log(component_id, f"using cached {destination.name} (checksum ok)")
                return destination
        temp = destination.with_suffix(destination.suffix + ".part")
        self._log(component_id, f"download {label or url}")
        try:
            with httpx.Client(trust_env=False, follow_redirects=True, timeout=60.0) as client:
                with client.stream("GET", url) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length") or 0)
                    got = 0
                    last = 0.0
                    self._set_progress(
                        component_id,
                        state="downloading",
                        phase="download",
                        percent=0.0,
                        downloaded=0,
                        total=total,
                        message=f"Downloading {label or destination.name}",
                    )
                    with open(temp, "wb") as handle:
                        for chunk in response.iter_bytes(CHUNK):
                            if not chunk:
                                continue
                            handle.write(chunk)
                            got += len(chunk)
                            now = time.time()
                            if now - last >= 0.25:
                                last = now
                                percent = (got * 100.0 / total) if total else 0.0
                                self._set_progress(
                                    component_id,
                                    state="downloading",
                                    phase="download",
                                    percent=percent,
                                    downloaded=got,
                                    total=total,
                                    message=(
                                        f"Downloading {label or destination.name} "
                                        f"({_human_bytes(got)}"
                                        + (f" / {_human_bytes(total)})" if total else ")")
                                    ),
                                )
            if expected_sha:
                actual = self._sha256(temp)
                if actual.lower() != expected_sha.lower():
                    temp.unlink(missing_ok=True)
                    raise ValueError(
                        f"checksum mismatch for {destination.name}: expected {expected_sha}, got {actual}"
                    )
            os.replace(temp, destination)
            self._log(component_id, f"downloaded {destination} ({_human_bytes(destination.stat().st_size)})")
            self._set_progress(
                component_id,
                state="downloading",
                phase="download",
                percent=100.0,
                downloaded=destination.stat().st_size,
                total=destination.stat().st_size,
                message=f"Downloaded {destination.name}",
            )
            return destination
        finally:
            try:
                if temp.exists():
                    temp.unlink()
            except OSError:
                pass

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _fetch_text(self, url: str) -> str:
        import httpx

        with httpx.Client(trust_env=False, follow_redirects=True, timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text

    def _expected_sha(self, component_id: str, url: str, filename: str) -> str | None:
        """Look up a published checksum, or return None when the project has none."""
        try:
            if component_id == "ovms":
                text = self._fetch_text(url).strip()
                match = re.search(r"\b([0-9a-fA-F]{64})\b", text)
                return match.group(1) if match else None
            if component_id == "gimp":
                text = self._fetch_text(url)
                for line in text.splitlines():
                    if line.strip().endswith(filename):
                        return line.split()[0].strip()
            if component_id == "node":
                text = self._fetch_text(url)
                for line in text.splitlines():
                    if line.strip().endswith(filename):
                        return line.split()[0].strip()
        except Exception as exc:
            self._log(component_id, f"checksum lookup failed: {type(exc).__name__}: {exc}")
        return None

    # ------------------------------------------------------------------ component definitions
    def _definitions(self) -> list[dict]:
        data = self.paths.data_dir
        tools = self.paths.tools_dir
        return [
            {
                "id": "ovms",
                "name": "OpenVINO Model Server",
                "description": "Runs the local chat, embedding and image models.",
                "required": True,
                "version": OVMS_VERSION,
                "size_mb": 140,
                "path": "tools/ovms",
            },
            {
                "id": "node",
                "name": "Node.js (portable)",
                "description": "Used by REA/npx. Keeps its own copy instead of touching a system Node.",
                "required": True,
                "version": NODE_VERSION,
                "size_mb": 35,
                "path": "tools/node",
            },
            {
                "id": "opencode",
                "name": "OpenCode server",
                "description": "Embedded coding agent, with its chats and settings under this app only.",
                "required": True,
                "version": OPENCODE_VERSION,
                "size_mb": 90,
                "path": "tools/opencode",
            },
            {
                "id": "python",
                "name": "Python 3.12 (private)",
                "description": "Runs the Intel OpenVINO GIMP plugin without using a system Python.",
                "required": True,
                "version": PYTHON_VERSION,
                "size_mb": 12,
                "path": "tools/python",
            },
            {
                "id": "gimp",
                "name": "GIMP 3",
                "description": "Private GIMP 3.0.6 installation used by the embedded dock.",
                "required": True,
                "version": GIMP_VERSION,
                "size_mb": 310,
                "path": "tools/gimp",
            },
            {
                "id": "gimp_plugin",
                "name": "Intel OpenVINO GIMP plugin",
                "description": "Super-resolution, segmentation and Stable Diffusion tools inside GIMP.",
                "required": True,
                "version": GIMP_PLUGIN_VERSION,
                "size_mb": 1200,
                "path": "tools/gimp-plugin",
            },
        ]

    # ------------------------------------------------------------------ detection
    def _detect(self, component_id: str) -> dict:
        settings = self.settings_store.load()
        tools = self.paths.tools_dir
        data = self.paths.data_dir
        if component_id == "ovms":
            candidates: list[Path] = []
            configured = (settings.ovms_path or "").strip().strip('"')
            if configured:
                candidates.append(Path(configured))
            found = _find_exe(tools / "ovms", ("ovms.exe", "ovms"), depth=5)
            if found:
                candidates.append(found)
            for env_name in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
                base = os.environ.get(env_name)
                if base:
                    for relative in ("OVMS/ovms.exe", "OpenVINO/OVMS/ovms.exe", "Intel/OVMS/ovms.exe"):
                        candidates.append(Path(base) / relative)
            for name in ("ovms", "ovms.exe"):
                which = shutil.which(name)
                if which:
                    candidates.append(Path(which))
            for candidate in candidates:
                path = candidate
                if path.is_dir():
                    path = path / "ovms.exe"
                if path.is_file():
                    version = _run_version(path, ["--version"]) or self._folder_version(path, f"ovms_{OVMS_VERSION}")
                    return {"installed": True, "path": str(path), "detected_version": version}
            return {"installed": False, "path": "", "detected_version": ""}

        if component_id == "node":
            # Private copy only. A system Node would write its npx cache into the
            # user profile, which breaks the portable-folder promise.
            candidates = [tools / "node" / "node.exe"]
            candidates += sorted((tools / "node").glob("node-*/node.exe"))
            for path in candidates:
                if path.is_file():
                    return {"installed": True, "path": str(path), "detected_version": _run_version(path, ["--version"])}
            return {"installed": False, "path": "", "detected_version": ""}

        if component_id == "opencode":
            from .code import CodeWorkspace

            workspace = CodeWorkspace(self.paths, self.settings_store, base_url="")
            exe = workspace.executable()
            if exe:
                return {
                    "installed": True,
                    "path": exe,
                    "detected_version": workspace.version() or "",
                }
            return {"installed": False, "path": "", "detected_version": ""}

        if component_id == "python":
            # Deliberately private: never fall back to a system Python, because the
            # plugin installer must not write packages into the user's environment.
            candidates = [tools / "python" / "python.exe"]
            candidates += sorted((tools / "python").glob("python-*/python.exe"))
            for path in candidates:
                if not path.is_file():
                    continue
                version = _run_version(path, ["--version"])
                if not self._pip_works(path):
                    # A half-extracted embeddable Python has no pip. The plugin
                    # cannot install anything until it is repaired, so do not
                    # report this runtime as ready.
                    return {
                        "installed": False,
                        "path": str(path),
                        "detected_version": f"{version} (pip missing)".strip(),
                        "portable": True,
                    }
                return {"installed": True, "path": str(path), "detected_version": version, "portable": True}
            return {"installed": False, "path": "", "detected_version": "", "portable": False}

        if component_id == "gimp":
            candidates = []
            configured = (settings.gimp_path or "").strip().strip('"')
            if configured:
                candidates.append(Path(configured))
            for relative in ("bin/gimp-3.0.exe", "bin/gimp.exe", "gimp-3.0.exe", "gimp.exe"):
                candidates.append(tools / "gimp" / relative)
            for candidate in (
                Path(r"C:\Program Files\GIMP 3\bin\gimp-3.0.exe"),
                Path(r"C:\Program Files\GIMP 3\bin\gimp.exe"),
            ):
                candidates.append(candidate)
            for path in candidates:
                if path.is_file():
                    return {"installed": True, "path": str(path), "detected_version": _run_version(path, ["--version"])}
            return {"installed": False, "path": "", "detected_version": ""}

        if component_id == "gimp_plugin":
            marker = tools / "gimp-plugin" / "installed.json"
            plugin_dir = self.gimp_personal / "plug-ins" / "openvino_utils"
            config = plugin_dir / "tools" / "gimp_openvino_config.json"
            if plugin_dir.is_dir() and config.is_file():
                version = ""
                try:
                    version = str(json.loads(marker.read_text(encoding="utf-8")).get("version") or "")
                except (OSError, ValueError, json.JSONDecodeError):
                    pass
                return {
                    "installed": True,
                    "path": str(plugin_dir),
                    "detected_version": version or GIMP_PLUGIN_VERSION,
                }
            return {"installed": False, "path": "", "detected_version": ""}

        return {"installed": False, "path": "", "detected_version": ""}

    @staticmethod
    def _folder_version(path: Path, fallback: str) -> str:
        for part in path.parts:
            match = re.search(r"(\d{4}\.\d+\.\d+)", part)
            if match:
                return match.group(1)
        return fallback

    def _refresh(self) -> None:
        with self._check_lock:
            status: dict[str, dict] = {}
            for definition in self._definitions():
                detected = self._detect(definition["id"])
                status[definition["id"]] = {**definition, **detected}
            with self._lock:
                self._status = status
                self._last_check = time.time()
        self._publish_ready()

    def _publish_ready(self) -> None:
        if not self._publish:
            return
        try:
            self._publish("requirements_ready", {"components": self.status()})
        except Exception:
            pass

    def ensure_fresh(self, max_age: float = 15.0) -> None:
        if not self._status or time.time() - self._last_check > max_age:
            self._refresh()

    def status(self) -> list[dict]:
        self.ensure_fresh()
        with self._lock:
            result = []
            for component_id, entry in self._status.items():
                progress = dict(self._progress.get(component_id) or {})
                result.append(
                    {
                        **entry,
                        "state": progress.get("state") or ("installed" if entry.get("installed") else "missing"),
                        "percent": progress.get("percent", 100.0 if entry.get("installed") else 0.0),
                        "phase": progress.get("phase", ""),
                        "message": progress.get("message", ""),
                        "downloaded": progress.get("downloaded", 0),
                        "total": progress.get("total", 0),
                        "error": self._errors.get(component_id, ""),
                        "installing": component_id in self._threads,
                        "log": str(self.component_log(component_id)),
                        "data_dir": str(self.paths.data_dir),
                    }
                )
            return result

    def summary(self) -> dict:
        self.ensure_fresh()
        components = self.status()
        missing = [c for c in components if c["required"] and not c.get("installed")]
        busy = [c for c in components if c.get("installing")]
        return {
            "components": components,
            "missing": len(missing),
            "installing": len(busy),
            "ready": not missing,
            "tools_dir": str(self.paths.tools_dir),
            "gimp_dir": str(self.gimp_personal.parent),
            "checked_at": int(self._last_check * 1000),
        }

    def check(self) -> dict:
        self._refresh()
        return self.summary()

    # ------------------------------------------------------------------ install orchestration
    def install(self, component_id: str) -> dict:
        definitions = {d["id"]: d for d in self._definitions()}
        definition = definitions.get(component_id)
        if not definition:
            raise ValueError("unknown component")
        with self._lock:
            existing = self._threads.get(component_id)
            if existing and existing.is_alive():
                return {"ok": True, "already_running": True, "component": component_id}
            self._errors.pop(component_id, None)
            self._progress[component_id] = {
                "state": "queued",
                "percent": 0.0,
                "phase": "queued",
                "downloaded": 0,
                "total": int(definition.get("size_mb", 0)) * 1024 * 1024,
                "message": "Waiting to start",
                "updated_at": int(time.time() * 1000),
            }
            thread = threading.Thread(
                target=self._install_worker,
                args=(component_id,),
                name=f"install-{component_id}",
                daemon=True,
            )
            self._threads[component_id] = thread
            thread.start()
        self._publish_progress(component_id, state="queued")
        return {"ok": True, "started": True, "component": component_id}

    def install_missing(self) -> dict:
        """Install missing components sequentially in the background.

        Sequential order matters: the GIMP plugin needs both GIMP and the private
        Python runtime to exist before its installer can run.
        """
        missing = [
            component["id"]
            for component in self.status()
            if component["required"] and not component.get("installed") and not component.get("installing")
        ]
        if not missing:
            return {"ok": True, "started": [], "queued": 0}
        with self._lock:
            if self._all_thread and self._all_thread.is_alive():
                return {"ok": True, "started": [], "already_running": True}
            self._all_thread = threading.Thread(
                target=self._install_all_worker,
                args=(missing,),
                name="install-all",
                daemon=True,
            )
            self._all_thread.start()
        return {"ok": True, "started": missing, "queued": len(missing)}

    def _install_all_worker(self, component_ids: list[str]) -> None:
        try:
            for component_id in component_ids:
                try:
                    self.install(component_id)
                except ValueError:
                    continue
                while True:
                    with self._lock:
                        thread = self._threads.get(component_id)
                    if thread is None:
                        break
                    thread.join(timeout=1.0)
        finally:
            with self._lock:
                self._all_thread = None

    def _install_worker(self, component_id: str) -> None:
        try:
            self.component_log(component_id).write_text("", encoding="utf-8")
            self._log(component_id, "install started")
            self._set_progress(
                component_id,
                state="running",
                phase="prepare",
                percent=1.0,
                message="Preparing",
            )
            handler = getattr(self, f"_install_{component_id}", None)
            if handler is None:
                raise ValueError(f"no installer for {component_id}")
            handler()
            self._set_progress(
                component_id,
                state="installed",
                phase="done",
                percent=100.0,
                message="Installed",
            )
            self._log(component_id, "install finished")
            self._refresh()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"[:1200]
            self._errors[component_id] = message
            self._set_progress(
                component_id,
                state="failed",
                phase="failed",
                message=message,
            )
            self._log(component_id, f"FAILED {message}")
        finally:
            with self._lock:
                self._threads.pop(component_id, None)

    # ------------------------------------------------------------------ installers
    def _install_ovms(self) -> None:
        component = "ovms"
        filename = Path(urlsplit(OVMS_URL).path).name
        destination = self.downloads / filename
        expected = self._expected_sha(component, OVMS_SHA_URL, filename)
        archive = self._download(component, OVMS_URL, destination, expected_sha=expected, label=filename)
        self._set_progress(component, state="installing", phase="extract", percent=100.0, message="Extracting OVMS")
        target = self.paths.tools_dir / "ovms"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        _safe_extract(archive, target)
        exe = _find_exe(target, ("ovms.exe",), depth=5)
        if not exe:
            raise ValueError("the OVMS archive did not contain ovms.exe")
        settings = self.settings_store.load()
        settings.ovms_path = str(exe)
        self.settings_store.save(settings)
        self._log(component, f"OVMS ready at {exe}")

    def _install_node(self) -> None:
        component = "node"
        filename = Path(urlsplit(NODE_URL).path).name
        destination = self.downloads / filename
        expected = self._expected_sha(component, NODE_SHA_URL, filename)
        archive = self._download(component, NODE_URL, destination, expected_sha=expected, label=filename)
        self._set_progress(component, state="installing", phase="extract", percent=100.0, message="Extracting Node.js")
        target = self.paths.tools_dir / "node"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        _safe_extract(archive, target)
        exe = _find_exe(target, ("node.exe",), depth=4)
        if not exe:
            raise ValueError("the Node.js archive did not contain node.exe")
        self._log(component, f"Node.js ready at {exe}")

    def _install_opencode(self) -> None:
        component = "opencode"
        filename = Path(urlsplit(OPENCODE_URL).path).name
        destination = self.downloads / filename
        archive = self._download(component, OPENCODE_URL, destination, label=filename)
        self._set_progress(component, state="installing", phase="extract", percent=100.0, message="Extracting OpenCode")
        target = self.paths.tools_dir / "opencode"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        _safe_extract(archive, target)
        exe = _find_exe(target, ("opencode.exe",), depth=4)
        if not exe:
            raise ValueError("the OpenCode archive did not contain opencode.exe")
        self._log(component, f"OpenCode ready at {exe}")

    def _python_exe(self) -> Path:
        found = _find_exe(self.paths.tools_dir / "python", ("python.exe",), depth=4)
        if found:
            return found
        raise ValueError("the private Python 3.12 runtime is not installed yet")

    @staticmethod
    def _pip_env() -> dict:
        env = dict(os.environ)
        env["PYTHONNOUSERSITE"] = "1"
        env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
        env["PIP_NO_INPUT"] = "1"
        return env

    def _pip_works(self, python_exe: Path) -> bool:
        try:
            proc = subprocess.run(
                [str(python_exe), "-m", "pip", "--version"],
                capture_output=True,
                timeout=20,
                creationflags=0x08000000 if os.name == "nt" else 0,
                env=self._pip_env(),
            )
            return proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def _ensure_pip(self, component_id: str, python_exe: Path) -> None:
        """Make `python -m pip` work, including on a bare embeddable Python.

        The python.org embeddable archive ships no pip and no ensurepip module.
        get-pip.py is a standalone script, so it must be run as
        `python get-pip.py`; `python -m pip get-pip.py` can never work there.
        """
        if self._pip_works(python_exe):
            self._log(component_id, _run_version(python_exe, ["-m", "pip", "--version"]))
            return
        get_pip = self.downloads / "get-pip.py"
        if not get_pip.is_file() or get_pip.stat().st_size < 100_000:
            self._download(component_id, GET_PIP_URL, get_pip, label="get-pip.py")
        self._set_progress(
            component_id,
            state="installing",
            phase="pip",
            message="Bootstrapping pip into the private Python runtime",
        )
        self._run_process(
            component_id,
            [str(python_exe), str(get_pip)],
            "Installing pip",
            timeout=900,
            env=self._pip_env(),
        )
        if not self._pip_works(python_exe):
            raise ValueError("pip was bootstrapped but `python -m pip` still does not start")
        # setup.py-based plugins need setuptools; wheel keeps later installs fast.
        self._run_pip(
            component_id,
            python_exe,
            ["install", "--upgrade", "setuptools", "wheel"],
            "Installing build tools",
            timeout=1200,
        )
        self._log(component_id, _run_version(python_exe, ["-m", "pip", "--version"]))

    def _install_python(self) -> None:
        component = "python"
        filename = Path(urlsplit(PYTHON_URL).path).name
        destination = self.downloads / filename
        archive = self._download(component, PYTHON_URL, destination, label=filename)
        self._set_progress(component, state="installing", phase="extract", percent=100.0, message="Extracting Python")
        target = self.paths.tools_dir / "python"
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
        _safe_extract(archive, target)
        exe = _find_exe(target, ("python.exe",), depth=3)
        if not exe:
            raise ValueError("the Python archive did not contain python.exe")
        # The embeddable build disables site-packages by default.
        for pth in exe.parent.glob("python*._pth"):
            text = pth.read_text(encoding="utf-8", errors="replace")
            if "#import site" in text:
                pth.write_text(text.replace("#import site", "import site"), encoding="utf-8")
        # Bootstrap pip. The embeddable archive has neither pip nor ensurepip, so
        # get-pip.py has to run directly; it cannot be launched as `-m pip`.
        self._ensure_pip(component, exe)
        self._log(component, f"Python ready at {exe}")

    def _install_gimp(self) -> None:
        component = "gimp"
        filename = Path(urlsplit(GIMP_URL).path).name
        destination = self.downloads / filename
        expected = self._expected_sha(component, GIMP_SHA_URL, filename)
        installer = self._download(component, GIMP_URL, destination, expected_sha=expected, label=filename)
        target = self.paths.tools_dir / "gimp"
        target.mkdir(parents=True, exist_ok=True)
        self._set_progress(
            component,
            state="installing",
            phase="installer",
            percent=100.0,
            message="Running the GIMP installer (this can take several minutes)",
        )
        # GIMP's Windows installer is Inno Setup. Install into the app's own folder so
        # no shared Program Files copy is touched, then fall back to the NSIS switches
        # used by a few older builds.
        attempts = [
            [
                str(installer),
                "/VERYSILENT",
                "/SUPPRESSMSGBOXES",
                "/NORESTART",
                "/SP-",
                f"/DIR={target}",
            ],
            [str(installer), "/S", f"/D={target}"],
        ]
        last_error = ""
        for args in attempts:
            try:
                self._run_process(component, args, "Running the GIMP installer", timeout=1800)
                if _find_exe(target, ("gimp-3.0.exe", "gimp.exe"), depth=4):
                    self._log(component, f"GIMP ready at {target}")
                    return
                last_error = "the installer finished but gimp-3.0.exe was not found"
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
        raise ValueError(
            f"GIMP could not be installed automatically ({last_error}). "
            f"The installer is saved at {installer}; run it and choose {target}."
        )

    def _install_gimp_plugin(self) -> None:
        component = "gimp_plugin"
        gimp = self._detect("gimp")
        if not gimp.get("installed"):
            raise ValueError("install GIMP 3 first")
        python_exe = self._python_exe()
        self._ensure_pip(component, python_exe)
        filename = Path(urlsplit(GIMP_PLUGIN_URL).path).name
        destination = self.downloads / filename
        archive = self._download(component, GIMP_PLUGIN_URL, destination, label=filename)
        source_root = self.paths.tools_dir / "gimp-plugin" / "src"
        if source_root.exists():
            shutil.rmtree(source_root, ignore_errors=True)
        source_root.mkdir(parents=True, exist_ok=True)
        self._set_progress(component, state="installing", phase="extract", percent=100.0, message="Extracting the plugin source")
        _safe_extract(archive, source_root)
        setup_py = next(source_root.glob("openvino-ai-plugins-gimp-*/setup.py"), None)
        if setup_py is None:
            raise ValueError("the plugin archive did not contain setup.py")
        plugin_root = setup_py.parent
        self._run_pip(component, python_exe, ["install", "wmi"], "Installing wmi")
        self._run_pip(
            component,
            python_exe,
            ["install", str(plugin_root)],
            "Installing the plugin and its OpenVINO dependencies",
            timeout=7200,
        )
        weights = plugin_root / "weights"
        env = dict(os.environ)
        env["GIMP_OPENVINO_MODELS_PATH"] = str(self.gimp_plugin_models)
        env["PYTHONNOUSERSITE"] = "1"
        code = (
            "from gimpopenvino import install_utils;"
            f"install_utils.complete_install(repo_weights_dir={str(weights)!r})"
        )
        self._run_process(
            component,
            [str(python_exe), "-c", code],
            "Copying bundled OpenVINO models",
            timeout=1800,
            env=env,
        )
        # Copy the five GIMP plug-in folders into the app's private GIMP profile.
        site_packages = self._site_packages(python_exe)
        source_plugins = site_packages / "gimpopenvino" / "plugins"
        if not source_plugins.is_dir():
            raise ValueError("gimpopenvino was installed but its plugins folder is missing")
        target = self.gimp_personal / "plug-ins"
        target.mkdir(parents=True, exist_ok=True)
        for name in ("openvino_utils", "semseg_ov", "stable_diffusion_ov", "superresolution_ov", "fastsd_ov"):
            source = source_plugins / name
            if not source.is_dir():
                continue
            destination_dir = target / name
            if destination_dir.exists():
                shutil.rmtree(destination_dir, ignore_errors=True)
            shutil.copytree(source, destination_dir)
            self._log(component, f"copied plugin {name}")
        marker = self.paths.tools_dir / "gimp-plugin" / "installed.json"
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps(
                {
                    "version": GIMP_PLUGIN_VERSION,
                    "python": str(python_exe),
                    "models": str(self.gimp_plugin_models),
                    "gimp_profile": str(self.gimp_personal),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._log(component, "plugin installed into the private GIMP profile")

    def _site_packages(self, python_exe: Path) -> Path:
        try:
            proc = subprocess.run(
                [str(python_exe), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                capture_output=True,
                text=True,
                timeout=30,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            if proc.returncode == 0:
                return Path(proc.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
        return python_exe.parent / "Lib" / "site-packages"

    # ------------------------------------------------------------------ process helpers
    def _run_pip(
        self,
        component_id: str,
        python_exe: Path,
        args: list[str],
        message: str,
        timeout: float = 3600,
    ) -> None:
        command = [str(python_exe), "-m", "pip", *args]
        self._run_process(
            component_id,
            command,
            message,
            timeout=timeout,
            env=self._pip_env(),
        )

    def _run_process(
        self,
        component_id: str,
        command: list[str],
        message: str,
        *,
        timeout: float = 3600,
        env: dict | None = None,
    ) -> None:
        self._log(component_id, f"$ {' '.join(command)}")
        self._set_progress(
            component_id,
            state="installing",
            phase="install",
            message=message,
        )
        creationflags = 0x08000000 if os.name == "nt" else 0
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env or dict(os.environ),
            creationflags=creationflags,
        )
        deadline = time.time() + max(30.0, timeout)
        last = time.time()
        tail: list[str] = []
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                text = line.rstrip()
                if not text:
                    continue
                tail.append(text[:500])
                if len(tail) > 40:
                    del tail[:20]
                self._log(component_id, text)
                now = time.time()
                if now - last >= 0.5:
                    last = now
                    self._set_progress(component_id, message=f"{message}: {text[:180]}")
                if time.time() > deadline:
                    proc.kill()
                    raise ValueError(f"{message} timed out")
            code = proc.wait(timeout=30)
        except Exception:
            try:
                proc.kill()
            except OSError:
                pass
            raise
        if code != 0:
            detail = "\n".join(tail[-8:]) or f"exit code {code}"
            raise ValueError(f"{message} failed (exit {code}):\n{detail}")

    # ------------------------------------------------------------------ logs
    def log_tail(self, component_id: str, lines: int = 120) -> list[str]:
        path = self.component_log(component_id)
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []
        rows = [line for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n") if line.strip()]
        return rows[-max(1, min(int(lines), 1000)):]
