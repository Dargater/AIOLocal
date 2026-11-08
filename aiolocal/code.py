"""Code workspace integration.

OpenCode is a separate local server. This module owns its lifecycle, keeps its data in
the app's own folders (so it never sees the user's desktop OpenCode chats) and exposes
a read-only view of the selected workspace.

The generated provider config is written to the app's config folder, not into the
project, and contains the loopback API token. Treat that file as a secret.
"""

from __future__ import annotations

import base64
import http.client
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import quote, urlencode

from .config import Paths, SettingsStore
from .db import now_ms

IGNORED_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", ".venv", "venv",
    "dist", "build", ".next", ".cache", "target",
}
TEXT_EXTENSIONS = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".json", ".jsonc", ".md", ".mdx",
    ".txt", ".toml", ".yaml", ".yml", ".ini", ".cfg", ".css", ".scss", ".html",
    ".htm", ".xml", ".sql", ".sh", ".ps1", ".bat", ".cmd", ".c", ".h", ".cc",
    ".cpp", ".hpp", ".cs", ".java", ".kt", ".go", ".rs", ".rb", ".php", ".lua",
    ".swift", ".r", ".jl", ".vue", ".svelte", ".dockerfile", ".gitignore",
}
SECRET_NAMES = re.compile(
    r"(^|/)(\.env(\..*)?|\.git-credentials|id_rsa.*|id_ed25519.*|.*\.(pem|key|pfx|p12|kdbx)|"
    r"credentials(\..*)?|secrets?(\..*)?)$",
    re.IGNORECASE,
)
MAX_PREVIEW_BYTES = 512 * 1024
MAX_LISTED_FILES = 800
PASSWORD_RE = re.compile(r"server password\s+(\S+)")
LISTEN_RE = re.compile(r"server listening on\s+(https?://\S+)")


def _port_free(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(preferred: int, attempts: int = 30) -> int:
    for offset in range(attempts):
        candidate = preferred + offset
        if 0 < candidate < 65536 and _port_free(candidate):
            return candidate
    raise ValueError(
        f"no free port in range {preferred}-{preferred + attempts - 1} for OpenCode"
    )


def _b64url(text: str) -> str:
    return base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")


class CodeWorkspace:
    def __init__(self, paths: Paths, settings_store: SettingsStore, base_url: str) -> None:
        self.paths = paths
        self.settings_store = settings_store
        self.base_url = base_url.rstrip("/")
        self._lock = threading.RLock()
        self._process: subprocess.Popen | None = None
        self._log = None
        self._reader: threading.Thread | None = None
        self._version_cache: str | None = None
        self._password = ""
        self._port = 0
        self._session_id = ""
        self._start_error = ""
        self._ready = threading.Event()
        self._log_tail_lines: list[str] = []
        self._log_lock = threading.Lock()
        self._session_lock = threading.Lock()

    # ------------------------------------------------------------------ discovery
    def executable(self) -> str | None:
        settings = self.settings_store.load()
        configured = (settings.opencode_path or "").strip().strip('"')
        if configured:
            path = Path(configured)
            if path.is_file():
                return str(path)
            return None
        # Portable copy installed by the requirements manager.
        portable = self.paths.tools_dir / "opencode"
        for candidate in (
            portable / "opencode.exe",
            portable / "opencode",
            portable / "opencode-windows-x64" / "opencode.exe",
        ):
            if candidate.is_file():
                return str(candidate)
        for name in ("opencode", "opencode.cmd", "opencode.exe"):
            found = shutil.which(name)
            if found:
                return found
        # The OpenCode desktop app ships a CLI/server that is not added to PATH.
        local = os.environ.get("LOCALAPPDATA")
        if local:
            for relative in (
                r"Programs\ai.opencode.desktop\resources\opencode-cli.exe",
                r"Programs\opencode\opencode.exe",
                r"Programs\OpenCode\opencode.exe",
            ):
                candidate = Path(local) / relative
                if candidate.is_file():
                    return str(candidate)
        return None

    def node_executable(self) -> str | None:
        portable = self.paths.tools_dir / "node"
        candidates = [portable / "node.exe"]
        try:
            candidates += sorted(portable.glob("node-*/node.exe"))
        except OSError:
            pass
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        for name in ("node", "node.exe"):
            found = shutil.which(name)
            if found:
                return found
        return None

    def workspace(self) -> Path:
        settings = self.settings_store.load()
        root = Path(settings.workspace_dir).expanduser() if settings.workspace_dir else self.paths.data_dir / "workspace"
        root.mkdir(parents=True, exist_ok=True)
        return root.resolve()

    def set_workspace(self, value: str) -> Path:
        raw = (value or "").strip().strip('"')
        if not raw:
            raise ValueError("choose a workspace folder")
        root = Path(raw).expanduser()
        if not root.exists():
            raise ValueError("that folder does not exist")
        if not root.is_dir():
            raise ValueError("that path is a file, not a folder")
        return root.resolve()

    def version(self) -> str | None:
        if self._version_cache is not None:
            return self._version_cache
        exe = self.executable()
        if not exe:
            return None
        try:
            proc = subprocess.run(
                [exe, "--version"],
                capture_output=True,
                timeout=12,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            text = (proc.stdout or proc.stderr).decode("utf-8", "replace").strip()
            self._version_cache = text[:120] or "installed"
        except (OSError, subprocess.SubprocessError):
            self._version_cache = None
        return self._version_cache

    def node_version(self) -> str | None:
        node = self.node_executable()
        if not node:
            return None
        try:
            proc = subprocess.run(
                [node, "--version"],
                capture_output=True,
                timeout=8,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
            return (proc.stdout or proc.stderr).decode("ascii", "replace").strip()[:40] or None
        except (OSError, subprocess.SubprocessError):
            return None

    # ------------------------------------------------------------------ isolation
    def runtime_dir(self) -> Path:
        root = self.paths.data_dir / "opencode"
        for child in ("data", "config", "cache", "state"):
            (root / child).mkdir(parents=True, exist_ok=True)
        return root

    def _env(self, config_path: Path) -> dict:
        runtime = self.runtime_dir()
        env = dict(os.environ)
        env["XDG_DATA_HOME"] = str(runtime / "data")
        env["XDG_CONFIG_HOME"] = str(runtime / "config")
        env["XDG_CACHE_HOME"] = str(runtime / "cache")
        env["XDG_STATE_HOME"] = str(runtime / "state")
        env["OPENCODE_CONFIG"] = str(config_path)
        env["OPENCODE_SERVER_PASSWORD"] = self._server_password()
        env["OPENCODE_DISABLE_TELEMETRY"] = "1"
        env["OPENCODE_DISABLE_AUTOUPDATE"] = "1"
        env["DO_NOT_TRACK"] = "1"
        # Never let a child inherit Python paths from the frozen app.
        for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(key, None)
        return env

    def _server_password(self) -> str:
        """Return a stable password for the embedded server.

        OpenCode prints a new random password on every start unless
        ``OPENCODE_SERVER_PASSWORD`` is set. A stable value keeps the browser
        session signed in after restarts and never leaves the app data folder.
        """
        path = self.runtime_dir() / "server-password.txt"
        try:
            existing = path.read_text(encoding="utf-8").strip()
            if 16 <= len(existing) <= 256:
                return existing
        except OSError:
            pass
        password = secrets.token_urlsafe(32)
        try:
            path.write_text(password + "\n", encoding="utf-8")
        except OSError:
            pass
        return password

    def _api(self, method: str, path: str, port: int, payload: dict | None = None) -> object:
        """Call OpenCode's local HTTP API with the stable server password."""
        if not self._password or not port:
            return None
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        try:
            auth = base64.b64encode(f"opencode:{self._password}".encode("utf-8")).decode("ascii")
            headers = {"Authorization": f"Basic {auth}"}
            body = None
            if payload is not None:
                body = json.dumps(payload).encode("utf-8")
                headers["Content-Type"] = "application/json"
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            data = response.read()
            if response.status >= 400:
                raise ValueError(f"OpenCode API {method} {path} returned {response.status}")
            return json.loads(data.decode("utf-8")) if data else None
        finally:
            try:
                connection.close()
            except OSError:
                pass

    def _register_project(self, workspace: Path, port: int) -> None:
        """Add the workspace to OpenCode's project list.

        OpenCode 2.x creates the project record on the first location lookup. Without
        this, the embedded UI does not know the folder exists and its New session
        button has no project to use.
        """
        try:
            query = urlencode({"location[directory]": str(workspace)})
            self._api("GET", f"/api/location?{query}", port)
        except Exception as exc:  # registration is a convenience, never a hard failure
            try:
                self._record_log(f"workspace project registration failed: {exc}\n".encode("utf-8"))
            except Exception:
                pass

    def _ensure_session(self, workspace: Path, port: int) -> str:
        """Return the newest workspace session, creating one when the list is empty.

        OpenCode's Home view lists only projects that the browser has "used" before.
        A freshly copied app therefore opens on an empty Home screen with a New
        session button that does nothing. Opening a real session instead gives the
        embedded UI a working composer immediately; the session also makes OpenCode
        remember the workspace project for later visits.
        """
        with self._session_lock:
            try:
                query = urlencode({"directory": str(workspace), "limit": 50})
                payload = self._api("GET", f"/api/session?{query}", port)
                sessions = payload.get("data", []) if isinstance(payload, dict) else payload
                for session in sessions or []:
                    session_id = session.get("id") if isinstance(session, dict) else None
                    if session_id:
                        return str(session_id)
                created = self._api(
                    "POST",
                    "/api/session",
                    port,
                    {"title": "Workspace", "location": {"directory": str(workspace)}},
                )
                data = created.get("data", {}) if isinstance(created, dict) else {}
                session_id = data.get("id") if isinstance(data, dict) else None
                return str(session_id) if session_id else ""
            except Exception as exc:
                try:
                    self._record_log(f"workspace session setup failed: {exc}\n".encode("utf-8"))
                except Exception:
                    pass
                return ""

    # ------------------------------------------------------------------ lifecycle
    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _record_log(self, line: str) -> None:
        text = line.rstrip("\r\n")
        if not text:
            return
        with self._log_lock:
            self._log_tail_lines.append(text[:2000])
            if len(self._log_tail_lines) > 400:
                del self._log_tail_lines[:200]
        if "password" in text.lower():
            match = PASSWORD_RE.search(text)
            if match:
                self._password = match.group(1).strip()
                self._ready.set()

    def _read_output(self, proc: subprocess.Popen, log_path: Path) -> None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(log_path, "ab")
        except OSError:
            self._log = None
        stream = proc.stdout
        if stream is None:
            return
        while True:
            try:
                raw = stream.readline()
            except (OSError, ValueError):
                break
            if not raw:
                break
            if isinstance(raw, str):
                data = raw.encode("utf-8", "replace")
            else:
                data = raw
            if self._log is not None:
                try:
                    self._log.write(data)
                    self._log.flush()
                except OSError:
                    pass
            self._record_log(data.decode("utf-8", "replace"))
        try:
            if self._log is not None:
                self._log.close()
        except OSError:
            pass
        self._log = None

    def start(self, wait_seconds: float = 35.0) -> dict:
        with self._lock:
            if self.running():
                if not self._session_id:
                    self._session_id = self._ensure_session(self.workspace(), self._port)
                return {
                    "ok": True,
                    "already_running": True,
                    "pid": self._process.pid,
                    "port": self._port or int(self.settings_store.load().opencode_port),
                    "embed_url": self.embed_url(),
                }
            exe = self.executable()
            if not exe:
                raise ValueError(
                    "OpenCode was not found. Install it from Setup, or set the executable in Settings."
                )
            settings = self.settings_store.load()
            workspace = self.workspace()
            port = _pick_port(int(settings.opencode_port))
            config_path = self.write_config()
            self._password = self._server_password()
            self._start_error = ""
            self._ready.clear()
            with self._log_lock:
                self._log_tail_lines.clear()
            self._port = port
            self.paths.log_dir.mkdir(parents=True, exist_ok=True)
            log_path = self.paths.log_dir / "opencode.log"
            env = self._env(config_path)
            try:
                self._process = subprocess.Popen(
                    [exe, "serve", "--hostname", "127.0.0.1", "--port", str(port)],
                    cwd=str(workspace),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    env=env,
                    creationflags=0x08000000 if os.name == "nt" else 0,
                )
            except OSError as exc:
                self._process = None
                raise ValueError(f"could not start OpenCode: {exc}") from exc
            self._reader = threading.Thread(
                target=self._read_output,
                args=(self._process, log_path),
                name="opencode-log",
                daemon=True,
            )
            self._reader.start()
            proc = self._process
            deadline = time.time() + max(5.0, wait_seconds)
            while time.time() < deadline:
                if proc.poll() is not None:
                    self._start_error = (
                        f"OpenCode exited during startup (code {proc.returncode}). "
                        f"See {log_path}."
                    )
                    break
                if self._password and _port_free(port) is False:
                    break
                time.sleep(0.15)
            if proc.poll() is not None:
                raise ValueError(self._start_error or "OpenCode exited during startup")
            if not self._ready.is_set():
                # Still allow startup if the server is listening but printed no password.
                if _port_free(port):
                    self._start_error = (
                        "OpenCode did not report a server password within "
                        f"{int(wait_seconds)} seconds. See {log_path}."
                    )
                    self.stop()
                    raise ValueError(self._start_error)
            self._register_project(workspace, port)
            self._session_id = self._ensure_session(workspace, port)
            return {
                "ok": True,
                "pid": proc.pid,
                "port": port,
                "workspace": str(workspace),
                "embed_url": self.embed_url(),
                "isolated": True,
                "data_dir": str(self.runtime_dir()),
            }

    def stop(self) -> dict:
        with self._lock:
            proc = self._process
            self._process = None
            self._ready.clear()
            self._password = ""
            self._port = 0
            self._session_id = ""
            if not proc or proc.poll() is not None:
                return {"ok": True, "already_stopped": True}
            try:
                proc.terminate()
                proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                if os.name == "nt":
                    try:
                        subprocess.run(
                            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                            capture_output=True,
                            timeout=10,
                            creationflags=0x08000000,
                        )
                    except (OSError, subprocess.SubprocessError):
                        pass
                else:
                    try:
                        proc.kill()
                    except OSError:
                        pass
            except OSError:
                pass
            try:
                if self._reader is not None:
                    self._reader.join(timeout=3)
            except RuntimeError:
                pass
            return {"ok": True, "stopped": True}

    def log_tail(self, lines: int = 80) -> list[str]:
        with self._log_lock:
            return list(self._log_tail_lines[-max(1, min(int(lines), 400)):])

    def embed_url(self) -> str:
        """Return the OpenCode web UI URL with its one-time auth token.

        OpenCode's UI accepts ``?auth_token=<base64url("opencode:<password>")>``. It
        consumes the token, stores the server as the canonical local connection and
        removes it from the address bar. That removes the pairing screen entirely.
        The app opens the newest workspace session so the embedded UI has a working
        composer even on a browser profile that has never seen OpenCode before.
        """
        if not self.running() or not self._port:
            return ""
        token = _b64url(f"opencode:{self._password}") if self._password else ""
        base = f"http://127.0.0.1:{self._port}/"
        if self._session_id:
            server = f"http://127.0.0.1:{self._port}"
            base = (
                f"{base}server/{_b64url(server)}/session/"
                f"{quote(self._session_id, safe='')}"
            )
        return f"{base}?auth_token={quote(token, safe='')}" if token else base

    def status(self) -> dict:
        settings = self.settings_store.load()
        workspace = self.workspace()
        exe = self.executable()
        runtime = self.runtime_dir()
        return {
            "installed": bool(exe),
            "executable": exe or "",
            "version": self.version(),
            "node_version": self.node_version(),
            "running": self.running(),
            "pid": self._process.pid if self.running() else None,
            "port": self._port or int(settings.opencode_port),
            "url": f"http://127.0.0.1:{self._port or int(settings.opencode_port)}",
            "embed_url": self.embed_url(),
            "auth_ready": bool(self._password),
            "isolated": True,
            "data_dir": str(runtime),
            "workspace": str(workspace),
            "config_path": str(self.paths.config_dir / "opencode.json"),
            "log": str(self.paths.log_dir / "opencode.log"),
            "log_tail": self.log_tail(60),
            "start_error": self._start_error,
            "models": self._models(settings),
        }

    def _models(self, settings) -> list[dict]:
        from .models import CATALOG

        return [
            {"id": m["id"], "name": m["name"]}
            for m in CATALOG
            if m["task"] == "text_generation"
        ]

    # ------------------------------------------------------------------ provider config
    def write_config(self) -> Path:
        """Write an OpenCode config that points at the local gateway.

        It lives in the app config folder so a project is never modified behind the
        user's back. The token inside is the same loopback secret the UI uses.
        """
        settings = self.settings_store.load()
        token = self._token()
        config = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "aiolocal": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "AIO Local",
                    "options": {
                        "baseURL": f"{self.base_url}/v1",
                        "apiKey": token,
                    },
                    "models": {
                        m["id"]: {"name": m["name"]}
                        for m in self._models(settings)
                    },
                }
            },
            "model": f"aiolocal/{settings.chat_model}",
        }
        path = self.paths.config_dir / "opencode.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(config, indent=2), encoding="utf-8")
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, path)
        return path

    def _token(self) -> str:
        from .security import load_or_create_token

        return load_or_create_token(self.paths.token_path)

    # ------------------------------------------------------------------ read-only files
    def list_files(self, limit: int = MAX_LISTED_FILES) -> list[dict]:
        root = self.workspace()
        limit = max(1, min(int(limit), MAX_LISTED_FILES))
        out: list[dict] = []
        for current, dirs, files in os.walk(root):
            dirs[:] = sorted(d for d in dirs if d.lower() not in IGNORED_DIRS and not d.startswith("."))
            for name in sorted(files):
                if len(out) >= limit:
                    return out
                path = Path(current) / name
                try:
                    stat = path.stat()
                except OSError:
                    continue
                rel = path.relative_to(root).as_posix()
                out.append(
                    {
                        "path": rel,
                        "size": stat.st_size,
                        "ext": path.suffix.lower(),
                        "secret": bool(SECRET_NAMES.search(rel)),
                    }
                )
        out.sort(key=lambda f: f["path"].lower())
        return out

    def read_file(self, relative: str) -> dict:
        from .security import safe_join

        if not relative or len(relative) > 500:
            raise ValueError("invalid file path")
        if SECRET_NAMES.search(relative):
            raise ValueError("this file looks like a secret and is not shown")
        root = self.workspace()
        path = safe_join(root, *relative.split("/"))
        if not path.is_file():
            raise ValueError("file not found")
        size = path.stat().st_size
        if size > MAX_PREVIEW_BYTES:
            raise ValueError("file is too large to preview (512 KB max)")
        suffix = path.suffix.lower()
        if suffix not in TEXT_EXTENSIONS and path.name.lower() not in ("dockerfile", "makefile"):
            raise ValueError("only text files can be previewed")
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise ValueError(f"could not read the file: {exc}") from exc
        return {"path": relative, "size": size, "text": text[:200_000], "truncated": len(text) > 200_000}


def check_node_version(version: str | None, minimum: tuple[int, int, int] = (22, 23, 4)) -> tuple[bool, str]:
    """REA (and some OpenCode installs) need a modern Node. Returns (ok, message)."""
    if not version:
        return False, "Node.js was not found on PATH."
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return False, f"Could not read the Node.js version ({version})."
    current = tuple(int(x) for x in match.groups())
    if current >= minimum:
        return True, f"Node.js {version} is supported."
    return False, f"Node.js {version} is too old; {'.'.join(map(str, minimum))}+ is required."


def check_rea_node(version: str | None) -> tuple[bool, str]:
    """REA supports specific Node lines; return a precise verdict.

    Upstream requirements (at the time of writing): 22.23.4+, 24.22.1+ or 26.12.1+.
    Odd-numbered and older lines are refused rather than guessed at.
    """
    if not version:
        return False, "Node.js was not found. REA needs Node 22.23.4+, 24.22.1+ or 26.12.1+."
    match = re.search(r"v?(\d+)\.(\d+)\.(\d+)", version)
    if not match:
        return False, f"Could not read the Node.js version ({version})."
    major, minor, patch = (int(x) for x in match.groups())
    current = (major, minor, patch)
    minimums = {22: (22, 23, 4), 24: (24, 22, 1), 26: (26, 12, 1)}
    need = minimums.get(major)
    if need and current >= need:
        return True, f"Node.js {version} is supported by REA."
    if need:
        return False, (
            f"Node.js {version} is too old for REA; use 22.23.4+, 24.22.1+ or 26.12.1+."
        )
    return False, (
        f"Node.js {version} is not a supported REA line; use 22.23.4+, 24.22.1+ or 26.12.1+."
    )
