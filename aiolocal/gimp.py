"""GIMP dock.

GIMP 3 is a GTK3 application and GTK4 removed the old socket-embed API, so the only
practical way to put it inside this app is the Win32 route: launch a dedicated GIMP
instance, find its top-level window, and re-parent it into a native host window with
``SetParent``.

The host is a real Windows window, not a browser element, so it opens next to the app.
If anything fails (no window within 25 seconds, a non-Windows build, an unsupported
GIMP build), the launcher falls back to leaving GIMP in its own window and reports that
honestly. Closing the dock never kills GIMP or discards unsaved work.

GIMP's personal folder, plug-ins and OpenVINO models all live below the app's data
directory so nothing is shared with a normal GIMP installation.
"""

from __future__ import annotations

import ctypes
import json
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from ctypes import wintypes
from pathlib import Path

from .config import Paths, SettingsStore

WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
WM_SIZE = 0x0005
WM_TIMER = 0x0113
WS_CHILD = 0x40000000
WS_POPUP = 0x80000000
WS_CAPTION = 0x00C00000
WS_THICKFRAME = 0x00040000
WS_VISIBLE = 0x10000000
WS_CLIPCHILDREN = 0x02000000
WS_CLIPSIBLINGS = 0x04000000
WS_OVERLAPPEDWINDOW = 0x00CF0000
SW_HIDE = 0
SW_SHOW = 5
SWP_FRAMECHANGED = 0x0020
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001
SWP_SHOWWINDOW = 0x0040
SWP_NOZORDER = 0x0004
GWL_STYLE = -16
GWL_EXSTYLE = -20
GA_ROOT = 2


class GimpError(Exception):
    pass


class _DockThread(threading.Thread):
    def __init__(self, gimp_proc: subprocess.Popen, before: set[int]) -> None:
        super().__init__(name="gimp-dock", daemon=True)
        self.gimp_proc = gimp_proc
        self.before = before
        self.host_hwnd = 0
        self.gimp_hwnd = 0
        self.error = ""
        self.embedded = threading.Event()
        self.stopping = threading.Event()
        self._original_style = 0
        self._original_exstyle = 0
        self._wndproc_ref = None

    # -- helpers ----------------------------------------------------------
    @staticmethod
    def _class_name(hwnd: int) -> str:
        user32 = ctypes.windll.user32
        buf = ctypes.create_unicode_buffer(256)
        user32.GetClassNameW(hwnd, buf, 256)
        return buf.value

    @staticmethod
    def _title(hwnd: int) -> str:
        user32 = ctypes.windll.user32
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        return buf.value

    def _process_id(self, hwnd: int) -> int:
        user32 = ctypes.windll.user32
        pid = wintypes.DWORD(0)
        try:
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        except Exception:
            return 0
        return int(pid.value)

    @staticmethod
    def _is_splash(title: str, klass: str) -> bool:
        haystack = f"{title} {klass}".lower()
        return any(word in haystack for word in ("splash", "startup", "starting", "loading"))

    def _find_gimp_window(self) -> int:
        if self.gimp_proc.poll() is not None:
            return 0
        user32 = ctypes.windll.user32
        found: list[tuple[int, int]] = []

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def callback(hwnd, lparam):
            if hwnd in self.before:
                return True
            if not user32.IsWindowVisible(hwnd):
                return True
            if user32.GetParent(hwnd):
                return True
            title = self._title(hwnd)
            klass = self._class_name(hwnd)
            if self._is_splash(title, klass):
                return True
            pid = self._process_id(hwnd)
            looks_gimp = "gimp" in title.lower() or klass.lower().startswith("gimp")
            if not looks_gimp:
                return True
            # Prefer the process we launched, but accept a child process too. This
            # avoids grabbing an unrelated GIMP window from the user's session.
            if pid != self.gimp_proc.pid and pid not in self._child_pids():
                return True
            rect = wintypes.RECT()
            user32.GetWindowRect(hwnd, ctypes.byref(rect))
            area = max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)
            if area < 120_000:
                return True
            found.append((area, int(hwnd)))
            return True

        user32.EnumWindows(callback, 0)
        if not found:
            return 0
        found.sort(reverse=True)
        return found[0][1]

    def _child_pids(self) -> set[int]:
        """GIMP may re-exec or keep the window in a child process on some builds."""
        pids = {self.gimp_proc.pid}
        if os.name != "nt":
            return pids
        try:
            import ctypes.wintypes as wt

            class PROCESSENTRY32W(ctypes.Structure):
                _fields_ = [
                    ("dwSize", wt.DWORD),
                    ("cntUsage", wt.DWORD),
                    ("th32ProcessID", wt.DWORD),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                    ("th32ModuleID", wt.DWORD),
                    ("cntThreads", wt.DWORD),
                    ("th32ParentProcessID", wt.DWORD),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", wt.DWORD),
                    ("szExeFile", wt.WCHAR * 260),
                ]

            kernel32 = ctypes.windll.kernel32
            snapshot = kernel32.CreateToolhelp32Snapshot(2, 0)  # TH32CS_SNAPPROCESS
            if snapshot in (0, -1):
                return pids
            entry = PROCESSENTRY32W()
            entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
            try:
                if kernel32.Process32FirstW(snapshot, ctypes.byref(entry)):
                    while True:
                        if int(entry.th32ParentProcessID) in pids:
                            pids.add(int(entry.th32ProcessID))
                        if not kernel32.Process32NextW(snapshot, ctypes.byref(entry)):
                            break
            finally:
                kernel32.CloseHandle(snapshot)
        except Exception:
            pass
        return pids

    # -- thread -----------------------------------------------------------
    def run(self) -> None:
        try:
            self._run()
        except Exception as exc:  # never let a dock bug touch the server
            self.error = f"{type(exc).__name__}: {exc}"[:500]
            self._release()
            if self.host_hwnd:
                try:
                    ctypes.windll.user32.DestroyWindow(self.host_hwnd)
                except Exception:
                    pass

    def _run(self) -> None:
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        WNDPROC = ctypes.WINFUNCTYPE(
            ctypes.c_ssize_t, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM
        )

        class WNDCLASSW(ctypes.Structure):
            _fields_ = [
                ("style", wintypes.UINT),
                ("lpfnWndProc", WNDPROC),
                ("cbClsExtra", ctypes.c_int),
                ("cbWndExtra", ctypes.c_int),
                ("hInstance", wintypes.HINSTANCE),
                ("hIcon", wintypes.HICON),
                ("hCursor", wintypes.HANDLE),
                ("hbrBackground", wintypes.HBRUSH),
                ("lpszMenuName", wintypes.LPCWSTR),
                ("lpszClassName", wintypes.LPCWSTR),
            ]

        def wndproc(hwnd, msg, wparam, lparam):
            if msg == WM_TIMER:
                self._tick()
                return 0
            if msg == WM_SIZE:
                self._resize_child()
                return 0
            if msg == WM_CLOSE:
                self.stopping.set()
                self._release()
                user32.DestroyWindow(hwnd)
                return 0
            if msg == WM_DESTROY:
                user32.PostQuitMessage(0)
                return 0
            return user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        self._wndproc_ref = WNDPROC(wndproc)
        hinst = kernel32.GetModuleHandleW(None)
        class_name = f"AIOLocalGimpDock-{os.getpid()}"
        wc = WNDCLASSW()
        wc.lpfnWndProc = self._wndproc_ref
        wc.hInstance = hinst
        wc.lpszClassName = class_name
        wc.hbrBackground = 0  # no white flash while the child is being attached
        atom = user32.RegisterClassW(ctypes.byref(wc))
        if not atom:
            raise GimpError("could not register the dock window class")

        # Start hidden: a visible empty host window is what made the old dock look
        # like a trapped white rectangle while GIMP was still starting.
        self.host_hwnd = user32.CreateWindowExW(
            0,
            class_name,
            "AIO Local — GIMP",
            WS_OVERLAPPEDWINDOW | WS_CLIPCHILDREN | WS_CLIPSIBLINGS,
            80,
            80,
            1200,
            820,
            None,
            None,
            hinst,
            None,
        )
        if not self.host_hwnd:
            raise GimpError("could not create the dock window")

        user32.SetTimer(self.host_hwnd, 1, 200, None)
        self._tick()
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))
        user32.KillTimer(self.host_hwnd, 1)

    def _tick(self) -> None:
        if self.stopping.is_set():
            self._release()
            if self.host_hwnd:
                ctypes.windll.user32.DestroyWindow(self.host_hwnd)
                self.host_hwnd = 0
            return
        if self.gimp_proc.poll() is not None:
            self._release()
            if self.host_hwnd:
                ctypes.windll.user32.DestroyWindow(self.host_hwnd)
                self.host_hwnd = 0
            return
        if not self.gimp_hwnd:
            hwnd = self._find_gimp_window()
            if hwnd:
                self._embed(hwnd)
            return
        if not ctypes.windll.user32.IsWindow(self.gimp_hwnd):
            self.gimp_hwnd = 0
            self._resize_child()

    def _embed(self, gimp_hwnd: int) -> None:
        user32 = ctypes.windll.user32
        user32.GetWindowLongPtrW.restype = ctypes.c_ssize_t
        user32.SetWindowLongPtrW.restype = ctypes.c_ssize_t
        self._original_style = int(user32.GetWindowLongPtrW(gimp_hwnd, GWL_STYLE))
        self._original_exstyle = int(user32.GetWindowLongPtrW(gimp_hwnd, GWL_STYLE - 4)) if False else 0
        # Hide first so the reparent is not visible as a white frame.
        user32.ShowWindow(gimp_hwnd, SW_HIDE)
        style = self._original_style
        style &= ~WS_POPUP
        style &= ~WS_CAPTION
        style &= ~WS_THICKFRAME
        style |= WS_CHILD | WS_VISIBLE
        user32.SetWindowLongPtrW(gimp_hwnd, GWL_STYLE, ctypes.c_ssize_t(style))
        if not user32.SetParent(gimp_hwnd, self.host_hwnd):
            self.error = "SetParent failed; GIMP stays in its own window"
            user32.SetWindowLongPtrW(gimp_hwnd, GWL_STYLE, ctypes.c_ssize_t(self._original_style))
            user32.ShowWindow(gimp_hwnd, SW_SHOW)
            self.stopping.set()
            return
        self.gimp_hwnd = gimp_hwnd
        self._resize_child()
        user32.ShowWindow(gimp_hwnd, SW_SHOW)
        user32.ShowWindow(self.host_hwnd, SW_SHOW)
        user32.SetForegroundWindow(self.host_hwnd)
        self.embedded.set()

    def _resize_child(self) -> None:
        if not (self.host_hwnd and self.gimp_hwnd):
            return
        user32 = ctypes.windll.user32
        rect = wintypes.RECT()
        user32.GetClientRect(self.host_hwnd, ctypes.byref(rect))
        user32.MoveWindow(
            self.gimp_hwnd,
            0,
            0,
            max(1, rect.right - rect.left),
            max(1, rect.bottom - rect.top),
            True,
        )

    def _release(self) -> None:
        hwnd = self.gimp_hwnd
        self.gimp_hwnd = 0
        if not hwnd:
            return
        user32 = ctypes.windll.user32
        try:
            if user32.IsWindow(hwnd):
                user32.ShowWindow(hwnd, SW_HIDE)
                user32.SetParent(hwnd, None)
                if self._original_style:
                    user32.SetWindowLongPtrW(hwnd, GWL_STYLE, ctypes.c_ssize_t(self._original_style))
                user32.ShowWindow(hwnd, SW_SHOW)
                user32.SetWindowPos(
                    hwnd, 0, 120, 120, 1100, 760, SWP_FRAMECHANGED | SWP_SHOWWINDOW
                )
        except Exception:
            pass


class GimpDock:
    def __init__(self, paths: Paths, settings_store: SettingsStore, image_lookup=None) -> None:
        self.paths = paths
        self.settings_store = settings_store
        self._image_lookup = image_lookup
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        self._dock: _DockThread | None = None
        self._last_error = ""
        self._started_at = 0.0
        self._version_cache: str | None = None
        self._log_handle = None
        self._log_path = paths.log_dir / "gimp.log"

    # ------------------------------------------------------------------ discovery
    def executable(self) -> str | None:
        settings = self.settings_store.load()
        configured = (settings.gimp_path or "").strip().strip('"')
        if configured:
            return configured if Path(configured).is_file() else None
        for name in ("gimp-3.0", "gimp-3.0.exe", "gimp", "gimp.exe"):
            found = shutil.which(name)
            if found:
                return found
        for candidate in (
            self.paths.tools_dir / "gimp" / "bin" / "gimp-3.0.exe",
            self.paths.tools_dir / "gimp" / "bin" / "gimp.exe",
            self.paths.tools_dir / "gimp" / "gimp-3.0.exe",
            self.paths.tools_dir / "gimp" / "gimp.exe",
            Path(r"C:\Program Files\GIMP 3\bin\gimp-3.0.exe"),
            Path(r"C:\Program Files\GIMP 3\bin\gimp.exe"),
        ):
            if Path(candidate).is_file():
                return str(candidate)
        return None

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
            match = re.search(r"(\d+\.\d+(?:\.\d+)?)", text)
            self._version_cache = (match.group(1) if match else text[:80]) or "installed"
        except (OSError, subprocess.SubprocessError):
            self._version_cache = None
        return self._version_cache

    def personal_dir(self) -> Path:
        path = self.paths.data_dir / "gimp" / "3.0"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def plugin_models_dir(self) -> Path:
        path = self.paths.data_dir / "gimp" / "openvino-models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def plugin_installed(self) -> bool:
        config = self.personal_dir() / "plug-ins" / "openvino_utils" / "tools" / "gimp_openvino_config.json"
        return config.is_file()

    def plugin_tools_dir(self) -> Path:
        return self.personal_dir() / "plug-ins" / "openvino_utils" / "tools"

    def fastsd_config_path(self) -> Path:
        return self.plugin_tools_dir() / "fastsd_models.json"

    @staticmethod
    def _port_open(port: int) -> bool:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.35):
                return True
        except OSError:
            return False

    def plugin_ports(self) -> dict:
        return {
            "65432": self._port_open(65432),
            "65433": self._port_open(65433),
            "65434": self._port_open(65434),
        }

    def installed_weights(self) -> list[str]:
        base = self.plugin_models_dir() / "weights"
        if not base.is_dir():
            return []
        try:
            return sorted(p.name for p in base.iterdir() if p.is_dir())
        except OSError:
            return []

    @staticmethod
    def has_sd_weights(names: list[str]) -> bool:
        marks = ("sd", "stable", "dream", "sdxl", "lcm", "turbo", "flux", "sana")
        return any(any(mark in name.lower() for mark in marks) for name in names)

    def issues(self, weights: list[str] | None = None, ports: dict | None = None) -> list[str]:
        weights = weights if weights is not None else self.installed_weights()
        ports = ports if ports is not None else self.plugin_ports()
        issues: list[str] = []
        if os.name != "nt":
            issues.append("GIMP docking is Windows-only; GIMP launches in its own window on Linux.")
        if not self.plugin_installed():
            issues.append("The Intel OpenVINO GIMP plugin is not installed. Install it from Setup.")
        if not self.has_sd_weights(weights):
            issues.append(
                "No Stable Diffusion weights were found in the plugin models folder. "
                "Download an SD model in the plugin's model manager before generating."
            )
        if ports.get("65432") and not self.running():
            issues.append(
                "Port 65432 is held by another process (an orphaned FastSD server or "
                "another GIMP instance). Close it or run Repair before generating."
            )
        return issues

    def sanitize_plugin_config(self) -> dict:
        """Remove URL entries the FastSD model picker cannot load.

        The plugin accepts Hugging Face repository names and local model folders,
        not HTTP URLs. A bad entry makes model selection fail with a confusing
        error, so repair removes them and keeps a backup.
        """
        path = self.fastsd_config_path()
        if not path.is_file():
            return {"ok": False, "reason": "the plugin config does not exist yet", "path": str(path), "removed": []}
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            return {"ok": False, "reason": f"could not read the config: {exc}", "path": str(path), "removed": []}
        if isinstance(raw, dict):
            models = raw.get("models")
        elif isinstance(raw, list):
            models = raw
        else:
            return {"ok": False, "reason": "unexpected config shape", "path": str(path), "removed": []}
        if not isinstance(models, list):
            return {"ok": False, "reason": "the models list is missing", "path": str(path), "removed": []}

        removed: list[str] = []
        kept: list[str] = []
        for item in models:
            name = str(item or "").strip()
            if not name or name.lower().startswith(("http://", "https://")):
                removed.append(name or "(empty)")
            elif name not in kept:
                kept.append(name)
        if not kept:
            kept = ["rupeshs/sd-turbo-openvino"]
        if not removed:
            return {"ok": True, "removed": [], "kept": len(kept), "path": str(path)}

        backup = path.with_suffix(path.suffix + ".bak")
        try:
            if not backup.exists():
                shutil.copy2(path, backup)
            payload = {"models": kept}
            if isinstance(raw, dict):
                payload = {**raw, "models": kept}
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, path)
        except OSError as exc:
            return {"ok": False, "reason": f"could not write the config: {exc}", "path": str(path), "removed": removed}
        return {"ok": True, "removed": removed, "kept": len(kept), "path": str(path), "backup": str(backup)}

    def repair(self) -> dict:
        config = self.sanitize_plugin_config()
        weights = self.installed_weights()
        ports = self.plugin_ports()
        stale = []
        try:
            for part in self.plugin_models_dir().rglob("*.part"):
                part.unlink(missing_ok=True)
                stale.append(str(part))
        except OSError:
            pass
        return {
            "ok": True,
            "config": config,
            "weights": weights,
            "ports": ports,
            "issues": self.issues(weights, ports),
            "cleaned": stale,
        }

    def plugin_version(self) -> str:
        marker = self.paths.tools_dir / "gimp-plugin" / "installed.json"
        try:
            return str(json.loads(marker.read_text(encoding="utf-8")).get("version") or "")
        except (OSError, ValueError, json.JSONDecodeError):
            return ""

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def status(self) -> dict:
        with self._lock:
            dock = self._dock
            embedded = bool(dock and dock.embedded.is_set())
            error = self._last_error or (dock.error if dock else "")
            version = self.version()
            weights = self.installed_weights()
            ports = self.plugin_ports()
            return {
                "installed": bool(self.executable()),
                "executable": self.executable() or "",
                "version": version or "",
                "version_ok": bool(version and version.startswith("3.0.6")),
                "running": self.running(),
                "pid": self._proc.pid if self.running() else None,
                "embedded": embedded,
                "supported": os.name == "nt",
                "host_window": dock.host_hwnd if dock else 0,
                "last_error": error,
                "started_at": int(self._started_at * 1000) if self._started_at else 0,
                "plugin_installed": self.plugin_installed(),
                "plugin_version": self.plugin_version(),
                "plugin_ports": ports,
                "plugin_server_ready": bool(ports.get("65432")),
                "weights": weights,
                "sd_weights": self.has_sd_weights(weights),
                "issues": self.issues(weights, ports),
                "log": str(self._log_path),
                "personal_dir": str(self.personal_dir()),
                "models_dir": str(self.plugin_models_dir()),
            }

    # ------------------------------------------------------------------ lifecycle
    def _env(self) -> dict:
        env = dict(os.environ)
        # Everything GIMP writes stays inside the app's data folder.
        env["GIMP3_DIRECTORY"] = str(self.personal_dir())
        env["GIMP_OPENVINO_MODELS_PATH"] = str(self.plugin_models_dir())
        # Force, do not inherit: a variable set by another GIMP/plugin install must
        # never redirect this dock away from its private profile.
        env["GIMP_OPENVINO_CONFIG_PATH"] = str(
            self.personal_dir() / "plug-ins" / "openvino_utils" / "tools"
        )
        env["GIMP3_PLUGIN_DIR"] = str(self.personal_dir() / "plug-ins")
        env["GIMP3_TEMP_DIR"] = str(self.paths.data_dir / "gimp" / "temp")
        Path(env["GIMP3_TEMP_DIR"]).mkdir(parents=True, exist_ok=True)
        return env

    def launch(self, *, image_id: str | None = None, embed: bool = True) -> dict:
        exe = self.executable()
        if not exe:
            raise GimpError(
                "GIMP was not found. Install GIMP 3.0.6 from Setup, or set its executable in Settings."
            )
        version = self.version() or ""
        warning = ""
        if version and not version.startswith("3.0.6"):
            warning = (
                f"GIMP {version} is installed, but the Intel OpenVINO plugin requires "
                "GIMP 3.0.6. Install the tested version from Setup."
            )
        if self.running():
            if image_id:
                self.open_image(image_id)
                return {"ok": True, "already_running": True, "opened": image_id, "warning": warning}
            return {"ok": True, "already_running": True, "pid": self._proc.pid, "warning": warning}

        before = self._windows()
        args = [exe, "--new-instance"]
        if image_id:
            args.append(str(self._image_path(image_id)))
        try:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = open(self._log_path, "ab", buffering=0)
            self._log_handle.write(
                f"\n--- {time.strftime('%Y-%m-%d %H:%M:%S')} launching GIMP ---\n".encode("utf-8")
            )
            self._proc = subprocess.Popen(
                args,
                cwd=str(Path(exe).parent),
                env=self._env(),
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            raise GimpError(f"could not start GIMP: {exc}") from exc
        self._started_at = time.time()
        self._last_error = ""

        if embed and os.name == "nt":
            dock = _DockThread(self._proc, before)
            self._dock = dock
            dock.start()
            if not dock.embedded.wait(timeout=30.0):
                # Leave GIMP running in its own window; never kill it.
                dock.stopping.set()
                if not dock.error:
                    self._last_error = (
                        "GIMP did not expose a main window to dock within 30 seconds; "
                        "it is running in its own window."
                    )
                else:
                    self._last_error = dock.error
        return {
            "ok": True,
            "pid": self._proc.pid,
            "embedding": bool(embed and os.name == "nt"),
            "note": "" if (embed and os.name == "nt") else "Docking is only available on Windows.",
            "warning": warning,
        }

    def _windows(self) -> set[int]:
        if os.name != "nt":
            return set()
        found: set[int] = set()

        @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
        def callback(hwnd, lparam):
            found.add(int(hwnd))
            return True

        try:
            ctypes.windll.user32.EnumWindows(callback, 0)
        except Exception:
            return set()
        return found

    def _image_path(self, image_id: str) -> Path:
        return self._resolve_image(image_id)

    def open_image(self, image_id: str) -> dict:
        exe = self.executable()
        if not exe:
            raise GimpError("GIMP was not found. Install GIMP 3.0.6 from Setup.")
        path = self._resolve_image(image_id)
        try:
            subprocess.Popen([exe, str(path)], cwd=str(Path(exe).parent), env=self._env())
        except OSError as exc:
            raise GimpError(f"could not open the image in GIMP: {exc}") from exc
        return {"ok": True, "image": image_id}

    def _resolve_image(self, image_id: str) -> Path:
        if self._image_lookup is None:
            raise GimpError("image lookup is not configured")
        try:
            return self._image_lookup(str(image_id))
        except ValueError as exc:
            raise GimpError(str(exc)) from exc

    def release(self) -> dict:
        with self._lock:
            dock = self._dock
            self._dock = None
            if dock:
                dock.stopping.set()
                try:
                    dock.join(timeout=5)
                except RuntimeError:
                    pass
            return {"ok": True, "released": bool(dock)}

    def shutdown(self) -> None:
        self.release()
        try:
            if self._log_handle is not None:
                self._log_handle.close()
        except OSError:
            pass
        self._log_handle = None
