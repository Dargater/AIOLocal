"""REA integration over MCP stdio.

REA is an external agent server (`npx -y rea-agents mcp`). This module speaks the MCP
stdio transport directly: newline-delimited JSON-RPC 2.0, one request at a time.

Safety choices:

* the process is only started when the user presses Start;
* tools are listed, never auto-invoked; every call is an explicit action with JSON
  arguments;
* output and stderr are capped, and the command is never passed through a shell.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path

from .code import check_rea_node

PROTOCOL_VERSION = "2025-06-18"
MAX_ARGS_BYTES = 24 * 1024
MAX_RESULT_BYTES = 256 * 1024
TOOL_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]{1,80}$")


class MCPError(Exception):
    pass


class MCPClient:
    """Minimal MCP stdio client with one blocking request at a time."""

    def __init__(self, command: list[str], env: dict | None = None) -> None:
        self.command = command
        self.env = env or {}
        self._proc: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._stderr: threading.Thread | None = None
        self._lock = threading.RLock()
        self._pending: dict[int, dict] = {}
        self._next_id = 1
        self._stderr_tail: deque[str] = deque(maxlen=60)
        self._closed = threading.Event()
        self.server_info: dict = {}

    # ------------------------------------------------------------------ process
    def start(self, timeout: float = 90.0) -> dict:
        if self.running():
            return {"ok": True, "already_running": True}
        env = dict(os.environ)
        env.update(self.env)
        env.setdefault("NO_COLOR", "1")
        env.setdefault("CI", "1")
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=env,
                creationflags=0x08000000 if os.name == "nt" else 0,
            )
        except OSError as exc:
            raise MCPError(f"could not start REA ({type(exc).__name__}: {exc})") from exc

        self._closed.clear()
        self._reader = threading.Thread(target=self._read_stdout, name="rea-stdout", daemon=True)
        self._reader.start()
        self._stderr = threading.Thread(target=self._read_stderr, name="rea-stderr", daemon=True)
        self._stderr.start()

        result = self._request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "aiolocal", "version": "0.1.0"},
            },
            timeout=timeout,
        )
        if isinstance(result, dict):
            self.server_info = {
                "serverInfo": result.get("serverInfo") or {},
                "protocolVersion": result.get("protocolVersion") or "",
                "capabilities": result.get("capabilities") or {},
            }
        self._notify("notifications/initialized", {})
        return {"ok": True, "server": self.server_info}

    def running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> dict:
        proc = self._proc
        self._proc = None
        self._closed.set()
        if not proc or proc.poll() is not None:
            return {"ok": True, "already_stopped": True}
        try:
            proc.terminate()
            proc.wait(timeout=6)
        except (subprocess.TimeoutExpired, OSError):
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
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            try:
                if stream:
                    stream.close()
            except OSError:
                pass
        return {"ok": True, "stopped": True}

    # ------------------------------------------------------------------ transport
    def _write(self, payload: dict) -> None:
        proc = self._proc
        if not proc or proc.poll() is not None or not proc.stdin:
            raise MCPError("REA is not running")
        line = json.dumps(payload, ensure_ascii=False) + "\n"
        with self._lock:
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (BrokenPipeError, OSError) as exc:
                raise MCPError("REA closed the connection") from exc

    def _notify(self, method: str, params: dict) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _request(self, method: str, params: dict, timeout: float = 90.0) -> object:
        with self._lock:
            request_id = self._next_id
            self._next_id += 1
            event = threading.Event()
            slot: dict = {"event": event}
            self._pending[request_id] = slot
        try:
            self._write({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params})
            if not event.wait(timeout):
                raise MCPError(f"REA did not answer {method} within {int(timeout)}s")
            if "error" in slot:
                raise MCPError(str(slot["error"]))
            return slot.get("result")
        finally:
            self._pending.pop(request_id, None)

    def _read_stdout(self) -> None:
        proc = self._proc
        if not proc or not proc.stdout:
            return
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(message, dict):
                continue
            if "id" in message and message["id"] in self._pending:
                slot = self._pending[message["id"]]
                if "error" in message:
                    err = message["error"]
                    if isinstance(err, dict):
                        slot["error"] = str(err.get("message") or err)[:800]
                    else:
                        slot["error"] = str(err)[:800]
                else:
                    slot["result"] = message.get("result")
                slot["event"].set()
        self._closed.set()
        # Unblock any waiting request so the UI gets a clear failure.
        for slot in list(self._pending.values()):
            slot.setdefault("error", "REA exited")
            slot["event"].set()

    def _read_stderr(self) -> None:
        proc = self._proc
        if not proc or not proc.stderr:
            return
        for line in proc.stderr:
            text = line.strip()
            if text:
                self._stderr_tail.append(text[:500])

    def stderr_tail(self) -> list[str]:
        return list(self._stderr_tail)

    # ------------------------------------------------------------------ tools
    def list_tools(self) -> list[dict]:
        result = self._request("tools/list", {}, timeout=45)
        tools = result.get("tools") if isinstance(result, dict) else []
        out = []
        for tool in tools or []:
            if not isinstance(tool, dict):
                continue
            name = str(tool.get("name") or "")
            if not TOOL_NAME_RE.fullmatch(name):
                continue
            out.append(
                {
                    "name": name,
                    "description": str(tool.get("description") or "")[:1500],
                    "inputSchema": tool.get("inputSchema") or {},
                }
            )
        return out

    def call_tool(self, name: str, arguments: dict, timeout: float = 180.0) -> dict:
        if not TOOL_NAME_RE.fullmatch(name or ""):
            raise MCPError("invalid tool name")
        if not isinstance(arguments, dict):
            raise MCPError("arguments must be a JSON object")
        encoded = json.dumps(arguments, ensure_ascii=False)
        if len(encoded.encode("utf-8")) > MAX_ARGS_BYTES:
            raise MCPError("arguments are too large")
        result = self._request(
            "tools/call", {"name": name, "arguments": arguments}, timeout=timeout
        )
        text_parts: list[str] = []
        truncated = False
        if isinstance(result, dict):
            content = result.get("content")
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text":
                        text_parts.append(str(part.get("text") or ""))
                    elif part.get("type") == "resource":
                        text_parts.append(str(part.get("resource") or ""))
            if result.get("isError"):
                text_parts.insert(0, "The REA tool reported an error.")
        text = "\n".join(text_parts)
        if len(text.encode("utf-8")) > MAX_RESULT_BYTES:
            text = text[: MAX_RESULT_BYTES // 4]
            truncated = True
        return {
            "ok": not (isinstance(result, dict) and result.get("isError")),
            "text": text,
            "truncated": truncated,
            "raw": result if not truncated else None,
        }


class ReverseEngineering:
    """Lifecycle + policy wrapper around :class:`MCPClient`."""

    def __init__(self, settings_store, paths=None) -> None:
        self.settings_store = settings_store
        self.paths = paths
        self.client: MCPClient | None = None
        self._lock = threading.RLock()
        self._last_error = ""
        self._tools: list[dict] = []

    def _portable_node(self) -> Path | None:
        """Return the app's own node.exe, never a system install."""
        root = getattr(self.paths, "tools_dir", None)
        if root is None:
            return None
        node_root = Path(root) / "node"
        candidates = [node_root / "node.exe"]
        try:
            candidates += sorted(node_root.glob("node-*/node.exe"))
        except OSError:
            pass
        for candidate in candidates:
            if candidate.is_file():
                return candidate
        return None

    def _node_executable(self) -> str | None:
        portable = self._portable_node()
        if portable:
            return str(portable)
        return shutil.which("node") or shutil.which("node.exe")

    def _npx_prefix(self) -> list[str]:
        """Build a command that runs npx without touching a system Node.

        The portable Node archive includes ``npx-cli.js``; running it through the
        portable ``node.exe`` avoids the usual Windows ``.cmd``/shell quoting trap.
        A system npx is only a last-resort fallback.
        """
        node = self._portable_node()
        if node:
            for cli in (
                node.parent / "node_modules" / "npm" / "bin" / "npx-cli.js",
                node.parent / "node_modules" / "npm" / "bin" / "npm-cli.js",
            ):
                if cli.is_file():
                    return [str(node), str(cli)]
        npx = shutil.which("npx") or shutil.which("npx.cmd")
        if not npx and node:
            for name in ("npx.cmd", "npx.bat", "npx"):
                candidate = node.parent / name
                if candidate.is_file():
                    npx = str(candidate)
                    break
        if not npx:
            raise MCPError(
                "Node.js/npx was not found. Open Setup and install the portable Node.js "
                "(REA needs 22.23.4+, 24.22.1+ or 26.12.1+)."
            )
        if os.name == "nt" and npx.lower().endswith((".cmd", ".bat")):
            comspec = os.environ.get("COMSPEC") or "cmd.exe"
            return [comspec, "/d", "/s", "/c", npx]
        return [npx]

    def _command(self) -> tuple[list[str], str]:
        settings = self.settings_store.load()
        custom = (settings.rea_command or "").strip()
        if custom:
            parts = shlex.split(custom, posix=os.name != "nt")
            return [p.strip('"') for p in parts], custom
        target = "rea-agents"
        if settings.rea_version.strip():
            version = settings.rea_version.strip()
            if not re.fullmatch(r"[0-9][0-9A-Za-z.\-+]{0,30}", version):
                raise MCPError("the REA version looks invalid")
            target = f"rea-agents@{version}"
        prefix = self._npx_prefix()
        return [*prefix, "-y", target, "mcp"], f"npx -y {target} mcp"

    def _env(self) -> dict:
        settings = self.settings_store.load()
        env = {}
        ghidra = (settings.ghidra_path or "").strip().strip('"')
        if ghidra:
            env["GHIDRA_INSTALL_DIR"] = ghidra
            env["GHIDRA_HOME"] = ghidra
        return env

    def status(self) -> dict:
        settings = self.settings_store.load()
        node_version = None
        node = self._node_executable()
        if node:
            try:
                proc = subprocess.run(
                    [node, "--version"],
                    capture_output=True,
                    timeout=8,
                    creationflags=0x08000000 if os.name == "nt" else 0,
                )
                node_version = (proc.stdout or proc.stderr).decode("ascii", "replace").strip() or None
            except (OSError, subprocess.SubprocessError):
                node_version = None
        node_ok, node_message = check_rea_node(node_version)
        try:
            command, display = self._command()
            command_error = ""
        except MCPError as exc:
            command, display, command_error = [], "", str(exc)
        client = self.client
        return {
            "node_version": node_version or "",
            "node_ok": node_ok,
            "node_message": node_message,
            "node_path": node or "",
            "node_portable": self._portable_node() is not None,
            "command": display,
            "command_error": command_error,
            "running": bool(client and client.running()),
            "server": (client.server_info if client else {}),
            "tools_cached": len(self._tools),
            "last_error": self._last_error,
            "stderr": client.stderr_tail() if client else [],
            "ghidra_path": settings.ghidra_path,
            "rea_version": settings.rea_version,
        }

    def start(self) -> dict:
        with self._lock:
            if self.client and self.client.running():
                return {"ok": True, "already_running": True}
            command, display = self._command()
            client = MCPClient(command, env=self._env())
            try:
                result = client.start()
            except Exception as exc:
                client.stop()
                self._last_error = f"{type(exc).__name__}: {exc}"[:800]
                raise
            self.client = client
            self._last_error = ""
            try:
                self._tools = client.list_tools()
            except Exception:
                self._tools = []
            return {"ok": True, "command": display, "server": result.get("server") or {}, "tools": len(self._tools)}

    def stop(self) -> dict:
        with self._lock:
            if not self.client:
                return {"ok": True, "already_stopped": True}
            result = self.client.stop()
            self.client = None
            self._tools = []
            return result

    def tools(self, refresh: bool = False) -> list[dict]:
        client = self.client
        if not client or not client.running():
            raise MCPError("REA is not running")
        if refresh or not self._tools:
            self._tools = client.list_tools()
        return self._tools

    def call(self, name: str, arguments: dict) -> dict:
        client = self.client
        if not client or not client.running():
            raise MCPError("REA is not running")
        return client.call_tool(name, arguments)
