from __future__ import annotations

import base64
import http.server
import json
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlencode

from aiolocal.code import CodeWorkspace
from aiolocal.config import Paths, Settings, SettingsStore


def test_opencode_environment_is_isolated(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    workspace = CodeWorkspace(paths, store, base_url="http://127.0.0.1:7317")
    config = paths.config_dir / "opencode.json"
    env = workspace._env(config)
    runtime = paths.data_dir / "opencode"
    assert env["XDG_DATA_HOME"] == str(runtime / "data")
    assert env["XDG_CONFIG_HOME"] == str(runtime / "config")
    assert env["XDG_CACHE_HOME"] == str(runtime / "cache")
    assert env["XDG_STATE_HOME"] == str(runtime / "state")
    assert env["OPENCODE_CONFIG"] == str(config)
    assert env["OPENCODE_DISABLE_TELEMETRY"] == "1"


def test_embed_url_contains_base64url_token(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    workspace = CodeWorkspace(paths, store, base_url="http://127.0.0.1:7317")
    workspace._process = SimpleNamespace(poll=lambda: None)
    workspace._port = 4123
    workspace._password = "secret value"
    workspace._session_id = "ses_test"
    expected = base64.urlsafe_b64encode(b"opencode:secret value").decode("ascii").rstrip("=")
    server_key = base64.urlsafe_b64encode(b"http://127.0.0.1:4123").decode("ascii").rstrip("=")
    assert workspace.embed_url() == (
        f"http://127.0.0.1:4123/server/{server_key}/session/ses_test?auth_token={expected}"
    )


def test_embed_url_without_a_session_opens_home(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    workspace = CodeWorkspace(paths, store, base_url="http://127.0.0.1:7317")
    workspace._process = SimpleNamespace(poll=lambda: None)
    workspace._port = 4123
    workspace._password = "secret value"
    expected = base64.urlsafe_b64encode(b"opencode:secret value").decode("ascii").rstrip("=")
    assert workspace.embed_url() == f"http://127.0.0.1:4123/?auth_token={expected}"


def test_embed_url_is_empty_when_stopped(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    workspace = CodeWorkspace(paths, store, base_url="http://127.0.0.1:7317")
    assert workspace.embed_url() == ""


def test_provider_config_points_at_the_app_gateway(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings(chat_model="qwen3.5-9b"))
    workspace = CodeWorkspace(paths, store, base_url="http://127.0.0.1:7317")
    path = workspace.write_config()
    text = path.read_text(encoding="utf-8")
    assert '"baseURL": "http://127.0.0.1:7317/v1"' in text
    assert '"model": "aiolocal/qwen3.5-9b"' in text
    assert "http://127.0.0.1:8000" not in text


def test_workspace_status_reports_isolation(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    workspace = CodeWorkspace(paths, store, base_url="http://127.0.0.1:7317")
    status = workspace.status()
    assert status["isolated"] is True
    assert status["data_dir"].endswith("opencode")
    assert status["config_path"].endswith("opencode.json")


def test_workspace_is_registered_as_an_opencode_project(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    workspace = CodeWorkspace(paths, store, base_url="http://127.0.0.1:7317")
    workspace._password = "secret value"
    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        target = paths.data_dir / "workspace"
        workspace._register_project(target, server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    query = urlencode({"location[directory]": str(target)})
    assert seen["path"] == f"/api/location?{query}"
    expected = base64.b64encode(b"opencode:secret value").decode("ascii")
    assert seen["auth"] == f"Basic {expected}"


def test_ensure_session_reuses_the_newest_workspace_session(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    workspace = CodeWorkspace(paths, store, base_url="http://127.0.0.1:7317")
    workspace._password = "secret value"
    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            body = json.dumps({"data": [{"id": "ses_existing"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        target = paths.data_dir / "workspace"
        session_id = workspace._ensure_session(target, server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert session_id == "ses_existing"
    assert seen["path"] == "/api/session?" + urlencode({"directory": str(target), "limit": 50})
    expected = base64.b64encode(b"opencode:secret value").decode("ascii")
    assert seen["auth"] == f"Basic {expected}"


def test_ensure_session_creates_one_when_the_list_is_empty(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    workspace = CodeWorkspace(paths, store, base_url="http://127.0.0.1:7317")
    workspace._password = "secret value"
    seen = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = json.dumps({"data": []}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            seen["body"] = json.loads(self.rfile.read(length).decode("utf-8"))
            body = json.dumps({"data": {"id": "ses_new"}}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        target = paths.data_dir / "workspace"
        session_id = workspace._ensure_session(target, server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert session_id == "ses_new"
    assert seen["body"] == {"title": "Workspace", "location": {"directory": str(target)}}
