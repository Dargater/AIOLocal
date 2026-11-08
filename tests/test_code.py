from __future__ import annotations

from pathlib import Path

from aiolocal.code import CodeWorkspace
from aiolocal.config import Paths, SettingsStore


def test_opencode_desktop_cli_is_detected(tmp_path: Path, monkeypatch):
    local = tmp_path / "Local"
    exe = local / "Programs" / "ai.opencode.desktop" / "resources" / "opencode-cli.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"stub")

    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr("aiolocal.code.shutil.which", lambda name: None)

    paths = Paths.create(tmp_path / "data")
    workspace = CodeWorkspace(paths, SettingsStore(paths), base_url="http://127.0.0.1:8000")
    assert workspace.executable() == str(exe)
