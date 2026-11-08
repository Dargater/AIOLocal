from __future__ import annotations

from pathlib import Path

from aiolocal.config import Paths, Settings, SettingsStore
from aiolocal.reverse import ReverseEngineering


def _stub(path: Path, text: str = "stub") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_rea_uses_portable_node_and_npx_cli(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings(rea_version="1.2.3"))
    node = _stub(paths.tools_dir / "node" / "node-v24.22.2-win-x64" / "node.exe")
    cli = _stub(
        paths.tools_dir
        / "node"
        / "node-v24.22.2-win-x64"
        / "node_modules"
        / "npm"
        / "bin"
        / "npx-cli.js"
    )
    reverse = ReverseEngineering(store, paths)
    command, display = reverse._command()
    assert command == [str(node), str(cli), "-y", "rea-agents@1.2.3", "mcp"]
    assert display == "npx -y rea-agents@1.2.3 mcp"
    assert reverse._portable_node() == node


def test_rea_status_marks_portable_node(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    _stub(paths.tools_dir / "node" / "node-v24.22.2-win-x64" / "node.exe")
    reverse = ReverseEngineering(store, paths)
    status = reverse.status()
    assert status["node_portable"] is True
    assert status["node_path"].endswith("node.exe")


def test_rea_prefers_app_node_over_path_node(tmp_path: Path, monkeypatch):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    portable = _stub(paths.tools_dir / "node" / "node-v24.22.2-win-x64" / "node.exe")
    monkeypatch.setattr(
        "aiolocal.reverse.shutil.which",
        lambda name: str(tmp_path / "system-node.exe") if name in ("node", "node.exe") else None,
    )
    reverse = ReverseEngineering(store, paths)
    assert reverse._node_executable() == str(portable)
