from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from aiolocal.config import Paths, SettingsStore
from aiolocal.requirements import RequirementsManager, _safe_extract


def _stub(path: Path, text: str = "stub") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _manager(tmp_path: Path) -> tuple[RequirementsManager, Paths]:
    paths = Paths.create(tmp_path / "data")
    return RequirementsManager(paths, SettingsStore(paths)), paths


def test_missing_components_are_reported(tmp_path: Path, monkeypatch):
    # Do not let a real system install make the test machine look complete.
    monkeypatch.setattr("aiolocal.requirements.shutil.which", lambda name: None)
    monkeypatch.setattr("aiolocal.code.shutil.which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "no-local-app-data"))
    monkeypatch.setenv("ProgramFiles", str(tmp_path / "no-program-files"))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path / "no-program-files-x86"))
    monkeypatch.setenv("ProgramW6432", str(tmp_path / "no-program-files-64"))
    manager, _ = _manager(tmp_path)
    summary = manager.summary()
    components = {component["id"]: component for component in summary["components"]}
    missing = {k for k, v in components.items() if not v["installed"]}
    assert summary["components"]
    assert summary["missing"] == len(missing)
    assert summary["ready"] is False
    assert {"ovms", "node", "python", "gimp_plugin"} <= missing


def test_portable_components_are_detected(tmp_path: Path, monkeypatch):
    manager, paths = _manager(tmp_path)
    # The stub python.exe cannot execute; treat its pip as healthy for this test.
    monkeypatch.setattr(manager, "_pip_works", lambda path: True)
    tools = paths.tools_dir

    _stub(tools / "node" / "node-v24.22.2-win-x64" / "node.exe")
    _stub(
        tools / "node" / "node-v24.22.2-win-x64" / "node_modules" / "npm" / "bin" / "npx-cli.js"
    )
    _stub(tools / "opencode" / "opencode.exe")
    _stub(tools / "python" / "python.exe")
    _stub(tools / "gimp" / "bin" / "gimp-3.0.exe")
    _stub(tools / "ovms" / "ovms.exe")
    plugin = paths.data_dir / "gimp" / "3.0" / "plug-ins" / "openvino_utils"
    _stub(plugin / "tools" / "gimp_openvino_config.json", "{}")
    _stub(tools / "gimp-plugin" / "installed.json", '{"version": "3.3.0"}')

    summary = manager.summary()
    components = {component["id"]: component for component in summary["components"]}
    assert components["node"]["installed"] is True
    assert components["opencode"]["installed"] is True
    assert components["python"]["installed"] is True
    assert components["gimp"]["installed"] is True
    assert components["ovms"]["installed"] is True
    assert components["gimp_plugin"]["installed"] is True
    assert components["gimp_plugin"]["detected_version"] == "3.3.0"
    assert summary["ready"] is True


def test_private_python_never_falls_back_to_system_python(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "aiolocal.requirements.shutil.which",
        lambda name: str(tmp_path / "system-python.exe") if name in ("python", "python.exe") else None,
    )
    manager, _ = _manager(tmp_path)
    detected = manager._detect("python")
    assert detected["installed"] is False


def test_private_python_without_pip_is_not_ready(tmp_path: Path):
    manager, paths = _manager(tmp_path)
    exe = _stub(paths.tools_dir / "python" / "python.exe")
    manager._pip_works = lambda path: False  # type: ignore[method-assign]
    detected = manager._detect("python")
    assert detected["installed"] is False
    assert detected["path"] == str(exe)
    assert "pip missing" in detected["detected_version"]


def test_private_python_with_pip_is_ready(tmp_path: Path):
    manager, paths = _manager(tmp_path)
    _stub(paths.tools_dir / "python" / "python.exe")
    manager._pip_works = lambda path: True  # type: ignore[method-assign]
    assert manager._detect("python")["installed"] is True


def test_pip_bootstrap_runs_get_pip_directly(tmp_path: Path):
    manager, paths = _manager(tmp_path)
    exe = _stub(paths.tools_dir / "python" / "python.exe")
    get_pip = _stub(paths.tools_dir / "downloads" / "get-pip.py", "x" * 100_001)
    commands: list[list[str]] = []
    checks = iter([False, True])

    manager._pip_works = lambda path: next(checks)  # type: ignore[method-assign]
    manager._run_process = lambda component, command, message, **kwargs: commands.append(command)  # type: ignore[method-assign]
    manager._run_pip = lambda component, python, args, message, timeout=3600: commands.append(  # type: ignore[method-assign]
        [str(python), "-m", "pip", *args]
    )

    manager._ensure_pip("python", exe)
    assert commands[0] == [str(exe), str(get_pip)]
    assert "-m" not in commands[0]
    assert commands[1][:4] == [str(exe), "-m", "pip", "install"]


def test_safe_extract_rejects_zip_slip(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "no")
    with pytest.raises(ValueError):
        _safe_extract(archive, tmp_path / "target")
    assert not (tmp_path / "escaped.txt").exists()


def test_safe_extract_handles_normal_archive(tmp_path: Path):
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("folder/file.txt", "hello")
    target = tmp_path / "target"
    _safe_extract(archive, target)
    assert (target / "folder" / "file.txt").read_text(encoding="utf-8") == "hello"
