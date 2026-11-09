from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest

from aiolocal.config import Paths, Settings, SettingsStore
from aiolocal import photogimp
from aiolocal.gimp import GimpDock
from aiolocal.photogimp import PhotoGIMP, PhotoGimpError


def make_pack(files: dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return buffer.getvalue()


@pytest.fixture()
def installer(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(photogimp.os, "name", "nt")
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    dock = GimpDock(paths, store)
    manager = PhotoGIMP(paths, dock)
    pack = make_pack(
        {
            "3.0/theme.css": b"photogimp theme",
            "3.0/images/logo.png": b"png-bytes",
            "3.0/__MACOSX/ignored": b"junk",
        }
    )
    monkeypatch.setattr(manager, "_latest_release", lambda: ("9.9", "https://github.com/Diolinux/PhotoGIMP/releases/download/9.9/PhotoGIMP.zip"))
    monkeypatch.setattr(manager, "_download", lambda url: pack)
    return manager, dock


def test_install_backs_up_and_remove_restores(installer):
    manager, dock = installer
    profile = dock.personal_dir()
    (profile / "theme.css").write_text("original theme", encoding="utf-8")

    result = manager.install()

    assert result["ok"] is True
    assert result["installed_files"] == 2
    assert result["backed_up"] == 1
    assert (profile / "theme.css").read_text(encoding="utf-8") == "photogimp theme"
    assert (profile / "images" / "logo.png").read_bytes() == b"png-bytes"
    manifest = json.loads((manager.manifest_path).read_text(encoding="utf-8"))
    assert manifest["version"] == "9.9"
    assert manifest["license"] == "GPL-3.0"
    assert "theme.css" in manifest["files"]

    status = manager.status()
    assert status["installed"] is True
    assert status["files_present"] == 2

    removed = manager.remove()

    assert removed["ok"] is True
    assert removed["removed"] == 2
    assert removed["restored"] == 1
    assert (profile / "theme.css").read_text(encoding="utf-8") == "original theme"
    assert not (profile / "images" / "logo.png").exists()
    assert not manager.manifest_path.exists()
    assert manager.status()["installed"] is False


def test_install_rejects_unsafe_archive_entries(installer):
    manager, dock = installer
    manager._download = lambda url: make_pack({"../escape.txt": b"bad"})

    result = manager.install()

    assert result["installed_files"] == 0
    assert not (dock.personal_dir().parent / "escape.txt").exists()
    assert not (dock.personal_dir().parent.parent / "escape.txt").exists()


def test_remove_without_manifest_is_a_no_op(installer):
    manager, _ = installer
    result = manager.remove()
    assert result["ok"] is False


def test_safe_relative_paths():
    manager = PhotoGIMP.__new__(PhotoGIMP)
    assert str(manager._safe_relative("3.0/theme.css")) == "theme.css"
    assert str(manager._safe_relative("theme.css")) == "theme.css"
    assert manager._safe_relative("../evil") is None
    assert manager._safe_relative("/etc/passwd") is None
    assert manager._safe_relative("__MACOSX/meta") is None


def test_install_is_windows_only(monkeypatch, installer):
    manager, _ = installer
    monkeypatch.setattr(photogimp.os, "name", "posix")
    with pytest.raises(PhotoGimpError, match="Windows-only"):
        manager.install()
