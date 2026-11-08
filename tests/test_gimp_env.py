from __future__ import annotations

from pathlib import Path

from aiolocal.config import Paths, Settings, SettingsStore
from aiolocal.gimp import GimpDock


def test_gimp_environment_stays_in_app_data(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    dock = GimpDock(paths, store)
    env = dock._env()
    assert env["GIMP3_DIRECTORY"] == str(paths.data_dir / "gimp" / "3.0")
    assert env["GIMP_OPENVINO_MODELS_PATH"] == str(paths.data_dir / "gimp" / "openvino-models")
    assert env["GIMP3_PLUGIN_DIR"] == str(paths.data_dir / "gimp" / "3.0" / "plug-ins")
    assert env["GIMP3_TEMP_DIR"] == str(paths.data_dir / "gimp" / "temp")


def test_gimp_plugin_detection_uses_private_profile(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    dock = GimpDock(paths, store)
    assert dock.plugin_installed() is False
    config = paths.data_dir / "gimp" / "3.0" / "plug-ins" / "openvino_utils" / "tools" / "gimp_openvino_config.json"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("{}", encoding="utf-8")
    assert dock.plugin_installed() is True
