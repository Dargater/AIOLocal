from __future__ import annotations

import json
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


def test_sanitize_plugin_config_removes_url_entries(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    dock = GimpDock(paths, store)
    config = dock.fastsd_config_path()
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(
        json.dumps(
            {
                "device_name": "GPU",
                "models": [
                    "rupeshs/sd-turbo-openvino",
                    "http://localhost:8000/v3/",
                    "rupeshs/sd-turbo-openvino",
                    "http://localhost:8000/v3/models/sd15-inpaint",
                ],
            }
        ),
        encoding="utf-8",
    )

    first = dock.sanitize_plugin_config()

    assert first["ok"] is True
    assert first["removed"] == [
        "http://localhost:8000/v3/",
        "http://localhost:8000/v3/models/sd15-inpaint",
    ]
    assert config.with_suffix(".json.bak").is_file()
    saved = json.loads(config.read_text(encoding="utf-8"))
    assert saved["device_name"] == "GPU"
    assert saved["models"] == ["rupeshs/sd-turbo-openvino"]

    again = dock.sanitize_plugin_config()
    assert again["ok"] is True
    assert again["removed"] == []


def test_repair_cleans_partial_weights_and_reports_issues(tmp_path: Path):
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings())
    dock = GimpDock(paths, store)
    partial = dock.plugin_models_dir() / "weights" / "sd15" / "model.bin.part"
    partial.parent.mkdir(parents=True, exist_ok=True)
    partial.write_bytes(b"half")

    result = dock.repair()

    assert result["ok"] is True
    assert not partial.exists()
    assert str(partial) in result["cleaned"]
    assert result["weights"] == ["sd15"]
    assert result["config"]["ok"] is False  # no config yet, but repair still succeeds
    assert any("plugin is not installed" in issue for issue in result["issues"])
