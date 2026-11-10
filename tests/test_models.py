from __future__ import annotations

import json
from pathlib import Path

from aiolocal.config import Paths, Settings, SettingsStore
from aiolocal.models import CATALOG, GRAPH_TEMPLATE, ModelManager


def make_manager(tmp_path: Path) -> ModelManager:
    paths = Paths.create(tmp_path / "data")
    store = SettingsStore(paths)
    store.save(Settings(models_dir=str(tmp_path / "models")))
    return ModelManager(paths, store)


def model_dir(manager: ModelManager) -> Path:
    entry = next(m for m in CATALOG if m["id"] == "qwen3.5-9b")
    directory = manager.model_dir(entry)
    directory.mkdir(parents=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    (directory / "graph.pbtxt").write_text(GRAPH_TEMPLATE, encoding="utf-8")
    return directory


def test_generated_config_uses_graph_path_and_devices(tmp_path: Path):
    manager = make_manager(tmp_path)
    directory = model_dir(manager)
    (directory / "openvino_language_model.bin").write_bytes(b"0" * 2_000_000)

    manager.load("qwen3.5-9b", "CPU")
    payload = json.loads(manager.config_path.read_text(encoding="utf-8"))
    config = payload["model_config_list"][0]["config"]

    assert config["name"] == "qwen3.5-9b"
    assert config["graph_path"] == "graph-CPU.pbtxt"
    assert "task" not in config
    assert "plugin_config" not in config
    graph = (directory / "graph-CPU.pbtxt").read_text(encoding="utf-8")
    assert 'device: "CPU"' in graph
    # The original graph is left untouched.
    assert 'device: "GPU"' in (directory / "graph.pbtxt").read_text(encoding="utf-8")

    manager.load("qwen3.5-9b", "GPU")
    payload = json.loads(manager.config_path.read_text(encoding="utf-8"))
    assert payload["model_config_list"][0]["config"]["graph_path"] == "graph-GPU.pbtxt"


def test_installed_rejects_partial_downloads(tmp_path: Path):
    manager = make_manager(tmp_path)
    directory = model_dir(manager)
    entry = {"repo": "OpenVINO/Qwen3.5-9B-int4-ov"}

    # config.json but no weights is a partial download, not an installed model.
    assert manager.installed(entry) is False

    (directory / "openvino_language_model.bin").write_bytes(b"0" * 2_000_000)
    manager._installed_cache.clear()  # what a completed pull does
    assert manager.installed(entry) is True

    # Files inside the Hugging Face cache do not count as model weights.
    cached = directory / ".cache" / "huggingface" / "download"
    cached.mkdir(parents=True)
    (cached / "partial.bin").write_bytes(b"0" * 3_000_000)
    (directory / "openvino_language_model.bin").unlink()
    manager._installed_cache.clear()
    assert manager.installed(entry) is False


def test_installed_check_is_cached_briefly(tmp_path: Path):
    manager = make_manager(tmp_path)
    directory = model_dir(manager)
    entry = {"repo": "OpenVINO/Qwen3.5-9B-int4-ov"}
    assert manager.installed(entry) is False

    # A direct folder change is not seen until the short cache expires; this keeps
    # the 1.5 s status polling from re-walking every model folder each time.
    (directory / "openvino_language_model.bin").write_bytes(b"0" * 2_000_000)
    assert manager.installed(entry) is False
    manager._installed_cache[entry["repo"]] = (0.0, False)
    assert manager.installed(entry) is True


def test_ovms_environment_is_sanitised(tmp_path: Path, monkeypatch):
    manager = make_manager(tmp_path)
    root = tmp_path / "OVMS"
    (root / "python" / "Scripts").mkdir(parents=True)
    (root / "espeak-ng-data").mkdir()
    exe = root / "ovms.exe"
    exe.write_bytes(b"stub")

    monkeypatch.setenv("PYTHONPATH", r"C:\polluted")
    monkeypatch.setenv("PYTHONHOME", r"C:\wrong-python")
    env = manager._ovms_env(exe)

    assert "PYTHONPATH" not in env
    assert env["PYTHONHOME"] == str(root / "python")
    assert env["ESPEAK_DATA_PATH"] == str(root / "espeak-ng-data")
    assert env["OVMS_DIR"] == str(root)
    assert env["PATH"].startswith(str(root))
    assert env["PYTHONNOUSERSITE"] == "1"


def test_ovms_executable_auto_detection(tmp_path: Path, monkeypatch):
    manager = make_manager(tmp_path)
    exe = tmp_path / "ovms.exe"
    exe.write_bytes(b"stub")
    monkeypatch.setattr(manager, "_ovms_detected", lambda: [str(exe)])
    assert manager._ovms_executable() == exe

def test_lfm_defaults_to_cpu_with_npu_fallback():
    from aiolocal.models import catalog_entry

    entry = catalog_entry("lfm2.5-1.2b")
    assert entry is not None
    assert entry["default_device"] == "CPU"
    assert entry["devices"][0] == "CPU"
    assert "CPU" in entry["fallback_devices"]
