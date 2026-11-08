"""Model catalog, OVMS configuration, process control and downloads.

The app never bundles model weights. Each entry below is a Hugging Face repository the
user can pull explicitly; only catalog entries are allowed, which keeps the download
path from becoming an arbitrary-URL fetch.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlsplit

from .config import Paths
from .llm import api_root, api_url
from .security import valid_repo

# OpenVINO GenAI servables are served through a MediaPipe graph. OVMS can only load
# these models from a graph file; the older "task" config key is rejected by the
# 2026 schema. This template matches the graph OVMS itself writes for an LLM export.
GRAPH_TEMPLATE = """\
# File created with: OpenVINO Model Server
# OVMS_GRAPH_QUEUE_MAX_SIZE: AUTO

    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    node: {
    name: "LLMExecutor"
    calculator: "HttpLLMCalculator"
    input_stream: "LOOPBACK:loopback"
    input_stream: "HTTP_REQUEST_PAYLOAD:input"
    input_side_packet: "LLM_NODE_RESOURCES:llm"
    input_side_packet: "LLM_NODE_EXECUTION_CONTEXTS:llm_ctx"
    output_stream: "LOOPBACK:loopback"
    output_stream: "HTTP_RESPONSE_PAYLOAD:output"
    input_stream_info: {
        tag_index: 'LOOPBACK:0',
        back_edge: true
    }
    node_options: {
        [type.googleapis.com / mediapipe.LLMCalculatorOptions]: {
            max_num_seqs:256,
            device: "GPU",
            models_path: "./",
            enable_prefix_caching: true,
            cache_size: 0,
        }
    }
    input_stream_handler {
        input_stream_handler: "SyncSetInputStreamHandler",
        options {
        [mediapipe.SyncSetInputStreamHandlerOptions.ext] {
            sync_set {
            tag_index: "LOOPBACK:0"
            }
        }
        }
    }
    }
"""

WEIGHT_SUFFIXES = {".bin", ".xml", ".safetensors", ".gguf", ".onnx"}

# Fallbacks used only when ``ovms --configure`` cannot run. They mirror the graphs
# OVMS 2026 generates for the same tasks and keep a previously downloaded model
# usable while a dependency is being repaired.
EMBEDDINGS_GRAPH_TEMPLATE = """\
# File created with: OpenVINO Model Server
# OVMS_GRAPH_QUEUE_MAX_SIZE: AUTO

input_stream: "REQUEST_PAYLOAD:input"
output_stream: "RESPONSE_PAYLOAD:output"
node {
    name: "EmbeddingsExecutor"
    calculator: "EmbeddingsCalculatorOV"
    input_side_packet: "EMBEDDINGS_NODE_RESOURCES:embeddings_servable"
    input_stream: "REQUEST_PAYLOAD:input"
    output_stream: "RESPONSE_PAYLOAD:output"
    node_options: {
        [type.googleapis.com / mediapipe.EmbeddingsCalculatorOVOptions]: {
            models_path: ".",
            normalize_embeddings: true,
            truncate: false,
            pooling: LAST,
            target_device: "CPU",
            plugin_config: '{"NUM_STREAMS":"1"}',
        }
    }
}
"""

IMAGE_GEN_GRAPH_TEMPLATE = """\
# File created with: OpenVINO Model Server
# OVMS_GRAPH_QUEUE_MAX_SIZE: AUTO

input_stream: "HTTP_REQUEST_PAYLOAD:input"
output_stream: "HTTP_RESPONSE_PAYLOAD:output"

node: {
  name: "ImageGenExecutor"
  calculator: "ImageGenCalculator"
  input_stream: "HTTP_REQUEST_PAYLOAD:input"
  input_side_packet: "IMAGE_GEN_NODE_RESOURCES:pipes"
  output_stream: "HTTP_RESPONSE_PAYLOAD:output"
  node_options: {
      [type.googleapis.com / mediapipe.ImageGenCalculatorOptions]: {
          models_path: "."
          device: "GPU"
      }
  }
}
"""


def _human_bytes(value: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024 or unit == "TB":
            return f"{value:.0f} {unit}" if unit in ("B", "KB") else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"

CATALOG: list[dict] = [
    {
        "id": "qwen3.5-9b",
        "repo": "OpenVINO/Qwen3.5-9B-int4-ov",
        "name": "Qwen3.5 9B (int4)",
        "task": "text_generation",
        "devices": ["GPU", "CPU"],
        "default_device": "GPU",
        "context": 8192,
        "size_gb": 5.8,
        "license": "Apache-2.0",
        "role": "Main chat model",
        "description": "Best local quality on Arc graphics. Needs about 8 GB of free RAM while loaded.",
    },
    {
        "id": "qwen3-4b",
        "repo": "OpenVINO/Qwen3-4B-Instruct-2507-int4-cw-ov",
        "name": "Qwen3 4B Instruct (int4)",
        "task": "text_generation",
        "devices": ["GPU", "CPU", "NPU"],
        "default_device": "GPU",
        "context": 8192,
        "size_gb": 2.8,
        "license": "Apache-2.0",
        "role": "Faster chat / NPU fallback",
        "description": "Smaller and quicker. A good choice if 9B is too slow on the iGPU.",
    },
    {
        "id": "lfm2.5-1.2b",
        "repo": "OpenVINO/LFM2.5-1.2B-Instruct-int4-ov",
        "name": "LFM 2.5 1.2B (int4)",
        "task": "text_generation",
        "devices": ["NPU", "GPU", "CPU"],
        "default_device": "NPU",
        "context": 4096,
        "size_gb": 0.9,
        "license": "Check upstream",
        "role": "Memory filing on the NPU",
        "description": "Tiny model reserved for background memory filing so chat stays responsive.",
    },
    {
        "id": "qwen3-embedding-0.6b",
        "repo": "OpenVINO/Qwen3-Embedding-0.6B-int8-ov",
        "name": "Qwen3 Embedding 0.6B (int8)",
        "task": "embeddings",
        "devices": ["CPU", "GPU", "NPU"],
        "default_device": "CPU",
        "context": 32768,
        "size_gb": 0.7,
        "license": "Apache-2.0",
        "role": "Semantic memory search",
        "description": "Replaces the built-in hashed embeddings with real semantic vectors.",
    },
    {
        "id": "sd15",
        "repo": "OpenVINO/stable-diffusion-v1-5-int8-ov",
        "name": "Stable Diffusion 1.5 (int8)",
        "task": "image_generation",
        "devices": ["GPU", "CPU"],
        "default_device": "GPU",
        "context": 512,
        "size_gb": 2.2,
        "license": "CreativeML OpenRAIL-M",
        "role": "Image generation",
        "description": "512x512 image generation. Read the OpenRAIL-M use restrictions before use.",
    },
    {
        "id": "sd15-inpaint",
        "repo": "OpenVINO/dreamshaper-8-inpainting-int8-ov",
        "name": "DreamShaper 8 Inpainting (int8, SD 1.5)",
        "task": "image_generation",
        "devices": ["GPU", "CPU"],
        "default_device": "GPU",
        "context": 512,
        "size_gb": 2.2,
        "license": "CreativeML OpenRAIL-M",
        "role": "Image editing",
        "description": "SD 1.5-compatible inpainting/editing. Uses the DreamShaper 8 inpainting OpenVINO repo, which is the working replacement for the removed Intel namespace repo.",
    },
    {
        "id": "qwen3-vl-4b",
        "repo": "OpenVINO/Qwen3-VL-4B-Instruct-int4-ov",
        "name": "Qwen3-VL 4B (int4, vision)",
        "task": "text_generation",
        "devices": ["GPU", "CPU"],
        "default_device": "GPU",
        "context": 8192,
        "size_gb": 3.0,
        "license": "Apache-2.0",
        "role": "Vision chat / image understanding",
        "description": "Attach an image in Chat to describe, read, or reason about it. A standard text model ignores the image.",
        "vision": True,
    },
]

# Practical memory/spec guidance shown next to each model. These are guidance, not
# hard limits: OVMS reports the real failure if the machine is too small.
_MODEL_SPECS = {
    "qwen3.5-9b": (8, "Best on Arc GPU with about 8 GB free RAM; 16 GB system RAM recommended."),
    "qwen3-4b": (4, "About 4 GB free RAM; a good fit for 8 GB systems and slower iGPUs."),
    "lfm2.5-1.2b": (2, "About 1.5-2 GB free RAM; designed for NPU background work."),
    "qwen3-embedding-0.6b": (1, "About 1 GB free RAM; runs quietly on CPU or NPU."),
    "sd15": (4, "About 4 GB free RAM for 512x512; larger sizes need more."),
    "sd15-inpaint": (4, "About 4 GB free RAM for 512x512 editing; larger sizes need more."),
    "qwen3-vl-4b": (6, "About 5-6 GB free RAM while loaded; image prompts add some context cost."),
}
for _entry in CATALOG:
    _ram, _spec = _MODEL_SPECS.get(_entry["id"], (0, ""))
    _entry.setdefault("ram_gb", _ram)
    _entry.setdefault("spec", _spec)

_BY_ID = {m["id"]: m for m in CATALOG}


def catalog_entry(model_id: str) -> dict | None:
    return _BY_ID.get(model_id)


_DLL_DIRECTORY_LOCK = threading.Lock()


@contextmanager
def _clear_windows_dll_directory():
    """Temporarily remove the frozen app's DLL directory before spawning a child.

    PyInstaller calls ``SetDllDirectoryW(_internal)`` so its own bundled DLLs win.
    Windows passes this directory to child processes, which made OVMS load the
    app's ``python312.dll`` instead of its own. OVMS then skipped its
    ``python312._pth`` and failed with "failed to get the Python codec of the
    filesystem encoding". The directory is restored immediately after the child
    process is created.
    """
    if os.name != "nt":
        yield
        return

    import ctypes
    from ctypes import wintypes

    with _DLL_DIRECTORY_LOCK:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        get_directory = kernel32.GetDllDirectoryW
        get_directory.argtypes = (wintypes.DWORD, wintypes.LPWSTR)
        get_directory.restype = wintypes.DWORD
        set_directory = kernel32.SetDllDirectoryW
        set_directory.argtypes = (wintypes.LPCWSTR,)
        set_directory.restype = wintypes.BOOL

        buffer = ctypes.create_unicode_buffer(32768)
        length = get_directory(len(buffer), buffer)
        previous = buffer.value if length else None
        if not set_directory(None):
            yield
            return
        try:
            yield
        finally:
            set_directory(previous)


class ModelManager:
    def __init__(self, paths: Paths, settings_store, publish=None) -> None:
        self.paths = paths
        self.settings_store = settings_store
        self.state_path = paths.config_dir / "models.json"
        self._lock = threading.RLock()
        self._pull_threads: set[str] = set()
        self._pull_errors: dict[str, str] = {}
        self._pull_progress: dict[str, dict] = {}
        self._installed_cache: dict[str, tuple[float, bool]] = {}
        self._graph_errors: dict[str, str] = {}
        self._publish = publish
        self._process: subprocess.Popen | None = None
        self._log_handle = None
        self._log_offset = 0
        self._ovms_state = "stopped"  # stopped | starting | loading | ready | error
        self._ovms_error = ""
        self._ovms_stage = ""
        self._ovms_started_at = 0.0
        self._ovms_ready_at = 0.0
        self._ovms_port = 8000
        self._ovms_generation = 0
        self._watch_thread: threading.Thread | None = None
        self._state = self._load_state()

    # ------------------------------------------------------------------ state
    def _load_state(self) -> dict:
        try:
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("active"), dict):
                return {"active": raw["active"]}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
        return {"active": {}}

    def _save_state(self) -> None:
        tmp = self.state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2), encoding="utf-8")
        os.replace(tmp, self.state_path)

    def model_dir(self, entry: dict) -> Path:
        name = entry["repo"].split("/", 1)[-1]
        return self.models_root() / name

    def models_root(self) -> Path:
        settings = self.settings_store.load()
        if settings.models_dir:
            return Path(settings.models_dir).expanduser()
        return self.paths.models_dir

    @property
    def config_path(self) -> Path:
        return self.models_root() / "config.json"

    def installed(self, entry: dict) -> bool:
        """A model is installed only when its metadata *and* weights are complete.

        ``snapshot_download`` writes into the target folder as it goes, so an
        interrupted download leaves a non-empty directory. Treating that as installed
        is what allowed OVMS to be started with half a model, so require a config file
        plus at least one complete weight file outside the Hugging Face cache.
        """
        now = time.time()
        cached = self._installed_cache.get(entry["repo"])
        if cached and now - cached[0] < 5.0:
            return cached[1]
        result = self._installed_uncached(entry)
        self._installed_cache[entry["repo"]] = (now, result)
        return result

    def _installed_uncached(self, entry: dict) -> bool:
        directory = self.model_dir(entry)
        if not directory.is_dir():
            return False
        has_config = any(
            (directory / name).is_file()
            for name in ("config.json", "openvino_config.json", "model_index.json", "graph.pbtxt")
        )
        if not has_config:
            return False
        weight_bytes = 0
        try:
            if any(directory.rglob("*.incomplete")):
                return False
        except OSError:
            return False
        try:
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                if ".cache" in path.parts or path.suffix.lower() not in WEIGHT_SUFFIXES:
                    continue
                weight_bytes += path.stat().st_size
                if weight_bytes > 1_000_000:
                    return True
        except OSError:
            return False
        return False

    def pull_active(self, model_id: str) -> bool:
        return model_id in self._pull_threads

    def pull_progress(self, model_id: str) -> dict | None:
        return self._pull_progress.get(model_id)

    # ------------------------------------------------------------------ config
    def _graph_source(self, entry: dict) -> Path | None:
        directory = self.model_dir(entry)
        preferred = directory / "graph.pbtxt"
        if preferred.is_file():
            return preferred
        for candidate in sorted(directory.glob("graph*.pbtxt")):
            if candidate.is_file():
                return candidate
        return None

    def _graph_for(self, entry: dict, device: str) -> Path | None:
        """Return a graph file whose device line matches ``device``.

        OVMS 2026 loads GenAI models through ``graph_path``. The exported graph often
        hard-codes ``device: "GPU"``, so changing the device in the UI used to have no
        effect. A per-device copy changes the graph path too, which also guarantees
        that OVMS notices the change on a config reload.
        """
        directory = self.model_dir(entry)
        source = self._graph_source(entry)
        if source is None:
            return None
        text = source.read_text(encoding="utf-8", errors="replace")
        patched, replacements = re.subn(
            r'(device\s*:\s*")[^"]*(")', lambda m: m.group(1) + device + m.group(2), text, count=1
        )
        if replacements == 0:
            return source
        target = directory / f"graph-{device}.pbtxt"
        try:
            if not target.is_file() or target.read_text(encoding="utf-8", errors="replace") != patched:
                target.write_text(patched, encoding="utf-8")
        except OSError:
            return source
        return target

    def _generate_graph(self, entry: dict, device: str) -> Path:
        """Run OVMS's own graph generator for a model that has no graph yet."""
        exe = self._ovms_executable()
        directory = self.model_dir(entry)
        task = entry["task"]
        if exe is not None:
            args = [
                str(exe),
                "--configure",
                "--model_path", str(directory),
                "--task", task,
                "--target_device", device,
                "--log_level", "ERROR",
            ]
            if task == "embeddings":
                args += ["--pooling", "LAST"]
            try:
                proc = subprocess.run(
                    args,
                    cwd=str(exe.parent),
                    env=self._ovms_env(exe),
                    capture_output=True,
                    timeout=600,
                    creationflags=0x08000000 if os.name == "nt" else 0,
                )
                output = (proc.stdout or b"") + (proc.stderr or b"")
                detail = output.decode("utf-8", "replace").strip()[-800:]
                generated = directory / "graph.pbtxt"
                if proc.returncode == 0 and generated.is_file():
                    return generated
                if proc.returncode == 0:
                    detail = detail or "OVMS did not write graph.pbtxt"
            except (OSError, subprocess.SubprocessError) as exc:
                detail = f"{type(exc).__name__}: {exc}"
        else:
            detail = "OVMS is not installed"

        # Generate the known-good graph ourselves if the command is unavailable.
        template = {
            "text_generation": GRAPH_TEMPLATE,
            "embeddings": EMBEDDINGS_GRAPH_TEMPLATE,
            "image_generation": IMAGE_GEN_GRAPH_TEMPLATE,
        }.get(task)
        if template is None:
            raise ValueError(f"no graph template exists for the {task!r} task")
        target = directory / "graph.pbtxt"
        text = template.replace('device: "GPU"', f'device: "{device}"')
        text = text.replace('target_device: "CPU"', f'target_device: "{device}"')
        try:
            target.write_text(text, encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"could not create graph.pbtxt: {exc}") from exc
        if not target.is_file():
            raise ValueError(detail or "OVMS did not create a graph")
        return target

    def _ensure_graph(self, entry: dict, device: str) -> Path | None:
        graph = self._graph_for(entry, device)
        if graph is not None:
            return graph
        if entry["task"] not in ("text_generation", "embeddings", "image_generation"):
            return None
        generated = self._generate_graph(entry, device)
        graph = self._graph_for(entry, device)
        if graph is None:
            raise ValueError(
                f"OVMS created {generated.name} but not a usable graph for {entry['name']}"
            )
        return graph

    def write_config(self) -> Path:
        entries: list[dict] = []
        errors: dict[str, str] = {}
        with self._lock:
            active = dict(self._state.get("active") or {})
        for model_id, info in active.items():
            entry = _BY_ID.get(model_id)
            if not entry or not self.installed(entry):
                continue
            device = str(info.get("device") or entry["default_device"]).upper()
            config = {
                "name": model_id,
                "base_path": str(self.model_dir(entry)),
            }
            try:
                graph = self._ensure_graph(entry, device)
            except Exception as exc:
                errors[model_id] = str(exc)[:800]
                continue
            if graph is not None:
                config["graph_path"] = graph.name
            elif entry["task"] in ("text_generation", "embeddings", "image_generation"):
                errors[model_id] = (
                    f"{entry['name']} has no graph.pbtxt. Delete it in Models and download it again."
                )
                continue
            else:
                config["target_device"] = device
            entries.append({"config": config})
        self._graph_errors = errors
        payload = {"model_config_list": entries}
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.config_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, self.config_path)
        return self.config_path

    async def reload(self, upstream) -> tuple[bool, str]:
        try:
            import httpx

            async with httpx.AsyncClient(trust_env=False, timeout=15.0) as client:
                resp = await client.post(f"{api_root(upstream.base_url)}/v1/config/reload")
            if resp.status_code < 400:
                return True, "OVMS reloaded the configuration"
            return False, f"OVMS returned HTTP {resp.status_code}: {resp.text[:300]}"
        except Exception as exc:
            return False, f"OVMS is not reachable ({type(exc).__name__}). Start it, then reload."

    def load(self, model_id: str, device: str) -> dict:
        entry = _BY_ID.get(model_id)
        if not entry:
            raise ValueError("unknown model")
        if device not in entry["devices"]:
            raise ValueError(f"device must be one of {', '.join(entry['devices'])}")
        if self.pull_active(model_id):
            raise ValueError("model is still downloading")
        if not self.installed(entry):
            raise ValueError("model is not downloaded yet")
        try:
            graph = self._ensure_graph(entry, device)
        except Exception as exc:
            raise ValueError(str(exc)) from exc
        if graph is None and entry["task"] in ("text_generation", "embeddings", "image_generation"):
            raise ValueError(f"{entry['name']} has no usable graph.pbtxt")
        with self._lock:
            self._state.setdefault("active", {})[model_id] = {"device": device}
            self._save_state()
        self.write_config()
        return {"ok": True, "model": model_id, "device": device}

    def unload(self, model_id: str) -> dict:
        with self._lock:
            self._state.setdefault("active", {}).pop(model_id, None)
            self._save_state()
        self.write_config()
        return {"ok": True, "model": model_id}

    def status(self, upstream_health: dict | None = None) -> dict:
        active = dict(self._state.get("active") or {})
        models = []
        for entry in CATALOG:
            info = active.get(entry["id"]) or {}
            progress = self._pull_progress.get(entry["id"]) or {}
            models.append(
                {
                    **{k: entry[k] for k in ("id", "repo", "name", "task", "devices", "default_device",
                                             "context", "size_gb", "license", "role", "description",
                                             "ram_gb", "spec")},
                    "vision": bool(entry.get("vision", False)),
                    "installed": self.installed(entry),
                    "loaded": entry["id"] in active,
                    "graph_ready": self._graph_source(entry) is not None,
                    "graph_error": self._graph_errors.get(entry["id"], ""),
                    "device": info.get("device") or entry["default_device"],
                    "pulling": self.pull_active(entry["id"]),
                    "pull_percent": progress.get("percent"),
                    "pull_downloaded": progress.get("downloaded"),
                    "pull_total": progress.get("total"),
                    "pull_state": progress.get("state"),
                    "pull_error": self._pull_errors.get(entry["id"]),
                }
            )
        exe = self._ovms_executable()
        return {
            "models": models,
            "active": active,
            "models_dir": str(self.models_root()),
            "config_path": str(self.config_path),
            "config_exists": self.config_path.exists(),
            "ovms_running": self.ovms_running(),
            "ovms_pid": self._process.pid if self._process and self._process.poll() is None else None,
            "ovms_state": self._ovms_state,
            "ovms_ready": self._ovms_state == "ready",
            "ovms_stage": self._ovms_stage,
            "ovms_error": self._ovms_error,
            "ovms_log_tail": self._log_tail() if self._ovms_state in ("error", "starting", "loading") else [],
            "ovms_started_at": int(self._ovms_started_at * 1000) if self._ovms_started_at else 0,
            "ovms_elapsed_s": round(time.time() - self._ovms_started_at, 1) if self._ovms_started_at else 0,
            "ovms_port": self._ovms_port,
            "ovms_executable": str(exe) if exe else "",
            "ovms_detected": self._ovms_detected(),
            "config_errors": dict(self._graph_errors),
            "upstream": upstream_health or {},
        }

    # ------------------------------------------------------------------ downloads
    @staticmethod
    def _dir_size(directory: Path) -> int:
        total = 0
        try:
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                if ".cache" in path.parts:
                    continue
                total += path.stat().st_size
        except OSError:
            return total
        return total

    @staticmethod
    def _remote_size(repo_id: str) -> int:
        try:
            from huggingface_hub import HfApi

            info = HfApi().model_info(repo_id, files_metadata=True)
            sizes = [int(getattr(s, "size", 0) or 0) for s in (info.siblings or [])]
            return sum(sizes)
        except Exception:
            return 0

    def start_pull(self, model_id: str, publish=None) -> dict:
        entry = _BY_ID.get(model_id)
        if not entry:
            raise ValueError("unknown model")
        if not valid_repo(entry["repo"]):
            raise ValueError("catalog entry has an invalid repository name")
        if model_id in self._pull_threads:
            return {"ok": True, "already_running": True, "model": model_id}
        if self.installed(entry):
            return {"ok": True, "already_installed": True, "model": model_id}

        target = self.model_dir(entry)
        self.models_root().mkdir(parents=True, exist_ok=True)
        free_gb = shutil.disk_usage(self.models_root()).free / (1024 ** 3)
        if free_gb < float(entry["size_gb"]) + 2.0:
            raise ValueError(
                f"not enough free disk space: about {entry['size_gb']:.1f} GB is needed, "
                f"{free_gb:.1f} GB is free"
            )

        self._pull_errors.pop(model_id, None)
        self._pull_progress[model_id] = {
            "state": "starting",
            "percent": 0.0,
            "downloaded": self._dir_size(target) if target.exists() else 0,
            "total": max(1, int(float(entry["size_gb"]) * (1024 ** 3))),
            "started_at": time.time(),
        }

        def publish_progress(**extra) -> None:
            if not publish:
                return
            payload = {"model": model_id, **self._pull_progress.get(model_id, {}), **extra}
            publish("model_pull", payload)

        def worker() -> None:
            expected = self._remote_size(entry["repo"])
            if expected > 0:
                self._pull_progress[model_id]["total"] = expected
            stop = threading.Event()

            def monitor() -> None:
                while not stop.wait(1.5):
                    got = self._dir_size(target) if target.exists() else 0
                    total = max(1, int(self._pull_progress[model_id].get("total") or 1))
                    percent = min(99.0, round(got * 100.0 / total, 1))
                    self._pull_progress[model_id].update(
                        {"state": "running", "downloaded": got, "percent": percent}
                    )
                    publish_progress()

            monitor_thread = threading.Thread(
                target=monitor, name=f"pull-progress-{model_id}", daemon=True
            )
            monitor_thread.start()
            try:
                from huggingface_hub import snapshot_download

                self._pull_progress[model_id]["state"] = "running"
                publish_progress()
                snapshot_download(
                    repo_id=entry["repo"],
                    local_dir=str(target),
                )
                got = self._dir_size(target)
                self._pull_progress[model_id].update(
                    {"state": "done", "percent": 100.0, "downloaded": got}
                )
                self._pull_errors.pop(model_id, None)
                self._installed_cache.clear()
                publish_progress()
            except Exception as exc:
                self._pull_errors[model_id] = f"{type(exc).__name__}: {exc}"[:400]
                self._pull_progress[model_id]["state"] = "failed"
                publish_progress(error=self._pull_errors[model_id])
            finally:
                stop.set()
                monitor_thread.join(timeout=3)
                self._pull_threads.discard(model_id)

        self._pull_threads.add(model_id)
        threading.Thread(target=worker, name=f"pull-{model_id}", daemon=True).start()
        return {"ok": True, "started": True, "model": model_id}

    # ------------------------------------------------------------------ OVMS process
    def _ovms_detected(self) -> list[str]:
        """Common install locations, so "optional" is never shown for a required path."""
        candidates: list[str] = []
        for env_name in ("ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"):
            base = os.environ.get(env_name)
            if not base:
                continue
            for relative in ("OVMS/ovms.exe", "OpenVINO/OVMS/ovms.exe", "Intel/OVMS/ovms.exe"):
                path = Path(base) / relative
                if path.is_file():
                    candidates.append(str(path))
        for name in ("ovms", "ovms.exe"):
            found = shutil.which(name)
            if found:
                candidates.append(found)
        # Preserve order but drop duplicates.
        return list(dict.fromkeys(candidates))

    def _ovms_executable(self) -> Path | None:
        settings = self.settings_store.load()
        raw = (settings.ovms_path or "").strip().strip('"')
        if raw:
            path = Path(raw).expanduser()
            if path.is_dir():
                path = path / "ovms.exe"
            if path.is_file():
                return path
            # A stale configured path should not hide a valid installation elsewhere.
        for candidate in self._ovms_detected():
            path = Path(candidate)
            if path.is_file():
                return path
        return None

    def ovms_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _ovms_env(self, exe: Path) -> dict:
        """Reproduce OVMS's setupvars before launching it.

        The bundled Python backend needs OVMS's own PYTHONHOME. Without this, the
        frozen app's Python paths leak in and ``import pyovms`` fails, which is the
        exact failure shown in the old ``ovms.log``.
        """
        env = dict(os.environ)
        for key in (
            "PYTHONPATH",
            "PYTHONHOME",
            "PYTHONSTARTUP",
            "PYTHONUSERBASE",
            "VIRTUAL_ENV",
        ):
            env.pop(key, None)
        root = exe.parent
        python_dir = root / "python"
        path_parts = [str(root)]
        if python_dir.is_dir():
            env["PYTHONHOME"] = str(python_dir)
            env["PYTHONNOUSERSITE"] = "1"
            path_parts += [str(python_dir), str(python_dir / "Scripts")]
        espeak = root / "espeak-ng-data"
        if espeak.is_dir():
            env["ESPEAK_DATA_PATH"] = str(espeak)
        env["OVMS_DIR"] = str(root)
        env["PATH"] = os.pathsep.join(path_parts + [env.get("PATH", "")])
        return env

    def _log_tail(self, lines: int = 20, max_bytes: int = 16_384) -> list[str]:
        path = self.paths.log_dir / "ovms.log"
        try:
            with open(path, "rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - max_bytes))
                data = fh.read()
        except OSError:
            return []
        text = data.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")
        return [line for line in text.split("\n") if line.strip()][-lines:]

    def _emit_ovms(self) -> None:
        if not self._publish:
            return
        try:
            self._publish(
                "ovms_status",
                {
                    "state": self._ovms_state,
                    "stage": self._ovms_stage,
                    "error": self._ovms_error,
                    "ready": self._ovms_state == "ready",
                    "pid": self._process.pid if self.ovms_running() else None,
                    "port": self._ovms_port,
                },
            )
        except Exception:
            pass

    def _set_ovms_state(self, state: str, stage: str = "", error: str = "") -> None:
        self._ovms_state = state
        self._ovms_stage = stage
        if error:
            self._ovms_error = error
        if state == "ready":
            self._ovms_ready_at = time.time()
        self._emit_ovms()

    def _watch_ovms(self, proc: subprocess.Popen, port: int, generation: int, timeout: float) -> None:
        deadline = time.time() + timeout
        active_names = set((self._state.get("active") or {}).keys())
        self._set_ovms_state("starting", "Starting the OVMS process")
        while time.time() < deadline:
            if generation != self._ovms_generation:
                return
            if proc.poll() is not None:
                tail = "\n".join(self._log_tail(12))
                message = (
                    f"OVMS exited during startup (code {proc.returncode})."
                    + (f"\n{tail}" if tail else " Open Models and check logs\\ovms.log.")
                )
                self._set_ovms_state("error", "OVMS exited", message)
                return
            try:
                import httpx

                with httpx.Client(trust_env=False, timeout=2.0) as client:
                    response = client.get(f"http://127.0.0.1:{port}/v1/models")
                if response.status_code < 400:
                    payload = response.json()
                    ids = {
                        str(item.get("id"))
                        for item in (payload.get("data") or [])
                        if isinstance(item, dict)
                    }
                    if not active_names or (active_names & ids):
                        self._set_ovms_state("ready", "Models are loaded" if ids else "REST endpoint is up")
                        self._watch_after_ready(proc, generation)
                        return
                    waiting = ", ".join(sorted(active_names))
                    self._set_ovms_state("loading", f"Loading model(s): {waiting}")
            except Exception:
                pass
            time.sleep(1.5)

        tail = "\n".join(self._log_tail(12))
        message = (
            f"OVMS did not become ready within {int(timeout)} seconds."
            + (f"\n{tail}" if tail else " Check logs\\ovms.log for details.")
        )
        self._set_ovms_state("error", "Startup timed out", message)

    def _watch_after_ready(self, proc: subprocess.Popen, generation: int) -> None:
        """Keep the status honest if OVMS dies after it became ready."""
        while generation == self._ovms_generation and proc.poll() is None:
            time.sleep(2.0)
        if generation != self._ovms_generation:
            return
        tail = "\n".join(self._log_tail(12))
        self._set_ovms_state(
            "error",
            "OVMS stopped unexpectedly",
            f"OVMS exited (code {proc.returncode})." + (f"\n{tail}" if tail else ""),
        )

    def start_ovms(self) -> dict:
        if self.ovms_running():
            return {
                "ok": True,
                "already_running": True,
                "pid": self._process.pid,
                "state": self._ovms_state,
                "port": self._ovms_port,
            }
        exe = self._ovms_executable()
        if not exe:
            raise ValueError(
                "OVMS was not found. Install OpenVINO Model Server, then set the "
                "ovms.exe path in Settings (it is required for local models)."
            )
        if not (self._state.get("active") or {}):
            raise ValueError("Load at least one model before starting OVMS.")
        # Always rewrite: older builds wrote an invalid "task" field that OVMS 2026 rejects.
        self.write_config()
        config_errors = dict(self._graph_errors)
        try:
            configured = json.loads(self.config_path.read_text(encoding="utf-8"))
            if not (configured.get("model_config_list") or []):
                detail = "; ".join(f"{k}: {v}" for k, v in config_errors.items())
                raise ValueError(
                    "No model could be configured for OVMS."
                    + (f" {detail}" if detail else " Check that the model folders are complete.")
                )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, ValueError) and str(exc).startswith("No model"):
                raise
            raise ValueError(f"Could not read the OVMS config: {exc}") from exc
        settings = self.settings_store.load()
        port = 8000
        try:
            parts = urlsplit(settings.upstream_base_url)
            if parts.port:
                port = parts.port
        except ValueError:
            pass
        args = [
            str(exe),
            "--config_path", str(self.config_path),
            "--rest_port", str(port),
            "--log_level", "INFO",
        ]
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.paths.log_dir / "ovms.log"
        try:
            self._log_offset = log_path.stat().st_size
        except OSError:
            self._log_offset = 0
        self._log_handle = open(log_path, "ab")
        creationflags = 0x08000000 if os.name == "nt" else 0  # CREATE_NO_WINDOW
        child_env = self._ovms_env(exe)
        try:
            with self._lock, _clear_windows_dll_directory():
                self._process = subprocess.Popen(
                    args,
                    cwd=str(exe.parent),
                    env=child_env,
                    stdout=self._log_handle,
                    stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
        except OSError:
            try:
                self._log_handle.close()
            except OSError:
                pass
            self._log_handle = None
            raise
        self._ovms_port = port
        self._ovms_started_at = time.time()
        self._ovms_ready_at = 0.0
        self._ovms_error = ""
        self._ovms_generation += 1
        generation = self._ovms_generation
        self._set_ovms_state("starting", "Starting the OVMS process")
        self._watch_thread = threading.Thread(
            target=self._watch_ovms,
            args=(self._process, port, generation, 300.0),
            name="ovms-startup",
            daemon=True,
        )
        self._watch_thread.start()
        return {
            "ok": True,
            "pid": self._process.pid,
            "port": port,
            "state": self._ovms_state,
            "executable": str(exe),
            "warnings": config_errors,
        }

    def stop_ovms(self) -> dict:
        self._ovms_generation += 1
        proc = self._process
        if not proc or proc.poll() is not None:
            self._set_ovms_state("stopped", "Stopped")
            return {"ok": True, "already_stopped": True}
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
        finally:
            if self._log_handle:
                try:
                    self._log_handle.close()
                except OSError:
                    pass
                self._log_handle = None
        self._set_ovms_state("stopped", "Stopped")
        return {"ok": True, "stopped": True}
