# 03 — Models, Devices and Memory Budget

## 1. The machine

| Component | Spec | Consequence |
|---|---|---|
| CPU | Core Ultra 5 225H (Arrow Lake-H), 14 cores (4 P + 8 E + 2 LP-E) | good for embeddings, Docling layout, audio post |
| iGPU | Arc 130T, ~7 Xe cores, shared system RAM | "VRAM" is RAM; no separate budget |
| NPU | ~13 TOPS (INT8), driver via OpenVINO | one request at a time, `int4-cw` only; prompt cap is measured in Phase 0 (2 k–16 k) |
| RAM | 16 GB LPDDR5 | the binding constraint |
| Disk | ~40 GB free | models ≈ 20 GB, leave 10 GB headroom |

**Reality check.** 16 GB shared means a 9 B int4 model plus a diffusion model plus Windows is not
simultaneously possible. Any plan that pretends otherwise fails at the first image generation. The
governor (`01` §4) is not optional.

*Exact core/TOPS numbers are re-probed at runtime and shown in Settings; the plan does not depend on
marketing figures.*

---

## 2. Memory budget (16 GB)

| Bucket | Typical RSS |
|---|---|
| Windows 11 + Defender + shell | 3.0 – 4.0 GB |
| AIO app (UI shell, control plane, gateway, SQLite) | 0.8 – 1.2 GB |
| Embedding model (CPU, int8, 0.6 B) | 0.7 – 0.9 GB |
| Small NPU model (1.2 B int4-cw) | 0.9 – 1.3 GB |
| **Baseline resident** | **5.5 – 7.5 GB** |
| + Qwen3.5-9B int4 on GPU (weights + runtime + 8 k KV) | +7.2 – 8.2 GB |
| + SD 1.5 int8 on GPU (peak during denoise) | +3.5 – 4.5 GB |
| + Whisper small int8 | +0.6 – 0.9 GB |
| + OpenCode server (Node) | +0.25 – 0.5 GB |
| + GIMP 3 + OpenVINO plugin (embedded dock) | +1.2 – 2.5 GB |
| + REA (Node) + Ghidra (Java) | +2.0 – 3.0 GB |

**Verdict**

- *Chat/Code/RE baseline*: 13 – 15 GB → workable but tight. In RE mode the small NPU model may need
  unloading if Ghidra is running.
- *Image mode*: 9 B **must** be unloaded (5.5 + 4.0 = 9.5 GB, leaving ~5 GB for Windows — still tight,
  so the 9 B is evicted and the SD pipeline gets the full GPU).
- *GIMP mode*: GIMP + the OpenVINO plugin + the 9 B is a three-way squeeze. The governor offers to
  unload the 9 B or to run the plugin on CPU; it does not start all three silently.
- *Audio*: co-exists with chat.
- An 8 GB page file is configured; the governor treats commit charge, not just RSS, as the metric.

---

## 3. Model matrix

### 3.1 Main brain

| Field | Value |
|---|---|
| HF repo | `OpenVINO/Qwen3.5-9B-int4-ov` (verified present, Apache-2.0) |
| Task | `text_generation`, VLM capable (vision encoder + merger present in the repo) |
| Device | **GPU** (iGPU) |
| Why not NPU | NPU is int4-cw only and serialises requests; a 9 B model with 8 k context is not viable there |
| Why not CPU | ~1–2 tok/s; unusable for chat or coding |
| Tool parser | auto-detected since OVMS 2026.3 (`hermes3` for Qwen); set explicitly as a fallback |
| Reasoning parser | `qwen3` (OVMS 2026.4 also honours top-level `reasoning_effort`) |
| Tool reliability | `--enable_tool_guided_generation true` constrains output to the tool schema |
| Context | start at 8 192; measure 16 384 |
| Notes | vision tower is used for Docling image pages and image Q&A |

### 3.2 Memory-filing model (NPU)

| Candidate | Size | Verdict |
|---|---|---|
| `OpenVINO/LFM2.5-1.2B-Instruct-int4-cw-ov` | ~0.9 GB | **primary** — fast, good at short JSON, fits a 2 k prompt comfortably |
| `OpenVINO/Qwen3-4B-Instruct-2507-int4-cw-ov` | ~2.4 GB | fallback if filing quality is poor; slower |
| `OpenVINO/SmolLM3-3B-int4-cw-ov` | ~1.8 GB | alternative |

Must be an `int4-cw` export (`--sym --ratio 1.0 --group-size -1`) — the NPU rejects others.

**Do not hardcode a 2 000-token ceiling.** The NPU demo uses `--max_prompt_len 2000`, but Intel's
own VSCode assistant recipe runs `Qwen3-8B-int4-cw` on NPU with `--max_prompt_len 16384` plus
`NPUW_LLM_PREFILL_ATTENTION_HINT: PYRAMID`. Phase 0 measures 2 k / 8 k / 16 k on the chosen model
and records the largest value that stays stable; the gateway then enforces *that* number.

### 3.3 Embeddings

`OpenVINO/Qwen3-Embedding-0.6B-int8-ov` — CPU, always resident, ~25 ms per query.
Alternative if multilingual quality is insufficient: `BAAI/bge-m3` int8 (larger, ~1.2 GB).
Dimension is stored per model in `embeddings.model` so switching models does not corrupt the index
(re-embed job runs in the background, old vectors kept until complete).

Optional reranker (Phase 6+): `BAAI/bge-reranker-v2-m3` int8 on CPU, top-20 only, +40 ms.
Off by default because retrieval already meets the latency budget.

### 3.4 Image

| Purpose | Model | Device | Size |
|---|---|---|---|
| Text→image | `OpenVINO/stable-diffusion-v1-5-int8-ov` | GPU | ~2.0 GB |
| Inpaint / outpaint | `Intel/stable-diffusion-v1-5-inpainting-openvino` | GPU | ~2.0 GB |
| Optional faster | `rupeshs/sd-turbo-openvino` or LCM | GPU | ~2.0 GB |
| Optional upscale | `OpenVINO/super-resolution-model` (or GIMP plugin) | GPU/CPU | ~0.1 GB |

Not included: SDXL / SD3.5 / FLUX — 4–12 GB weights, unusable at 16 GB shared.
Generation profile: 512×512, 20 steps, batch 1, ≈ 25–45 s on Arc 130T. Documented in the UI so
expectations are correct.

### 3.5 Audio

| Purpose | Model | Device |
|---|---|---|
| ASR + translation | Whisper base/small int8 (`OpenVINO/whisper-small-int8-ov` class) | GPU or CPU |
| TTS | Kokoro-82M (`OpenVINO/Kokoro-82M-int8-ov` class) | CPU (GPU if free) |

Streaming transcription is supported by OVMS (`audio/transcriptions` with stream). Kokoro's
`language` field selects phonemisation; only supported languages are exposed in the UI.

### 3.6 Document parsing

Docling layout/table models on CPU; the VLM pipeline points at the main Qwen through the gateway.
This keeps one copy of the 9 B in memory instead of a second VLM.

### 3.7 Video (v2, not in v1)

The earlier assumption that OpenVINO has no text-to-video path was wrong. As of OpenVINO GenAI 2026.1:

- `openvino_genai.Text2VideoPipeline` supports **LTX-Video** (`Lightricks/LTX-Video`) and the
  LTX-2 family, including image-to-video;
- export is `optimum-cli export openvino --model Lightricks/LTX-Video --task text-to-video
  --weight-format int8`;
- Stable Video Diffusion image-to-video is also documented, with AnimateLCM for speed.

The catch: **OVMS has no video endpoint**, so this cannot be served the way chat/image/audio are. v2
adds a separate `video-worker` process that imports `openvino_genai`, owns the GPU exclusively, and
exposes a small local API to the control plane. Disk and RAM cost is significant (multi-GB weights,
720p generation is slow on an Arc 130T), so it stays out of v1 and out of the 40 GB model budget.

Licence check for LTX-Video weights is required before shipping v2.

---

## 4. OVMS configuration

One process, models added/removed by editing `ovms-config.json` and calling
`POST /v1/config/reload`. Illustrative shape:

```json
{
  "model_config_list": [
    {
      "config": {
        "name": "qwen3.5-9b",
        "base_path": "OpenVINO/Qwen3.5-9B-int4-ov",
        "target_device": "GPU",
        "plugin_config": {
          "NUM_STREAMS": "1",
          "PERFORMANCE_HINT": "LATENCY"
        }
      }
    },
    {
      "config": {
        "name": "lfm2.5-1.2b",
        "base_path": "OpenVINO/LFM2.5-1.2B-Instruct-int4-cw-ov",
        "target_device": "NPU",
        "plugin_config": { "MAX_PROMPT_LEN": "2000", "NPUW_DEVICES": "NPU" }
      }
    },
    {
      "config": {
        "name": "qwen3-embedding-0.6b",
        "base_path": "OpenVINO/Qwen3-Embedding-0.6B-int8-ov",
        "target_device": "CPU"
      }
    }
  ]
}
```

Plus the generative pipeline settings OVMS requires (`task`, `max_num_seqs`, `cache_size`,
`tool_parser`, `reasoning_parser`, `enable_tool_guided_generation`, `enable_prefix_caching`). The
NPU entry's `MAX_PROMPT_LEN` is written from the Phase 0 measurement, not guessed. Exact keys are
validated against the installed OVMS version at startup; the app ships a versioned template per OVMS
minor release and refuses to start with a mismatched template rather than emitting a confusing OVMS
error.

> OVMS config reload unloads a removed model **after** in-flight requests finish — exactly the
> semantics needed for safe unload while a reply is streaming.

---

## 5. Load / unload UX (per the brief)

Every model card in Settings shows:

```
┌──────────────────────────────────────────────────────────────┐
│ Qwen3.5 9B (vision)                      [ GPU ▾ ]  [ Load ] │
│ 6.4 GB weights · 8k context · chat, vision, tools            │
│ ○ Not loaded                                                 │
└──────────────────────────────────────────────────────────────┘
```

- Device dropdown: `AUTO | NPU | GPU | CPU`, with unsupported options disabled and a tooltip saying
  why (e.g. "NPU requires int4-cw quantisation").
- `Load` → governor check → config edit → reload → progress events → `● Ready (7.6 GB)`.
- `Unload` → confirm if streaming → config edit → reload → `○ Not loaded`.
- `Pull` → HF download with progress, resumable, checksum verified.
- `Upload` → a folder of OpenVINO IR files (`.xml`/`.bin` + tokenizer) or a GGUF; validated, then
  added to the catalog. Per model type: an LLM folder, an embedding folder, an SD pipeline folder,
  a Whisper folder. Wrong-type uploads are rejected with the expected file list.
- `Quantise` (advanced) → runs `optimum-intel` export with device-appropriate flags; NPU preset
  hardcodes `--sym --ratio 1.0 --group-size -1`.

**Startup policy** (per the brief "only models selected at startup get loaded"): a Startup tab with
checkboxes per model; default = embeddings + small NPU model only. A "Load main model when AIO
starts" toggle exists for people who want chat ready immediately.

---

## 6. Device assignment policy

| Workload | Default | Reason |
|---|---|---|
| 9B chat/VLM | GPU | only viable option |
| filing model | NPU | offloads work from GPU, always on |
| embeddings | CPU | small, predictable, avoids GPU contention |
| SD 1.5 | GPU | diffusion needs it |
| Whisper | GPU if free, else CPU | |
| Kokoro | CPU | |
| Docling layout | CPU | |
| Docling VLM | GPU via gateway | reuses the 9B |

The UI lets the user override any of these; the governor re-checks feasibility after an override.

---

## 7. Context measurement protocol (Phase 0, mandatory)

We do not guess the usable context.

1. Load the 9B with `cache_size` = 1, 2, 4 GB.
2. Send increasing prompt sizes (1 k → 32 k in 1 k steps) at `max_tokens=64`.
3. Record: TTFT, tokens/s, peak RSS, commit charge, OOM/crash.
4. Choose the largest window where p95 generation stays > 6 tok/s and commit stays < 90 %.
5. Write the result into `models.json` as `context_tokens` and surface it in the UI.

Expected: 8 k safe, 16 k possible with 4 GB cache, 32 k not viable on 16 GB.

---

## 8. Model catalog governance

- Each entry carries `license`, `source_url`, `sha256`, `size_bytes`, `min_ram_gb`.
- A model cannot be added without a recognised licence; the UI shows a licence badge and, for
  OpenRAIL models (SD 1.5), the use-restriction notice before first use.
- The catalog is data, not code: adding a model never requires an app update.
- `openvino` and `optimum-intel` versions are pinned; a compatibility check runs on startup and
  warns if the installed OVMS is older than the template requires (v2026.4.1 is the current baseline).
