# AIO Local — Master Plan

**A single, local, multipurpose AI desktop application for Intel hardware.**
Working name: **AIO Local** (rename later; nothing depends on it).

Status: **planning only. No application code has been written.**
Date: 2026-11-03
Target machine: Core Ultra 5 225H · Arc 130T iGPU · NPU · 16 GB RAM · ~40 GB free disk · Windows 11 first, Bazzite/Linux later.

**Locked decisions** (confirmed 2026-11-03):

1. Desktop shell = **pywebview + WebView2**.
2. **Open WebUI is dropped.** Our React UI is the only frontend; the gateway stays OpenAI-compatible
   so any future client can attach without changes.
3. **Video generation is v2**, not v1. (It is feasible via OpenVINO GenAI `Text2VideoPipeline`
   + LTX-Video; it is *not* an OVMS endpoint. See `03` §3.7.)
4. **GIMP is embedded in v1**, implemented as an owned native "GIMP dock" window (not an iframe)
   with an automatic detach fallback. See `04` §4.

---

## 1. What this app is

One installable desktop app with a launcher and a shared memory brain, exposing seven workspaces:

| Workspace | What it does | Engine |
|---|---|---|
| **Chat** | Text + vision chat, any topic, streaming | Qwen3.5-9B on GPU |
| **Knowledge** | Ask questions over your own books/papers/notes, with citations | Docling + embeddings + Qwen |
| **Documents** | Drop a PDF/.py/.docx/folder, parse, summarise, query, archive | Docling + Qwen |
| **Images** | Generate, edit, inpaint, outpaint, upscale; optional GIMP round-trip | SD 1.5 int8 on GPU |
| **Audio** | Speech-to-text, translation, text-to-speech | Whisper + Kokoro |
| **Code** | Agentic coding in a repo, diffs, approvals, terminal | OpenCode + Qwen |
| **Reverse Engineering** | Analyse binaries/apps/APKs/firmware, trace behaviour | REA (MCP) + Qwen |

All seven share **one memory system** (Section 3), one model fleet, one settings surface, one event log.

Non-goals for v1: cloud sync, multi-user accounts, training/fine-tuning, video generation (see §9), mobile, Docker.

---

## 2. The five principles everything is built on

1. **The model never accumulates context.** Every request is rebuilt from scratch:
   `current input + capped recent history + retrieved memory + mode instructions`.
   The gateway owns the context window; frontends are not trusted to.
2. **Memory is a gateway, not a feature of a UI.** Any client that speaks OpenAI
   (`/v1/chat/completions`) gets the same memory: our UI, OpenCode, REA, curl.
3. **Fast path is deterministic.** Embedding + full-text + SQL on CPU. The NPU model is
   only allowed to *file* memory in the background (and, rarely, rewrite an ambiguous query).
   No LLM is ever in the reply-critical path.
4. **Everything is a process with a port.** Supervisor starts/stops OVMS, gateway, Docling,
   OpenCode, REA. Nothing is embedded in-process that can crash the UI.
5. **16 GB is the design constraint, not an afterthought.** A RAM governor owns model
   load/unload and refuses impossible combinations instead of swapping to death.

---

## 3. Architecture in one picture

```
┌───────────────────────────── UI layer (React in pywebview / any browser) ─────────────────────────────┐
│  Chat · Knowledge · Documents · Images · Audio · Code · Reverse Engineering · Memory Inspector · Settings │
└───────────────────────────────────────────────┬───────────────────────────────────────────────────────┘
                                                │ HTTP + SSE  (127.0.0.1 only)
┌───────────────────────────────────────────────▼───────────────────────────────────────────────────────┐
│  CONTROL PLANE  :7317   FastAPI                                                                       │
│  model catalog · load/unload · device picker · uploads · library · jobs · diagnostics · event bus      │
└──────┬───────────────────────┬──────────────────────────┬──────────────────────┬───────────────────────┘
       │                       │                          │                      │
┌──────▼──────────┐   ┌────────▼───────────┐   ┌──────────▼─────────┐   ┌────────▼──────────────┐
│ MEMORY GATEWAY  │   │  OVMS  :8000       │   │  DOCLING WORKER    │   │  SIDECARS             │
│ :8002           │──▶│  one process,      │   │  :8003             │   │  OpenCode :4096       │
│ OpenAI-compat   │   │  per-model device  │   │  parse → markdown  │   │  REA MCP  :4097       │
│ + retrieval     │   │  + hot reload      │   │  + DocTags sanitiser│  │  (on demand)          │
└──────┬──────────┘   └────────┬───────────┘   └──────────┬─────────┘   └────────┬──────────────┘
       │                       │                          │                      │
┌──────▼───────────────────────▼──────────────────────────▼──────────────────────▼───────────────────────┐
│  STORAGE   %LOCALAPPDATA%\AIOLocal\                                                                     │
│  memory.db (SQLite WAL) · knowledge.db · artifacts\ · models\ · logs\ · config\                          │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Port map**

| Port | Process | Bind | Notes |
|---|---|---|---|
| 7317 | Control plane (FastAPI) | 127.0.0.1 | serves API + built UI |
| 8002 | Memory gateway | 127.0.0.1 | the only endpoint frontends point at |
| 8000 | OVMS | 127.0.0.1 | OpenAI `/v3`, config reload |
| 8003 | Docling worker | 127.0.0.1 | internal only |
| 4096 | OpenCode server | 127.0.0.1 | started on demand |
| 4097 | REA MCP (stdio) | 127.0.0.1 | started on demand |
| 5173 | Vite dev server | 127.0.0.1 | development only |

**Why one OVMS process, not two.** Device choice in OVMS is per model, not per process, and the
NPU constraints (`int4-cw`, `max_prompt_len`, no batching) are model-level settings. One process
saves 300–500 MB of Python/runtime overhead on a 16 GB machine and makes load/unload a single
config-reload transaction. A second instance (`--rest_port 8001`) remains a documented fallback if
NPU and GPU models prove unstable together.

---

## 4. Memory in one page (full detail in `02-memory.md`)

Two databases, exactly as specified:

- **Exact DB** — append-only, word-for-word turns with ULID + timestamp. Nothing is ever edited
  or deleted. Edits/regenerations are new rows linked by `revises_id`.
- **Summary DB** — many small, context-scoped summaries (not one per conversation), each linked
  to many exact messages (`summary_messages` M2M), organised into topics.

**Read path** (before the model answers, target < 150 ms):
`normalise → embed on CPU → vector top-k + FTS5 top-k → RRF fuse → topic/recency/salience boost
→ pick summaries → expand only the referenced message IDs → token-budget trim → assemble`.

**Write path** (after the reply, NPU, non-blocking):
`turn lands in Exact DB immediately → background job asks the small NPU model for
{topics, summary, message_ids} → schema + ID allowlist validation → insert/merge summary →
link IDs → embed new summaries → invalid output retries once, then falls into topic "unsorted"`.

The NPU model **cannot** write message text, invent IDs, or delete anything. Code enforces this.

**Frontends are not trusted with history.** The gateway ignores the client's `messages` array by
default and rebuilds history from its own Exact DB using the conversation key. That single decision
solves full-history resending, client-side compaction, and injected RAG blocks at once — which is
what makes the gateway safe to expose to OpenCode and to any future client.
Exception: the current turn's tool-call chain is passed through verbatim (a tool loop must not be
broken).

---

## 5. Models and devices (full matrix in `03-models-and-devices.md`)

| Role | Model | Device | Resident? |
|---|---|---|---|
| Main brain (chat, vision, code, RE, Docling VLM) | `OpenVINO/Qwen3.5-9B-int4-ov` | GPU (iGPU, shared RAM) | on demand |
| Memory filing (small, JSON) | `LFM2.5-1.2B-Instruct-int4-cw-ov` (fallback `Qwen3-4B-Instruct-2507-int4-cw-ov`) | NPU | always |
| Embeddings | `Qwen3-Embedding-0.6B-int8-ov` | CPU | always |
| Image generation | `OpenVINO/stable-diffusion-v1-5-int8-ov` | GPU | exclusive |
| Inpaint/outpaint | `Intel/stable-diffusion-v1-5-inpainting-openvino` | GPU | exclusive |
| ASR / translate | Whisper base or small, int8 | GPU/CPU | on demand |
| TTS | Kokoro-82M | CPU/NPU | on demand |

Because the iGPU has no dedicated VRAM, **model weights and system RAM are the same pool**. The
governor therefore treats "GPU free" and "RAM free" as one calculation. Chat (9B) and image
generation **cannot** be resident together on 16 GB; switching modes unloads one and loads the other,
with progress in the UI.

---

## 6. Document set

| File | Contents |
|---|---|
| `README.md` | this overview, decisions, principles |
| `01-architecture.md` | processes, lifecycle, supervisor, config, storage, UI shell |
| `02-memory.md` | schemas, retrieval/write algorithms, budgets, failure modes, dedupe |
| `03-models-and-devices.md` | hardware budget, model matrix, OVMS config, load/unload, quantisation |
| `04-integrations.md` | OpenCode, REA, Docling bridge, GIMP embedding, audio, images |
| `05-data-and-api.md` | SQLite DDL, REST/SSE contracts, event schema |
| `06-roadmap-testing-risks.md` | Phase 0 measurements, P1–P10, tests, risks, licensing |
| `07-toolchain.md` | every language, library, CLI and external tool, with why |
| `08-build-workflow.md` | how Qwen writes the app, how you debug it, first ten tasks |

---

## 7. Decisions

**Locked** (see the top of this file): pywebview shell · Open WebUI dropped · video in v2 ·
embedded GIMP dock in v1.

**Still open, but not blocking Phase 0:**

1. **App name** — "AIO Local" is a placeholder; renaming is cheap until the installer exists.
2. **Default main model** — Qwen3.5-9B is the plan; if Phase 0 shows the 8 k context is too tight,
   fall back to a 4B-class model for chat and keep the 9B for documents/code.
3. **Medical mode default** — off by default, with a first-run explanation of what it does and does
   not do.
4. **REA depth in v1** — JS/Electron, .NET, APK and PyInstaller work without Ghidra. Native binary
   analysis needs Ghidra/IDA/Hopper; decide after the Phase 0 readiness check whether to ship the
   Ghidra wizard in v1 or v1.1.

Everything else in this document set is frozen enough to build against. Phase 0 exists to replace
the remaining assumptions with measurements, not to reopen the design.
