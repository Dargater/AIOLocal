# 01 — Architecture

## 1. Process model

Everything is a child process of a single supervisor. The UI is a window over HTTP; it holds no
state that matters.

```
AIOLocal.exe  (pywebview + WebView2)
└── supervisor (Python, same exe, background thread)
    ├── control-plane      FastAPI/Uvicorn  :7317   (always)
    ├── memory-gateway     FastAPI/Uvicorn  :8002   (always)
    ├── ovms.exe                            :8000   (always, models loaded per policy)
    ├── docling-worker     FastAPI          :8003   (on demand / warm)
    ├── opencode.exe serve                  :4096   (Code workspace only)
    ├── rea MCP server      (node)          :4097   (RE workspace only)
    └── gimp-host.exe                       (embedded GIMP dock, on demand)
```

**Why separate processes**

| Boundary | Reason |
|---|---|
| control plane ↔ gateway | a crash in retrieval must not kill the UI; the gateway must also be usable by OpenCode/REA with the app window closed |
| gateway ↔ OVMS | OVMS is a C++ server we do not modify; config reload is its public API |
| Docling worker | CPU-heavy, GIL-heavy, large model files; isolated so a bad PDF cannot freeze the app |
| OpenCode / REA | Node runtimes with their own dependency trees and crash modes |
| GIMP | separate GPL process, always; the embedder is a separate Win32 host process so a GIMP hang cannot freeze the app |

**Supervision rules**

- Each child has: start timeout, health probe, restart policy (`on-failure`, max 3 in 10 min),
  log file, and a circuit breaker that surfaces a red banner in the UI instead of silently retrying.
- Ports are allocated by the supervisor and written to `config/runtime.json`; nothing hardcodes a
  port except the documented defaults.
- Shutdown order: stop accepting requests → drain gateway → stop sidecars → unload OVMS models →
  stop OVMS → flush logs. A 10-second hard kill follows.
- Orphan guard: children are started in a Windows Job Object so closing the app cannot leave
  `ovms.exe` holding 8 GB.

## 2. Storage layout

```
%LOCALAPPDATA%\AIOLocal\
  config\
    app.json            UI prefs, mode, theme, ports
    models.json         model catalog (source of truth, UI-editable)
    ovms-config.json    generated, never hand-edited
    secrets.bin         DPAPI-encrypted (API keys, HF token)
  models\               OVMS model repository (can be relocated to another drive)
    <org>\<model>\...
    .ov_cache\
  data\
    memory.db           exact + summaries + jobs + embeddings
    knowledge.db        documents, chunks, collections
    app.db              conversations index, gallery, sessions, settings
  artifacts\
    images\  audio\  documents\  re\  code\
  logs\
    supervisor.log  gateway.jsonl  ovms.log  ui.log
  backups\              nightly sqlite .backup, 7-day rotation
```

`models\` is the only large directory and is relocatable from Settings (needed on a 40 GB disk).
SQLite runs in WAL mode with `synchronous=NORMAL`; a nightly `VACUUM INTO` backup runs when idle.

## 3. Supervisor and lifecycle

**Startup sequence**

1. Load `config/app.json`; validate schema; migrate if older.
2. Detect hardware: CPU name, RAM, GPU, NPU, driver versions (`openvino` device query via a tiny
   helper shipped with OVMS). Cache result; re-probe on demand.
3. Open databases, run migrations, start background queue.
4. Start OVMS with a config containing **only** models marked `auto_load` (default: embeddings +
   small NPU model). The 9B is *not* auto-loaded unless the user enables "load main model on start".
5. Start gateway → control plane → wait for health.
6. Show window. Everything else starts on demand.

**Model state machine** (owned by the control plane, executed through OVMS config reload)

```
absent → pulling → present → loading → ready → unloading → present
                    ↘ error (reason, retry)
```

- `load` appends the model to `ovms-config.json` and calls `POST /v1/config/reload`.
- `unload` removes it and reloads; OVMS drains in-flight requests first.
- A load is **refused** (HTTP 409 with a human explanation) if the RAM governor says it will not fit.
- Progress is streamed to the UI over the event bus (`model.state` events).

**Crash recovery**: on restart, the supervisor compares desired state (`models.json`) with actual
(`GET /v1/config`) and repairs. Half-downloaded models are detected by a `.incomplete` marker and
resumed or purged.

## 4. RAM governor

The single most important non-AI component on this machine.

**Model cost model** (measured in Phase 0, stored in `models.json`)

```json
{ "id": "OpenVINO/Qwen3.5-9B-int4-ov", "weights_gb": 6.4, "runtime_gb": 1.2,
  "kv_per_1k_tokens_gb": 0.09, "device": "GPU", "group": "llm" }
```

**Residency groups**

| Group | Members | Rule |
|---|---|---|
| `always` | embeddings, small NPU model | never evicted |
| `llm` | Qwen3.5-9B | one at a time |
| `image` | SD1.5 / inpaint | mutually exclusive with `llm` |
| `audio` | whisper, kokoro | may co-exist with `llm` if budget allows |
| `tooling` | OpenCode, REA, Ghidra | counted as fixed overhead |

**Budget**: `available = total_ram − OS_reserve(2.5 GB) − app_overhead(1.0 GB) − always(1.6 GB)`.
On 16 GB that is ≈ 10.9 GB for on-demand models. SD1.5 needs ~4 GB and the 9B ~7.6 GB, hence the
hard exclusion.

**Behaviour**: before every load the governor computes `projected = resident + new`. If
`projected > available`, the UI shows a modal: *"Image generation needs 4.1 GB; Qwen3.5-9B is using
7.6 GB. Unload it and continue?"* — never a silent swap. A `Memory pressure` warning appears above
85 % system commit.

## 5. Control plane (FastAPI, :7317)

Responsibilities:

- model catalog, pull/quantise/upload, load/unload, device assignment
- hardware + memory telemetry
- conversations index (title, timestamps, pinned) — content lives in memory.db
- knowledge collections and document ingestion
- gallery, audio library, code sessions, RE sessions
- job queue status, diagnostics bundle, log tail
- unified SSE event bus `/api/events`
- serves the built SPA at `/`

It does **not** proxy chat. Chat goes to :8002 so that external frontends work identically.

## 6. Memory gateway (:8002)

An OpenAI-compatible server. Endpoints: `/v1/models`, `/v1/chat/completions`, `/v1/embeddings`,
`/v3/...` aliases, `/health`, plus internal `/internal/*` for archive/flush/admin.

Request pipeline:

```
auth (local token) → parse → resolve conversation key → classify request
  ├─ side request (title/tags/probe) → passthrough, no memory, no archive
  ├─ tool-loop continuation         → passthrough current chain + rebuilt history
  ├─ document/VLM request           → attach document context, route to docling pipeline
  └─ normal chat                    → retrieve → assemble → stream → archive
```

Full algorithm in `02-memory.md`.

## 7. Frontend shell

**Decided: pywebview 5 + WebView2 (Edge runtime, present on Windows 11).**

- RAM: ~80–120 MB for the shell versus 250–350 MB for Electron.
- Native window, tray icon, file dialogs, "Open with", single-instance lock.
- One language stack (Python) for the shell and backend; no Rust toolchain.
- Dev mode runs the same UI in a normal browser against Vite + :7317.

```python
webview.create_window("AIO Local", "http://127.0.0.1:7317",
                      width=1440, height=900, min_size=(1024, 700),
                      background_color="#0b0d10")
```

**Stack**: React 19 + TypeScript, Vite 6, Tailwind 4, shadcn/ui (Radix), Zustand, TanStack Query,
`react-markdown` + Shiki for code, `wavesurfer.js` for audio, `monaco-editor` for code/diffs.

**Performance rules on 16 GB**

- One WebView, no nested webviews; Code/RE panels are routes, not iframes.
- Virtualise long transcripts and the gallery.
- Stream with `fetch` + `ReadableStream` (POST SSE), not `EventSource`.
- Images are thumbnailed on the backend; the UI never loads a 4 K PNG into the transcript.
- No background animation loops; charts render on demand.

### 7.1 The embedded GIMP dock (v1 requirement)

A foreign window cannot live inside WebView2's surface without fighting its z-order, so the dock is
implemented as a **native sibling window owned by the app**, positioned against the main window:

```
┌────────────────────────── AIO Local (main, WebView2) ──────────────────────────┐
│  Chat │ Images │ …                                                             │
└────────────────────────────────────────────────────────────────────────────────┘
┌──────────────── GIMP dock (owned window, no taskbar entry) ────────────────────┐
│  native Win32 child host  ← SetParent(gimp_main_hwnd)                          │
│  GIMP 3 (GTK3, single-window mode)                                             │
└────────────────────────────────────────────────────────────────────────────────┘
```

- The dock follows the main window's move/minimise/restore and snaps left or right.
- A detach button returns GIMP to a normal top-level window at the same size and position.
- If reparenting fails or GIMP misbehaves, the app automatically detaches and shows a one-time
  explanation — it never leaves a half-broken window.
- The Win32 work lives in a small `gimp-host` process (a shim), so a hung GIMP cannot block the UI
  thread. Full design and fallbacks in `04` §4.

## 8. Configuration

`config/models.json` is the single source of truth and looks like:

```json
{
  "version": 3,
  "models": [
    {
      "id": "qwen3.5-9b",
      "name": "Qwen3.5 9B (vision)",
      "hf_repo": "OpenVINO/Qwen3.5-9B-int4-ov",
      "task": "text_generation",
      "capabilities": ["chat", "vision", "tools"],
      "device": "GPU",
      "auto_load": false,
      "context_tokens": 16384,
      "tool_parser": "hermes3",
      "weights_gb": 6.4,
      "runtime_gb": 1.2,
      "group": "llm",
      "is_main": true
    }
  ]
}
```

The control plane renders `ovms-config.json` from this file. Users never edit OVMS JSON by hand,
but an "Advanced → open config" button exists for debugging, with schema validation on save.

## 9. Failure design

| Failure | Behaviour |
|---|---|
| OVMS crashes | supervisor restarts it, reloads `auto_load` models, marks main model `absent`, UI banner + Retry |
| Model OOM on load | load marked failed with captured stderr tail; governor lowers the estimate; UI suggests a smaller model or CPU |
| Gateway crashes | auto-restart; in-flight streams fail with a clear error; archive jobs are durable in SQLite so nothing is lost |
| NPU model unavailable | memory filing degrades to CPU small model or heuristic filing; chat is unaffected |
| Disk full | refuse pull/generate before writing; show required vs free; cleanup helper for artifacts |
| Port in use | supervisor picks the next free port and rewrites runtime config |
| DB locked | WAL + busy timeout 5 s; long jobs run in a writer thread, readers never block |

## 10. Threading / concurrency model

- FastAPI async endpoints for I/O.
- All SQLite writes funnel through **one** writer thread per database (avoids `database is locked`
  and keeps append-only ordering trivial).
- CPU work (embeddings, tokenisation) runs in a `ProcessPoolExecutor` with 2 workers; the iGPU
  model is unaffected.
- Docling ingestion is a bounded worker pool (1 process) so a 500-page PDF cannot starve chat.
- The NPU filing worker is strictly serial (the NPU handles one request at a time).

## 11. Observability

- `logs/gateway.jsonl`: one line per request — request id, conversation key, frontend, classification,
  tokens in/out, retrieval latency, candidates considered, memory ids chosen, trim decisions.
- `logs/supervisor.log`: process lifecycle.
- **Memory Inspector** (UI): for any answer, "why did the model see this?" — shows the exact assembled
  prompt, retrieved summaries, expanded messages, and every trimming step.
- Diagnostics bundle: config (secrets redacted), versions, hardware, last 2000 log lines, DB stats.
  Zipping is explicit and local; nothing is uploaded.
