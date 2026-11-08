# 05 — Data and API Contracts

## 1. Conventions

- All services bind `127.0.0.1` only. There is no remote mode in v1.
- Every process requires `Authorization: Bearer <token>`; the token is generated at first run,
  stored DPAPI-encrypted, and injected into the UI and the OpenCode/REA configs.
- JSON everywhere; timestamps are Unix milliseconds (UTC); IDs are ULIDs unless noted.
- Errors use one shape:

```json
{ "error": { "code": "MODEL_NOT_LOADED", "message": "qwen3.5-9b is not loaded",
             "details": { "model": "qwen3.5-9b" }, "request_id": "01J…" } }
```

- API version is in the path (`/v1` for OpenAI compatibility, `/api/v1` for the control plane).
- Breaking changes bump `/api/v2`; the UI and backend ship together, so this is mostly for plugins.

---

## 2. Memory gateway

Base: `http://127.0.0.1:8002`

### 2.1 `POST /v1/chat/completions`

Standard OpenAI body. Additional optional fields:

| Field | Type | Meaning |
|---|---|---|
| `conversation_id` | string | overrides header |
| `mode` | `chat\|knowledge\|code\|re\|medical` | retrieval policy |
| `memory` | `auto\|off\|only` | `off` = no read, still archive; `only` = retrieval preview |
| `client` | string | `native\|opencode\|rea\|api` |

Headers:

| Header | Purpose |
|---|---|
| `X-AIO-Conversation` | conversation key (preferred) |
| `X-AIO-Mode` | same as body `mode` |
| `X-AIO-Client` | source, for logging and injected-pattern detection |
| `X-AIO-Request-Id` | client-supplied idempotency key |
| `X-AIO-No-Memory` | `1` disables read + write |
| `X-AIO-Side` | `1` marks a side request |

Streaming (`stream:true`) returns SSE identical to OpenAI plus our own events:

```
data: {"id":"…","choices":[{"delta":{"content":"Hel"}}]}
data: {"id":"…","choices":[{"delta":{"content":"lo"}}]}
data: {"aio":{"memory":{"retrieved":3,"summaries":2,"messages":7,"tokens":1840,"latency_ms":83}}}
data: [DONE]
```

The `aio` frame is additive and ignored by standard clients; our UI uses it to render the memory
chip. If the client disconnects mid-stream, partial output is archived and filing runs normally.

### 2.2 `GET /v1/models`

Merged view: loaded models first, then available. `owned_by` = `aio`.

### 2.3 `POST /v1/embeddings`

Proxies to OVMS embeddings, with caching by `sha256(model + input)` (LRU, 10 k entries) because
re-embedding the same query is common.

### 2.4 `POST /v1/images/generations` · `POST /v1/images/edits`

OpenAI-compatible passthrough with additions: `seed`, `negative_prompt`, `steps`, `width`, `height`,
`save` (default true). Response includes `artifact_id` and `file_path` alongside `b64_json`.

### 2.5 `POST /v1/audio/transcriptions` · `/translations` · `/speech`

Multipart pass-through; adds `artifact_id`, `duration_ms`, `language`, `segments[]` with timestamps.

### 2.6 Internal (never exposed to frontends)

| Endpoint | Purpose |
|---|---|
| `POST /internal/archive` | force flush of a conversation |
| `POST /internal/reindex` | rebuild embeddings/FTS |
| `GET /internal/config` | effective retrieval config (for debugging) |
| `POST /internal/docling/vlm` | DocTags sanitiser endpoint |
| `GET /internal/health` | component-level health |

---

## 3. Control plane

Base: `http://127.0.0.1:7317/api/v1`

### 3.1 System

| Method | Path | Purpose |
|---|---|---|
| GET | `/system/info` | versions, hardware, uptime, paths |
| GET | `/system/health` | per-process status |
| GET | `/system/memory` | RSS/commit per process, governor projection |
| GET | `/system/logs?source=gateway&tail=200` | log tail |
| POST | `/system/diagnostics` | build a local zip (never uploaded) |
| GET | `/system/events` | SSE stream (see §5) |

### 3.2 Models

| Method | Path | Body / notes |
|---|---|---|
| GET | `/models` | catalog + state + device + footprint |
| POST | `/models` | add custom entry |
| PATCH | `/models/{id}` | device, auto_load, context_tokens, display name |
| POST | `/models/{id}/load` | 409 if governor refuses |
| POST | `/models/{id}/unload` | drains in-flight |
| POST | `/models/pull` | `{ "hf_repo": "...", "revision": "main" }` → job id |
| POST | `/models/upload` | multipart folder/zip; validated by task type |
| DELETE | `/models/{id}` | removes files after confirmation |
| POST | `/models/{id}/quantize` | advanced; preset by target device |
| GET | `/models/jobs/{job_id}` | progress |

### 3.3 Memory

| Method | Path | Purpose |
|---|---|---|
| GET | `/memory/search?q=&k=&mode=` | debug retrieval, returns scores and reasons |
| GET | `/memory/summaries?topic=&cursor=` | browse |
| POST | `/memory/summaries` | manual summary |
| PATCH | `/memory/summaries/{id}` | edit text/salience/topic (never touches Exact DB) |
| GET | `/memory/messages/{id}` | exact row + lineage |
| GET | `/memory/graph?root=` | nodes/edges for the graph view |
| GET | `/memory/topics` | tree + counts |
| POST | `/memory/merge` | manual merge of two summaries |
| POST | `/memory/forget` | tombstone a summary (Exact DB untouched) |
| GET | `/memory/queue` | filing jobs, failures, retries |
| POST | `/memory/jobs/{id}/retry` | |
| POST | `/memory/reindex` | re-embed everything (background) |
| GET | `/memory/inspector/{request_id}` | full prompt + decisions |

### 3.4 Conversations

`GET /conversations` · `GET /conversations/{id}` (messages from Exact DB) · `PATCH /conversations/{id}`
(title, pinned, archived) · `DELETE /conversations/{id}` (hides; rows stay for memory integrity).

### 3.5 Knowledge

| Method | Path |
|---|---|
| GET/POST | `/knowledge/collections` |
| POST | `/knowledge/collections/{id}/documents` (multipart) |
| GET | `/knowledge/documents/{id}` (status, pages, chunks) |
| GET | `/knowledge/search?q=&collection=` |
| DELETE | `/knowledge/documents/{id}` (soft) |

### 3.6 Images / audio / code / RE

| Area | Endpoints |
|---|---|
| Images | `POST /images/generate`, `POST /images/edit`, `GET /images`, `GET /images/{id}`, `DELETE /images/{id}`, `POST /images/{id}/upscale`, `POST /images/{id}/open-gimp` |
| Audio | `POST /audio/transcribe`, `POST /audio/speak`, `GET /audio`, `GET /audio/{id}` |
| Code | `POST /code/sessions`, `GET /code/sessions`, `GET /code/sessions/{id}`, `POST /code/sessions/{id}/message`, `GET /code/sessions/{id}/events` (SSE proxy), `POST /code/sessions/{id}/permission` |
| RE | `POST /re/sessions`, `GET /re/sessions/{id}`, `POST /re/sessions/{id}/target`, `POST /re/sessions/{id}/message`, `GET /re/sessions/{id}/events`, `POST /re/sessions/{id}/report`, `GET /re/tools` |
| Settings | `GET/PATCH /settings`, `GET/POST /settings/secrets` (write-only), `POST /settings/reset` |

---

## 4. Example: one chat turn end-to-end

```
POST /v1/chat/completions
Authorization: Bearer …
X-AIO-Conversation: conv_01J…
X-AIO-Client: native
{
  "model": "qwen3.5-9b",
  "messages": [
    {"role":"user","content":"why did we pick tokio over async-std?"}
  ],
  "stream": true,
  "max_tokens": 800
}
```

Internal sequence:

```
1  classify            chat
2  archive user msg    id=01JA… (INSERT OR IGNORE)
3  key                 conv_01J…
4  intent              lookup (entity "tokio", "async-std")
5  embed               q → 768-d, 18 ms
6  candidates          summaries 6, messages 22
7  fuse/score          top: "Rust CLI: runtime decision" 0.91
8  expand              3 messages (the actual decision turn)
9  plan tokens         120 system + 2100 memory + 640 history + 14 input = 2874/8192
10 stream              → OVMS (GPU), first token ~380 ms
11 on [DONE]           archive assistant msg, enqueue file_turn
12 background          NPU adds summary link: this question → same decision summary
```

Response to the client includes the normal chunks plus:

```json
{"aio":{"memory":{"retrieved":6,"summaries":1,"messages":3,"tokens":2100,"latency_ms":83}}}
```

---

## 5. Event bus (`GET /system/events`)

```json
{ "type": "model.state",  "ts": 1762…, "data": { "id": "qwen3.5-9b", "state": "loading", "progress": 0.42 } }
{ "type": "model.pull",   "ts": …, "data": { "id": "…", "bytes": 123, "total": 4567, "eta_s": 90 } }
{ "type": "memory.job",   "ts": …, "data": { "id": "01J…", "state": "failed", "reason": "invalid_ids" } }
{ "type": "memory.filed", "ts": …, "data": { "conversation_id": "…", "summary_id": "…", "topic": "rust/async" } }
{ "type": "doc.progress", "ts": …, "data": { "document_id": "…", "page": 12, "total": 240 } }
{ "type": "code.permission", "ts": …, "data": { "session": "…", "tool": "edit", "path": "src/main.rs" } }
{ "type": "system.memory", "ts": …, "data": { "commit_pct": 0.87, "resident_gb": 12.4 } }
{ "type": "system.error",  "ts": …, "data": { "source": "ovms", "message": "…" } }
```

Events are ephemeral (ring buffer of 500, replayed from `Last-Event-ID`). Anything that must survive
a restart is a database row, not an event.

---

## 6. Upload rules

| Kind | Accepted | Limit | Validation |
|---|---|---|---|
| Document | pdf, docx, pptx, xlsx, md, txt, html, epub, images | 200 MB / file | magic bytes, not just extension |
| Code project | folder or zip | 500 MB | path traversal check, symlink refusal |
| Image | png, jpg, webp, bmp | 50 MB | decode before store |
| Audio | wav, mp3, flac, m4a, ogg | 500 MB | duration probe |
| Model | OVMS IR folder or zip, GGUF | 20 GB | required files per task, size check against free disk |

All uploads go to a temp directory, are validated, then atomically moved into `artifacts/` or
`models/`. Filenames are replaced with IDs; the original name is metadata only.

---

## 7. Settings schema (excerpt)

```json
{
  "ui": { "theme": "dark", "density": "comfortable", "language": "en" },
  "startup": { "open_window": true, "load_main_model": false, "auto_start_gateway": true },
  "models": { "models_dir": "C:\\AI\\models", "auto_unload_idle_min": 20 },
  "memory": {
    "enabled": true,
    "short_history_turns": 6,
    "summary_share": 0.35,
    "exact_share": 0.40,
    "filing_model": "lfm2.5-1.2b",
    "consolidate_hour": 3
  },
  "retrieval": { "top_k": 40, "tau_low": 0.22, "ambiguity_delta": 0.06, "rrf_k": 60 },
  "privacy": { "telemetry": false, "network": "models-only", "redact_logs": true },
  "medical": { "enabled": false, "require_citations": true }
}
```

`network: models-only` means the app makes outbound requests **only** to Hugging Face for model
pulls, and only when the user clicks Pull. Everything else is loopback.

---

## 8. Versioning and migrations

- `PRAGMA user_version` drives SQL migrations, applied forward-only at startup inside a transaction.
- A downgrade attempt refuses to start and offers the nightly backup instead of corrupting data.
- Model catalog has its own `version`; unknown future fields are preserved when re-saving.
- The gateway and control plane expose `/health` with a build hash so a stale process is detectable
  after an update.
