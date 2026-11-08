# AIO Local

**A local, memory-first AI workspace for Intel laptops.** Chat, image generation, model
management, and a transparent long-term memory system — all on `127.0.0.1`, with no cloud calls.

This is **v0.1.0**, a deliberately small but real vertical slice. It is not the finished product
described in `PLAN/`. What exists is listed below; what does not exist is listed just as plainly.

---

## What works in v0.1

| Area | Status |
|---|---|
| Local server + browser UI | ✅ |
| Chat with streaming through any OpenAI-compatible upstream (OVMS by default) | ✅ |
| Exact memory DB (append-only, deduplicated, SQLite WAL) | ✅ |
| Summary memory + topics + many-to-many links | ✅ |
| Hybrid retrieval (BM25 + local hashed embeddings + reciprocal rank fusion) | ✅ |
| Token-budgeted prompt assembly with the documented trim order | ✅ |
| Background memory filing with validation and `unsorted` fallback | ✅ |
| Memory Inspector (see exactly what the model saw) | ✅ |
| Model catalog, device choice, load/unload via OVMS config reload | ✅ |
| Image generation/edit proxy + local gallery | ✅ |
| Image attachments to a vision chat model (Qwen3-VL 4B) | ✅ |
| Offline math rendering (bundled KaTeX) and PDF text extraction (pypdf) | ✅ |
| Local-only security: loopback bind, per-install token, origin/host checks, CSP | ✅ |
| Windows build (PyInstaller onedir) | ✅ |
| Documents / Knowledge RAG (separate DB, FTS5 + vectors, inbox, upload isolation) | ✅ v1 |
| Code workspace (OpenCode lifecycle + workspace preview) | ✅ v1 |
| Reverse engineering (REA MCP stdio client, tool console, lifecycle) | ✅ v1 |
| Embedded GIMP dock (Win32 host + embed + external fallback) | ✅ v1 |
| Audio / video | ⛔ planned (v2 — Stable Diffusion-class model for 16 GB) |

The UI groups Documents, Code, Reverse engineering and GIMP under **Workspace** because they are
real v1 views. Only **Audio & video** is under **Planned modules**.

---

## Requirements

- Windows 10/11 x64, 8 GB RAM minimum (16 GB recommended for the 9B model).
- **App bundle** includes server, UI, memory, KB, web search, image proxy, and the GIMP/REA/OpenCode lifecycle managers. It does **not** include OVMS/GIMP/REA/OpenCode binaries or model weights (those are GB-scale/externally licensed).
- **Automatic setup**: every launch checks the six runtime components (OVMS, portable Node.js, OpenCode, private Python, GIMP 3, Intel OpenVINO GIMP plugin). The Setup page downloads, verifies and unpacks anything missing into app-owned folders under the data directory, with per-component progress and logs. No system-wide install is required.
- **Automatic hosting**: the app can start/stop/reload its own server (`uvicorn`), manage the GIMP dock (`Win32 SetParent`), and launch REA (`npx -y rea-agents mcp`) when the user presses Start. Settings allow custom executable paths.
- **Specs-aware model notes**: the Models page shows licence tag, size (`~X GB`), install/load status, and device options for each model. The 9B chat model is recommended for 16 GB + GPU/NPU; the small filing model (`LFM2.5-1.2B`) fits easily alongside.
- **Models** are not bundled (several GB each). Pull them from the Models page; Stable Diffusion-class generation (planned v2) needs its own disk budget.

Without OVMS the app still starts, the UI works, and memory is fully testable — chat returns a clear
`UPSTREAM_UNAVAILABLE` message instead of a fake answer.

---

## Run from source

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -e ".[dev]"

# start (opens your browser)
python -m aiolocal

# or without opening a browser
python -m aiolocal --no-browser --port 7317
```

Then open <http://127.0.0.1:7317>.

### First-launch setup

OVMS is required for local models and is **not bundled** (it is a separate, externally
licensed server). Open **Setup** on first launch and press **Install all**. The app downloads
and verifies its own copies of OVMS, Node.js, OpenCode, Python, GIMP and the Intel OpenVINO
GIMP plugin into `data\tools` under its data folder. You can also install or repair one
component at a time and read its full log from the same page.

Then:

1. Open **Models**, download a model, choose a device and press **Load**. Model weights are
   the only large downloads you choose explicitly; they are never bundled.
2. Press **Start**. The app writes the OVMS config itself, reproduces OVMS's `setupvars`
   environment (its bundled Python backend needs `PYTHONHOME`/`PATH`), and shows a live
   startup status with the log tail if anything fails.
3. Model downloads show real byte progress; server startup shows an indeterminate progress
   bar plus the current stage (starting → loading model → ready).

The previous build wrote a config with a `task` key, which OVMS 2026 rejects, and launched
`ovms.exe` without its Python environment. Both are fixed: GenAI models are configured with
`graph_path`, and the generated graph is patched per device so GPU/CPU/NPU changes actually
take effect.

### Direct OpenVINO GenAI (no server)

OpenVINO GenAI does have an in-process Python API (`openvino_genai.LLMPipeline`) that needs
no OVMS server. This build does **not** bundle that engine; OVMS remains the default because
it provides one OpenAI-compatible endpoint for chat, embeddings and image generation, plus
model reload without restarting the app. Bundling a second engine would add another large
native dependency and a different streaming API, so it is not claimed here.

---

## Build the Windows binary

```powershell
.\packaging\build.ps1
```

Output: `dist\AIOLocal\AIOLocal.exe` (and a zip next to it). The build is a `onedir` bundle so
startup is fast and the web assets are visible for auditing. Copy the whole `dist\AIOLocal`
folder to another Windows PC; there is no installer. The first launch creates its own data
and tool folders, then the Setup page installs anything the machine is missing.

---

## Data and privacy

Everything lives in `%LOCALAPPDATA%\AIOLocal\`:

```
config\app.json      settings (no secrets)
config\token.bin     DPAPI-encrypted local API token
config\secrets.bin   DPAPI-encrypted optional keys (Brave Search, etc.)
data\memory.db       conversations, summaries, jobs, embeddings
data\knowledge.db    separate knowledge base (FTS5 + chunk vectors)
knowledge\inbox\    raw JSON/MD files picked up by the filing LLM
artifacts\images\    generated images
logs\                rotating logs (prompts are not logged at info level)
models\              OVMS model repository (relocatable)
tools\               app-owned runtimes installed by Setup (OVMS, Node, OpenCode, Python, GIMP)
opencode\            isolated OpenCode data, config, cache and state
workspace\           code workspace
```

- The server binds **127.0.0.1 only** and rejects non-local `Host`/`Origin` headers.
- Every API call needs the per-install token; the UI receives it in a same-origin page.
- No telemetry, no analytics, no remote logging, no auto-update.
- The only outbound traffic is model download, triggered explicitly from the Models page.

See `SECURITY.md` for the threat model and reporting instructions.

---

## Licence

AGPL-3.0-or-later. See `LICENSE` and `NOTICE`.

**Before uploading to GitHub:**
- Confirm the source is complete (`git add -A`; no `node_modules`, `.venv`, `dist`, or model weights).
- `NOTICE` already lists all third-party licences and the AGPL obligations. Do not bundle Open WebUI, OVMS binaries, or model files.
- The AGPL requires that anyone who receives the binary also gets the source. Provide the repository link and the exact version tag in releases.
- The `websearch.py` feature references third-party backends (DuckDuckGo/SeArXNG/Brave); include a note in the release that Brave needs a user-provided API key and the app does not ship keys.
- Stable Diffusion 1.5 weights use CreativeML OpenRAIL-M; do not bundle them and mention that restriction in the release notes.

Third-party components keep their own licences; `NOTICE` lists them and the obligations. In
particular, Stable Diffusion 1.5 weights are **CreativeML OpenRAIL-M**, not Apache-2.0, and carry
use restrictions that pass through to you.

---

## Project layout

```
aiolocal/
  __main__.py        launcher (server + browser)
  app.py             FastAPI app, middleware, static files
  config.py          paths, settings, token bootstrap
  db.py              SQLite, migrations, single-writer transactions
  security.py        DPAPI, token checks, origin/host validation, safe paths
  gateway.py         OpenAI-compatible /v1/chat/completions + memory
  control.py         /api/v1 control plane
  llm.py             upstream client and SSE tee
  models.py          OVMS catalog, config generation, load/unload
  images.py          image proxy and gallery
  events.py          server-sent events
  memory/            text, store, retrieve, assemble, filing, jobs
  web/               no-build HTML/CSS/JS UI
tests/               unit + integration tests (fake upstream, no GPU needed)
packaging/           PyInstaller spec and build script
PLAN/                the full architecture plan (design source of truth)
```
