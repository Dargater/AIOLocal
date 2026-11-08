# 07 — Toolchain

Everything below is chosen for one of four reasons: it runs on Windows without Docker, it is small
enough for 16 GB, it is debuggable by a human, or it is the only thing that does the job.

## 1. Languages

| Language | Used for | Why |
|---|---|---|
| **Python 3.12** | supervisor, control plane, gateway, Docling worker, memory engine | OVMS/Docling/OpenVINO ecosystem is Python-first; fastest to iterate and debug |
| **TypeScript + React** | UI | the only sane option for a modern desktop UI |
| **PowerShell** | installer glue, Windows service bits | native, no extra runtime |
| **Minimal Rust** | *not used in v1* | Tauri would add a second toolchain for little gain; revisit only if pywebview disappoints |

No Docker. No WSL requirement. No Electron.

## 2. Python dependencies

| Package | Purpose |
|---|---|
| `fastapi`, `uvicorn[standard]` | HTTP services |
| `pydantic` v2 | schemas, settings, validation |
| `sqlalchemy` 2 + `alembic` | DB access and migrations (or raw `sqlite3` if SQLAlchemy proves heavy — decided in P1) |
| `aiosqlite` | async reads; writes go through one writer thread |
| `httpx` | OVMS/HF calls, streaming |
| `numpy` | vector search, audio buffers |
| `tokenizers` | exact token counts with the model's own tokenizer |
| `orjson` | fast JSON for logs and hot paths |
| `structlog` | JSONL logs with request ids |
| `tenacity` | retries with backoff (filing, pulls) |
| `psutil` | RAM/commit telemetry for the governor |
| `pywin32` | DPAPI secrets, job objects, window handles |
| `keyring` | fallback secret storage |
| `pywebview` | desktop shell (WebView2) |
| `python-multipart` | uploads |
| `Pillow`, `opencv-python-headless` | image IO, masks, thumbnails |
| `soundfile`, `pydub` | audio IO |
| `docling` | document parsing (layout, OCR, tables, VLM pipeline) |
| `openvino` (probe only) | enumerate NPU/GPU/CPU and driver versions at startup |
| `sqlite-vec` *(optional)* | ANN index if brute force stops being fast enough; fallback is numpy |
| `pytest`, `hypothesis`, `respx` | tests |

Deliberately **not** used:

- **Mem0** — the whole point of the custom design; also adds a cloud-shaped abstraction we do not want.
- **LangChain / LlamaIndex** — too much hidden prompt behaviour for a system whose main feature is
  controlling the prompt exactly.
- **Qdrant / Chroma / Milvus** — a second server process and hundreds of MB for a corpus that fits in
  SQLite + numpy.
- **Ollama** — the brief mandates OVMS.
- **Electron** — RAM.

## 3. Node side

| Tool | Version | Purpose |
|---|---|---|
| Node.js | **24 LTS ≥ 24.22.1** (or ≥ 26.12.1) | OpenCode, REA. *Current install is 26.7.0 — below REA's minimum, must be upgraded or a second runtime bundled.* |
| pnpm | 9+ | UI dependencies |
| `opencode` | latest | coding agent server (`opencode serve`) |
| `rea-agents` | pinned per release | reverse-engineering MCP server (`npx -y rea-agents@<v> mcp`) |

Node is bundled as a portable runtime in the installer so the user does not manage versions, and
pinned so REA's minimum is always satisfied.

## 4. Frontend stack

| Package | Purpose |
|---|---|
| React 19 + TypeScript | UI |
| Vite 6 | dev server + build |
| Tailwind CSS 4 | styling |
| shadcn/ui (Radix) | accessible primitives, owned in-repo |
| Zustand | local UI state |
| TanStack Query | server state, caching, retries |
| `react-markdown` + `remark-gfm` + `rehype-highlight`/Shiki | transcripts and code |
| Monaco Editor | diffs, file viewer, code workspace |
| `wavesurfer.js` | audio waveforms and playback |
| `lucide-react` | icons |
| Zod | runtime validation of API payloads |
| Vitest + Testing Library | component tests |
| Playwright | E2E against the real app |

## 5. External applications

| App | Version | Role | Install |
|---|---|---|---|
| **OVMS** | 2026.4.1 Windows zip | the inference server | bundled, `--rest_port 8000` |
| **GIMP** | 3.0.8+ | image editing, plugin host, embedded dock | optional, detected |
| **OpenVINO AI plugins for GIMP** | latest | in-GIMP AI tools | optional |
| **Ghidra** | 11.x | native binary analysis for REA | optional, ~1.5 GB |
| **JADX** | latest | APK decompilation for REA | optional |
| **JDK** | 21 | required by JADX/Ghidra | optional |
| **.NET SDK / ILSpy CLI** | 8/9 | .NET analysis for REA | optional |

## 6. Build, packaging, release

| Tool | Purpose |
|---|---|
| `uv` | Python env + lockfile (`uv.lock`), far faster than pip on Windows |
| PyInstaller (onedir) | ship the Python app; onedir avoids slow onefile extraction and plays well with OVMS binaries |
| Inno Setup | per-user installer, no admin, uninstaller, file associations |
| `pnpm build` | static UI into `backend/static` |
| GitHub Actions (windows-latest) | lint, test, build installer artifact |
| `ruff` | lint + format |
| `mypy --strict` | types on core modules (memory, gateway, governor) |
| `eslint` + `tsc --noEmit` | UI checks |
| `pre-commit` | run the above before commit |

## 7. Development and debugging tools

| Tool | Use |
|---|---|
| **OpenCode** | writes the app; the user debugs — the plan assumes an AI-authored codebase, so readability and tests matter more than cleverness |
| `mitmproxy` | capture third-party OpenAI client traffic during integration tests (loopback only) |
| DB Browser for SQLite / `sqlite3` CLI | inspect memory DB, run the eval queries |
| Process Explorer / RAMMap | find the real memory ceiling, detect leaks |
| WPR/WPA (optional) | only if GPU/NPU stalls need deep analysis |
| Chrome DevTools via `--devtools` | profile the WebView |
| `curl` / `hey` | latency and load checks on the gateway |
| Playwright trace viewer | reproduce UI bugs |

## 8. What gets installed on this machine, in order

```powershell
# 1. toolchain
winget install --id Git.Git -e
winget install --id astral-sh.uv -e
winget install --id OpenJS.NodeJS.LTS -e        # 24.x, satisfies REA ≥ 24.22.1
winget install --id OpenJS.NodeJS -e            # only if you want 26.12.1+ instead
npm install -g pnpm

# 2. OVMS
# download ovms_windows_2026.4.1_python_off.zip from the GitHub release,
# extract to C:\AI\ovms, add to PATH

# 3. optional
winget install --id GIMP.GIMP.3 -e
winget install --id Oracle.JDK.21 -e
# Ghidra + JADX from their releases, only if you want native/APK RE

# 4. app dev
git init
uv init --python 3.12
uv add fastapi uvicorn pydantic sqlalchemy alembic aiosqlite httpx numpy tokenizers orjson structlog tenacity psutil pywin32 keyring pywebview python-multipart pillow opencv-python-headless soundfile docling
```

## 9. Dependency risks and pinning

- **Docling moves fast.** Pin an exact version; the VLM API changed recently (`ApiVlmEngineOptions` /
  `VlmEngineType.API`). The sanitiser is coupled to its `DocTags` output, so upgrades require
  re-running the 30-document fixture suite.
- **OVMS ↔ OpenVINO version pairing.** Use the OVMS Windows zip as shipped; do not mix a separately
  installed OpenVINO runtime for inference. `openvino` in Python is used only to enumerate devices.
- **REA Node minimum** is a hard gate; the installer checks and refuses to enable the RE workspace
  with an incompatible runtime rather than failing at first use.
- **GIMP 3 / GTK3** is the embedding target. A future GIMP GTK4 migration would invalidate the
  reparenting approach; the dock therefore probes capability at launch and falls back automatically.
- **Video (v2)** needs `openvino_genai` pinned separately from the OVMS runtime; keep them in the
  same OpenVINO release train to avoid plugin ABI mismatches.

## 10. Repository layout (target)

```
aio-local/
├─ apps/
│  ├─ desktop/                 # pywebview shell
│  ├─ control-plane/           # FastAPI :7317
│  ├─ gateway/                 # FastAPI :8002 (memory lives here)
│  └─ docling-worker/          # :8003
├─ packages/
│  ├─ memory/                  # schema, retrieval, filing, validation
│  ├─ models/                  # catalog, governor, ovms config generation
│  ├─ integrations/            # opencode, rea, gimp, docling adapters
│  └─ shared/                  # types, logging, config, ids
├─ ui/                         # React app
├─ tests/
│  ├─ unit/ integration/ e2e/ fixtures/ bench/
├─ installer/
├─ docs/                       # this plan + ADRs
└─ pyproject.toml
```

Monorepo, one lockfile per ecosystem (`uv.lock`, `pnpm-lock.yaml`), ADRs in `docs/adr/` for every
decision that is expensive to reverse (gateway-owns-history, single OVMS process, SQLite over a
vector DB, pywebview over Electron).
