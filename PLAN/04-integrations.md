# 04 — Integrations

Every integration connects to the **gateway** (:8002), not directly to OVMS, except raw image/audio
calls, which the gateway proxies without memory-wrapping.

---

## 1. Frontend compatibility

Open WebUI is **not part of this project** (decided). Our React UI is the only shipped frontend.

The gateway still speaks OpenAI `/v1/chat/completions`, because that is the cheapest insurance
against lock-in and it is how OpenCode and REA attach. That means:

- any future client (a CLI, a VS Code extension, a phone bridge) gets memory for free;
- the UI is not special — it cannot do anything the API cannot;
- tests exercise the API directly, not through the UI.

Because the gateway **rebuilds history from its own Exact DB** and ignores client history, a new
client does not need to implement compaction, trimming, or memory at all. It only needs to send the
current turn and a conversation key.

---

## 2. OpenCode (Code workspace)

### 2.1 Process

`opencode serve --hostname 127.0.0.1 --port 4096`, started when the Code workspace opens, kept warm
if the user pins it.

### 2.2 Provider configuration

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "aio": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "AIO Local",
      "options": {
        "baseURL": "http://127.0.0.1:8002/v1",
        "apiKey": "local-gateway-token",
        "headers": { "X-AIO-Client": "opencode" }
      },
      "models": {
        "qwen3.5-9b": {
          "name": "Qwen3.5 9B",
          "tool_call": true,
          "limit": { "context": 8192, "output": 2048 }
        }
      }
    }
  },
  "model": "aio/qwen3.5-9b",
  "permission": { "edit": "ask", "bash": "ask", "webfetch": "ask" }
}
```

If OpenCode's provider options do not forward custom headers, the app runs a 40-line header-injecting
proxy on `:8004` in front of `:8002`. Verify in Phase 0 (5-minute test), because it decides whether
that proxy exists at all.

### 2.3 How the conversation key is derived

OpenCode session id → `X-AIO-Conversation: code/<session-id>`. The adapter that creates the session
writes the mapping into `app.db`, so the gateway can resolve it even if the header is lost.

### 2.4 UI

- Left: repo file tree (read-only preview).
- Centre: chat + tool-call timeline (each bash/edit call is a collapsible card).
- Right: diff viewer (Monaco) with Accept/Reject per hunk.
- Bottom: tool output stream, read-only; a separate "open terminal" button spawns PowerShell bound
  to the repo.
- Approvals: OpenCode permission events surface as modal cards. Nothing auto-approves writes or
  shell commands.

### 2.5 Memory for code

Namespace `code/<repo-slug>`. The filing model extracts decisions, conventions, failed approaches
and file-level facts. Retrieval injects a short "project memory" block before the task prompt.

Phase 8 ships passthrough first (no memory) to prove the integration, then adds the namespace. This
matches the brief and reduces the number of moving parts in one step.

### 2.6 Model fit

A 9B model at 8 k context is enough for focused edits, not for whole-repo refactors. The UI sets
expectations: the agent is scoped to the open file + selected context by default, with "expand
context" as an explicit action. When a task exceeds the window the app says so instead of silently
truncating.

---

## 3. REA (Reverse Engineering workspace)

`https://github.com/morluto/rea` — MIT, npm `rea-agents`, MCP server, local-only analysis.

### 3.1 Verified constraints

- **Node**: 22.x ≥ 22.23.4, 24.x ≥ 24.22.1, or 26.12.1+. The machine has **26.7.0**, which is below
  the minimum. Fix: bundle portable Node 24 LTS (≥ 24.22.1) or upgrade.
- **MCP entry point**: `npx -y rea-agents@<pinned> mcp`, **stdio** transport (confirmed in REA's
  client-configuration docs).
- **Analysis engines**: native binaries need Hopper, Ghidra, IDA or Kuna. Ghidra on Windows is
  described by REA as experimental. Everything else — JS/Electron, desktop installs, .NET, APK
  (JADX + JDK), PyInstaller, Go, SQLite — works without one.
- **Context control**: `REA_MCP_TOOL_FAMILIES` limits the exposed tool set;
  `REA_MCP_OUTPUT_SCHEMA_PROFILE=compact` and `REA_MCP_MAX_RESPONSE_BYTES` keep results from blowing
  the 8 k window.
- We do **not** run `rea setup` (it rewrites agent configs). We write our own MCP registration in
  the app's config directory and pass env vars explicitly.

### 3.2 Integration shape

REA gets its own workspace and its own agent loop; it is not handed to OpenCode in v1.

```
UI → control plane → RE session
      → gateway /v1/chat/completions with tools = REA tool subset
      → OVMS (Qwen, tool parser auto-detected in 2026.3+, guided generation on)
      → tool_calls → MCP client (stdio) → rea-agents mcp
      → results trimmed by size → back to OVMS → final answer
```

**Tool-call reliability**: OVMS 2026.3+ detects the tool parser from the model chat template, and
2026.4 supports `--enable_tool_guided_generation`, which constrains output to the tool JSON schema.
Turn it on for the RE and Code models. This turns the biggest integration risk (R3) into a
configuration detail — but Phase 0 still tests it end-to-end.

### 3.3 Workspace UX

- **Target picker**: file/folder/APK/installation directory + an authorization checkbox.
- **Session view**: chat left; evidence right (code, strings, symbols, file/offset references).
- **Tool timeline**: tool name, arguments, duration, result size, truncated flag.
- **Report export**: Markdown with findings, evidence, unknowns → `artifacts/re/`.
- **Memory**: pinned findings become `kind='re'` summaries linked to the artifact hash and target
  path, retrievable in later sessions on the same target.

### 3.4 Safety

- Read-only by default; write-capable tools disabled unless "allow modifications" is enabled per
  session.
- No network egress from the RE workspace; web-recording features off in v1.
- Files are hashed and recorded; analysing outside the chosen root needs a second confirmation.
- The app never executes an analysed binary and says so in the UI. It does not pretend to sandbox.

### 3.5 RAM / disk

Node + REA ≈ 300 MB. Ghidra ≈ 1.5 GB disk, 2–3 GB RAM. The governor may unload the 9B while Ghidra
is running; the UI explains the trade instead of silently degrading.

---

## 4. GIMP — embedded in v1

### 4.1 What is actually true

- GIMP 3.x uses **GTK3** (not GTK4) and ships **single-window mode by default** — verified in the
  3.0 release notes and manual.
- GTK3's `GtkSocket`/`GtkPlug` are **X11-only** and are gone in GTK4, so widget-level embedding is a
  dead end on Windows.
- What *can* work is **Win32 cross-process window reparenting**: take GIMP's top-level `HWND` and
  `SetParent` it into a host window we own. This is the same mechanism used by many "app inside an
  app" tools, and it is explicitly not a supported Windows scenario — so it is built as a
  best-effort layer with a guaranteed fallback, never as the only path.

### 4.2 Architecture: the GIMP dock

```
AIO Local main window (pywebview / WebView2)
   └── owned window "GIMP dock" (native, no taskbar entry)
         └── child host HWND (created by gimp-host.exe)
               └── GIMP main HWND (SetParent'd, WS_CHILD)
```

- `gimp-host.exe` is a tiny Win32 host (Python + pywin32, message pump) so a GIMP hang cannot freeze
  the UI's WebView thread.
- The dock window is **owned** by the main window: it follows move, minimise, restore, and closes
  with the app.
- GIMP is launched with a dedicated profile directory (`--gimpdir`) so single-window mode, theme and
  plugin paths are controlled by the app rather than by whatever the user last did.
- Files are opened via GIMP's command line (`gimp --new-instance <file>` or the running instance's
  remote interface), so the dock and the file round-trip share one mechanism.

### 4.3 Reparenting sequence (and why each step exists)

1. Start GIMP if not running; wait for a top-level window whose class/title matches GIMP.
2. Read `GWL_STYLE`; clear `WS_POPUP` and `WS_OVERLAPPEDWINDOW`, set `WS_CHILD` **before**
   `SetParent` (otherwise Windows leaves the styles inconsistent).
3. `SetParent(gimpHwnd, hostHwnd)`, then `SetWindowPos` to fill the host, `ShowWindow(SW_SHOW)`.
4. Match DPI awareness between host and GIMP to avoid blurry or mis-scaled rendering.
5. On host resize, forward the new client size and let GIMP lay out.
6. Verify: `GetParent(gimpHwnd) == hostHwnd` and the window has non-zero client size after 2 s;
   otherwise declare failure and detach.

Known failure modes and the response:

| Failure | Response |
|---|---|
| Focus/accelerator weirdness | keep the dock as an owned window, not a child of the WebView; forward only resize/close; let GIMP own keyboard focus inside its HWND |
| Modal dialogs open as separate windows | accept it in v1 (they appear above the dock); optionally hook `EVENT_OBJECT_CREATE` later to reparent dialogs |
| DPI change / monitor move | re-apply DPI awareness and resize; if it breaks, auto-detach with a one-time notice |
| GIMP theme ignores dark mode | ship a matching GIMP theme via the dedicated profile; cosmetic only |
| Reparent causes GIMP to crash or freeze | watchdog restarts GIMP detached; the dock shows "GIMP detached — click to re-embed" |
| User prefers a normal window | Detach button, persisted per user |

### 4.4 Fallback ladder (automatic, in order)

1. **Native dock** — reparented, resizable, follows the main window.
2. **Owned window** — same window, not reparented, positioned beside the app; still managed by AIO.
3. **External GIMP** — plain launch, with the file round-trip bridge.

The UI never shows a broken half-embedded window: a failed embed transitions to level 2 or 3 and
tells the user once.

### 4.5 File round-trip (still built, for reliability)

`Edit in GIMP` writes `artifacts/gimp/in/<id>.png` + sidecar JSON (prompt, mask, source), opens it in
the dock, and watches `artifacts/gimp/out/<id>.png` for the result with an "Import result" action.
This path works even if every embedding trick fails.

### 4.6 Plugin

`intel/openvino-ai-plugins-gimp` targets GIMP 3 / Python 3 / Windows. It runs its own OpenVINO
inference, so it duplicates models and memory. Policy:

- detect and report whether the plugin is installed;
- point it at the app's model directory where compatible, otherwise keep it CPU-only;
- warn before launching GIMP while the 9B is loaded (GIMP + plugin + 9B can exceed 16 GB);
- PhotoGIMP is an optional cosmetic profile, not a dependency.

### 4.7 Phase 0 spike (1 day, before commitment)

Reparent Notepad, then GIMP 3.0.8, into a throwaway host window. Measure: input latency, keyboard
focus, DPI behaviour, dialog behaviour, resize cost, crash recovery. If the spike shows the dock is
unusable, the plan still ships level 2/3 — the user gets "GIMP managed by the app" and we revisit
true embedding with real data instead of hope.

---

## 5. Docling bridge (documents)

### 5.1 Pipeline

```
upload → type detect → Docling (layout, OCR, tables) → Markdown + JSON + page/bbox provenance
       → chunk (400–800 tokens, 15 % overlap) → embeddings → knowledge.db
       → optional VLM pass for image/diagram pages (main Qwen, GPU)
       → summary + tags (NPU filing model) → archive artifact
```

### 5.2 The ID problem and the sanitiser

Docling's VLM pipeline expects output in its `DocTags` grammar with valid references. Local VLMs
frequently emit malformed location tokens or hallucinated IDs, which makes Docling reject the page.
This is the exact issue the brief flags.

Design: a dedicated endpoint on the gateway, `/v1/docling/vlm`, used via Docling's
`ApiVlmEngineOptions`:

```
Docling → POST /v1/docling/vlm (image + DocTags instruction)
        → gateway forwards to OVMS (Qwen VLM, temperature 0, max 1024)
        → sanitiser:
            1. strip prose preamble / markdown fences
            2. normalise <loc_…> and [[x1,y1,x2,y2]] into Docling's expected token
            3. clamp coordinates to page bounds; drop refs with area < 0.1 % or > 100 %
            4. drop unknown ID references; renumber surviving refs contiguously
            5. if < 60 % of refs survive → retry once with a stricter prompt
            6. if still invalid → emit a page-level text block (graceful degradation)
        → Docling parses the cleaned DocTags
```

Every pass logs before/after snippets. Thirty real documents become the fixture suite.

### 5.3 Isolation

Docling runs in its own process with a per-document timeout (5 min), a memory cap, and
kill-and-restart. The UI shows page-level progress.

---

## 6. Image generation

- `POST /v3/images/generations` — text→image, SD 1.5 int8, 512×512 default.
- `POST /v3/images/edits` with `mask` — inpaint and outpaint (OVMS 2026.4 exposes both).
- Gateway adds: optional prompt enhancement, safety filter, PNG metadata (prompt, seed, model,
  steps), thumbnails, gallery indexing.
- Every generation becomes an artifact + a `kind='image'` summary, so "the logo I made yesterday" is
  retrievable in chat.
- Device exclusivity is handled by the RAM governor (`03` §2, `01` §4).

---

## 7. Audio

- `POST /v3/audio/transcriptions` — file or microphone chunks; streaming responses supported.
- `POST /v3/audio/translations` — non-English → English.
- `POST /v3/audio/speech` — Kokoro, voice + language, streamed to the player.
- Transcripts are archived with `modality='audio'` and timestamps, so retrieval can point at a
  moment in a recording.
- Voice notes become memory ("what did I say about the budget?" retrieves the segment).

---

## 8. Knowledge (RAG)

Deliberately separate from conversational memory:

| | Memory | Knowledge |
|---|---|---|
| Source | the user's turns | uploaded documents |
| Written by | system, automatically | user, explicitly |
| Scope | personal, cross-conversation | collection-based |
| Ranking | recency + salience + vector | vector + BM25 (+ optional rerank) |
| UI | invisible, Inspector on demand | visible library with citations |

Collections carry name, description, tags, embedding model and chunking profile. Deleting a document
is soft (`deleted_at`, 30-day purge) because summaries and chunks reference it. Citations carry
`document_id`, `page`, `bbox`, `char_range`; the UI opens the source at the right position.

### 8.1 Domain packs (the brief's knowledge list)

The seven domains are shipped as **collection templates**, not hardcoded prompts. Each template sets
a default system instruction, retrieval hints, citation policy and an optional starter reading list.
The user can create their own on top, and nothing is downloaded without consent.

| Domain | Template behaviour | Extra |
|---|---|---|
| STEM | derivation-friendly answers, units checked, formulas rendered | LaTeX rendering on |
| Coding | language-aware, runnable snippets, references repo memory when in a code session | links to Code workspace |
| LLMs & AI | architecture/paper mode, distinguishes fact from speculation | arxiv-friendly citations |
| Networking | protocol/OSI framing, commands shown per OS | config snippets |
| Computer architecture | microarchitecture vocabulary, latency/throughput tables | |
| Operating systems | kernel/scheduler/memory framing, Windows + Linux commands | |
| Medical diagnosis and treatment | **citation-only**, conservative, refuses to diagnose from insufficient evidence | not-a-doctor disclaimer forced; mode disabled until the user reads the notice |

A domain pack never changes the memory algorithm — it only changes the system instruction, the
retrieval filter (`collection tag = domain`) and the output contract (e.g. medical requires
citations). That is what keeps "the memory plan is followed for each part" true as features are
added.
