# 06 — Roadmap, Testing, Risks, Licensing

## 1. How the phases are ordered, and why

Three rules drive the sequence:

1. **Measure before building.** The brief's open questions (Qwen's usable context, NPU model choice,
   NPU speed, NPU prompt ceiling) plus two integration unknowns (tool-call reliability, GIMP
   embedding). Guessing any of them costs a rewrite later.
2. **Memory first, features after.** Every workspace depends on it; retrofitting it is the classic
   failure mode of this kind of app.
3. **Vertical slice early.** By the end of P3 the app must genuinely chat with memory, even if every
   other workspace is a placeholder. That proves the architecture before the long tail.

Durations are "focused weeks" for one developer with an AI writing most of the code and the user
debugging. They are planning estimates, not commitments.

---

## 2. Phase 0 — Measurement and spikes (1–2 weeks)

No product code. Produce numbers and fixtures.

| # | Task | Output |
|---|---|---|
| 0.1 | Install OVMS 2026.4.1 Windows binary, pull the 9B, confirm `/v3` chat + streaming from curl | `bench/ovms.md`: working commands, config keys |
| 0.2 | Context ladder: 1 k→32 k prompts at `cache_size` 1/2/4 GB | `bench/context.md`: usable window, TTFT, tok/s, RSS |
| 0.3 | Benchmark 3 NPU candidates on filing prompts; test `MAX_PROMPT_LEN` 2 k/8 k/16 k | `bench/npu.md`: tok/s, JSON validity %, best model, safe prompt ceiling |
| 0.4 | RAM ceiling per mode (chat, image, code, RE, GIMP) | `bench/ram.md`: resident/commit per process |
| 0.5 | SD 1.5 int8 generation profile on Arc 130T | `bench/image.md`: seconds/image by steps/resolution |
| 0.6 | Docling on 20 documents, VLM sanitiser failure rate | `bench/docling.md`: pages/min, failure taxonomy |
| 0.7 | OpenCode ↔ OVMS tool-calling loop with `--enable_tool_guided_generation` | `bench/tools.md`: does the loop complete reliably? |
| 0.8 | Verify Node/REA on this machine | `bench/rea.md`: Node 26.7.0 < 26.12.1 → bundle 24 LTS; Ghidra feasibility |
| 0.9 | **GIMP embedding spike**: reparent Notepad, then GIMP 3.0.8 into a throwaway host | `bench/gimp.md`: focus, DPI, dialogs, resize, crash recovery, go/no-go for the dock |

**Exit criteria**: all numbers written down; the model matrix in `03` updated with real values;
no unresolved "we'll find out later" that affects architecture.

**If 0.7 fails** (tool calling is still unreliable even with guided generation), the Code and RE
workspaces switch to an app-level tool loop with a stricter JSON protocol. Better to know in week 2
than month 5.

**If 0.9 fails**, the GIMP feature ships as an app-managed owned window (level 2) rather than a
reparented dock, and the plan's GIMP section is updated with the measured reason. The user still gets
an embedded-feeling experience without a fragile hack.

---

## 3. Phase 1 — Skeleton (1–2 weeks)

- Repo, `uv` env, lint/format/type-check, CI (local pre-commit + GitHub Actions).
- Supervisor with health checks, job objects, port allocation, log rotation.
- pywebview shell + React app shell + design tokens + routing.
- Control plane with `/system/*`, event bus, settings, DPAPI secrets.
- OVMS launcher with a pinned template; `GET /v1/config` parsed into model state.
- Diagnostics bundle.

**Done when**: the app opens, starts OVMS, shows real hardware and process health, and shuts down
cleanly with no orphan processes.

---

## 4. Phase 2 — Model fleet and load/unload (2 weeks)

- Catalog (`models.json`), pull with resume + checksum, upload validation, delete.
- OVMS config generator + reload + state machine.
- RAM governor with refusal UX and projection display.
- Device picker with capability explanations.
- Chat UI streaming **directly to OVMS** (no memory yet) to validate the pipe.

**Done when**: the user can pull the 9B, pick GPU, load it, chat, unload it, load SD 1.5, generate
an image — and the governor correctly refuses to hold both.

---

## 5. Phase 3 — Memory v1: exact recall (2–3 weeks)

- `memory.db` schema, migrations, single-writer queue.
- Archive on request/response, dedupe, edits/regeneration lineage.
- Embeddings (CPU) + FTS5 + RRF retrieval.
- Conversation-key resolution, request classifier, side-request skip.
- Prompt assembly + token planner + trimming, with the Inspector.
- Gateway becomes the frontend's endpoint.
- Latency benchmark at 50 k messages.

**Done when**: a fresh conversation can recall a fact from a week-old conversation, p95 retrieval
< 150 ms, and the Inspector explains exactly which rows were used.

---

## 6. Phase 4 — Memory v2: summaries and NPU filing (2–3 weeks)

- Topics, summaries, M2M links, edges.
- NPU filing worker with strict JSON, validation, retry, `unsorted` fallback.
- Consolidation, salience decay, contradiction pass.
- Memory browser UI (topics, summaries, graph, queue, retry).
- Adversarial tests for ID hallucination.

**Done when**: summaries are created automatically after replies, every link validates, a killed
NPU model degrades gracefully, and the summary DB stays under ~40 summaries/topic.

---

## 7. Phase 5 — Documents and knowledge (2–3 weeks)

- Docling worker + progress + timeouts.
- DocTags sanitiser with 30-document fixture suite.
- Chunking, embeddings, `knowledge.db`, collections, citations.
- Document workspace UI + library.
- Document Q&A through the gateway (mode `doc`).

**Done when**: a 200-page PDF is ingested without freezing chat, cited answers point to the right
page, and the sanitiser handles malformed VLM output without losing the page.

---

## 8. Phase 6 — Images and embedded GIMP (3 weeks)

- SD 1.5 generation UI, seeds, presets, gallery, metadata.
- Inpaint/outpaint canvas (mask brush, expand canvas).
- **GIMP dock**: `gimp-host` process, owned window, reparenting, detach, watchdog, fallback ladder.
- GIMP profile management (`--gimpdir`), single-window mode, plugin detection.
- File round-trip bridge as the reliability floor.
- Image memory: artifacts + `kind='image'` summaries.
- Governor exclusivity UX.

**Done when**: generation works at 512×512 in a predictable time, inpaint respects the mask,
"the logo from yesterday" is retrievable in chat, and GIMP opens *inside* the app with a working
detach and an automatic fallback if embedding fails.

---

## 9. Phase 7 — Audio (1–2 weeks)

- Whisper transcription + streaming, translation, timestamps.
- Kokoro TTS with voice/language picker and playback.
- Audio artifacts + memory integration.
- Microphone capture with device selection and level meter.

**Done when**: a 10-minute recording transcribes with timestamps, is searchable, and can be spoken
back.

---

## 10. Phase 8 — Code workspace (3–4 weeks)

- OpenCode sidecar lifecycle, config generation, health.
- Chat + tool timeline + diff viewer + approvals + terminal.
- Header injection so memory applies (`X-AIO-Conversation`).
- Repo-scoped memory namespace.
- Failure handling: model too small for a task → clear message, not a broken loop.

**Done when**: a real bug in a real repo can be fixed through the UI with visible diffs and
approvals, and the app survives OpenCode crashing mid-edit.

---

## 11. Phase 9 — Reverse engineering (3 weeks)

- Node 24 LTS ≥ 24.22.1 bundled or required.
- REA MCP client (stdio), tool allowlist, session lifecycle.
- RE workspace UI, evidence panel, report export.
- Authorization gate, read-only default, no egress.
- Optional Ghidra setup wizard (experimental on Windows).

**Done when**: a JS/Electron app or a native binary can be analysed end-to-end, findings are cited
to files/offsets, and a report is exported.

---

## 12. Phase 10 — Hardening and packaging (2–3 weeks)

- Installer (per-user, no admin), uninstaller, model-dir relocation.
- First-run wizard: hardware probe, model download choices, disk check.
- Accessibility, keyboard shortcuts, i18n scaffolding.
- Crash reporting **off by default**, local only.
- Performance pass: startup < 5 s, idle RAM < 1.2 GB, no UI jank while streaming.
- Linux/Bazzite port: replace pywebview backend with GTK/Qt WebKit, service files, no Windows APIs
  outside a platform shim.

**Done when**: a clean Windows VM installs, runs the wizard, chats with memory, and uninstalls
without leftovers.

---

## 13. Testing strategy

### 13.1 Layers

| Layer | Tool | Scope |
|---|---|---|
| Unit | pytest | token planner, RRF, validators, sanitiser, config generation |
| Property | hypothesis | append-only, dedupe, trim invariants |
| Contract | pytest + respx | gateway ↔ OVMS, OpenAI schema compatibility |
| Golden | pytest | 30 prompt-assembly scenarios, 30 Docling pages |
| Integration | pytest + real OVMS | load/unload, streaming, tool calls |
| E2E | Playwright | chat, memory recall, image, upload, code approval |
| Performance | custom bench | retrieval p95, TTFT, tok/s, RSS over 2 h |
| Chaos | script | kill each process, fill disk, truncate WAL, unplug network |
| Security | manual + zap-lite | path traversal, SSRF via URL fields, token leakage in logs |

### 13.2 Quality gates per phase

1. All new unit tests pass; coverage of new modules ≥ 70 %.
2. No regression in retrieval latency or idle RAM beyond 10 %.
3. `ruff` + `mypy --strict` on new Python; `tsc --noEmit` + `eslint` on new TS.
4. A written manual test script executed once on the target laptop.
5. Logs contain no prompt text at `info` level (privacy).

### 13.3 The memory eval set

120 hand-written queries over a seeded 5 000-message corpus, labelled with the correct summary and
message IDs. Metrics: recall@5, MRR, and "wrong memory injected" rate (target < 2 %). This set is
built in P3 and grows with every bug report.

---

## 14. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | 9 B + SD + Windows cannot co-exist; user hits OOM | High | High | governor, refusal UX, mode switching, measured budgets in P0 |
| R2 | NPU filing model too weak → bad summaries | Medium | High | 3-candidate bake-off, fallback to Qwen3-4B, heuristic fallback, manual edit UI |
| R3 | Tool calling with Qwen 3.5 unreliable → Code/RE break | Low | High | auto-detected parser + guided generation; P0 spike; fallback to app-level tool loop |
| R4 | Docling VLM output invalid → documents fail | High | Medium | sanitiser, retry, page-level degradation, fixture suite |
| R5 | Embedded GIMP misbehaves (focus, DPI, dialogs, crash) | High | Medium | P0 spike, owned-window fallback, watchdog, detach button, file round-trip floor |
| R6 | REA Node/tooling requirements on Windows (Ghidra experimental) | High | Medium | bundle Node 24 LTS, start with JS/.NET/APK targets, Ghidra optional |
| R7 | 40 GB disk exhausted | Medium | High | size pre-checks, relocation, cleanup, "models you are not using" panel |
| R8 | Retrieval quality degrades as DB grows | Medium | High | eval set, RRF, consolidation, rerank option, Inspector for tuning |
| R9 | Memory writes duplicate/near-duplicate summaries | High | Medium | dedupe thresholds, consolidation, merge UI |
| R10 | Licence conflict (SD OpenRAIL, LTX-Video weights) | Medium | High | model licence gate, no bundled third-party UI, AGPL review before release |
| R11 | Streaming + archival race loses the last tokens | Medium | Medium | archive on `[DONE]` **and** on disconnect with partial flag |
| R12 | Windows Defender/antivirus slows model loads | Medium | Low | exclude model dir guidance, document it |
| R13 | pywebview/WebView2 inconsistencies | Low | Medium | browser fallback mode, Playwright runs against Chromium |
| R14 | GIMP reparenting breaks after a GIMP/GTK update | Medium | Medium | host shim isolated, capability probe at launch, automatic fallback ladder |
| R15 | Video expected but deferred | Low | Medium | explicitly v2; architecture (`video-worker`) documented now so it is not a rewrite |

---

## 15. Licensing

The app itself: **AGPL-3.0-or-later**.

| Component | Licence | Obligation / note |
|---|---|---|
| OVMS | Apache-2.0 | attribution, NOTICE file |
| OpenVINO runtime | Apache-2.0 | attribution |
| OpenCode | MIT (verify at P8) | attribution |
| REA | MIT | attribution |
| Docling | MIT | attribution |
| GIMP | GPL-3.0 | separate process, communicated via files/CLI/Win32 window handles; no linking |
| GIMP OpenVINO plugins | Apache-2.0 (verify) | separate process |
| Qwen3.5-9B | Apache-2.0 | attribution |
| SD 1.5 | CreativeML OpenRAIL-M | **use restrictions pass through to the user**; show before first generation; cannot claim unrestricted use |
| Whisper | MIT | attribution |
| Kokoro | Apache-2.0 | attribution |
| LFM2.5 / Qwen3 small | check per model | stored in catalog metadata |
| LTX-Video (v2 only) | check weights licence before shipping | do not assume Apache-2.0 |

AGPL note: separate processes communicating over loopback HTTP are generally separate works; the
app's own code stays AGPL. If we ever link a GPL library into the Python process, the whole app must
be GPL-compatible — the plan avoids this. A licence review is a P10 gate before any distribution.

---

## 16. What "done" means for v1

A non-technical user on this laptop can:

1. Install without admin rights, finish a wizard, and download only the models they need.
2. Chat, ask about a PDF they dropped in, generate and inpaint an image, edit it in the embedded
   GIMP dock, transcribe a meeting, fix a bug in a repo, and analyse a small binary — in one window.
3. Close the app, reopen it a week later, and ask "what did we decide about the Rust runtime?" and
   get the right answer with the original conversation quoted.
4. Never see a silent failure: every degraded path is visible, explained, and recoverable.

Explicitly **not** in v1: video generation (v2 via `openvino_genai` + LTX-Video), any third-party
web UI, cloud sync, multi-user accounts.
