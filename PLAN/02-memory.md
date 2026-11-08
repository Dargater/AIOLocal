# 02 — Memory System

This is the heart of the project and the part that must be exactly right. Everything else is a
client of it.

---

## 1. Invariants

1. **Exact DB is append-only.** No UPDATE of `content`; no DELETE. Edits and regenerations create
   new rows with `revises_id` / `superseded_by`.
2. **The LLM may only write topics, summary text, and links.** It can never write into the Exact DB,
   invent a message ID, or delete a row.
3. **Every ID the model returns is validated** against the exact set offered in that call. Invalid →
   one retry with the error → still invalid → the turn is filed under `unsorted` with a heuristic
   summary. Chat is never blocked by this.
4. **The model never sees client history.** The gateway rebuilds history from its own DB.
5. **Failure to retrieve is not failure to answer.** Any retrieval error degrades to "no memory"
   and is logged.
6. **Latency budget on the read path is hard**: 150 ms p95 without the small model, 400 ms p95 with it.

---

## 2. Data model

```sql
-- ────────────────────────────── EXACT (append-only) ──────────────────────────────
CREATE TABLE messages (
  id              TEXT PRIMARY KEY,          -- ULID (time-sortable)
  conversation_id TEXT NOT NULL,
  parent_id       TEXT,                      -- turn threading
  role            TEXT NOT NULL,             -- user|assistant|tool|system|document
  content         TEXT NOT NULL,
  content_hash    TEXT NOT NULL,             -- sha256(normalise(content))
  tokens          INTEGER,
  modality        TEXT NOT NULL DEFAULT 'text',
  attachments     TEXT,                      -- JSON array of artifact refs
  source          TEXT NOT NULL,             -- native|opencode|rea|api|import
  client_msg_id   TEXT,
  request_id      TEXT,
  revises_id      TEXT,                      -- this row replaces that row (edit/regenerate)
  superseded_by   TEXT,                      -- set once, when a revision arrives
  duplicate_of    TEXT,
  injected        INTEGER NOT NULL DEFAULT 0,-- detected boilerplate/RAG/system injection
  created_at      INTEGER NOT NULL,          -- unix ms
  meta            TEXT                       -- JSON: model, params, tool name, page refs…
);
CREATE INDEX ix_msg_conv  ON messages(conversation_id, created_at);
CREATE INDEX ix_msg_time  ON messages(created_at);
CREATE INDEX ix_msg_hash  ON messages(content_hash);
CREATE UNIQUE INDEX ux_msg_dedupe
  ON messages(conversation_id, role, content_hash, IFNULL(revises_id,''));

CREATE TABLE conversations (
  id          TEXT PRIMARY KEY,
  title       TEXT,
  source      TEXT NOT NULL,
  external_id TEXT,                -- client-side chat id, if the client sends one
  started_at  INTEGER,
  last_at     INTEGER,
  meta        TEXT
);
CREATE UNIQUE INDEX ux_conv_ext ON conversations(source, external_id);

-- ────────────────────────────── SUMMARY ──────────────────────────────
CREATE TABLE topics (
  id          TEXT PRIMARY KEY,
  slug        TEXT NOT NULL UNIQUE,        -- 'rust/async', 'medical/cardiology'
  name        TEXT NOT NULL,
  parent_id   TEXT REFERENCES topics(id),
  description TEXT,
  created_at  INTEGER NOT NULL
);

CREATE TABLE summaries (
  id          TEXT PRIMARY KEY,
  topic_id    TEXT REFERENCES topics(id),
  title       TEXT NOT NULL,
  body        TEXT NOT NULL,
  kind        TEXT NOT NULL,             -- episodic|profile|project|decision|fact|code|re
  entities    TEXT,                      -- JSON array of names/versions/paths
  salience    REAL NOT NULL DEFAULT 0.5, -- 0..1, decays, boosted on use
  confidence  REAL NOT NULL DEFAULT 0.7,
  valid_from  INTEGER,
  valid_to    INTEGER,                   -- NULL = current
  merged_into TEXT,                      -- consolidation pointer
  source      TEXT NOT NULL,             -- llm|manual|import
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL
);
CREATE INDEX ix_sum_topic ON summaries(topic_id, salience DESC);

-- M2M: one summary ↔ many exact messages, many summaries ↔ one message
CREATE TABLE summary_messages (
  summary_id TEXT NOT NULL REFERENCES summaries(id),
  message_id TEXT NOT NULL REFERENCES messages(id),
  weight     REAL NOT NULL DEFAULT 1.0,
  PRIMARY KEY (summary_id, message_id)
);
CREATE INDEX ix_sm_msg ON summary_messages(message_id);

-- relations between summaries (supersedes, contradicts, elaborates)
CREATE TABLE summary_edges (
  src TEXT NOT NULL, dst TEXT NOT NULL, rel TEXT NOT NULL, weight REAL DEFAULT 1.0,
  PRIMARY KEY (src, dst, rel)
);

-- ────────────────────────────── INDEX ──────────────────────────────
CREATE VIRTUAL TABLE messages_fts USING fts5(
  content, content='messages', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2');
CREATE VIRTUAL TABLE summaries_fts USING fts5(
  title, body, content='summaries', content_rowid='rowid', tokenize='unicode61 remove_diacritics 2');

CREATE TABLE embeddings (
  owner_type TEXT NOT NULL,             -- message|summary|chunk
  owner_id   TEXT NOT NULL,
  model      TEXT NOT NULL,
  dim        INTEGER NOT NULL,
  vec        BLOB NOT NULL,             -- float32 little-endian
  norm       REAL NOT NULL,
  PRIMARY KEY (owner_type, owner_id, model)
);

-- ────────────────────────────── OPERATIONS ──────────────────────────────
CREATE TABLE memory_jobs (
  id         TEXT PRIMARY KEY,
  kind       TEXT NOT NULL,             -- file_turn|consolidate|reindex|decay
  payload    TEXT NOT NULL,
  state      TEXT NOT NULL,             -- queued|running|done|failed|dead
  attempts   INTEGER NOT NULL DEFAULT 0,
  run_after  INTEGER,
  last_error TEXT,
  created_at INTEGER NOT NULL
);

CREATE TABLE retrieval_log (
  id         TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  query      TEXT,
  plan       TEXT,     -- JSON: candidates, scores, chosen, trimmed
  latency_ms INTEGER,
  created_at INTEGER NOT NULL
);
```

`knowledge.db` is structurally separate (`documents`, `chunks`, `collections`, `chunks_fts`,
`embeddings` with `owner_type='chunk'`) so conversational memory and uploaded knowledge never
contaminate each other's rankings. Fusion across them is explicit and mode-driven.

---

## 3. Conversation key resolution

The gateway must know which conversation a request belongs to before it can rebuild history.

Precedence:

1. `X-AIO-Conversation` header (our UI, OpenCode adapter — always set).
2. Body `chat_id` or `metadata.chat_id` (some external clients send one).
3. `conversations.external_id` lookup by `(source, external_id)`.
4. Fallback fingerprint = `sha256(source + model + first_user_message_normalised)`; stable for a
   conversation that starts with the same first message.
5. Last resort: a new conversation per request (memory read still works globally, history is empty).

The fingerprint path is what makes a third-party client work without any plugin: as long as the
first user message of the chat is unchanged, the key is stable.

---

## 4. Request classification

Cheap, deterministic, before any model call:

| Class | Signals | Behaviour |
|---|---|---|
| `side` | `max_tokens ≤ 64`, prompt matches title/tag/summarise-chat patterns, `stream:false`, or `X-AIO-Side: 1` | passthrough, **no memory, no archive** |
| `probe` | `GET /v1/models`, empty messages | passthrough |
| `tool_loop` | last user turn followed by `assistant.tool_calls` + `tool` messages | passthrough chain + rebuilt *pre-turn* history |
| `doc` | attachment/doc reference, or `X-AIO-Mode: document` | document pipeline + knowledge retrieval |
| `image` | `/v1/images/*` | separate route, no text memory read; result archived as artifact + summary |
| `chat` | default | full memory pipeline |

Client-generated title/tag requests are the main source of noise; they are distinguishable by their
short output budget and their template. The classifier is a table of predicates, not model calls,
and is unit-tested against captured real payloads (Phase 0).

---

## 5. Read path (reply-critical)

Target: **p95 < 150 ms** end-to-end before the first token reaches OVMS.

### 5.1 Inputs

- `q` = current user text (attachments replaced by short descriptors)
- `mode` = chat | knowledge | code | re | medical
- `conversation_key`
- `budget_memory_tokens` (from the token planner)

### 5.2 Steps

```
1  normalise(q)                     lowercase, strip markdown, collapse whitespace, keep code fences
2  classify intent                  heuristic: continuation | lookup | project | document | medical
3  embed(q)                         CPU, 384–1024 dim, ~8–25 ms
4  candidates:
     a) vector top-40 over summaries      (brute-force numpy on ≤ 50k vectors, ~5 ms)
     b) FTS5 top-40 over summaries+msgs   (BM25, ~2 ms)
     c) topic filter                      (entities/slugs found in q)
     d) recency window                    (last 30 messages of this conversation)
5  fuse                              RRF (k=60) across a,b,c
6  score                             score = rrf · (0.75 + 0.25·salience) · recency_boost
7  gate                              if top score < τ_low → return "no memory"
8  ambiguous?                        only if (top1 − top2) < δ and mode ∈ {chat,medical}
                                     → optional NPU query rewrite, 120 ms cap, else skip
9  select summaries                  until 60 % of memory budget
10 expand                            messages linked to selected summaries, newest-first,
                                     capped at 40 % of budget
11 order                             exact conversations first, then summaries, then recent history
12 trim                              see §7 planner; current input always wins
```

**Why RRF and not weighted cosine.** BM25 and cosine are not on the same scale; rank fusion is
stable without per-corpus calibration, which matters when the DB grows from 100 to 100 000 rows.

**Why no LLM in the read path.** On this NPU, a 1.2 B model costs 150–400 ms for a short completion
and serialises with filing. Putting it in front of every reply would double perceived latency for
little gain. It is used only for the ambiguous-query case, and only when the deterministic score
gap is small.

### 5.3 The "make it shorter" problem

Short follow-ups have no retrievable content. Handled by:

- intent = `continuation` when the query is < 6 tokens, contains a pronoun/deictic, or matches a
  small phrase list ("make it shorter", "again", "why", "continue", "fix that").
- For `continuation`, retrieval is **skipped** and instead the last N turns of *this conversation*
  are loaded (N configurable, default 6 messages / 1200 tokens).
- Summaries are still attached if the conversation has a project summary.

### 5.4 Assembly format

```
[system]
You are AIO Local. <mode instructions>
<medical disclaimer if mode=medical>

## What you remember (retrieved)
### Summary — <title>  [<topic>]
<body>
### Exact excerpt — <date>  (conversation <id>, messages <id>…<id>)
<verbatim messages, oldest→newest>

## Recent conversation (same chat)
<last N turns>

## Rules
- The current user message overrides anything remembered.
- If memory conflicts with the current message, follow the current message.
- Cite memory ids only when asked to explain what you remember.
[/system]

<messages: current turn (+ tool chain)>
```

Client-supplied `system` messages are dropped except for an explicit allowlist (user-authored
system prompt set in Settings). This removes client-injected RAG/context blocks generically, without
per-client rules.

### 5.5 Retrieval logging

Every request writes one `retrieval_log` row (candidates, scores, chosen ids, trims, latency) so the
Memory Inspector can answer "why did the model see this?" and so we can tune τ and δ on real data.

---

## 6. Write path

### 6.1 Immediate (synchronous, < 5 ms)

On request arrival:

- archive the user message (`source`, `request_id`, `client_msg_id`); `INSERT OR IGNORE` on the
  dedupe index makes retries idempotent.
- detect `injected=1` if content matches known injected patterns (client RAG scaffolds,
  "### Task Instructions", RAG citation scaffolds) — such rows are stored but excluded from
  retrieval and never summarised as the user's words.

On stream completion (including client abort with partial text):

- archive the assistant message; if `revises_id` points at an older assistant row, set its
  `superseded_by`.
- create a `file_turn` job.

### 6.2 Background filing (NPU, serial)

Input to the small model, kept under ~1500 tokens:

```
TURN (verbatim, truncated to 700 tokens)
CANDIDATE TOPICS (max 40 slugs + 1-line descriptions)
CANDIDATE EXISTING SUMMARIES (max 5, title + 40-token body)
ALLOWED MESSAGE IDS (the ids just archived + up to 10 recent ids)
```

Output (strict JSON, constrained by grammar / structured output):

```json
{
  "topics": [{"slug": "rust/async", "name": "Rust async", "parent": "rust"}],
  "summaries": [
    { "id": null,
      "title": "Tokio runtime selection for the CLI",
      "body": "User chose multi-thread Tokio; single-thread rejected because …",
      "kind": "decision",
      "message_ids": ["01J...","01J..."],
      "links": [{"to": "01J...", "rel": "elaborates"}] }
  ],
  "merge_into": "01J...",
  "salience": 0.7
}
```

**Validation** (all server-side):

1. JSON schema (types, lengths, max 3 topics, max 2 summaries per turn).
2. `message_ids ⊆ allowed` — any unknown id rejects the whole response.
3. `merge_into` must exist and be in the candidate set.
4. `slug` matches `^[a-z0-9][a-z0-9/_-]{1,60}$`; parent must exist or be created at depth ≤ 3.
5. body length 20–400 tokens; no content copied verbatim > 30 tokens (prevents summariser echo).
6. On failure: retry once with the validator error appended. On second failure: heuristic filing —
   first 2 topics from `topics` table by keyword overlap, summary = first 60 tokens of the user turn,
   `unsorted` flag. Mark job `failed` with reason for the Inspector.

**Apply transaction** (single writer):

```
BEGIN IMMEDIATE
  upsert topics
  if merge_into: append body to existing summary, bump updated_at, add links
  else: insert summary, link message_ids with weight 1.0
  insert summary_edges
  enqueue embed_summary
COMMIT
```

### 6.3 Consolidation (nightly, idle)

- merge summaries with cosine > 0.92 and same topic (keep the older id, `merged_into` the newer);
- decay `salience *= 0.95` for summaries unused in 30 days; floor 0.05;
- mark `valid_to` on summaries contradicted by newer ones (contradiction detected by a small-model
  pass restricted to candidates sharing entities);
- cap 40 active summaries per topic; fold the lowest-salience tail into a topic digest;
- rebuild FTS if row counts drift; `VACUUM INTO` backup.

### 6.4 Dedupe and edits

| Case | Handling |
|---|---|
| Same request retried | `ux_msg_dedupe` ignores the duplicate; `request_id` recorded |
| User edits a message | new user row with `revises_id`; linked summaries flagged `stale=1` in `meta`, re-filed next idle window |
| Regenerate | new assistant row with `revises_id`; old row `superseded_by` set; retrieval prefers non-superseded |
| Near-duplicate (> 0.97 cosine, same conversation, 24 h) | second row kept (append-only) but `duplicate_of` set and excluded from summary links |
| Imported/duplicated chat (same content, new conversation) | `duplicate_of` set; summaries not duplicated |

Append-only is preserved: "supersede" is a forward pointer, never a delete.

---

## 7. Token budget planner

The single place that decides what fits. Uses the **model's own tokenizer** (`tokenizer.json` from
the model directory via `tokenizers`), not an approximation.

```
context_window   = measured per model (Phase 0), stored in models.json
reserve_output   = request.max_tokens or 1024
reserve_system   = tokens(system prompt)
available        = context_window - reserve_output - reserve_system - 64 safety

allocate, in order, stopping when full:
  1. current user input            (never trimmed; if it alone exceeds → 413 with explanation)
  2. tool chain of the current turn (never trimmed)
  3. recent history                min(6 turns, 25 % of available)
  4. summaries                     min(35 %)
  5. exact excerpts                remainder
```

**Trim order when over budget** (the brief's order): exact excerpts first, then summaries, then
recent history. Inside exact excerpts: drop oldest first; inside summaries: drop lowest score first;
inside history: drop oldest turn first, but keep the immediately preceding assistant answer if the
current message is a continuation.

Every trim is logged with the reason. The Inspector shows a visual bar of the allocation.

Phase 0 must measure the real usable window at 16 GB. Starting assumption: **8 k tokens** with a
2 GB KV cache; a 16 k setting is offered only if the measurement supports it.

---

## 8. Latency budget

| Stage | Budget | Notes |
|---|---|---|
| classify | 0.2 ms | table predicates |
| embed query | 25 ms | CPU, 0.6 B int8 |
| vector top-k | 6 ms | 50 k × 768 floats in numpy |
| FTS5 | 3 ms | |
| fuse + score | 2 ms | |
| expand + tokenise | 40 ms | tokenising 4 k tokens is the expensive part |
| assemble | 5 ms | |
| **total** | **~80 ms p50 / 150 ms p95** | |
| optional NPU rewrite | +120 ms cap | only when ambiguous |

If any stage exceeds its budget it is skipped, logged, and the reply proceeds. A slow DB never
delays a reply by more than 150 ms.

---

## 9. Fast path vs background — what runs where

| Work | Device | When | Blocking? |
|---|---|---|---|
| query embedding | CPU | every chat request | yes (25 ms) |
| vector + FTS search | CPU | every chat request | yes |
| query rewrite | NPU | ambiguous only | yes, capped |
| turn filing / summarisation | NPU | after reply | no |
| consolidation | NPU | idle, nightly | no |
| re-embedding summaries | CPU | after filing | no |
| document chunking | CPU | ingestion | no |
| Docling VLM calls | GPU (main model) | ingestion | queued behind chat |

Docling's VLM calls compete with chat for the GPU. The queue gives chat priority: document jobs
pause while a chat stream is active and resume after 3 s of idle.

---

## 10. Medical mode

Not a separate memory, a **retrieval contract**:

- retrieval is restricted to collections tagged `medical` plus general knowledge; conversational
  memory is included only as "patient-reported history", never as evidence.
- every claim must be followed by a source marker `[doc:chunk-id]`; the UI renders it as a link.
- if no source clears a relevance threshold, the answer must say it cannot answer from the provided
  sources, give general safety guidance, and stop.
- a fixed disclaimer is appended server-side (not model-generated) and shown as a persistent banner:
  *"This is not medical advice. I am not a doctor. Consult a qualified clinician. In an emergency,
  call your local emergency number."*
- the mode can be disabled in Settings but the disclaimer component cannot be removed from a
  medical-mode answer.

---

## 11. Memory per workspace

One memory system, seven policies. This table is the contract that stops a workspace from quietly
inventing its own behaviour.

| Workspace | Read mode | Namespace | Archived after the turn | Notes |
|---|---|---|---|---|
| **Chat** | `chat` | personal | user + assistant messages | the reference implementation |
| **Knowledge** | `knowledge` | collection + personal | question, answer, cited chunk ids | citations required; answered chunks are not re-summarised as user facts |
| **Documents** | `doc` | document + personal | parsed artifact, document summary, key entities | document chunks live in `knowledge.db`, not in the summary DB |
| **Images** | none before generation | personal | prompt, params, artifact id, `kind='image'` summary | retrieval can *inform* a prompt ("same style as yesterday") but never blocks generation |
| **Audio** | `chat` | personal | transcript segments as messages with timestamps | TTS output is an artifact, not a memory of the user's words |
| **Code** | `code` | `code/<repo>` | task, plan, accepted diffs, decisions | passthrough in P8; whole-repo dumps are never archived; tool output is summarised only if it changed a decision |
| **Reverse engineering** | `re` | `re/<target-hash>` | findings, evidence refs, tool calls, report | artifact hash scopes the namespace; nothing leaves the target folder |
| **Medical** | `medical` | medical collections | Q + A + citations | disclaimer enforced; memory is history, never evidence |

Rules that apply everywhere:

- side requests (titles, tags, probes) are never retrieved, never archived;
- the current input always outranks memory;
- a workspace cannot read another workspace's namespace unless the user explicitly links them
  (e.g. "use what we decided in the code session" is an explicit cross-namespace query);
- every workspace uses the same token planner and the same Inspector.

---

## 12. Test plan for memory (must exist before Phase 5)

| Test | Type | Pass condition |
|---|---|---|
| ID hallucination | unit | 100 adversarial model outputs; zero unvalidated IDs applied |
| append-only | property | random edit/regenerate sequences never change a `content` value |
| idempotent archive | unit | same request replayed 5× → 1 user row, 1 assistant row |
| trim order | golden | 20 budget scenarios match expected allocations |
| retrieval quality | eval set | 120 hand-labelled queries; recall@5 ≥ 0.85 on summaries |
| latency | benchmark | p95 < 150 ms at 50 k messages |
| frontend parity | integration | identical answer shape from our UI, OpenCode, and curl |
| injection filtering | unit | client RAG block never summarised as user content |
| NPU fallback | chaos | kill NPU model mid-job → job retries on CPU, chat unaffected |
| DB corruption | chaos | WAL truncated → app starts read-only, offers backup restore |
