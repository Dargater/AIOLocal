# 08 — Build Workflow (Qwen writes, you debug)

The brief says: *"OpenCode with local Qwen 3.5 9B writes the app; I debug it."* That is a real
constraint on how the code must be shaped. A 9B model at 8 k context cannot hold a large codebase in
its head, so the plan optimises for **small, isolated, testable units with explicit contracts**.

---

## 1. Bootstrap order (before the app exists)

You cannot build the app with the app. The first two weeks use the tools standalone:

```
OVMS (Windows binary, :8000)
  └── Qwen3.5-9B on GPU
        └── OpenCode CLI (not the app) in the repo
              └── writes: supervisor, gateway skeleton, memory schema + tests
```

OpenCode points at `http://127.0.0.1:8000/v1` directly during this phase. It switches to the gateway
(`:8002`) only after the gateway can serve a chat request — at which point the gateway starts
remembering the development sessions too, which is a useful early dogfood test.

---

## 2. Repository rules that make AI authorship safe

1. **One module per session.** Never ask for "the backend". Ask for `packages/memory/retrieval.py`
   with a stated input/output contract and the tests that must pass.
2. **Interfaces before implementations.** Every module starts as a typed stub with a docstring and
   failing tests. Qwen fills the body; the contract does not move.
3. **No hidden state.** Pure functions for retrieval scoring, token planning, validation and prompt
   assembly. Side effects live at the edges (DB writer, HTTP client, file system).
4. **One writer per database.** Stated in code as a class with a single public `write()`; nothing
   else opens a write connection.
5. **Errors are values at boundaries.** Parsers and validators return `Result`, not exceptions;
   only the HTTP layer turns them into responses.
6. **Every module has a runnable check.** `uv run pytest tests/unit/test_<module>.py` must work
   without OVMS, without a GPU, without network.
7. **`AGENTS.md` at the repo root** contains: the module map, the invariants (from `02` §1), the
   commands to run tests/lint, and the rule "do not change public signatures without updating the
   caller and the test".
8. **ADRs for irreversible choices.** Gateway owns history; one OVMS process; SQLite over a vector
   DB; pywebview over Electron; GIMP dock with fallback. Each is one page in `docs/adr/`.

---

## 3. Context strategy for an 8 k model

| Technique | Why |
|---|---|
| Keep files under ~300 lines | fits in context with room for the task |
| Type stubs + docstrings as the spec | the model reads the contract, not the whole repo |
| Give the model the failing test first | turns an open-ended request into a bounded one |
| One dependency at a time | prevents "while you're here, refactor everything" |
| Never paste the whole DB schema | paste only the tables the task touches |
| Use the Inspector's assembled-prompt view | shows exactly what the model saw, which is also how you debug it |
| Summarise decisions into `docs/decisions.md` | cheap long-term memory for the model, independent of the app's memory system |

The app's own memory system is not used to code the app in v1 (bootstrapping problem). Once the
gateway works, OpenCode gets the `code/<repo>` namespace and the two systems converge.

---

## 4. The loop

```
1  write/adjust the failing test           (human or Qwen, small)
2  Qwen implements the module              (OpenCode, one file)
3  run: ruff, mypy, pytest                 (automated)
4  if red: paste the failure, not the file (Qwen fixes precisely)
5  human reviews the diff                  (accept / reject hunks)
6  commit with a one-line why              (not "update file")
```

Rules:

- The human never accepts a diff that changes an invariant without reading the ADR.
- A task is not done because tests pass; it is done when the **manual check** in the task
  description also passes (e.g. "kill the NPU model mid-filing, chat still answers").
- Any bug found twice becomes a regression test before it is fixed.

---

## 5. First ten tasks (concrete, in order)

| # | Task | Files | Acceptance |
|---|---|---|---|
| 1 | Repo skeleton, `uv`, ruff/mypy, CI | `pyproject.toml`, `.github/` | `uv run pytest` passes with one dummy test |
| 2 | Config + paths + logging | `packages/shared/config.py`, `logging.py` | loads `app.json`, redacts secrets |
| 3 | SQLite bootstrap + migrations | `packages/memory/db.py` | creates `memory.db`, `PRAGMA user_version=1` |
| 4 | ULID + hashing + normalisation | `packages/shared/ids.py` | deterministic hashes, monotonic ULIDs |
| 5 | Message repository (append-only) | `packages/memory/messages.py` | dedupe test passes; no UPDATE of content |
| 6 | FTS5 + embedding store | `packages/memory/index.py` | search returns expected ids on a fixture |
| 7 | RRF + scoring | `packages/memory/rank.py` | golden ranking test |
| 8 | Token planner | `packages/memory/budget.py` | 20 golden allocation cases |
| 9 | Prompt assembler | `packages/memory/assemble.py` | snapshot test of the exact system block |
| 10 | Gateway `/v1/chat/completions` passthrough | `apps/gateway/main.py` | curl streams from OVMS unchanged |

Only after task 10 does memory get wired into the request path. That ordering means every layer is
provable on its own, which is exactly what an 8 k model needs.

---

## 6. Debugging aids you will actually use

- **Memory Inspector**: the exact prompt, the retrieved rows, every trim decision, the raw model
  response. If an answer is wrong, this tells you whether it was retrieval, assembly, or the model.
- **Golden prompt snapshots**: when assembly changes, the diff shows precisely what the model will
  now see. Review these like code, because they are the product.
- **Retrieval eval set**: 120 labelled queries; run after every ranking change.
- **Structured logs**: one JSON line per request with timings per stage; grep by `request_id`.
- **Process tree panel**: RSS and commit per child; the fastest way to catch a leak.
- **Chaos scripts**: kill OVMS, kill NPU model, fill disk, corrupt WAL. Each has an expected UI
  state; run them before every release.

---

## 7. What "the AI wrote it" changes about the plan

- More tests, fewer abstractions. Clever code is a liability when the author forgets it next session.
- Stricter module boundaries. A 9B model will happily duplicate logic across files; the module map
  and import rules prevent that.
- Smaller commits. A 600-line diff is unreviewable by a human debugging at the end of the day.
- Documentation as context. `AGENTS.md`, ADRs and docstrings are not overhead; they are the model's
  working memory.
- The human owns the invariants. The model can implement a validator; it cannot decide that
  append-only is negotiable.

---

## 8. Definition of done (per task)

1. Public contract documented (docstring or type stub).
2. Unit tests cover the happy path and at least two failure modes.
3. `ruff`, `mypy --strict`, `pytest` clean.
4. No new dependency without a line in `pyproject.toml` explaining why.
5. Manual check from the task description executed and noted in the commit message.
6. If it touches memory, the append-only and ID-validation invariants are explicitly asserted.
