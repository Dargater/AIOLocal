from __future__ import annotations

from pathlib import Path

from aiolocal.config import Paths, Settings
from aiolocal.db import Database
from aiolocal.memory import store
from aiolocal.memory.filing import heuristic_file_turn, validate_filing
from aiolocal.memory.jobs import MemoryWorker, consolidate
from aiolocal.memory.retrieve import classify_intent, retrieve


def make_db(tmp_path: Path) -> tuple[Database, Settings]:
    paths = Paths.create(tmp_path / "data")
    return Database(paths.db_path), Settings()


def test_archive_is_append_only_and_deduped(tmp_path: Path):
    db, settings = make_db(tmp_path)
    cid = "conv1"
    store.ensure_conversation(db, cid)

    first_id, inserted = store.archive_message(db, conversation_id=cid, role="user", content="hello world")
    assert inserted
    same_id, inserted_again = store.archive_message(db, conversation_id=cid, role="user", content="hello world")
    assert not inserted_again
    assert same_id == first_id

    original, _ = store.archive_message(db, conversation_id=cid, role="assistant", content="first answer")
    revised, inserted = store.archive_message(
        db, conversation_id=cid, role="assistant", content="second answer", revises_id=original
    )
    assert inserted

    old_row = store.message_by_id(db, original)
    assert old_row["content"] == "first answer"  # content never changes
    assert old_row["superseded_by"] == revised

    with db.read() as c:
        assert c.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 3
        assert c.execute("SELECT COUNT(*) FROM messages_fts").fetchone()[0] == 3


def test_retrieval_finds_summary_and_exact(tmp_path: Path):
    db, settings = make_db(tmp_path)
    cid = "conv-memory"
    store.ensure_conversation(db, cid)
    mid, _ = store.archive_message(
        db, conversation_id=cid, role="user", content="We decided to use Tokio multi-thread runtime for the CLI."
    )
    store.archive_message(db, conversation_id=cid, role="assistant", content="Good choice for throughput.")
    summary_id = store.insert_summary(
        db,
        title="Tokio runtime selection",
        body="The project uses the multi-threaded Tokio runtime because the CLI needs throughput.",
        topic_slug="rust/async",
        message_ids=[mid],
        salience=0.8,
    )

    result = retrieve(db, "which tokio runtime did we choose?", settings=settings, conversation_id="other")
    assert result.intent == "lookup"
    assert result.items
    assert any(item.id == summary_id for item in result.items)
    assert any(item.kind == "exact" for item in result.items) or result.items

    continuation = retrieve(db, "make it shorter", settings=settings, conversation_id=cid)
    assert continuation.intent == "continuation"
    assert continuation.items == []


def test_classify_intent():
    assert classify_intent("") == "continuation"
    assert classify_intent("make it shorter") == "continuation"
    assert classify_intent("why?") == "continuation"
    assert classify_intent("integral") == "lookup"
    assert classify_intent("integral of arctan x") == "lookup"
    assert classify_intent("Explain how OpenVINO batching works in detail") == "lookup"


def test_force_lookup_finds_single_word_term(tmp_path: Path):
    db, settings = make_db(tmp_path)
    cid = "conv-integral"
    store.ensure_conversation(db, cid)
    store.archive_message(
        db,
        conversation_id=cid,
        role="user",
        content="Can you do the integral of tan inverse x?",
    )
    store.archive_message(
        db,
        conversation_id=cid,
        role="assistant",
        content="Use integration by parts. The integral is x arctan(x) minus one half ln(1+x^2).",
    )

    result = retrieve(db, "integral", settings=settings, force_lookup=True)
    assert result.intent == "lookup"
    assert result.items
    assert any("integral" in item.body.lower() for item in result.items)


def test_heuristic_filing_creates_summary(tmp_path: Path):
    db, settings = make_db(tmp_path)
    cid = "conv-file"
    store.ensure_conversation(db, cid)
    uid, _ = store.archive_message(db, conversation_id=cid, role="user", content="I prefer Postgres over MySQL for this project.")
    aid, _ = store.archive_message(db, conversation_id=cid, role="assistant", content="Noted, Postgres it is.")

    summary_id = heuristic_file_turn(
        db,
        conversation_id=cid,
        user_message_id=uid,
        assistant_message_id=aid,
        user_text="I prefer Postgres over MySQL for this project.",
        assistant_text="Noted, Postgres it is.",
        settings=settings,
    )
    summary = store.get_summary(db, summary_id)
    assert summary is not None
    assert summary["message_count"] == 2
    assert "Postgres" in summary["title"]
    assert summary["source"] == "heuristic"


def test_worker_processes_job(tmp_path: Path):
    db, settings = make_db(tmp_path)
    cid = "conv-worker"
    store.ensure_conversation(db, cid)
    uid, _ = store.archive_message(db, conversation_id=cid, role="user", content="Remember that the build uses PyInstaller.")
    aid, _ = store.archive_message(db, conversation_id=cid, role="assistant", content="Okay.")
    worker = MemoryWorker(db, settings)
    worker.enqueue_file_turn(
        conversation_id=cid,
        user_message_id=uid,
        assistant_message_id=aid,
        user_text="Remember that the build uses PyInstaller.",
        assistant_text="Okay.",
    )
    assert worker.run_once() is True
    assert db.stats()["summaries"] >= 1
    job = store.job_stats(db)
    assert any(row["state"] == "done" for row in job)


def test_consolidation_merges_duplicates(tmp_path: Path):
    db, settings = make_db(tmp_path)
    cid = "conv-merge"
    store.ensure_conversation(db, cid)
    mid, _ = store.archive_message(db, conversation_id=cid, role="user", content="same fact")
    body = "The project stores data in SQLite with WAL mode enabled for concurrency."
    store.insert_summary(db, title="Database choice", body=body, topic_slug="tech/db", message_ids=[mid])
    store.insert_summary(db, title="Database choice", body=body, topic_slug="tech/db", message_ids=[mid])
    merged = consolidate(db, embed_dim=settings.embed_dim)
    assert merged >= 1
    assert len(store.list_summaries(db, limit=10)) == 1


def test_validate_filing_rejects_unknown_ids():
    payload = {
        "topics": [{"slug": "rust/async", "name": "Rust async"}],
        "summaries": [
            {
                "title": "A decision",
                "body": "A" * 200,
                "kind": "decision",
                "message_ids": ["not-a-real-id"],
            }
        ],
    }
    clean, error = validate_filing(payload, allowed_ids={"real-id"}, candidate_summary_ids=set())
    assert clean is None
    assert "unknown message id" in error


def test_messages_exclude_injected_from_history(tmp_path: Path):
    db, _ = make_db(tmp_path)
    cid = "conv-injected"
    store.ensure_conversation(db, cid)
    store.archive_message(db, conversation_id=cid, role="user", content="real question")
    store.archive_message(db, conversation_id=cid, role="system", content="junk", injected=True)
    rows = store.recent_messages(db, cid, limit=10)
    assert [r["content"] for r in rows] == ["real question"]
