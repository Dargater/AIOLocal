from __future__ import annotations

from pathlib import Path

from aiolocal.config import Paths, Settings
from aiolocal.db import Database
from aiolocal.memory import semantic, store


def make_db(tmp_path: Path) -> Database:
    paths = Paths.create(tmp_path / "data")
    return Database(paths.db_path)


def test_semantic_is_off_by_default(tmp_path: Path):
    db = make_db(tmp_path)
    settings = Settings()

    assert semantic.enabled(settings) is False
    assert semantic.model_name(settings) == "hash"
    assert semantic.index_message(db, settings, "msg1", "hello") is False
    assert semantic.embed_text(settings, "hello") is None

    result = semantic.reembed_all(db, settings)
    assert result["ok"] is False
    assert "disabled" in result["reason"]
    assert result["messages"] == 0
    assert result["summaries"] == 0


def test_semantic_enabled_uses_the_configured_model(tmp_path: Path):
    settings = Settings(embedding_backend="ovms", embedding_model="tiny-embed")

    assert semantic.enabled(settings) is True
    assert semantic.model_name(settings) == "tiny-embed"


def test_reembed_reports_server_failures_without_raising(tmp_path: Path, monkeypatch):
    db = make_db(tmp_path)
    settings = Settings(embedding_backend="ovms", embedding_model="tiny-embed")
    store.ensure_conversation(db, "conv")
    store.archive_message(db, conversation_id="conv", role="user", content="remember this")

    def offline(settings, texts, **kwargs):
        return [None] * len(texts), "connection refused"

    monkeypatch.setattr(semantic, "embed_batch", offline)
    result = semantic.reembed_all(db, settings)

    assert result["ok"] is False
    assert result["error"] == "connection refused"
    assert result["messages"] == 0
    assert result["summaries"] == 0
    # Hash vectors remain available for retrieval.
    rows = store.get_embeddings(db, "message", "hash")
    assert rows

def test_reembed_indexes_summaries_even_when_messages_fail(tmp_path: Path, monkeypatch):
    import numpy as np

    db = make_db(tmp_path)
    settings = Settings(embedding_backend="ovms", embedding_model="tiny-embed")
    store.ensure_conversation(db, "conv")
    store.archive_message(db, conversation_id="conv", role="user", content="remember this")
    sid = store.insert_summary(db, title="Durable summary", body="important fact")

    def fake_batch(settings, texts, **kwargs):
        if any("remember this" in t for t in texts):
            return [None] * len(texts), "HTTP 500: bad batch"
        return [np.ones(4, dtype=np.float32)] * len(texts), ""

    monkeypatch.setattr(semantic, "embed_batch", fake_batch)
    monkeypatch.setattr(semantic, "embed_text", lambda *args, **kwargs: None)
    result = semantic.reembed_all(db, settings)

    assert result["summaries"] == 1
    assert result["messages"] == 0
    assert result["error"]
    rows = store.get_embeddings(db, "summary", "tiny-embed")
    assert [r[0] for r in rows] == [sid]

def test_read_timeout_does_not_abort_reembed():
    assert semantic._server_unreachable("ReadTimeout: timed out") is False
    assert semantic._server_unreachable("ConnectError: connection refused") is True
