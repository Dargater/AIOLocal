"""Semantic embeddings through the configured local model server.

Hash embeddings always work and stay in the database, so retrieval never depends on
a network call. When the user selects the OVMS backend, the same text is also
embedded by the local embedding model and stored under that model's name. Retrieval
fuses both ranked lists. If the model server is down, everything falls back to hash
vectors instead of failing the turn.
"""

from __future__ import annotations

import numpy as np
import httpx

from . import store
from .text import embed

from ..config import Settings
from ..db import Database
from ..llm import api_url


def model_name(settings: Settings) -> str:
    if getattr(settings, "embedding_backend", "hash") == "ovms":
        return str(getattr(settings, "embedding_model", "") or "qwen3-embedding-0.6b")
    return "hash"


def enabled(settings: Settings) -> bool:
    return model_name(settings) != "hash"


def embed_text(settings: Settings, text: str, *, timeout: float = 30.0) -> np.ndarray | None:
    if not enabled(settings) or not str(text or "").strip():
        return None
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            response = client.post(
                api_url(settings.upstream_base_url, "embeddings"),
                json={"model": model_name(settings), "input": str(text)[:16_000]},
            )
        if response.status_code >= 400:
            return None
        return _first_embedding(response.json())
    except Exception:
        return None


def embed_batch(
    settings: Settings, texts: list[str], *, batch_size: int = 16, timeout: float = 120.0
) -> tuple[list[np.ndarray | None], str]:
    """Return one vector (or None) per text plus a short error string."""
    if not enabled(settings):
        return [None] * len(texts), "semantic embeddings are disabled"
    out: list[np.ndarray | None] = []
    try:
        with httpx.Client(timeout=timeout, trust_env=False) as client:
            for start in range(0, len(texts), batch_size):
                chunk = [str(t or "")[:16_000] or " " for t in texts[start : start + batch_size]]
                response = client.post(
                    api_url(settings.upstream_base_url, "embeddings"),
                    json={"model": model_name(settings), "input": chunk},
                )
                if response.status_code >= 400:
                    return out + [None] * (len(texts) - len(out)), (
                        f"HTTP {response.status_code}: {response.text[:200]}"
                    )
                vectors = _embedding_list(response.json())
                if len(vectors) != len(chunk):
                    return out + [None] * (len(texts) - len(out)), "the server returned the wrong number of vectors"
                out.extend(vectors)
    except Exception as exc:
        return out + [None] * (len(texts) - len(out)), f"{type(exc).__name__}: {exc}"
    return out, ""


def index_message(db: Database, settings: Settings, message_id: str, text: str) -> bool:
    if not enabled(settings):
        return False
    vector = embed_text(settings, text)
    if vector is None:
        return False
    store.put_embedding(db, "message", message_id, model_name(settings), vector)
    return True


def index_summary(
    db: Database, settings: Settings, summary_id: str, title: str, body: str
) -> bool:
    if not enabled(settings):
        return False
    vector = embed_text(settings, f"{title}\n{body}")
    if vector is None:
        return False
    store.put_embedding(db, "summary", summary_id, model_name(settings), vector)
    return True


def reembed_all(db: Database, settings: Settings, *, limit: int = 5000) -> dict:
    """Fill semantic vectors for existing messages and summaries.

    Hash vectors stay untouched, so retrieval keeps working while this job runs and
    after it finishes. ``limit`` caps each table to keep one job bounded.
    """
    if not enabled(settings):
        return {"ok": False, "reason": "semantic embeddings are disabled", "messages": 0, "summaries": 0}
    model = model_name(settings)
    with db.read() as c:
        message_rows = c.execute(
            "SELECT id, content FROM messages WHERE content != '' ORDER BY created_at DESC LIMIT ?",
            (max(1, min(int(limit), 50_000)),),
        ).fetchall()
        summary_rows = c.execute(
            "SELECT id, title, body FROM summaries WHERE merged_into IS NULL ORDER BY created_at DESC LIMIT ?",
            (max(1, min(int(limit), 50_000)),),
        ).fetchall()

    messages_done = 0
    summaries_done = 0
    error = ""

    message_ids = [r["id"] for r in message_rows]
    texts = [r["content"] for r in message_rows]
    for start in range(0, len(texts), 128):
        vectors, error = embed_batch(settings, texts[start : start + 128])
        if error:
            break
        for message_id, vector in zip(message_ids[start : start + 128], vectors):
            if vector is not None:
                store.put_embedding(db, "message", message_id, model, vector)
                messages_done += 1

    if not error:
        summary_ids = [r["id"] for r in summary_rows]
        summary_texts = [f"{r['title']}\n{r['body']}" for r in summary_rows]
        for start in range(0, len(summary_texts), 128):
            vectors, error = embed_batch(settings, summary_texts[start : start + 128])
            if error:
                break
            for summary_id, vector in zip(summary_ids[start : start + 128], vectors):
                if vector is not None:
                    store.put_embedding(db, "summary", summary_id, model, vector)
                    summaries_done += 1

    return {
        "ok": not error,
        "model": model,
        "messages": messages_done,
        "summaries": summaries_done,
        "error": error,
    }


# ---------------------------------------------------------------------- parsing
def _embedding_list(payload: object) -> list[np.ndarray]:
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if not isinstance(data, list):
        return []
    indexed = []
    for item in data:
        if not isinstance(item, dict):
            continue
        vector = item.get("embedding")
        if vector is None and isinstance(item.get("data"), list):
            vector = item.get("data")
        if not isinstance(vector, list) or not vector:
            continue
        try:
            indexed.append((int(item.get("index", len(indexed))), np.asarray(vector, dtype=np.float32)))
        except (TypeError, ValueError):
            continue
    indexed.sort(key=lambda pair: pair[0])
    return [vec for _, vec in indexed]


def _first_embedding(payload: object) -> np.ndarray | None:
    vectors = _embedding_list(payload)
    return vectors[0] if vectors else None
