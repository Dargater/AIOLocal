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


def embed_text(
    settings: Settings, text: str, *, timeout: float = 10.0
) -> np.ndarray | None:
    """Embed one short text.

    The default timeout is deliberately short because this is also called from the
    chat request path (retrieval and archiving). A cold OVMS embedding model can take
    minutes to initialise, and chat must not wait for that; the background re-embed
    job warms it and fills the vectors later. Callers that run off the request path
    pass a longer timeout.
    """
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
    settings: Settings, texts: list[str], *, batch_size: int = 16, timeout: float = 300.0
) -> tuple[list[np.ndarray | None], str]:
    """Return one vector (or None) per text plus a short error string.

    The timeout is generous: OVMS may need a few minutes to compile/load the
    embedding model on the first call, and CPU inference of a long document can take
    tens of seconds even when warm.
    """
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
    vector = embed_text(settings, f"{title}\n{body}", timeout=120.0)
    if vector is None:
        return False
    store.put_embedding(db, "summary", summary_id, model_name(settings), vector)
    return True


def reembed_all(db: Database, settings: Settings, *, limit: int = 5000) -> dict:
    """Fill semantic vectors for existing messages and summaries.

    Hash vectors stay untouched, so retrieval keeps working while this job runs and
    after it finishes. ``limit`` caps each table to keep one job bounded. Summaries
    are processed first and each batch is independent, so one oversized document
    cannot stop the rest of the memory from being indexed.
    """
    if not enabled(settings):
        return {"ok": False, "reason": "semantic embeddings are disabled", "messages": 0, "summaries": 0}
    model = model_name(settings)
    cap = max(1, min(int(limit), 50_000))
    with db.read() as c:
        summary_rows = c.execute(
            "SELECT id, title, body FROM summaries WHERE merged_into IS NULL ORDER BY created_at DESC LIMIT ?",
            (cap,),
        ).fetchall()
        message_rows = c.execute(
            "SELECT id, content FROM messages WHERE content != '' ORDER BY created_at DESC LIMIT ?",
            (cap,),
        ).fetchall()

    errors: list[str] = []

    def run_stage(owner_type: str, rows, text_of, batch_size: int) -> int:
        done = 0
        ids = [r["id"] for r in rows]
        texts = [text_of(r) for r in rows]
        for start in range(0, len(texts), batch_size):
            batch_ids = ids[start : start + batch_size]
            batch_texts = texts[start : start + batch_size]
            vectors, error = embed_batch(settings, batch_texts, timeout=300.0)
            if error:
                errors.append(error)
                if _server_unreachable(error):
                    return done
                # One unusually long item can make the whole batch fail. Retry the
                # batch one item at a time so the rest still get vectors.
                for owner_id, text in zip(batch_ids, batch_texts):
                    vector = embed_text(settings, text, timeout=300.0)
                    if vector is not None:
                        store.put_embedding(db, owner_type, owner_id, model, vector)
                        done += 1
                continue
            for owner_id, vector in zip(batch_ids, vectors):
                if vector is not None:
                    store.put_embedding(db, owner_type, owner_id, model, vector)
                    done += 1
        return done

    summaries_done = run_stage(
        "summary",
        summary_rows,
        lambda r: f"{r['title']}\n{r['body']}",
        batch_size=8,
    )
    messages_done = 0
    if not any(_server_unreachable(error) for error in errors):
        messages_done = run_stage(
            "message",
            message_rows,
            lambda r: r["content"],
            batch_size=24,
        )

    return {
        "ok": not errors,
        "model": model,
        "messages": messages_done,
        "summaries": summaries_done,
        "error": errors[0] if errors else "",
    }


def _server_unreachable(error: str) -> bool:
    """True only for errors that mean the process is not listening at all.

    A read timeout is *not* unreachable: OVMS can take minutes to compile a model on
    first use, and aborting the whole re-embed run on that would silently leave most
    of the memory unindexed. Only immediate connection failures stop the job.
    """
    text = str(error or "").lower()
    return any(
        marker in text
        for marker in ("connecterror", "connection refused", "not reachable", "name or service not known")
    )


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
