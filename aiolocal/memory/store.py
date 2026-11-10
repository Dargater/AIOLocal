"""Persistence operations for the exact and summary databases."""

from __future__ import annotations

import json
import sqlite3
from typing import Any, Iterable

from .text import content_hash, embed, estimate_tokens, new_id, pack

from ..db import Database, json_dumps, now_ms


def _row(row: sqlite3.Row | None) -> dict | None:
    return dict(row) if row is not None else None


# ------------------------------------------------------------------ conversations
def ensure_conversation(
    db: Database,
    conversation_id: str,
    *,
    source: str = "native",
    title: str | None = None,
    external_id: str | None = None,
) -> None:
    ts = now_ms()
    with db.write() as c:
        c.execute(
            """INSERT INTO conversations(id, title, source, external_id, started_at, last_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET last_at=excluded.last_at,
                 title=COALESCE(conversations.title, excluded.title)""",
            (conversation_id, title, source, external_id, ts, ts),
        )


def touch_conversation(db: Database, conversation_id: str) -> None:
    with db.write() as c:
        c.execute("UPDATE conversations SET last_at=? WHERE id=?", (now_ms(), conversation_id))


def list_conversations(db: Database, limit: int = 100) -> list[dict]:
    with db.read() as c:
        rows = c.execute(
            """SELECT id, title, source, started_at, last_at,
                      (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = conversations.id) AS message_count,
                      (SELECT content FROM messages m WHERE m.conversation_id = conversations.id
                         AND m.role='user' ORDER BY m.created_at ASC LIMIT 1) AS first_user
               FROM conversations ORDER BY last_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        if not d.get("title") and d.get("first_user"):
            d["title"] = (d["first_user"] or "").strip().splitlines()[0][:80]
        out.append(d)
    return out


def find_conversation_by_external(db: Database, source: str, external_id: str) -> str | None:
    with db.read() as c:
        row = c.execute(
            "SELECT id FROM conversations WHERE source=? AND external_id=?",
            (source, external_id),
        ).fetchone()
    return row["id"] if row else None


def delete_conversation(db: Database, conversation_id: str) -> bool:
    """Delete a chat and its messages, leaving summary memory untouched."""
    with db.write() as c:
        exists = c.execute(
            "SELECT 1 FROM conversations WHERE id=?", (conversation_id,)
        ).fetchone()
        if not exists:
            return False
        c.execute(
            "DELETE FROM messages_fts WHERE message_id IN "
            "(SELECT id FROM messages WHERE conversation_id=?)",
            (conversation_id,),
        )
        c.execute(
            "DELETE FROM embeddings WHERE owner_type='message' AND owner_id IN "
            "(SELECT id FROM messages WHERE conversation_id=?)",
            (conversation_id,),
        )
        c.execute(
            "DELETE FROM summary_messages WHERE message_id IN "
            "(SELECT id FROM messages WHERE conversation_id=?)",
            (conversation_id,),
        )
        c.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        c.execute("DELETE FROM retrieval_log WHERE conversation_id=?", (conversation_id,))
        c.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))
    return True


def delete_summary(db: Database, summary_id: str) -> bool:
    """Delete one summary and all of its indexes/links."""
    with db.write() as c:
        exists = c.execute("SELECT 1 FROM summaries WHERE id=?", (summary_id,)).fetchone()
        if not exists:
            return False
        c.execute("DELETE FROM summaries_fts WHERE summary_id=?", (summary_id,))
        c.execute(
            "DELETE FROM embeddings WHERE owner_type='summary' AND owner_id=?",
            (summary_id,),
        )
        c.execute("DELETE FROM summary_messages WHERE summary_id=?", (summary_id,))
        c.execute(
            "DELETE FROM summary_edges WHERE src=? OR dst=?", (summary_id, summary_id)
        )
        c.execute("DELETE FROM summaries WHERE id=?", (summary_id,))
    return True


def conversation_messages(db: Database, conversation_id: str, limit: int = 500) -> list[dict]:
    with db.read() as c:
        rows = c.execute(
            """SELECT id, role, content, created_at, superseded_by, revises_id, modality
               FROM messages WHERE conversation_id=? AND injected=0
               ORDER BY created_at ASC LIMIT ?""",
            (conversation_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def recent_messages(db: Database, conversation_id: str, limit: int = 12) -> list[dict]:
    with db.read() as c:
        rows = c.execute(
            """SELECT id, role, content, created_at, tokens FROM messages
               WHERE conversation_id=? AND injected=0 AND superseded_by IS NULL
               ORDER BY created_at DESC LIMIT ?""",
            (conversation_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]


# ------------------------------------------------------------------ exact messages
def archive_message(
    db: Database,
    *,
    conversation_id: str,
    role: str,
    content: str,
    source: str = "native",
    client_msg_id: str | None = None,
    request_id: str | None = None,
    revises_id: str | None = None,
    injected: bool = False,
    modality: str = "text",
    meta: dict | None = None,
    embed_dim: int = 256,
) -> tuple[str, bool]:
    """Append a message. Returns ``(id, inserted)``. Never overwrites content."""
    chash = content_hash(content)
    msg_id = new_id()
    ts = now_ms()
    tokens = estimate_tokens(content)
    with db.write() as c:
        try:
            cur = c.execute(
                """INSERT INTO messages(id, conversation_id, role, content, content_hash, tokens,
                       modality, source, client_msg_id, request_id, revises_id, injected, created_at, meta)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    msg_id, conversation_id, role, content, chash, tokens, modality, source,
                    client_msg_id, request_id, revises_id, 1 if injected else 0, ts,
                    json_dumps(meta or {}),
                ),
            )
            inserted = cur.rowcount == 1
        except sqlite3.IntegrityError:
            row = c.execute(
                """SELECT id FROM messages
                   WHERE conversation_id=? AND role=? AND content_hash=? AND IFNULL(revises_id,'')=?""",
                (conversation_id, role, chash, revises_id or ""),
            ).fetchone()
            inserted = False
            msg_id = row["id"] if row else msg_id

        if revises_id:
            c.execute("UPDATE messages SET superseded_by=? WHERE id=? AND superseded_by IS NULL",
                      (msg_id, revises_id))
        c.execute(
            """INSERT INTO conversations(id, title, source, started_at, last_at)
               VALUES(?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET last_at=excluded.last_at""",
            (conversation_id, None, source, ts, ts),
        )
    if inserted and not injected and content.strip():
        put_embedding(db, "message", msg_id, "hash", embed(content, embed_dim))
    return msg_id, inserted


def message_by_id(db: Database, message_id: str) -> dict | None:
    with db.read() as c:
        return _row(c.execute("SELECT * FROM messages WHERE id=?", (message_id,)).fetchone())


def messages_by_ids(db: Database, ids: Iterable[str]) -> list[dict]:
    ids = list(dict.fromkeys(ids))
    if not ids:
        return []
    q = ",".join("?" for _ in ids)
    with db.read() as c:
        rows = c.execute(
            f"""SELECT id, conversation_id, role, content, created_at, tokens
                FROM messages WHERE id IN ({q}) AND superseded_by IS NULL
                ORDER BY created_at ASC""",
            tuple(ids),
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ full-text search
def _fts_query(text: str) -> str:
    """Turn arbitrary user text into a safe FTS5 OR-query (never raw MATCH syntax)."""
    import re

    words = [w for w in re.findall(r"[A-Za-z0-9_\u4e00-\u9fff]{2,}", text or "")][:16]
    if not words:
        return ""
    return " OR ".join(f'"{w}"' for w in words)


def search_messages_fts(db: Database, text: str, limit: int = 40) -> list[tuple[str, float]]:
    q = _fts_query(text)
    if not q:
        return []
    with db.read() as c:
        try:
            rows = c.execute(
                """SELECT message_id, bm25(messages_fts) AS score
                   FROM messages_fts WHERE messages_fts MATCH ? ORDER BY score LIMIT ?""",
                (q, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    # bm25 is negative and smaller is better; convert to a positive, larger-is-better score.
    return [(r["message_id"], -float(r["score"])) for r in rows]


def search_summaries_fts(db: Database, text: str, limit: int = 40) -> list[tuple[str, float]]:
    q = _fts_query(text)
    if not q:
        return []
    with db.read() as c:
        try:
            rows = c.execute(
                """SELECT summary_id, bm25(summaries_fts) AS score
                   FROM summaries_fts WHERE summaries_fts MATCH ? ORDER BY score LIMIT ?""",
                (q, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
    return [(r["summary_id"], -float(r["score"])) for r in rows]


# ------------------------------------------------------------------ summaries
def get_or_create_topic(
    conn: sqlite3.Connection, slug: str, name: str | None = None, parent: str | None = None
) -> str:
    row = conn.execute("SELECT id FROM topics WHERE slug=?", (slug,)).fetchone()
    if row:
        return row["id"]
    topic_id = new_id()
    conn.execute(
        "INSERT INTO topics(id, slug, name, parent_id, created_at) VALUES(?,?,?,?,?)",
        (topic_id, slug, name or slug.replace("/", " / "), parent, now_ms()),
    )
    return topic_id


def insert_summary(
    db: Database,
    *,
    title: str,
    body: str,
    topic_slug: str = "general",
    kind: str = "episodic",
    message_ids: Iterable[str] = (),
    salience: float = 0.55,
    confidence: float = 0.7,
    source: str = "heuristic",
    entities: list[str] | None = None,
    embed_dim: int = 256,
) -> str:
    summary_id = new_id()
    ts = now_ms()
    with db.write() as c:
        topic_id = get_or_create_topic(c, topic_slug)
        c.execute(
            """INSERT INTO summaries(id, topic_id, title, body, kind, entities, salience,
                   confidence, valid_from, source, created_at, updated_at)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                summary_id, topic_id, title[:200], body, kind,
                json_dumps(entities or []), max(0.0, min(1.0, salience)),
                max(0.0, min(1.0, confidence)), ts, source, ts, ts,
            ),
        )
        for mid in dict.fromkeys(message_ids):
            c.execute(
                "INSERT OR IGNORE INTO summary_messages(summary_id, message_id) VALUES(?,?)",
                (summary_id, mid),
            )
    vec = embed(title + "\n" + body, embed_dim)
    put_embedding(db, "summary", summary_id, "hash", vec)
    return summary_id


def update_summary_body(
    db: Database,
    summary_id: str,
    body: str,
    *,
    title: str | None = None,
    embed_dim: int = 256,
) -> bool:
    """Replace (or append to) a summary body, used by incremental document jobs."""
    with db.write() as c:
        row = c.execute(
            "SELECT title, body FROM summaries WHERE id=? AND merged_into IS NULL",
            (summary_id,),
        ).fetchone()
        if not row:
            return False
        c.execute(
            "UPDATE summaries SET body=?, title=?, updated_at=? WHERE id=?",
            (body[:24_000], (title or row["title"])[:200], now_ms(), summary_id),
        )
    current_title = title or row["title"]
    put_embedding(db, "summary", summary_id, "hash", embed(current_title + "\n" + body[:8000], embed_dim))
    return True


def merge_summary(db: Database, summary_id: str, body: str, message_ids: Iterable[str]) -> bool:
    with db.write() as c:
        row = c.execute("SELECT id FROM summaries WHERE id=? AND merged_into IS NULL",
                        (summary_id,)).fetchone()
        if not row:
            return False
        c.execute("UPDATE summaries SET body=?, updated_at=? WHERE id=?",
                  (body, now_ms(), summary_id))
        for mid in dict.fromkeys(message_ids):
            c.execute(
                "INSERT OR IGNORE INTO summary_messages(summary_id, message_id) VALUES(?,?)",
                (summary_id, mid),
            )
        title = c.execute("SELECT title FROM summaries WHERE id=?", (summary_id,)).fetchone()["title"]
    put_embedding(db, "summary", summary_id, "hash", embed(title + "\n" + body, 256))
    return True


def get_summary(db: Database, summary_id: str) -> dict | None:
    with db.read() as c:
        row = c.execute(
            """SELECT s.*, t.slug AS topic_slug,
                      (SELECT COUNT(*) FROM summary_messages sm WHERE sm.summary_id=s.id) AS message_count
               FROM summaries s LEFT JOIN topics t ON t.id=s.topic_id WHERE s.id=?""",
            (summary_id,),
        ).fetchone()
    return _row(row)


def list_summaries(db: Database, limit: int = 50, offset: int = 0, topic: str | None = None) -> list[dict]:
    with db.read() as c:
        if topic:
            rows = c.execute(
                """SELECT s.*, t.slug AS topic_slug,
                          (SELECT COUNT(*) FROM summary_messages sm WHERE sm.summary_id=s.id) AS message_count
                   FROM summaries s LEFT JOIN topics t ON t.id=s.topic_id
                   WHERE s.merged_into IS NULL AND t.slug=?
                   ORDER BY s.created_at DESC LIMIT ? OFFSET ?""",
                (topic, limit, offset),
            ).fetchall()
        else:
            rows = c.execute(
                """SELECT s.*, t.slug AS topic_slug,
                          (SELECT COUNT(*) FROM summary_messages sm WHERE sm.summary_id=s.id) AS message_count
                   FROM summaries s LEFT JOIN topics t ON t.id=s.topic_id
                   WHERE s.merged_into IS NULL
                   ORDER BY s.created_at DESC LIMIT ? OFFSET ?""",
                (limit, offset),
            ).fetchall()
    return [dict(r) for r in rows]


def summaries_by_ids(db: Database, ids: Iterable[str]) -> list[dict]:
    ids = list(dict.fromkeys(ids))
    if not ids:
        return []
    q = ",".join("?" for _ in ids)
    with db.read() as c:
        rows = c.execute(
            f"""SELECT s.*, t.slug AS topic_slug
                FROM summaries s LEFT JOIN topics t ON t.id=s.topic_id
                WHERE s.id IN ({q}) AND s.merged_into IS NULL
                ORDER BY s.created_at DESC""",
            tuple(ids),
        ).fetchall()
    return [dict(r) for r in rows]


def insert_summary_edge(db: Database, src: str, dst: str, rel: str, weight: float = 1.0) -> None:
    if src == dst:
        return
    with db.write() as c:
        c.execute(
            """INSERT OR IGNORE INTO summary_edges(src, dst, rel, weight) VALUES(?,?,?,?)""",
            (src, dst, rel, max(0.0, min(1.0, weight))),
        )


def bump_salience(db: Database, summary_ids: Iterable[str], amount: float = 0.02) -> None:
    ids = list(dict.fromkeys(summary_ids))
    if not ids:
        return
    q = ",".join("?" for _ in ids)
    with db.write() as c:
        c.execute(
            f"UPDATE summaries SET salience=MIN(0.99, salience+?) WHERE id IN ({q})",
            (amount, *ids),
        )


def list_topics(db: Database) -> list[dict]:
    with db.read() as c:
        rows = c.execute(
            """SELECT t.id, t.slug, t.name,
                      (SELECT COUNT(*) FROM summaries s WHERE s.topic_id=t.id AND s.merged_into IS NULL) AS count
               FROM topics t ORDER BY t.slug"""
        ).fetchall()
    return [dict(r) for r in rows]


def summary_message_ids(db: Database, summary_id: str, limit: int = 30) -> list[str]:
    with db.read() as c:
        rows = c.execute(
            """SELECT message_id FROM summary_messages WHERE summary_id=?
               ORDER BY weight DESC LIMIT ?""",
            (summary_id, limit),
        ).fetchall()
    return [r["message_id"] for r in rows]


def summary_id_for_hash(db: Database, title: str) -> str | None:
    with db.read() as c:
        row = c.execute(
            "SELECT id FROM summaries WHERE title=? AND merged_into IS NULL ORDER BY created_at DESC LIMIT 1",
            (title[:200],),
        ).fetchone()
    return row["id"] if row else None


# ------------------------------------------------------------------ embeddings
def put_embedding(db: Database, owner_type: str, owner_id: str, model: str, vec) -> None:
    blob, norm = pack(vec)
    with db.write() as c:
        c.execute(
            """INSERT INTO embeddings(owner_type, owner_id, model, dim, vec, norm)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(owner_type, owner_id, model) DO UPDATE
                 SET dim=excluded.dim, vec=excluded.vec, norm=excluded.norm""",
            (owner_type, owner_id, model, int(vec.size), blob, norm),
        )


def get_embeddings(
    db: Database, owner_type: str, model: str = "hash", limit: int | None = None
) -> list[tuple[str, object]]:
    import numpy as np

    sql = (
        "SELECT owner_id, dim, vec FROM embeddings WHERE owner_type=? AND model=? "
        "ORDER BY rowid DESC"
    )
    args: tuple = (owner_type, model)
    if limit:
        sql += " LIMIT ?"
        args = (*args, int(limit))
    with db.read() as c:
        rows = c.execute(sql, args).fetchall()
    out = []
    for r in rows:
        out.append((r["owner_id"], np.frombuffer(r["vec"], dtype=np.float32)))
    return out


# ------------------------------------------------------------------ jobs
def queue_job(db: Database, kind: str, payload: dict, run_after: int | None = None) -> str:
    job_id = new_id()
    with db.write() as c:
        c.execute(
            "INSERT INTO memory_jobs(id, kind, payload, state, attempts, run_after, created_at) VALUES(?,?,?,?,?,?,?)",
            (job_id, kind, json_dumps(payload), "queued", 0, run_after or now_ms(), now_ms()),
        )
    return job_id


def claim_job(db: Database) -> dict | None:
    with db.write() as c:
        row = c.execute(
            """SELECT * FROM memory_jobs
               WHERE state='queued' AND IFNULL(run_after,0) <= ?
               ORDER BY created_at LIMIT 1""",
            (now_ms(),),
        ).fetchone()
        if not row:
            return None
        c.execute("UPDATE memory_jobs SET state='running' WHERE id=?", (row["id"],))
    return dict(row)


def finish_job(db: Database, job_id: str, *, ok: bool, error: str | None = None, retry: bool = False) -> None:
    with db.write() as c:
        row = c.execute("SELECT attempts FROM memory_jobs WHERE id=?", (job_id,)).fetchone()
        attempts = (row["attempts"] if row else 0) + 1
        if ok:
            c.execute("UPDATE memory_jobs SET state='done', attempts=? WHERE id=?", (attempts, job_id))
        elif retry and attempts < 3:
            c.execute(
                "UPDATE memory_jobs SET state='queued', attempts=?, last_error=?, run_after=? WHERE id=?",
                (attempts, error, now_ms() + 5000 * attempts, job_id),
            )
        else:
            c.execute(
                "UPDATE memory_jobs SET state='failed', attempts=?, last_error=? WHERE id=?",
                (attempts, error, job_id),
            )


def job_stats(db: Database) -> list[dict]:
    with db.read() as c:
        rows = c.execute(
            "SELECT state, COUNT(*) AS count FROM memory_jobs GROUP BY state"
        ).fetchall()
    return [dict(r) for r in rows]


# ------------------------------------------------------------------ inspector
def log_retrieval(
    db: Database,
    *,
    request_id: str,
    conversation_id: str | None,
    query: str,
    plan: dict,
    latency_ms: int,
) -> None:
    with db.write() as c:
        c.execute(
            """INSERT INTO retrieval_log(id, request_id, conversation_id, query, plan, latency_ms, created_at)
               VALUES(?,?,?,?,?,?,?)""",
            (new_id(), request_id, conversation_id, query[:2000], json_dumps(plan), latency_ms, now_ms()),
        )


def attach_retrieval_extra(db: Database, request_id: str, key: str, value: object) -> None:
    """Merge an extra section (e.g. the assembly plan) into a retrieval log row."""
    with db.write() as c:
        row = c.execute(
            "SELECT id, plan FROM retrieval_log WHERE request_id=? ORDER BY created_at DESC LIMIT 1",
            (request_id,),
        ).fetchone()
        if not row:
            return
        try:
            plan = json.loads(row["plan"] or "{}")
        except json.JSONDecodeError:
            plan = {}
        plan[key] = value
        c.execute("UPDATE retrieval_log SET plan=? WHERE id=?", (json_dumps(plan), row["id"]))


def retrieval_plan(db: Database, request_id: str) -> dict | None:
    with db.read() as c:
        row = c.execute(
            "SELECT * FROM retrieval_log WHERE request_id=? ORDER BY created_at DESC LIMIT 1",
            (request_id,),
        ).fetchone()
    if not row:
        return None
    d = dict(row)
    try:
        d["plan"] = json.loads(d["plan"] or "{}")
    except json.JSONDecodeError:
        d["plan"] = {}
    return d
