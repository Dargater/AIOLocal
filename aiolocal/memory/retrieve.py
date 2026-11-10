"""Memory read path.

The design goal is a deterministic, fast retrieval that never blocks a reply. Every
stage is best-effort: if one fails, the reply proceeds with less memory. The
retrieval plan is logged so the Memory Inspector can explain what the model saw.

Scoring is reciprocal rank fusion (RRF) across independent ranked lists. BM25 and
cosine scores are not comparable, so fusing ranks avoids per-corpus calibration as
the database grows.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from . import store, semantic
from .text import embed, estimate_tokens, keywords, normalise

from ..config import Settings
from ..db import Database, now_ms

RRF_K = 60
MAX_SUMMARIES = 8
MAX_EXACT_MESSAGES = 12
EXPAND_PER_SUMMARY = 6
GATE_NORMALISED = 0.06

_CONTINUATION_PHRASES = {
    "make it shorter", "make it longer", "again", "why", "why?", "continue", "go on",
    "fix that", "try again", "same", "and then", "what about that", "explain more",
    "more detail", "elaborate", "rewrite", "rephrase", "simplify", "expand",
    "keep going", "do it", "yes", "no", "ok", "okay", "thanks", "thank you",
}
# One-word follow-ups must stay continuations without turning every short lookup
# (for example "integral") into one.
_CONTINUATION_WORDS = {
    "again", "continue", "more", "less", "same", "why", "how", "what", "huh", "ok",
    "okay", "yes", "no", "thanks", "thank", "you", "it", "that", "this", "those",
    "these", "them", "there", "then", "go", "on",
}


@dataclass
class MemoryItem:
    kind: str  # "summary" | "exact"
    id: str
    title: str
    body: str
    score: float = 0.0
    topic: str | None = None
    message_ids: list[str] = field(default_factory=list)
    created_at: int = 0
    salience: float = 0.5
    tokens: int = 0
    summary_kind: str = ""  # summaries only: episodic/document/...

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "summary_kind": self.summary_kind,
            "id": self.id,
            "title": self.title,
            "body": self.body,
            "score": round(self.score, 5),
            "topic": self.topic,
            "message_ids": self.message_ids,
            "created_at": self.created_at,
            "salience": round(self.salience, 3),
            "tokens": self.tokens or estimate_tokens(self.body),
        }


@dataclass
class RetrievalResult:
    intent: str
    items: list[MemoryItem]
    plan: dict
    latency_ms: int

    @property
    def token_count(self) -> int:
        return sum(i.tokens or estimate_tokens(i.body) for i in self.items)


def classify_intent(query: str) -> str:
    text = normalise(query).lower()
    if not text:
        return "continuation"
    if text in _CONTINUATION_PHRASES:
        return "continuation"
    words = text.replace("?", "").replace("!", "").split()
    if words and all(word in _CONTINUATION_WORDS for word in words):
        return "continuation"
    if len(words) <= 4:
        deictic = {"it", "that", "this", "those", "these", "them", "again", "more", "less"}
        if deictic & set(words):
            return "continuation"
    return "lookup"


def _recency_boost(created_at: int) -> float:
    if not created_at:
        return 1.0
    age_days = max(0.0, (now_ms() - created_at) / 86_400_000.0)
    return 1.0 + 0.15 * math.exp(-age_days / 45.0)


def _topic_hits(db: Database, query: str) -> list[str]:
    words = set(keywords(query, limit=10))
    if not words:
        return []
    slugs: list[str] = []
    for topic in store.list_topics(db):
        hay = f"{topic['slug']} {topic['name']}".lower()
        if any(w in hay for w in words):
            slugs.append(topic["slug"])
    return slugs


def _topics_for_slugs(db: Database, slugs: list[str], limit: int = 8) -> list[str]:
    if not slugs:
        return []
    q = ",".join("?" for _ in slugs)
    with db.read() as c:
        rows = c.execute(
            f"""SELECT s.id FROM summaries s JOIN topics t ON t.id=s.topic_id
                WHERE t.slug IN ({q}) AND s.merged_into IS NULL
                ORDER BY s.salience DESC, s.created_at DESC LIMIT ?""",
            (*slugs, limit),
        ).fetchall()
    return [r["id"] for r in rows]


def _vector_rank(
    query: str, rows: list[tuple[str, object]], dim: int, threshold: float, limit: int
) -> list[str]:
    pairs = [(owner_id, np.asarray(vec, dtype=np.float32)) for owner_id, vec in rows]
    pairs = [(owner_id, vec) for owner_id, vec in pairs if vec.size == dim]
    if not pairs:
        return []
    qv = embed(query, dim)
    return _rank_pairs(qv, pairs, threshold, limit)


def _rank_pairs(
    qv: np.ndarray, pairs: list[tuple[str, np.ndarray]], threshold: float, limit: int
) -> list[str]:
    qnorm = float(np.linalg.norm(qv))
    if qnorm == 0 or not pairs:
        return []
    matrix = np.vstack([vec for _, vec in pairs])
    norms = np.linalg.norm(matrix, axis=1)
    denom = np.maximum(norms * qnorm, 1e-9)
    sims = (matrix @ qv) / denom
    order = np.argsort(-sims)[:limit]
    return [pairs[i][0] for i in order if float(sims[i]) > threshold]


def _vector_summaries(db: Database, query: str, settings: Settings, limit: int = 40) -> list[str]:
    return _vector_lists(
        db, "summary", query, settings, threshold=0.02, limit=limit, cap=50_000
    )


def _vector_messages(db: Database, query: str, settings: Settings, limit: int = 30) -> list[str]:
    return _vector_lists(
        db, "message", query, settings, threshold=0.10, limit=limit, cap=20_000
    )


def _vector_lists(
    db: Database,
    owner_type: str,
    query: str,
    settings: Settings,
    *,
    threshold: float,
    limit: int,
    cap: int,
) -> list[str]:
    """Fuse semantic vectors (when configured) with the always-present hash vectors."""
    out: list[str] = []
    seen: set[str] = set()

    def add(ids: list[str]) -> None:
        for owner_id in ids:
            if owner_id not in seen:
                seen.add(owner_id)
                out.append(owner_id)

    semantic_model = semantic.model_name(settings)
    if semantic_model != "hash":
        try:
            rows = store.get_embeddings(db, owner_type, semantic_model, limit=cap)
            if rows:
                vector = semantic.embed_text(settings, query)
                if vector is not None and vector.size:
                    pairs = [(owner_id, np.asarray(vec, dtype=np.float32)) for owner_id, vec in rows]
                    pairs = [(owner_id, vec) for owner_id, vec in pairs if vec.size == vector.size]
                    add(_rank_pairs(vector, pairs, threshold, limit))
        except Exception:
            pass

    try:
        rows = store.get_embeddings(db, owner_type, "hash", limit=cap)
        add(_vector_rank(query, rows, settings.embed_dim, threshold, limit))
    except Exception:
        pass
    return out[:limit]


def _rrf(lists: list[list[str]]) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranked in lists:
        for rank, key in enumerate(ranked):
            fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank + 1)
    return fused


def _group_exact(db: Database, message_ids: list[str], fused: dict[str, float]) -> list[MemoryItem]:
    rows = store.messages_by_ids(db, message_ids)
    if not rows:
        return []
    rows.sort(key=lambda r: r["created_at"])
    groups: list[list[dict]] = []
    for row in rows:
        if (
            groups
            and groups[-1][-1]["conversation_id"] == row["conversation_id"]
            and row["created_at"] - groups[-1][-1]["created_at"] < 30 * 60 * 1000
        ):
            groups[-1].append(row)
        else:
            groups.append([row])

    items: list[MemoryItem] = []
    for group in groups:
        body = "\n".join(f"{r['role']}: {r['content']}" for r in group)
        best = max(fused.get(r["id"], 0.0) for r in group)
        first, last = group[0], group[-1]
        date = time.strftime("%Y-%m-%d %H:%M", time.localtime(first["created_at"] / 1000))
        items.append(
            MemoryItem(
                kind="exact",
                id=first["conversation_id"],
                title=f"Exact excerpt — {date}",
                body=body,
                score=best,
                message_ids=[r["id"] for r in group],
                created_at=first["created_at"],
                salience=0.5,
                tokens=estimate_tokens(body),
            )
        )
    items.sort(key=lambda i: i.score, reverse=True)
    return items[:MAX_EXACT_MESSAGES]


def retrieve(
    db: Database,
    query: str,
    *,
    settings: Settings,
    conversation_id: str | None = None,
    exclude_message_ids: set[str] | None = None,
    request_id: str | None = None,
    force_lookup: bool = False,
) -> RetrievalResult:
    t0 = time.perf_counter()
    q = normalise(query)
    exclude = exclude_message_ids or set()
    # The Memory Inspector search box always searches. Intent classification only
    # exists to decide whether a normal chat turn needs retrieval at all.
    intent = "lookup" if force_lookup else classify_intent(q)
    plan: dict = {
        "intent": intent,
        "query": q[:400],
        "sources": {},
        "gate": None,
        "selected": [],
        "latency_ms": 0,
    }

    if not q or intent == "continuation":
        plan["skipped"] = "continuation"
        latency = int((time.perf_counter() - t0) * 1000)
        plan["latency_ms"] = latency
        if request_id:
            try:
                store.log_retrieval(
                    db,
                    request_id=request_id,
                    conversation_id=conversation_id,
                    query=q,
                    plan=plan,
                    latency_ms=latency,
                )
            except Exception:  # pragma: no cover
                pass
        return RetrievalResult(intent=intent, items=[], plan=plan, latency_ms=latency)

    summaries: dict[str, dict] = {}
    lists: list[list[str]] = []

    try:
        vec_ids = _vector_summaries(db, q, settings)
        lists.append(vec_ids)
        plan["sources"]["summary_vector"] = vec_ids[:10]
    except Exception as exc:  # pragma: no cover - defensive
        plan["sources"]["summary_vector_error"] = str(exc)[:200]
        vec_ids = []

    try:
        fts_sum = [sid for sid, _ in store.search_summaries_fts(db, q, limit=40)]
        lists.append(fts_sum)
        plan["sources"]["summary_fts"] = fts_sum[:10]
    except Exception as exc:  # pragma: no cover
        plan["sources"]["summary_fts_error"] = str(exc)[:200]
        fts_sum = []

    try:
        fts_msg = [mid for mid, _ in store.search_messages_fts(db, q, limit=40)]
        fts_msg = [m for m in fts_msg if m not in exclude]
        lists.append(fts_msg)
        plan["sources"]["message_fts"] = fts_msg[:10]
    except Exception as exc:  # pragma: no cover
        plan["sources"]["message_fts_error"] = str(exc)[:200]
        fts_msg = []

    try:
        topic_ids = _topics_for_slugs(db, _topic_hits(db, q))
        lists.append(topic_ids)
        plan["sources"]["topic"] = topic_ids[:10]
    except Exception as exc:  # pragma: no cover
        plan["sources"]["topic_error"] = str(exc)[:200]
        topic_ids = []

    try:
        vec_msg = [m for m in _vector_messages(db, q, settings) if m not in exclude]
        lists.append(vec_msg)
        plan["sources"]["message_vector"] = vec_msg[:10]
    except Exception as exc:  # pragma: no cover
        plan["sources"]["message_vector_error"] = str(exc)[:200]
        vec_msg = []

    fused_summary = _rrf(lists[:2] + [topic_ids])
    fused_message = _rrf([fts_msg, vec_msg])

    summary_rows = {s["id"]: s for s in store.summaries_by_ids(db, list(fused_summary))}
    max_possible = 3.0 / (RRF_K + 1)
    top_raw = max(fused_summary.values(), default=0.0)
    plan["gate"] = round(top_raw / max_possible, 4) if max_possible else 0.0

    ordered: list[MemoryItem] = []
    for sid, rrf in sorted(fused_summary.items(), key=lambda kv: kv[1], reverse=True):
        row = summary_rows.get(sid)
        if not row:
            continue
        salience = float(row.get("salience") or 0.5)
        score = rrf * (0.75 + 0.25 * salience) * _recency_boost(int(row.get("created_at") or 0))
        body = row.get("body") or ""
        ordered.append(
            MemoryItem(
                kind="summary",
                id=sid,
                title=row.get("title") or "Summary",
                body=body,
                score=score,
                topic=row.get("topic_slug"),
                created_at=int(row.get("created_at") or 0),
                salience=salience,
                tokens=estimate_tokens(body) + estimate_tokens(row.get("title") or ""),
                summary_kind=str(row.get("kind") or ""),
            )
        )
    ordered = ordered[:MAX_SUMMARIES]

    if top_raw / max_possible < GATE_NORMALISED:
        plan["gate"] = "below_threshold"
        ordered = []

    # Expand the messages behind the selected summaries, plus direct message hits.
    expand_ids: list[str] = []
    for item in ordered:
        for mid in store.summary_message_ids(db, item.id, limit=EXPAND_PER_SUMMARY):
            if mid not in exclude and mid not in expand_ids:
                expand_ids.append(mid)
    for mid in list(fused_message):
        if mid not in exclude and mid not in expand_ids:
            expand_ids.append(mid)

    exact_items: list[MemoryItem] = []
    if expand_ids:
        try:
            exact_items = _group_exact(db, expand_ids[:MAX_EXACT_MESSAGES * 4], fused_message)
        except Exception as exc:  # pragma: no cover
            plan["exact_error"] = str(exc)[:200]

    items = exact_items + ordered
    plan["selected"] = [i.to_dict() for i in items]
    latency = int((time.perf_counter() - t0) * 1000)
    plan["latency_ms"] = latency

    if request_id:
        try:
            store.log_retrieval(
                db,
                request_id=request_id,
                conversation_id=conversation_id,
                query=q,
                plan=plan,
                latency_ms=latency,
            )
        except Exception:  # pragma: no cover - logging must never break a reply
            pass

    return RetrievalResult(intent=intent, items=items, plan=plan, latency_ms=latency)
