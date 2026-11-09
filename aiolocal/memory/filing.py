"""Background filing: turn a finished exchange into summary memory.

The LLM is allowed to write topics, summary text and links — never the Exact DB.
Everything it returns is validated against the ids offered in that call. If validation
fails twice, we fall back to deterministic heuristic filing and mark the job failed so
the Inspector can show why. Chat is never blocked by this.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

import httpx

from . import semantic, store
from .text import estimate_tokens, first_sentence, keywords, normalise, truncate_tokens

from ..config import Settings
from ..db import Database
from ..llm import api_url

TOPIC_RE = re.compile(r"^[a-z0-9][a-z0-9/_-]{1,60}$")
ALLOWED_KINDS = {"episodic", "profile", "project", "decision", "fact", "code", "re"}
MAX_TOPICS = 3
MAX_SUMMARIES = 2


def _existing_topic(db: Database, words: set[str]) -> str | None:
    if not words:
        return None
    for topic in store.list_topics(db):
        hay = {w.strip("_-/") for w in re.split(r"[\s/_-]+", f"{topic['slug']} {topic['name']}".lower())}
        if hay & words:
            return topic["slug"]
    return None


def heuristic_file_turn(
    db: Database,
    *,
    conversation_id: str,
    user_message_id: str,
    assistant_message_id: str,
    user_text: str,
    assistant_text: str,
    settings: Settings,
) -> str:
    """Deterministic fallback filing. Always succeeds if the DB is writable."""
    words = keywords(user_text, limit=6)
    slug = _existing_topic(db, set(words))
    if not slug:
        slug = f"auto/{words[0]}" if words else "unsorted"
    title = first_sentence(user_text, max_chars=120) or "Untitled turn"
    body_parts = [f"User asked: {truncate_tokens(user_text, 180)}"]
    if normalise(assistant_text):
        body_parts.append(f"Assistant answered: {truncate_tokens(assistant_text, 240)}")
    body = "\n\n".join(body_parts)

    summary_id = store.insert_summary(
        db,
        title=title,
        body=body,
        topic_slug=slug,
        kind="episodic",
        message_ids=[user_message_id, assistant_message_id],
        salience=0.5,
        confidence=0.6,
        source="heuristic",
        entities=words,
        embed_dim=settings.embed_dim,
    )
    _link_previous(db, conversation_id, summary_id)
    semantic.index_summary(db, settings, summary_id, title, body)
    return summary_id


def _link_previous(db: Database, conversation_id: str, summary_id: str) -> None:
    with db.read() as c:
        row = c.execute(
            """SELECT s.id FROM summaries s
               JOIN summary_messages sm ON sm.summary_id = s.id
               JOIN messages m ON m.id = sm.message_id
               WHERE m.conversation_id = ? AND s.id != ? AND s.merged_into IS NULL
               ORDER BY s.created_at DESC LIMIT 1""",
            (conversation_id, summary_id),
        ).fetchone()
    if row:
        store.insert_summary_edge(db, summary_id, row["id"], "follows", 1.0)


# --------------------------------------------------------------------- model output
def model_file_turn(
    db: Database,
    *,
    conversation_id: str,
    user_message_id: str,
    assistant_message_id: str,
    user_text: str,
    assistant_text: str,
    settings: Settings,
) -> str | None:
    """Ask the configured filing model to file this turn.

    Returns a summary id, or ``None`` so the caller can use the deterministic
    heuristic. This intentionally never raises: filing is background work.
    """
    model = str(settings.filing_model or "").strip()
    if not model:
        return None
    try:
        topics = store.list_topics(db)
        candidates = store.list_summaries(db, limit=5)
        allowed_ids = [user_message_id, assistant_message_id]
        prompt = build_filing_prompt(
            user_text=user_text,
            assistant_text=assistant_text,
            allowed_ids=allowed_ids,
            topics=topics,
            summaries=candidates,
        )
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a precise memory filing engine. Output JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 900,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        with httpx.Client(timeout=120.0, trust_env=False) as client:
            response = client.post(api_url(settings.upstream_base_url, "chat/completions"), json=payload)
        if response.status_code >= 400:
            return None
        content = response.json()["choices"][0]["message"].get("content") or ""
        parsed = _extract_json(str(content))
        validated, error = validate_filing(
            parsed,
            allowed_ids=allowed_ids,
            candidate_summary_ids=[str(s["id"]) for s in candidates],
        )
        if error or not validated:
            return None
        created = apply_filing(db, validated, embed_dim=settings.embed_dim)
        if not created:
            return None
        for summary_id in created:
            row = store.get_summary(db, summary_id)
            if row:
                semantic.index_summary(
                    db, settings, summary_id, row.get("title") or "", row.get("body") or ""
                )
        _link_previous(db, conversation_id, created[0])
        return created[0]
    except Exception:
        return None


def _extract_json(text: str) -> object:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None


def build_filing_prompt(
    *,
    user_text: str,
    assistant_text: str,
    allowed_ids: list[str],
    topics: list[dict],
    summaries: list[dict],
) -> str:
    topic_lines = "\n".join(f"- {t['slug']}: {t.get('name') or ''}" for t in topics[:40])
    summary_lines = "\n".join(
        f"- id={s['id']} title={s['title']} body={truncate_tokens(s.get('body') or '', 40)}"
        for s in summaries[:5]
    )
    return (
        "You file one finished conversation turn into a memory database.\n"
        "Return ONLY a JSON object, no prose, matching this shape:\n"
        '{"topics":[{"slug":"area/topic","name":"Area topic","parent":null}],'
        '"summaries":[{"title":"...","body":"...","kind":"episodic|profile|project|decision|fact|code|re",'
        '"message_ids":["<allowed id>"],"links":[{"to":"<existing summary id>","rel":"elaborates"}]}],'
        '"merge_into":null,"salience":0.6}\n'
        "Rules: at most 3 topics and 2 summaries. Use ONLY message ids from the allowed list. "
        "Never copy more than 30 consecutive tokens from the turn. Write 20-400 tokens of body.\n\n"
        f"TURN (user):\n{truncate_tokens(user_text, 700)}\n\n"
        f"TURN (assistant):\n{truncate_tokens(assistant_text, 700)}\n\n"
        f"ALLOWED MESSAGE IDS: {', '.join(allowed_ids)}\n\n"
        f"CANDIDATE TOPICS:\n{topic_lines or '(none)'}\n\n"
        f"CANDIDATE SUMMARIES:\n{summary_lines or '(none)'}\n"
    )


def validate_filing(
    payload: object,
    *,
    allowed_ids: Iterable[str],
    candidate_summary_ids: Iterable[str],
) -> tuple[dict | None, str | None]:
    allowed = set(allowed_ids)
    candidates = set(candidate_summary_ids)
    if not isinstance(payload, dict):
        return None, "response must be a JSON object"

    topics_raw = payload.get("topics") or []
    summaries_raw = payload.get("summaries") or []
    if not isinstance(topics_raw, list) or not isinstance(summaries_raw, list):
        return None, "topics and summaries must be arrays"
    if len(topics_raw) > MAX_TOPICS:
        return None, f"at most {MAX_TOPICS} topics are allowed"
    if len(summaries_raw) > MAX_SUMMARIES:
        return None, f"at most {MAX_SUMMARIES} summaries are allowed"

    topics: list[dict] = []
    for t in topics_raw:
        if not isinstance(t, dict):
            return None, "each topic must be an object"
        slug = str(t.get("slug") or "").strip()
        if not TOPIC_RE.fullmatch(slug):
            return None, f"invalid topic slug: {slug!r}"
        name = str(t.get("name") or slug.replace("/", " / ")).strip()[:80]
        parent = t.get("parent")
        if parent is not None:
            parent = str(parent).strip()
            if not TOPIC_RE.fullmatch(parent):
                return None, f"invalid parent slug: {parent!r}"
        topics.append({"slug": slug, "name": name, "parent": parent})

    summaries: list[dict] = []
    for s in summaries_raw:
        if not isinstance(s, dict):
            return None, "each summary must be an object"
        title = normalise(str(s.get("title") or ""))[:200]
        body = normalise(str(s.get("body") or ""))
        if len(title) < 3:
            return None, "summary title is too short"
        if estimate_tokens(body) < 20:
            return None, "summary body is too short"
        if estimate_tokens(body) > 400:
            return None, "summary body is too long"
        kind = str(s.get("kind") or "episodic")
        if kind not in ALLOWED_KINDS:
            return None, f"invalid summary kind: {kind!r}"
        ids = s.get("message_ids") or []
        if not isinstance(ids, list) or not ids:
            return None, "a summary needs at least one message id"
        for mid in ids:
            if str(mid) not in allowed:
                return None, f"unknown message id: {mid!r}"
        links = s.get("links") or []
        clean_links: list[dict] = []
        if isinstance(links, list):
            for link in links[:4]:
                if not isinstance(link, dict):
                    continue
                to = str(link.get("to") or "")
                rel = str(link.get("rel") or "elaborates")[:30]
                if to in candidates:
                    clean_links.append({"to": to, "rel": rel})
        summaries.append(
            {
                "title": title,
                "body": body,
                "kind": kind,
                "message_ids": [str(m) for m in ids][:10],
                "links": clean_links,
            }
        )

    merge_into = payload.get("merge_into")
    if merge_into:
        merge_into = str(merge_into)
        if merge_into not in candidates:
            return None, "merge_into is not one of the candidate summaries"
    else:
        merge_into = None

    if not summaries and not merge_into:
        return None, "must create or merge at least one summary"

    try:
        salience = float(payload.get("salience", 0.55))
    except (TypeError, ValueError):
        return None, "salience must be a number"

    return {
        "topics": topics,
        "summaries": summaries,
        "merge_into": merge_into,
        "salience": max(0.0, min(1.0, salience)),
    }, None


def apply_filing(db: Database, payload: dict, *, embed_dim: int = 256) -> list[str]:
    """Apply a validated filing payload in one transaction. Returns summary ids."""
    created: list[str] = []
    with db.write() as c:
        slug_to_id: dict[str, str] = {}
        for topic in payload.get("topics", []):
            parent_id = None
            if topic.get("parent"):
                parent_id = store.get_or_create_topic(c, topic["parent"], topic["parent"])
            slug_to_id[topic["slug"]] = store.get_or_create_topic(
                c, topic["slug"], topic.get("name")
            )

        merge_into = payload.get("merge_into")
        salience = float(payload.get("salience", 0.55))

        if merge_into:
            row = c.execute(
                "SELECT id, title, body FROM summaries WHERE id=? AND merged_into IS NULL",
                (merge_into,),
            ).fetchone()
            if row:
                appended = row["body"]
                for s in payload.get("summaries", []):
                    appended += "\n\n" + s["body"]
                c.execute(
                    "UPDATE summaries SET body=?, updated_at=?, salience=MAX(salience, ?) WHERE id=?",
                    (appended[-12_000:], store.now_ms(), salience, merge_into),
                )
                for s in payload.get("summaries", []):
                    for mid in s["message_ids"]:
                        c.execute(
                            "INSERT OR IGNORE INTO summary_messages(summary_id, message_id) VALUES(?,?)",
                            (merge_into, mid),
                        )
                created.append(merge_into)

        for s in payload.get("summaries", []):
            summary_id = store.new_id()
            topic_id = next(iter(slug_to_id.values()), None)
            if not topic_id:
                topic_id = store.get_or_create_topic(c, "unsorted", None)
            c.execute(
                """INSERT INTO summaries(id, topic_id, title, body, kind, entities, salience,
                       confidence, valid_from, source, created_at, updated_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    summary_id, topic_id, s["title"], s["body"], s["kind"],
                    json.dumps([], ensure_ascii=False),
                    salience, 0.75, store.now_ms(), "llm",
                    store.now_ms(), store.now_ms(),
                ),
            )
            for mid in s["message_ids"]:
                c.execute(
                    "INSERT OR IGNORE INTO summary_messages(summary_id, message_id) VALUES(?,?)",
                    (summary_id, mid),
                )
            for link in s.get("links", []):
                c.execute(
                    "INSERT OR IGNORE INTO summary_edges(src, dst, rel, weight) VALUES(?,?,?,?)",
                    (summary_id, link["to"], link["rel"], 1.0),
                )
            created.append(summary_id)

    for summary_id in created:
        row = store.get_summary(db, summary_id)
        if row:
            from .text import embed

            store.put_embedding(
                db, "summary", summary_id, "hash",
                embed((row.get("title") or "") + "\n" + (row.get("body") or ""), embed_dim),
            )
    return created
