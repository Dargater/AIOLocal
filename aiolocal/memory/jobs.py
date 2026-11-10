"""Background memory worker.

One serial worker thread. Filing runs after the reply has been sent, so it never adds
latency to chat. Failures are retried up to three times with backoff, then recorded as
failed for the Memory Inspector.

The worker deliberately owns its own SQLite connections (the Database wrapper is
thread-local), so it can run while requests are being served.
"""

from __future__ import annotations

import json
import threading
import traceback
from typing import Callable

import numpy as np

from . import filing, semantic, store
from .text import cosine

from ..config import Settings
from ..db import Database, now_ms


class MemoryWorker:
    def __init__(
        self,
        db: Database,
        settings: Settings,
        *,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self.on_event = on_event
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="memory-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout=timeout)

    def notify(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                worked = self.run_once()
            except Exception:  # pragma: no cover - the loop must survive anything
                worked = False
            if not worked:
                self._wake.wait(0.75)
                self._wake.clear()

    # ------------------------------------------------------------------ jobs
    def enqueue_file_turn(
        self,
        *,
        conversation_id: str,
        user_message_id: str,
        assistant_message_id: str,
        user_text: str,
        assistant_text: str,
    ) -> str:
        job_id = store.queue_job(
            self.db,
            "file_turn",
            {
                "conversation_id": conversation_id,
                "user_message_id": user_message_id,
                "assistant_message_id": assistant_message_id,
                "user_text": user_text,
                "assistant_text": assistant_text,
            },
        )
        self.notify()
        return job_id

    def enqueue_reembed(self, limit: int = 5000) -> str:
        job_id = store.queue_job(self.db, "reembed", {"limit": int(limit)})
        self.notify()
        return job_id

    def run_once(self) -> bool:
        job = store.claim_job(self.db)
        if not job:
            return False
        try:
            payload = json.loads(job["payload"])
            self._dispatch(job["kind"], payload)
            store.finish_job(self.db, job["id"], ok=True)
            self._emit("job_done", {"id": job["id"], "kind": job["kind"]})
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            retry = job["kind"] not in ("consolidate",)
            store.finish_job(self.db, job["id"], ok=False, error=error[:500], retry=retry)
            self._emit("job_failed", {"id": job["id"], "kind": job["kind"], "error": error[:500]})
            if not retry:
                traceback.print_exc()
        return True

    def _dispatch(self, kind: str, payload: dict) -> None:
        if kind == "file_turn":
            filed = None
            if self.settings.filing_model:
                filed = filing.model_file_turn(
                    self.db,
                    conversation_id=payload["conversation_id"],
                    user_message_id=payload["user_message_id"],
                    assistant_message_id=payload["assistant_message_id"],
                    user_text=payload.get("user_text") or "",
                    assistant_text=payload.get("assistant_text") or "",
                    settings=self.settings,
                )
            if not filed:
                filing.heuristic_file_turn(
                    self.db,
                    conversation_id=payload["conversation_id"],
                    user_message_id=payload["user_message_id"],
                    assistant_message_id=payload["assistant_message_id"],
                    user_text=payload.get("user_text") or "",
                    assistant_text=payload.get("assistant_text") or "",
                    settings=self.settings,
                )
        elif kind == "consolidate":
            consolidate(self.db, embed_dim=self.settings.embed_dim)
        elif kind == "reindex":
            rebuild_fts(self.db)
        elif kind == "reembed":
            result = semantic.reembed_all(
                self.db, self.settings, limit=int(payload.get("limit", 5000))
            )
            # A partial run is fine (hash retrieval still works), but a run that
            # indexed nothing must be visible as a failed job instead of "done".
            if result.get("error") and not result.get("summaries") and not result.get("messages"):
                raise RuntimeError(result["error"])
        else:
            raise ValueError(f"unknown job kind: {kind}")

    def _emit(self, event: str, data: dict) -> None:
        if self.on_event:
            try:
                self.on_event(event, data)
            except Exception:
                pass


# ---------------------------------------------------------------------- maintenance
def rebuild_fts(db: Database) -> None:
    with db.write() as c:
        c.execute("DELETE FROM messages_fts")
        c.execute("INSERT INTO messages_fts(message_id, content) SELECT id, content FROM messages")
        c.execute("DELETE FROM summaries_fts")
        c.execute("INSERT INTO summaries_fts(summary_id, title, body) SELECT id, title, body FROM summaries")


def consolidate(db: Database, *, embed_dim: int = 256, merge_threshold: float = 0.92) -> int:
    """Merge near-duplicate summaries in the same topic and decay unused salience."""
    merged = 0
    rows = store.list_summaries(db, limit=2000)
    by_topic: dict[str, list[dict]] = {}
    for row in rows:
        by_topic.setdefault(row.get("topic_slug") or "unsorted", []).append(row)

    vecs: dict[str, np.ndarray] = {}
    for owner_id, vec in store.get_embeddings(db, "summary", "hash"):
        vecs[owner_id] = np.asarray(vec, dtype=np.float32)

    for topic_rows in by_topic.values():
        topic_rows.sort(key=lambda r: r["created_at"])
        for i, older in enumerate(topic_rows):
            if older.get("merged_into"):
                continue
            v_old = vecs.get(older["id"])
            if v_old is None:
                continue
            for newer in topic_rows[i + 1 :]:
                if newer.get("merged_into") or newer["id"] == older["id"]:
                    continue
                v_new = vecs.get(newer["id"])
                if v_new is None:
                    continue
                if cosine(v_old, v_new) >= merge_threshold:
                    with db.write() as c:
                        c.execute(
                            "UPDATE summaries SET merged_into=?, updated_at=? WHERE id=? AND merged_into IS NULL",
                            (newer["id"], now_ms(), older["id"]),
                        )
                    merged += 1
                    break

    cutoff = now_ms() - 30 * 86_400_000
    with db.write() as c:
        c.execute(
            "UPDATE summaries SET salience=MAX(0.05, salience*0.95) WHERE merged_into IS NULL AND updated_at < ?",
            (cutoff,),
        )
    return merged


def queue_consolidation(db: Database) -> str:
    return store.queue_job(db, "consolidate", {})
