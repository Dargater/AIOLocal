"""SQLite storage: schema, migrations, and a single-writer transaction helper.

The Exact DB is append-only: no code path updates ``messages.content``. Edits and
regenerations are new rows linked by ``revises_id``; supersession is a forward pointer.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
  id          TEXT PRIMARY KEY,
  title       TEXT,
  source      TEXT NOT NULL DEFAULT 'native',
  external_id TEXT,
  started_at  INTEGER NOT NULL,
  last_at     INTEGER NOT NULL,
  meta        TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS ux_conv_ext
  ON conversations(source, external_id) WHERE external_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS messages (
  id              TEXT PRIMARY KEY,
  conversation_id TEXT NOT NULL,
  role            TEXT NOT NULL,
  content         TEXT NOT NULL,
  content_hash    TEXT NOT NULL,
  tokens          INTEGER NOT NULL DEFAULT 0,
  modality        TEXT NOT NULL DEFAULT 'text',
  source          TEXT NOT NULL DEFAULT 'native',
  client_msg_id   TEXT,
  request_id      TEXT,
  revises_id      TEXT,
  superseded_by   TEXT,
  duplicate_of    TEXT,
  injected        INTEGER NOT NULL DEFAULT 0,
  created_at      INTEGER NOT NULL,
  meta            TEXT
);
CREATE INDEX IF NOT EXISTS ix_msg_conv ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS ix_msg_time ON messages(created_at);
CREATE INDEX IF NOT EXISTS ix_msg_hash ON messages(content_hash);
CREATE UNIQUE INDEX IF NOT EXISTS ux_msg_dedupe
  ON messages(conversation_id, role, content_hash, IFNULL(revises_id, ''));

CREATE TABLE IF NOT EXISTS topics (
  id          TEXT PRIMARY KEY,
  slug        TEXT NOT NULL UNIQUE,
  name        TEXT NOT NULL,
  parent_id   TEXT,
  description TEXT,
  created_at  INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
  id          TEXT PRIMARY KEY,
  topic_id    TEXT,
  title       TEXT NOT NULL,
  body        TEXT NOT NULL,
  kind        TEXT NOT NULL DEFAULT 'episodic',
  entities    TEXT,
  salience    REAL NOT NULL DEFAULT 0.5,
  confidence  REAL NOT NULL DEFAULT 0.7,
  valid_from  INTEGER,
  valid_to    INTEGER,
  merged_into TEXT,
  source      TEXT NOT NULL DEFAULT 'heuristic',
  created_at  INTEGER NOT NULL,
  updated_at  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_sum_topic ON summaries(topic_id, salience DESC);
CREATE INDEX IF NOT EXISTS ix_sum_time ON summaries(created_at);

CREATE TABLE IF NOT EXISTS summary_messages (
  summary_id TEXT NOT NULL,
  message_id TEXT NOT NULL,
  weight     REAL NOT NULL DEFAULT 1.0,
  PRIMARY KEY (summary_id, message_id)
);
CREATE INDEX IF NOT EXISTS ix_sm_msg ON summary_messages(message_id);

CREATE TABLE IF NOT EXISTS summary_edges (
  src TEXT NOT NULL,
  dst TEXT NOT NULL,
  rel TEXT NOT NULL,
  weight REAL NOT NULL DEFAULT 1.0,
  PRIMARY KEY (src, dst, rel)
);

CREATE TABLE IF NOT EXISTS embeddings (
  owner_type TEXT NOT NULL,
  owner_id   TEXT NOT NULL,
  model      TEXT NOT NULL,
  dim        INTEGER NOT NULL,
  vec        BLOB NOT NULL,
  norm       REAL NOT NULL,
  PRIMARY KEY (owner_type, owner_id, model)
);

CREATE TABLE IF NOT EXISTS memory_jobs (
  id         TEXT PRIMARY KEY,
  kind       TEXT NOT NULL,
  payload    TEXT NOT NULL,
  state      TEXT NOT NULL DEFAULT 'queued',
  attempts   INTEGER NOT NULL DEFAULT 0,
  run_after  INTEGER,
  last_error TEXT,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_jobs_state ON memory_jobs(state, run_after);

CREATE TABLE IF NOT EXISTS images (
  id         TEXT PRIMARY KEY,
  prompt     TEXT NOT NULL DEFAULT '',
  negative   TEXT,
  model      TEXT,
  path       TEXT NOT NULL,
  width      INTEGER,
  height     INTEGER,
  seed       INTEGER,
  created_at INTEGER NOT NULL,
  meta       TEXT
);
CREATE INDEX IF NOT EXISTS ix_images_time ON images(created_at DESC);

CREATE TABLE IF NOT EXISTS retrieval_log (
  id         TEXT PRIMARY KEY,
  request_id TEXT NOT NULL,
  conversation_id TEXT,
  query      TEXT,
  plan       TEXT,
  latency_ms INTEGER,
  created_at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_rlog_request ON retrieval_log(request_id);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
  message_id UNINDEXED,
  content,
  tokenize = 'unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
  summary_id UNINDEXED,
  title,
  body,
  tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS trg_msg_ai AFTER INSERT ON messages BEGIN
  INSERT INTO messages_fts(message_id, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS trg_sum_ai AFTER INSERT ON summaries BEGIN
  INSERT INTO summaries_fts(summary_id, title, body) VALUES (new.id, new.title, new.body);
END;

CREATE TRIGGER IF NOT EXISTS trg_sum_au AFTER UPDATE OF title, body ON summaries BEGIN
  DELETE FROM summaries_fts WHERE summary_id = old.id;
  INSERT INTO summaries_fts(summary_id, title, body) VALUES (new.id, new.title, new.body);
END;
"""


class Database:
    """Thread-safe-by-connection SQLite wrapper with a single writer lock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._write_lock = threading.RLock()
        self._setup()

    # -- connections -------------------------------------------------------
    def _setup(self) -> None:
        conn = sqlite3.connect(self.path, timeout=10.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA foreign_keys=ON")
            version = int(conn.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"database schema v{version} is newer than this build (v{SCHEMA_VERSION})"
                )
            if version < SCHEMA_VERSION:
                conn.executescript(_SCHEMA)
                conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
                conn.commit()
        finally:
            conn.close()

    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = sqlite3.connect(self.path, timeout=10.0, check_same_thread=False)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys=ON")
            c.execute("PRAGMA busy_timeout=8000")
            self._local.conn = c
            with self._conns_lock:
                self._conns.append(c)
        return c

    @contextmanager
    def read(self) -> Iterator[sqlite3.Connection]:
        yield self.conn()

    @contextmanager
    def write(self) -> Iterator[sqlite3.Connection]:
        conn = self.conn()
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
                conn.commit()
            except BaseException:
                conn.rollback()
                raise

    def close(self) -> None:
        with self._conns_lock:
            for c in self._conns:
                try:
                    c.close()
                except sqlite3.Error:
                    pass
            self._conns.clear()
        self._local = threading.local()

    # -- convenience -------------------------------------------------------
    def scalar(self, sql: str, args: tuple = ()) -> object:
        with self.read() as c:
            row = c.execute(sql, args).fetchone()
            return row[0] if row else None

    def stats(self) -> dict:
        with self.read() as c:
            def one(q: str) -> int:
                row = c.execute(q).fetchone()
                return int(row[0]) if row else 0

            return {
                "messages": one("SELECT COUNT(*) FROM messages"),
                "conversations": one("SELECT COUNT(*) FROM conversations"),
                "summaries": one("SELECT COUNT(*) FROM summaries WHERE merged_into IS NULL"),
                "topics": one("SELECT COUNT(*) FROM topics"),
                "jobs_pending": one("SELECT COUNT(*) FROM memory_jobs WHERE state='queued'"),
                "jobs_failed": one("SELECT COUNT(*) FROM memory_jobs WHERE state='failed'"),
            }


def now_ms() -> int:
    return int(time.time() * 1000)


def json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
