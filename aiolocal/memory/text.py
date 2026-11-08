"""Text utilities: ids, normalisation, token estimation, and local embeddings.

Token counts are an approximation. The plan calls for the model's own tokenizer; that
arrives with the tokenizer loader. Until then we use a deliberately conservative estimate
(never underestimate by much), so the assembler trims earlier rather than overflowing.
"""

from __future__ import annotations

import hashlib
import math
import re
import secrets
import time
from collections import Counter

import numpy as np

_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")
_SPACE_RE = re.compile(r"\s+")
_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "so", "of", "to", "in", "on",
    "for", "with", "without", "at", "by", "from", "as", "is", "are", "was", "were",
    "be", "been", "being", "it", "its", "this", "that", "these", "those", "i", "you",
    "he", "she", "we", "they", "me", "my", "your", "our", "their", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "should", "may", "might",
    "not", "no", "yes", "what", "why", "how", "when", "where", "who", "which", "just",
    "about", "into", "over", "under", "again", "also", "very", "really", "please",
}


def new_id() -> str:
    """Time-sortable, collision-resistant identifier (ULID-like, no dependency)."""
    return f"{int(time.time() * 1000):013d}{secrets.token_hex(8)}"


def normalise(text: str) -> str:
    return _SPACE_RE.sub(" ", (text or "").replace("\x00", " ")).strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(normalise(text).lower().encode("utf-8")).hexdigest()


def estimate_tokens(text: str) -> int:
    """Conservative estimate: ~4 chars/token and ~1.3 tokens/word, whichever is larger."""
    if not text:
        return 0
    words = len(text.split())
    return max(1, int(max(len(text) / 4.0, words * 1.3)))


def truncate_tokens(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if estimate_tokens(text) <= limit:
        return text
    # 4 chars/token is the planning ratio; leave room for the ellipsis.
    cut = max(0, limit * 4 - 1)
    out = text[:cut]
    if " " in out:
        out = out.rsplit(" ", 1)[0]
    return out.rstrip() + "…"


def keywords(text: str, limit: int = 8) -> list[str]:
    words = [w.lower() for w in _WORD_RE.findall(text or "")]
    words = [w for w in words if len(w) > 1 and w not in _STOPWORDS]
    return [w for w, _ in Counter(words).most_common(limit)]


def first_sentence(text: str, max_chars: int = 240) -> str:
    text = normalise(text)
    if not text:
        return ""
    match = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return (match[0] if match else text)[:max_chars].strip()


def _bucket(token: str, dim: int) -> tuple[int, float]:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "little")
    return value % dim, (1.0 if (value >> 63) & 1 else -1.0)


def embed(text: str, dim: int = 256) -> np.ndarray:
    """Deterministic hashed bag-of-words + character trigrams.

    This is not a semantic embedding. It exists so memory retrieval works with zero model
    downloads and zero network. When an OVMS embedding model is configured, the gateway can
    swap the backend without changing the storage format (``embeddings.model`` is stored).
    """
    dim = max(64, min(int(dim), 2048))
    vec = np.zeros(dim, dtype=np.float32)
    norm = normalise(text).lower()
    if not norm:
        return vec

    for word in _WORD_RE.findall(norm):
        b, s = _bucket("w:" + word, dim)
        vec[b] += s * 1.0

    compact = re.sub(r"\s+", " ", norm)
    for i in range(max(0, len(compact) - 2)):
        tri = compact[i : i + 3]
        if tri.strip():
            b, s = _bucket("t:" + tri, dim)
            vec[b] += s * 0.35

    n = float(np.linalg.norm(vec))
    if n > 0:
        vec /= n
    return vec


def pack(vec: np.ndarray) -> tuple[bytes, float]:
    v = np.asarray(vec, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v.tobytes(), n


def unpack(blob: bytes, dim: int) -> np.ndarray:
    v = np.frombuffer(blob, dtype=np.float32)
    if v.size != dim:
        return np.zeros(dim, dtype=np.float32)
    return v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def is_blank(text: str) -> bool:
    return not normalise(text)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))
