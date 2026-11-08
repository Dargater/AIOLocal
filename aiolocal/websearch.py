"""Optional web search with explicit, per-message opt-in.

The feature is off by default. Every search must be requested by the user (the UI
re-arms a toggle for each message), and the server records that in the retrieval plan.

Safety rules enforced here:

* the query is sanitised and length-limited;
* outbound requests have a timeout, a body cap and redirects disabled;
* result URLs are only displayed after scheme/host checks; the app never follows them;
* private, loopback and link-local destinations are rejected (except a SeArxNG instance
  the user deliberately points at loopback);
* results are labelled as untrusted data in the prompt, because a web page can contain
  text that looks like instructions.
"""

from __future__ import annotations

import ipaddress
import re
import socket
import threading
import time
from collections import deque
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, urlsplit

import httpx

MAX_QUERY_CHARS = 300
MAX_RESULTS = 10
MAX_BODY_BYTES = 2 * 1024 * 1024
USER_AGENT = "AIOLocal/0.1 (local desktop assistant)"

_BLOCKED_SUFFIXES = (
    ".local", ".localhost", ".internal", ".intranet", ".lan", ".home", ".corp",
    ".test", ".invalid", ".example", ".onion",
)


class WebSearchError(Exception):
    """Search failed or was refused."""


# --------------------------------------------------------------------------- safety
def sanitise_query(query: str) -> str:
    text = re.sub(r"[\x00-\x1f\x7f]", " ", str(query or ""))
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        raise WebSearchError("the search query is empty")
    if len(text) > MAX_QUERY_CHARS:
        text = text[:MAX_QUERY_CHARS].rstrip()
    return text


def _host_is_private(host: str) -> bool:
    host = (host or "").strip("[]").lower().rstrip(".")
    if not host:
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return not ip.is_global or ip.is_multicast or ip.is_reserved

    if host == "localhost" or host.endswith(_BLOCKED_SUFFIXES):
        return True
    if "." not in host:
        return True  # single-label names are intranet hosts
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True
    for info in infos:
        try:
            ip = ipaddress.ip_address(info[4][0])
        except ValueError:
            return True
        if not ip.is_global or ip.is_multicast or ip.is_reserved:
            return True
    return False


def validate_result_url(url: str) -> str | None:
    """Return a cleaned http(s) URL, or ``None`` when it must not be shown."""
    url = str(url or "").strip()
    if not url or len(url) > 2048:
        return None
    try:
        parts = urlsplit(url)
    except ValueError:
        return None
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    if parts.username or parts.password:
        return None
    if _host_is_private(parts.hostname):
        return None
    return url


def validate_searxng_url(value: str) -> str:
    """Allow https anywhere public, or http only on loopback."""
    value = (value or "").strip().rstrip("/")
    if not value:
        raise WebSearchError("a SeArXNG URL is required for that backend")
    try:
        parts = urlsplit(value)
    except ValueError as exc:
        raise WebSearchError("invalid SeArXNG URL") from exc
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise WebSearchError("the SeArXNG URL must be http(s)")
    if parts.username or parts.password:
        raise WebSearchError("the SeArXNG URL must not contain credentials")
    host = parts.hostname.lower()
    loopback = host in ("127.0.0.1", "localhost", "::1")
    if parts.scheme == "http" and not loopback:
        raise WebSearchError("use https for a remote SeArXNG instance (plain http is loopback-only)")
    if not loopback and _host_is_private(host):
        raise WebSearchError("the SeArXNG URL must not point at a private address")
    return value


# --------------------------------------------------------------------------- limiter
class RateLimiter:
    def __init__(self, min_interval: float = 1.0, per_minute: int = 20) -> None:
        self.min_interval = min_interval
        self.per_minute = per_minute
        self._lock = threading.Lock()
        self._times: deque[float] = deque()

    def check(self) -> None:
        now = time.monotonic()
        with self._lock:
            while self._times and now - self._times[0] > 60.0:
                self._times.popleft()
            if self._times and now - self._times[-1] < self.min_interval:
                raise WebSearchError("please wait a moment before searching again")
            if len(self._times) >= self.per_minute:
                raise WebSearchError("web search rate limit reached; try again in a minute")
            self._times.append(now)


# --------------------------------------------------------------------------- parsing
class _DuckParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[dict] = []
        self._mode: str | None = None
        self._href = ""
        self._buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        data = dict(attrs)
        classes = (data.get("class") or "").split()
        if "result__a" in classes:
            self._mode = "title"
            self._href = data.get("href") or ""
            self._buf = []
        elif "result__snippet" in classes:
            self._mode = "snippet"
            self._buf = []

    def handle_data(self, data: str) -> None:
        if self._mode:
            self._buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._mode:
            return
        text = unescape("".join(self._buf)).strip()
        if self._mode == "title":
            if self._href and text:
                self.results.append({"title": text, "url": self._href, "snippet": ""})
        elif self._mode == "snippet" and self.results:
            self.results[-1]["snippet"] = text
        self._mode = None

    def close(self) -> None:
        super().close()
        # DuckDuckGo wraps links in /l/?uddg=<encoded target>.
        for result in self.results:
            url = result.get("url") or ""
            try:
                parts = urlsplit(url)
                if parts.netloc.endswith("duckduckgo.com") and parts.path.startswith("/l/"):
                    target = parse_qs(parts.query).get("uddg", [""])[0]
                    if target:
                        result["url"] = target
                elif url.startswith("//"):
                    result["url"] = "https:" + url
            except ValueError:
                continue


def _clean_text(value: str, limit: int) -> str:
    text = re.sub(r"(?s)<[^>]+>", " ", str(value or ""))
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def parse_duckduckgo(markup: str) -> list[dict]:
    parser = _DuckParser()
    try:
        parser.feed(markup)
        parser.close()
    except Exception:
        return []
    return parser.results


# --------------------------------------------------------------------------- search
class WebSearch:
    def __init__(self, settings, secrets) -> None:
        self.settings = settings
        self.secrets = secrets
        self.limiter = RateLimiter()

    def available(self) -> tuple[bool, str]:
        if not self.settings.web_search_enabled:
            return False, "Web search is disabled in Settings."
        backend = self.settings.web_search_backend
        if backend == "searxng" and not self.settings.searxng_url:
            return False, "Set a SeArXNG URL in Settings."
        if backend == "brave" and not self.secrets.get("brave_api_key"):
            return False, "Add a Brave Search API key in Settings."
        return True, backend

    def search(self, query: str, *, limit: int | None = None) -> list[dict]:
        ok, detail = self.available()
        if not ok:
            raise WebSearchError(detail)
        query = sanitise_query(query)
        self.limiter.check()
        limit = max(1, min(int(limit or self.settings.web_search_max_results), MAX_RESULTS))

        backend = self.settings.web_search_backend
        if backend == "duckduckgo":
            raw = self._duckduckgo(query, limit)
        elif backend == "searxng":
            raw = self._searxng(query, limit)
        elif backend == "brave":
            raw = self._brave(query, limit)
        else:
            raise WebSearchError("unknown search backend")

        return _dedupe_and_validate(raw)[:limit]

    # -- backends ---------------------------------------------------------
    def _client(self, follow: bool = False) -> httpx.Client:
        return httpx.Client(
            timeout=httpx.Timeout(12.0, connect=5.0),
            follow_redirects=follow,
            trust_env=False,
            headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/json"},
        )

    def _duckduckgo(self, query: str, limit: int) -> list[dict]:
        with self._client() as client:
            resp = client.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "kl": "wt-wt"},
            )
        if resp.status_code >= 400:
            raise WebSearchError(f"DuckDuckGo returned HTTP {resp.status_code}")
        if len(resp.content) > MAX_BODY_BYTES:
            raise WebSearchError("search response was too large")
        return parse_duckduckgo(resp.text)[: limit * 3]

    def _searxng(self, query: str, limit: int) -> list[dict]:
        base = validate_searxng_url(self.settings.searxng_url)
        with self._client() as client:
            resp = client.get(
                f"{base}/search",
                params={
                    "q": query,
                    "format": "json",
                    "safesearch": "2",
                    "categories": "general",
                },
            )
        if resp.status_code >= 400:
            raise WebSearchError(f"SeArXNG returned HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise WebSearchError("SeArXNG did not return JSON (enable format=json)") from exc
        out = []
        for item in (data.get("results") or [])[: limit * 3]:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "title": _clean_text(item.get("title") or "", 300),
                    "url": str(item.get("url") or ""),
                    "snippet": _clean_text(item.get("content") or "", 800),
                }
            )
        return out

    def _brave(self, query: str, limit: int) -> list[dict]:
        key = self.secrets.get("brave_api_key")
        if not key:
            raise WebSearchError("Brave Search needs an API key")
        with self._client() as client:
            resp = client.get(
                "https://api.search.brave.com/res/v1/web/search",
                params={"q": query, "count": limit, "safesearch": "strict"},
                headers={"X-Subscription-Token": key, "Accept": "application/json"},
            )
        if resp.status_code in (401, 403):
            raise WebSearchError("Brave rejected the API key")
        if resp.status_code == 429:
            raise WebSearchError("Brave rate limit reached")
        if resp.status_code >= 400:
            raise WebSearchError(f"Brave returned HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError as exc:
            raise WebSearchError("Brave returned invalid JSON") from exc
        out = []
        for item in ((data.get("web") or {}).get("results") or [])[: limit * 3]:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "title": _clean_text(item.get("title") or "", 300),
                    "url": str(item.get("url") or ""),
                    "snippet": _clean_text(item.get("description") or "", 800),
                }
            )
        return out


def _dedupe_and_validate(items: list[dict]) -> list[dict]:
    out: list[dict] = []
    seen: set[str] = set()
    for item in items:
        url = validate_result_url(str(item.get("url") or ""))
        if not url:
            continue
        key = url.split("#", 1)[0].rstrip("/")
        if key in seen:
            continue
        seen.add(key)
        title = _clean_text(item.get("title") or "", 300) or url
        snippet = _clean_text(item.get("snippet") or "", 800)
        out.append({"title": title, "url": url, "snippet": snippet})
    return out


def render_results(results: list[dict]) -> str:
    lines = []
    for index, item in enumerate(results, start=1):
        lines.append(f"[{index}] {item['title']}\nURL: {item['url']}\n{item['snippet']}")
    return "\n\n".join(lines)
