"""HTTP client for the OpenAI-compatible upstream (OVMS by default).

The gateway never talks to a model directly. It talks to this client, which means the
same memory pipeline works with OVMS today and with any compatible server later.
"""

from __future__ import annotations

import json
from urllib.parse import urlsplit, urlunsplit

import httpx


class UpstreamError(Exception):
    def __init__(self, status: int, message: str, *, kind: str = "upstream_error") -> None:
        super().__init__(message)
        self.status = status
        self.message = message
        self.kind = kind


def validate_base_url(value: str) -> str:
    value = (value or "").strip().rstrip("/")
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("upstream URL must be an http(s) URL")
    if parts.username or parts.password:
        raise ValueError("upstream URL must not contain credentials")
    return value


def split_api_base(value: str) -> tuple[str, str]:
    """Split a configured URL into server root and API prefix.

    OVMS 2026 serves chat, embeddings and image generation under ``/v3`` while
    most OpenAI-compatible servers use ``/v1``.  Users paste either form (and
    sometimes include a trailing slash), so normalise once instead of producing
    paths like ``/v3/v1/chat/completions``.
    """
    value = validate_base_url(value)
    parts = urlsplit(value)
    path = parts.path.rstrip("/")
    prefix = "/v1"
    root_path = path
    for candidate in ("/v3", "/v1"):
        if path.endswith(candidate):
            prefix = candidate
            root_path = path[: -len(candidate)]
            break
    root = urlunsplit((parts.scheme, parts.netloc, root_path, "", "")).rstrip("/")
    return root, prefix


def api_root(value: str) -> str:
    return split_api_base(value)[0]


def api_url(value: str, path: str) -> str:
    root, prefix = split_api_base(value)
    return f"{root}{prefix}/{path.lstrip('/')}"


class Upstream:
    def __init__(self, base_url: str, *, read_timeout: float = 300.0) -> None:
        self._root, self._prefix = split_api_base(base_url)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(read_timeout, connect=5.0, write=30.0, pool=5.0),
            limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
            trust_env=False,
        )

    @property
    def base_url(self) -> str:
        return f"{self._root}{self._prefix}"

    @base_url.setter
    def base_url(self, value: str) -> None:
        self._root, self._prefix = split_api_base(value)

    @property
    def root_url(self) -> str:
        return self._root

    @property
    def api_prefix(self) -> str:
        return self._prefix

    async def aclose(self) -> None:
        await self._client.aclose()

    def _url(self, path: str) -> str:
        return f"{self._root}{self._prefix}/{path.lstrip('/')}"

    async def health(self) -> dict:
        try:
            resp = await self._client.get(self._url("/models"), timeout=3.0)
            if resp.status_code != 200:
                return {"ok": False, "error": f"HTTP {resp.status_code}"}
            data = resp.json()
            models = [m.get("id") for m in data.get("data", []) if isinstance(m, dict)]
            return {"ok": True, "models": models}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

    async def list_models(self) -> dict:
        resp = await self._client.get(self._url("/models"), timeout=10.0)
        if resp.status_code != 200:
            raise UpstreamError(resp.status_code, _short_body(resp))
        return resp.json()

    async def start_stream(self, payload: dict) -> httpx.Response:
        """Open a streaming completion. The caller owns the response and must close it."""
        headers = {"Accept": "text/event-stream"}
        request = self._client.build_request(
            "POST", self._url("/chat/completions"), json=payload, headers=headers
        )
        resp = await self._client.send(request, stream=True)
        if resp.status_code >= 400:
            body = await resp.aread()
            await resp.aclose()
            raise UpstreamError(resp.status_code, _decode_error(body))
        return resp

    async def complete(self, payload: dict) -> dict:
        resp = await self._client.post(self._url("/chat/completions"), json=payload)
        if resp.status_code >= 400:
            raise UpstreamError(resp.status_code, _decode_error(resp.content))
        return resp.json()

    async def images_generate(self, payload: dict) -> dict:
        resp = await self._client.post(
            self._url("/images/generations"), json=_image_payload(payload)
        )
        if resp.status_code >= 400:
            raise UpstreamError(resp.status_code, _decode_error(resp.content))
        return resp.json()

    async def images_edit(self, payload: dict, files: dict) -> dict:
        resp = await self._client.post(
            self._url("/images/edits"), data=_image_payload(payload, form=True), files=files
        )
        if resp.status_code >= 400:
            raise UpstreamError(resp.status_code, _decode_error(resp.content))
        return resp.json()


def _image_payload(payload: dict, *, form: bool = False) -> dict:
    """Translate the app's friendly fields to OVMS's OpenAI image schema.

    OVMS uses ``size: "512x512"`` and ``rng_seed``; the UI uses width/height and
    seed so the form can validate them as numbers.  Keep both forms accepted so
    other OpenAI-compatible servers continue to work.
    """
    data = {k: v for k, v in payload.items() if v is not None}
    width = data.pop("width", None)
    height = data.pop("height", None)
    if width and height and not data.get("size"):
        data["size"] = f"{int(width)}x{int(height)}"
    seed = data.pop("seed", None)
    if seed is not None and "rng_seed" not in data:
        data["rng_seed"] = int(seed)
    steps = data.pop("steps", None)
    if steps is not None and "num_inference_steps" not in data:
        data["num_inference_steps"] = int(steps)
    data.setdefault("response_format", "b64_json")
    if form:
        return {k: str(v) for k, v in data.items()}
    return data


def _short_body(resp: httpx.Response) -> str:
    return _decode_error(resp.content)


def _decode_error(body: bytes) -> str:
    text = body.decode("utf-8", "replace")[:2000]
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            err = data.get("error")
            if isinstance(err, dict):
                return str(err.get("message") or err)[:800]
            if err:
                return str(err)[:800]
            if data.get("detail"):
                return str(data["detail"])[:800]
    except json.JSONDecodeError:
        pass
    return text.strip()[:800] or "upstream returned an error"
