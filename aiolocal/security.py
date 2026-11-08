"""Local security primitives.

Design notes
------------
* The API token is a bearer secret generated once per installation and stored with
  Windows DPAPI (user scope). On non-Windows it falls back to a 0600 file so the code
  remains testable; the shipped target is Windows.
* Every HTTP request is checked for a loopback ``Host`` and, when present, a same-origin
  ``Origin``/``Referer``. This blocks DNS-rebinding and cross-site requests from a browser,
  which is the realistic attack surface for a localhost server.
* Filesystem access always goes through :func:`safe_join`, and identifiers are validated
  before they are used in a path or a subprocess argument.
"""

from __future__ import annotations

import base64
import ctypes
import hmac
import json
import os
import re
import secrets
import sys
import threading
from pathlib import Path
from urllib.parse import urlsplit

TOKEN_BYTES = 32
_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
_REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}/[A-Za-z0-9][A-Za-z0-9._-]{0,95}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


# --------------------------------------------------------------------------- DPAPI
class _DATA_BLOB(ctypes.Structure):
    _fields_ = [
        ("cbData", ctypes.c_uint32),
        ("pbData", ctypes.POINTER(ctypes.c_ubyte)),
    ]


def _blob(data: bytes):
    buf = ctypes.create_string_buffer(data, len(data))
    blob = _DATA_BLOB(len(data), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    return blob, buf


def _dpapi(data: bytes, protect: bool) -> bytes:
    crypt32 = ctypes.windll.crypt32  # type: ignore[attr-defined]
    fn = crypt32.CryptProtectData if protect else crypt32.CryptUnprotectData
    fn.restype = ctypes.c_int
    in_blob, _keep = _blob(data)
    out_blob = _DATA_BLOB()
    ok = fn(
        ctypes.byref(in_blob),
        None,
        None,
        None,
        None,
        0x1,  # CRYPTPROTECT_UI_FORBIDDEN
        ctypes.byref(out_blob),
    )
    if not ok:
        raise OSError(f"DPAPI {'encrypt' if protect else 'decrypt'} failed ({ctypes.get_last_error()})")
    try:
        return ctypes.string_at(out_blob.pbData, out_blob.cbData)
    finally:
        ctypes.windll.kernel32.LocalFree(out_blob.pbData)  # type: ignore[attr-defined]


def encrypt_secret(data: bytes) -> bytes:
    if sys.platform == "win32":
        return b"DPAPI1\x00" + _dpapi(data, protect=True)
    return b"PLAIN1\x00" + data


def decrypt_secret(data: bytes) -> bytes:
    if data.startswith(b"DPAPI1\x00"):
        if sys.platform != "win32":
            raise OSError("DPAPI blob cannot be read on this platform")
        return _dpapi(data[7:], protect=False)
    if data.startswith(b"PLAIN1\x00"):
        return data[7:]
    raise OSError("unknown secret format")


def load_or_create_token(path: Path) -> str:
    """Return the local API token, creating it on first run."""
    try:
        if path.exists():
            raw = decrypt_secret(path.read_bytes())
            token = raw.decode("ascii").strip()
            if len(token) >= 24:
                return token
    except (OSError, ValueError, UnicodeDecodeError):
        # A token we cannot decrypt (different user, corrupted file) is replaced.
        # The old token stops working; the UI re-bootstraps from the same origin.
        pass

    token = secrets.token_urlsafe(TOKEN_BYTES)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encrypt_secret(token.encode("ascii")))
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return token


def tokens_equal(expected: str, supplied: str | None) -> bool:
    if not supplied:
        return False
    return hmac.compare_digest(expected.encode("utf-8"), supplied.encode("utf-8", "ignore"))


# ----------------------------------------------------------------------- HTTP checks
def _split_host(value: str) -> tuple[str, int | None]:
    value = value.strip().lower()
    if value.startswith("["):  # [::1]:7317
        end = value.find("]")
        if end == -1:
            return value, None
        host = value[1:end]
        rest = value[end + 1 :]
        port = int(rest[1:]) if rest.startswith(":") and rest[1:].isdigit() else None
        return host, port
    if value.count(":") == 1:
        host, _, p = value.partition(":")
        return host, int(p) if p.isdigit() else None
    return value, None


def host_allowed(host_header: str | None, expected_port: int) -> bool:
    if not host_header:
        return False
    host, port = _split_host(host_header)
    if host not in _LOOPBACK_HOSTS:
        return False
    if port is not None and port != expected_port:
        return False
    return True


def origin_allowed(origin: str | None, expected_port: int) -> bool:
    """``Origin``/``Referer`` must be absent (non-browser) or same-origin loopback."""
    if not origin:
        return True
    try:
        parts = urlsplit(origin)
    except ValueError:
        return False
    if parts.scheme not in ("http", "https"):
        return False
    host = (parts.hostname or "").lower()
    if host not in _LOOPBACK_HOSTS:
        return False
    port = parts.port or (443 if parts.scheme == "https" else 80)
    return port == expected_port


# ---------------------------------------------------------------------- identifiers
def valid_repo(value: str) -> bool:
    return bool(_REPO_RE.fullmatch(value or ""))


def valid_id(value: str) -> bool:
    return bool(_ID_RE.fullmatch(value or ""))


def safe_join(root: Path, *parts: str) -> Path:
    """Join under *root*, rejecting absolute paths, traversal and symlink escapes."""
    root = root.resolve()
    for part in parts:
        if not part or part in (".", "..") or "/" in part or "\\" in part:
            raise ValueError("invalid path component")
    candidate = (root.joinpath(*parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("path escapes root") from exc
    return candidate


def redact(text: str, secrets_to_hide: list[str]) -> str:
    out = text
    for secret in secrets_to_hide:
        if secret:
            out = out.replace(secret, "[redacted]")
    return out


class SecretStore:
    """Small encrypted key/value store for optional third-party keys.

    Values are JSON, encrypted as one blob with the same DPAPI primitive as the API
    token. Nothing here is ever returned to the UI; callers only receive a boolean
    "configured" flag.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def _load(self) -> dict:
        try:
            if not self.path.exists():
                return {}
            raw = json.loads(decrypt_secret(self.path.read_bytes()).decode("utf-8"))
            return raw if isinstance(raw, dict) else {}
        except (OSError, ValueError, UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_bytes(encrypt_secret(payload))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self.path)

    def get(self, key: str) -> str:
        with self._lock:
            value = self._load().get(key)
        return str(value) if isinstance(value, str) else ""

    def set(self, key: str, value: str) -> None:
        with self._lock:
            data = self._load()
            if value:
                data[key] = value
            else:
                data.pop(key, None)
            self._save(data)

    def has(self, key: str) -> bool:
        return bool(self.get(key))


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")
