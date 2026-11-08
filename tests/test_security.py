from __future__ import annotations

from pathlib import Path

import pytest

from aiolocal.security import (
    decrypt_secret,
    encrypt_secret,
    host_allowed,
    load_or_create_token,
    origin_allowed,
    safe_join,
    tokens_equal,
    valid_id,
    valid_repo,
)


def test_host_allowed_loopback_only():
    assert host_allowed("127.0.0.1:7317", 7317)
    assert host_allowed("localhost:7317", 7317)
    assert host_allowed("[::1]:7317", 7317)
    assert not host_allowed("192.168.1.5:7317", 7317)
    assert not host_allowed("evil.example:7317", 7317)
    assert not host_allowed("127.0.0.1:8000", 7317)
    assert not host_allowed(None, 7317)


def test_origin_allowed():
    assert origin_allowed(None, 7317)
    assert origin_allowed("http://127.0.0.1:7317", 7317)
    assert origin_allowed("http://localhost:7317/page", 7317)
    assert not origin_allowed("https://evil.example", 7317)
    assert not origin_allowed("http://127.0.0.1:8000", 7317)
    assert not origin_allowed("file:///etc/passwd", 7317)


def test_tokens_equal():
    assert tokens_equal("abc", "abc")
    assert not tokens_equal("abc", "abd")
    assert not tokens_equal("abc", None)


def test_safe_join_rejects_escapes(tmp_path: Path):
    root = tmp_path / "root"
    root.mkdir()
    assert safe_join(root, "a.png").name == "a.png"
    for bad in ["..", ".", "../x", "a/b", "a\\b", "", "/abs"]:
        with pytest.raises(ValueError):
            safe_join(root, bad)


def test_identifiers():
    assert valid_repo("OpenVINO/Qwen3.5-9B-int4-ov")
    assert not valid_repo("../../etc/passwd")
    assert not valid_repo("no-slash")
    assert valid_id("01JABCDEFG0123456789")
    assert not valid_id("../../x")
    assert not valid_id("has space")


def test_token_round_trip(tmp_path: Path):
    path = tmp_path / "token.bin"
    token = load_or_create_token(path)
    assert len(token) >= 24
    again = load_or_create_token(path)
    assert again == token
    raw = path.read_bytes()
    assert token.encode("ascii") not in raw


def test_secret_format(tmp_path: Path):
    payload = b"secret-value"
    blob = encrypt_secret(payload)
    assert blob != payload
    assert decrypt_secret(blob) == payload
