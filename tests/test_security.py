"""Tests for JWT token generation and decoding."""
from __future__ import annotations

import time

import pytest

# `cryptography` (a pyjwt soft-dependency for asymmetric algorithms) panics on
# some sandboxed environments. Symmetric HS256 does not need it, but the JWT
# package eagerly imports it. Skip cleanly if that import explodes.
try:
    import jwt  # noqa: F401
    _IMPORT_OK = True
except BaseException:  # noqa: BLE001 — catches pyo3 PanicException too
    _IMPORT_OK = False

pytestmark = pytest.mark.skipif(not _IMPORT_OK, reason="pyjwt unavailable in this env")


def _load_security(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", "unit-secret")
    monkeypatch.setenv("JWT_TTL_HOURS", "1")
    import importlib
    import sys
    for mod in ("services.config", "backend.security"):
        sys.modules.pop(mod, None)
    return importlib.import_module("backend.security")


def test_token_roundtrip(monkeypatch):
    security = _load_security(monkeypatch)
    token = security.create_token("alice", "instructor")
    payload = security._decode(token)
    assert payload["sub"] == "alice"
    assert payload["role"] == "instructor"
    assert payload["exp"] > time.time()


def test_invalid_token_rejected(monkeypatch):
    security = _load_security(monkeypatch)
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        security._decode("not-a-jwt")
