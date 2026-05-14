"""Tests for the bcrypt-based auth module + legacy hash migration."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
from pathlib import Path

import pytest


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    db_path = tmp_path / "user.db"
    monkeypatch.setenv("USER_DB_PATH", str(db_path))
    # Force re-import so config picks up the new path
    for mod in ("auth", "config"):
        sys.modules.pop(mod, None)
    import auth  # noqa: WPS433

    auth.create_users_table()
    auth.create_chat_table()
    return auth, db_path


def test_register_and_login(fresh_db):
    auth, _ = fresh_db
    assert auth.register_user("alice", "secret123") is True
    ok, role = auth.login_user("alice", "secret123")
    assert ok is True
    assert role == "student"


def test_register_rejects_duplicate(fresh_db):
    auth, _ = fresh_db
    auth.register_user("bob", "pw1")
    assert auth.register_user("bob", "pw2") is False


def test_register_rejects_empty(fresh_db):
    auth, _ = fresh_db
    assert auth.register_user("", "pw") is False
    assert auth.register_user("user", "") is False


def test_wrong_password_fails(fresh_db):
    auth, _ = fresh_db
    auth.register_user("carol", "rightpw")
    ok, _ = auth.login_user("carol", "wrongpw")
    assert ok is False


def test_legacy_sha256_migrates_on_login(fresh_db):
    auth, db_path = fresh_db
    legacy_hash = hashlib.sha256(b"oldpassword").hexdigest()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("dave", legacy_hash, "student"),
        )

    ok, role = auth.login_user("dave", "oldpassword")
    assert ok is True
    assert role == "student"

    with sqlite3.connect(db_path) as conn:
        new_hash = conn.execute(
            "SELECT password FROM users WHERE username = ?", ("dave",)
        ).fetchone()[0]
    assert new_hash.startswith("$2")  # bcrypt prefix


def test_chat_history_roundtrip(fresh_db):
    auth, _ = fresh_db
    auth.register_user("eve", "pw")
    auth.save_message("eve", "user", "hello", sources="src.pdf p.1", images=["a.png"])
    auth.save_message("eve", "assistant", "hi back")
    history = auth.load_chat_history("eve")
    assert len(history) == 2
    assert history[0]["images"] == ["a.png"]
    auth.clear_chat_history("eve")
    assert auth.load_chat_history("eve") == []
