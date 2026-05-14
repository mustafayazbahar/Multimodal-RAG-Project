"""Auth, RBAC, and chat history with bcrypt + legacy-hash migration."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Optional, Tuple

import bcrypt

from services.config import settings
from services.logging_config import get_logger

log = get_logger(__name__)

_LEGACY_HEX_LEN = 64


def _db_path() -> str:
    path = settings.paths.user_db
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=settings.auth.bcrypt_rounds),
    ).decode("utf-8")


def _legacy_sha256(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _is_legacy_hash(stored: str) -> bool:
    return len(stored) == _LEGACY_HEX_LEN and not stored.startswith("$2")


def _verify_password(password: str, stored: str) -> bool:
    if not stored:
        return False
    if _is_legacy_hash(stored):
        return _legacy_sha256(password) == stored
    try:
        return bcrypt.checkpw(password.encode("utf-8"), stored.encode("utf-8"))
    except ValueError:
        return False


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(_db_path())


def create_users_table() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                role TEXT NOT NULL
            )
            """
        )


def create_default_admin() -> None:
    username = settings.auth.default_admin_username
    password = settings.auth.default_admin_password
    with get_connection() as conn:
        if conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            return
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, _hash_password(password), "instructor"),
        )
        log.info("Created default admin user '%s'", username)


def create_chat_table() -> None:
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT DEFAULT '',
                images TEXT DEFAULT '[]'
            )
            """
        )
        existing = {row[1] for row in conn.execute("PRAGMA table_info(chat_history)").fetchall()}
        if "sources" not in existing:
            conn.execute("ALTER TABLE chat_history ADD COLUMN sources TEXT DEFAULT ''")
        if "images" not in existing:
            conn.execute("ALTER TABLE chat_history ADD COLUMN images TEXT DEFAULT '[]'")


def register_user(username: str, password: str) -> bool:
    username = (username or "").strip()
    if not username or not password:
        return False
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, _hash_password(password), "student"),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def login_user(username: str, password: str) -> Tuple[bool, Optional[str]]:
    username = (username or "").strip()
    if not username or not password:
        return False, None

    with get_connection() as conn:
        row = conn.execute(
            "SELECT password, role FROM users WHERE username = ?", (username,)
        ).fetchone()
        if not row:
            return False, None
        stored_hash, role = row
        if not _verify_password(password, stored_hash):
            return False, None
        if _is_legacy_hash(stored_hash):
            try:
                conn.execute(
                    "UPDATE users SET password = ? WHERE username = ?",
                    (_hash_password(password), username),
                )
                log.info("Migrated user '%s' to bcrypt hash", username)
            except sqlite3.Error as exc:
                log.warning("Migration failed for '%s': %s", username, exc)
    return True, role


def save_message(username: str, role: str, content: str, sources: str = "", images=None) -> None:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_history (username, role, content, sources, images) VALUES (?, ?, ?, ?, ?)",
            (username, role, content, sources or "", json.dumps(images or [], ensure_ascii=False)),
        )


def load_chat_history(username: str) -> list[dict]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, sources, images FROM chat_history WHERE username = ? ORDER BY id ASC",
            (username,),
        ).fetchall()
    history: list[dict] = []
    for role, content, sources, images in rows:
        try:
            parsed = json.loads(images) if images else []
            if not isinstance(parsed, list):
                parsed = []
        except (json.JSONDecodeError, TypeError):
            parsed = []
        history.append({"role": role, "content": content, "sources": sources or "", "images": parsed})
    return history


def clear_chat_history(username: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM chat_history WHERE username = ?", (username,))
