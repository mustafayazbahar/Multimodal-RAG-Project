"""Authentication, RBAC, and chat history persistence.

Uses bcrypt for password hashing (with legacy SHA-256 migration support)
so existing user.db files keep working after upgrade.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Optional, Tuple

import bcrypt

from config import settings
from logging_config import get_logger

log = get_logger(__name__)

DB_NAME = str(settings.paths.user_db)
_LEGACY_HEX_LEN = 64  # SHA-256 hex digest length


def _hash_password(password: str) -> str:
    rounds = settings.auth.bcrypt_rounds
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=rounds)).decode("utf-8")


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
    db_path = settings.paths.user_db
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(db_path))


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
        cur = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,))
        if cur.fetchone():
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
        cur = conn.execute("PRAGMA table_info(chat_history)")
        existing = {row[1] for row in cur.fetchall()}
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
    password = password or ""
    if not username or not password:
        return False, None

    with get_connection() as conn:
        cur = conn.execute(
            "SELECT password, role FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        if not row:
            return False, None

        stored_hash, role = row
        if not _verify_password(password, stored_hash):
            return False, None

        # Transparent migration: upgrade legacy SHA-256 to bcrypt on successful login.
        if _is_legacy_hash(stored_hash):
            try:
                conn.execute(
                    "UPDATE users SET password = ? WHERE username = ?",
                    (_hash_password(password), username),
                )
                log.info("Migrated user '%s' to bcrypt hash", username)
            except sqlite3.Error as exc:
                log.warning("Failed to migrate password hash for '%s': %s", username, exc)

    return True, role


def save_message(username: str, role: str, content: str, sources: str = "", images=None) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_history (username, role, content, sources, images)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                username,
                role,
                content,
                sources or "",
                json.dumps(images or [], ensure_ascii=False),
            ),
        )


def load_chat_history(username: str) -> list[dict]:
    with get_connection() as conn:
        cur = conn.execute(
            """
            SELECT role, content, sources, images
            FROM chat_history
            WHERE username = ?
            ORDER BY id ASC
            """,
            (username,),
        )
        rows = cur.fetchall()

    history: list[dict] = []
    for role, content, sources, images in rows:
        try:
            parsed_images = json.loads(images) if images else []
            if not isinstance(parsed_images, list):
                parsed_images = []
        except (json.JSONDecodeError, TypeError):
            parsed_images = []
        history.append(
            {
                "role": role,
                "content": content,
                "sources": sources or "",
                "images": parsed_images,
            }
        )
    return history


def clear_chat_history(username: str) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM chat_history WHERE username = ?", (username,))
