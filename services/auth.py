"""Chat-history persistence (auth itself moved to Keycloak).

Historically this module also did SQLite + bcrypt user auth. We pulled
that out when we migrated to Keycloak for OAuth/OIDC user management —
see services/keycloak_auth.py for the new auth path. What's left here
is the chat_history table, which is per-user message log, not auth.
We keep it in SQLite (data/user.db) so users have continuity even if
their Keycloak account is recreated under the same username.
"""
from __future__ import annotations

import json
import sqlite3

from services.config import settings
from services.logging_config import get_logger

log = get_logger(__name__)


def _db_path() -> str:
    path = settings.paths.user_db
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_connection() -> sqlite3.Connection:
    return sqlite3.connect(_db_path())


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
