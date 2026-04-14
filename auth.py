import sqlite3
import hashlib
import json

DB_NAME = "user.db"


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def get_connection():
    return sqlite3.connect(DB_NAME)


def create_users_table():
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def create_default_admin():
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT 1 FROM users WHERE username = ?", ("admin",))
    if not c.fetchone():
        c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            ("admin", hash_password("admin123"), "instructor")
        )
    conn.commit()
    conn.close()


def create_chat_table():
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT DEFAULT '',
            images TEXT DEFAULT '[]'
        )
    """)

    c.execute("PRAGMA table_info(chat_history)")
    existing_columns = [row[1] for row in c.fetchall()]

    if "sources" not in existing_columns:
        c.execute("ALTER TABLE chat_history ADD COLUMN sources TEXT DEFAULT ''")

    if "images" not in existing_columns:
        c.execute("ALTER TABLE chat_history ADD COLUMN images TEXT DEFAULT '[]'")

    conn.commit()
    conn.close()


def register_user(username, password):
    username = (username or "").strip()
    password = password or ""

    if not username or not password:
        return False

    conn = get_connection()
    c = conn.cursor()
    try:
        c.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hash_password(password), "student")
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def login_user(username, password):
    username = (username or "").strip()
    password = password or ""

    conn = get_connection()
    c = conn.cursor()
    c.execute(
        "SELECT role FROM users WHERE username = ? AND password = ?",
        (username, hash_password(password))
    )
    result = c.fetchone()
    conn.close()

    if result:
        return True, result[0]
    return False, None


def save_message(username, role, content, sources="", images=None):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO chat_history (username, role, content, sources, images)
        VALUES (?, ?, ?, ?, ?)
    """, (
        username,
        role,
        content,
        sources or "",
        json.dumps(images or [], ensure_ascii=False)
    ))
    conn.commit()
    conn.close()


def load_chat_history(username):
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        SELECT role, content, sources, images
        FROM chat_history
        WHERE username = ?
        ORDER BY id ASC
    """, (username,))
    rows = c.fetchall()
    conn.close()

    history = []
    for role, content, sources, images in rows:
        try:
            parsed_images = json.loads(images) if images else []
            if not isinstance(parsed_images, list):
                parsed_images = []
        except (json.JSONDecodeError, TypeError):
            parsed_images = []

        history.append({
            "role": role,
            "content": content,
            "sources": sources or "",
            "images": parsed_images
        })

    return history
