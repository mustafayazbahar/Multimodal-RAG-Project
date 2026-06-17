"""Chat-session + history persistence (auth itself moved to Keycloak).

Two tables now:
  chat_sessions  - one row per "topic" (named chat thread) per user.
  chat_history   - one row per message, foreign-keyed to a session.

The General Chat session is auto-created for every user on first
access, is_default=1, and is protected from rename/delete so users
always have a fallback thread.

Migration story (old single-thread schema → multi-session):
on first boot after upgrade, create_chat_table() detects orphan
chat_history rows (session_id IS NULL) and reparents them to each
user's General Chat. Idempotent — running it again is a no-op.
"""
from __future__ import annotations

import json
import sqlite3
import uuid

from services.config import settings
from services.logging_config import get_logger

log = get_logger(__name__)

# Her kullanici icin otomatik olusturulan, silinemeyen/yeniden adlandirilamayan
# varsayilan sohbet basligi. Kullanici her zaman bir geri donus thread'ine sahip olur.
GENERAL_CHAT_TITLE = "General Chat"


def _db_path() -> str:
    # SQLite dosya yolunu dondurur ve ust dizinin var oldugundan emin olur.
    # mkdir(exist_ok=True): ilk caliştirmada data/ klasoru yoksa olusturur, varsa hata vermez.
    path = settings.paths.user_db
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def get_connection() -> sqlite3.Connection:
    # Tek noktadan SQLite baglantisi acar. Cagiranlar bunu "with" blogunda
    # kullanir; boylece blok sonunda commit/rollback otomatik yapilir.
    return sqlite3.connect(_db_path())


def create_chat_table() -> None:
    """Create and (idempotently) migrate the chat tables.

    Adds the chat_sessions table, the session_id column on chat_history,
    and back-fills any orphan messages from the pre-Topics schema into a
    newly-minted General Chat per affected user.
    """
    with get_connection() as conn:
        # Sohbet oturumlari (Topics) tablosu: kullanici basina bir veya cok thread.
        # is_default=1 satiri General Chat'i isaretler.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_sessions (
                session_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                title TEXT NOT NULL,
                is_default INTEGER NOT NULL DEFAULT 0,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        # Mesaj gecmisi tablosu: her satir bir mesaj. session_id ile bir oturuma baglidir.
        # sources ve images, yardimci cevap verisini (kaynaklar, gorseller) tasir.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                username TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT DEFAULT '',
                images TEXT DEFAULT '[]'
            )
            """
        )

        # MIGRATION (semaya gecis): CREATE TABLE yeni kurulumlari halleder, ancak
        # eski bir veritabaninda chat_history tablosu zaten varsa eksik sutunlari
        # eklemek gerekir. Once mevcut sutun adlarini PRAGMA ile okuyup, olmayanlari
        # tek tek ekliyoruz (SQLite'ta "ADD COLUMN IF NOT EXISTS" yok).
        existing_cols = {
            row[1] for row in conn.execute("PRAGMA table_info(chat_history)").fetchall()
        }
        if "session_id" not in existing_cols:
            conn.execute("ALTER TABLE chat_history ADD COLUMN session_id TEXT")
        if "sources" not in existing_cols:
            conn.execute("ALTER TABLE chat_history ADD COLUMN sources TEXT DEFAULT ''")
        if "images" not in existing_cols:
            conn.execute("ALTER TABLE chat_history ADD COLUMN images TEXT DEFAULT '[]'")

        # Eski tek-thread semasindan kalan sahipsiz (session_id IS NULL) mesajlar.
        # Bunlar Topics ozelligi gelmeden once yazilmis eski kayitlardir.
        orphan_users = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT username FROM chat_history WHERE session_id IS NULL"
            ).fetchall()
        ]
        # Etkilenen her kullanici icin General Chat olusturup sahipsiz mesajlari
        # ona baglariz. Boylece eski sohbet gecmisi kaybolmadan yeni semaya tasinir.
        for username in orphan_users:
            session_id = _ensure_general_chat(conn, username)
            conn.execute(
                "UPDATE chat_history SET session_id = ? WHERE username = ? AND session_id IS NULL",
                (session_id, username),
            )

        # Sik sorgulanan sutunlar icin indeksler: mesajlari oturuma gore filtreleme
        # ve oturumlari kullaniciya gore (General Chat once, sonra en yeni) siralama hizlanir.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chat_sessions_user ON chat_sessions(username, is_default DESC, created_at DESC)"
        )


# ─────────────────────────────────────────────────────────────────────────
# Sessions (chat threads / "topics")
# ─────────────────────────────────────────────────────────────────────────
def _ensure_general_chat(conn: sqlite3.Connection, username: str) -> str:
    # Kullanicinin General Chat oturumunu getirir; yoksa olusturur (get-or-create).
    # Mevcut bir baglanti aldigi icin migration gibi tek transaction icinde
    # tekrar tekrar cagrilabilir (ic/dahili surum).
    row = conn.execute(
        "SELECT session_id FROM chat_sessions WHERE username = ? AND is_default = 1",
        (username,),
    ).fetchone()
    # Zaten varsa mevcut id'yi dondur; yeni satir acmadan tekrar kullan.
    if row:
        return row[0]
    # Yoksa yeni bir UUID ile is_default=1 oturum olustur.
    session_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO chat_sessions (session_id, username, title, is_default) "
        "VALUES (?, ?, ?, 1)",
        (session_id, username, GENERAL_CHAT_TITLE),
    )
    return session_id


def ensure_general_chat(username: str) -> str:
    """Public wrapper: get-or-create the user's General Chat session id."""
    # Disa acik sarmalayici: kendi baglantisini acip _ensure_general_chat'i cagirir.
    with get_connection() as conn:
        return _ensure_general_chat(conn, username)


def list_sessions(username: str) -> list[dict]:
    """Return all sessions for a user. General Chat first, newest other sessions next."""
    # Kullanicinin tum oturumlarini dondurur. Siralama is_default DESC ile General Chat'i
    # en uste, ardindan created_at DESC ile en yeni oturumlari koyar (arayuz beklentisi).
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT session_id, title, is_default, created_at "
            "FROM chat_sessions WHERE username = ? "
            "ORDER BY is_default DESC, created_at DESC",
            (username,),
        ).fetchall()
    return [
        {
            "session_id": r[0],
            "title": r[1],
            "is_default": bool(r[2]),
            "created_at": r[3],
        }
        for r in rows
    ]


def create_session(username: str, title: str) -> dict:
    # Kullanici icin yeni (varsayilan olmayan, is_default=0) bir sohbet oturumu olusturur.
    # Bos veya yalniz bosluk olan baslik "Untitled"a duser, boylece baslik hep dolu olur.
    title = (title or "").strip() or "Untitled"
    session_id = str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (session_id, username, title, is_default) "
            "VALUES (?, ?, ?, 0)",
            (session_id, username, title),
        )
        row = conn.execute(
            "SELECT created_at FROM chat_sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return {
        "session_id": session_id,
        "title": title,
        "is_default": False,
        "created_at": row[0] if row else None,
    }


def _session_owner(conn: sqlite3.Connection, session_id: str) -> tuple[str | None, bool]:
    # Bir oturumun sahibini (username) ve varsayilan olup olmadigini (is_default) dondurur.
    # Yetki kontrolu icin ortak yardimci; oturum yoksa (None, False) doner.
    row = conn.execute(
        "SELECT username, is_default FROM chat_sessions WHERE session_id = ?",
        (session_id,),
    ).fetchone()
    if not row:
        return None, False
    return row[0], bool(row[1])


def update_session_title(username: str, session_id: str, new_title: str) -> bool:
    """Rename a session. Refuses on missing/foreign/default sessions."""
    # Bos baslik kabul edilmez; islem yapilmadan False doner.
    new_title = (new_title or "").strip()
    if not new_title:
        return False
    with get_connection() as conn:
        # Guvenlik kontrolu: oturum baskasina aitse (owner != username) ya da
        # korumali General Chat ise (is_default) yeniden adlandirmayi reddet.
        owner, is_default = _session_owner(conn, session_id)
        if owner != username or is_default:
            return False
        conn.execute(
            "UPDATE chat_sessions SET title = ? WHERE session_id = ?",
            (new_title, session_id),
        )
    return True


def delete_session(username: str, session_id: str) -> bool:
    """Delete a session and all its messages. Refuses to delete General Chat."""
    with get_connection() as conn:
        # Yeniden adlandirma ile ayni yetki kurali: yabanci ya da varsayilan oturum silinemez.
        owner, is_default = _session_owner(conn, session_id)
        if owner != username or is_default:
            return False
        # Once mesajlari, sonra oturumu sil. Yetim mesaj kalmamasi icin sira onemli.
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
        conn.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))
    return True


def resolve_session(username: str, session_id: str | None) -> str:
    """Validate session_id is owned by `username`; otherwise fall back to General Chat.

    Used by every history/query endpoint so the frontend never has to
    create the default session explicitly, and a stale session_id from
    localStorage (e.g. a session the user deleted in another tab) can't
    cause a 404.
    """
    # Verilen session_id kullaniciya ait mi dogrular; degilse (silinmis ya da
    # baskasinin) sessizce General Chat'e duser. Boylece localStorage'da kalan
    # eski bir session_id 404 yerine guvenli bir varsayilana yonlenir.
    if session_id:
        with get_connection() as conn:
            owner, _ = _session_owner(conn, session_id)
        if owner == username:
            return session_id
    return ensure_general_chat(username)


# ─────────────────────────────────────────────────────────────────────────
# Messages
# ─────────────────────────────────────────────────────────────────────────
def save_message(
    session_id: str,
    username: str,
    role: str,
    content: str,
    sources: str = "",
    images: list | None = None,
) -> None:
    # Tek bir mesaji (kullanici ya da asistan) ilgili oturuma kaydeder.
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO chat_history (session_id, username, role, content, sources, images) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                username,
                role,
                content,
                sources or "",
                # images listesini JSON metnine cevirip saklariz (SQLite'ta dizi tipi yok).
                # ensure_ascii=False: Turkce karakterler \uXXXX'e kacirilmadan okunabilir kalir.
                json.dumps(images or [], ensure_ascii=False),
            ),
        )


def load_chat_history(session_id: str) -> list[dict]:
    # Bir oturumun tum mesajlarini kronolojik sirada (id ASC) yukler.
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT role, content, sources, images FROM chat_history "
            "WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    history: list[dict] = []
    for role, content, sources, images in rows:
        # images sutunu JSON metin olarak saklanir; parse edip listeye ceviriyoruz.
        # Bozuk/gecersiz JSON ya da liste olmayan veri gelirse bos listeye duseriz;
        # boylece tek bir hatali kayit tum gecmisin yuklenmesini engellemez.
        try:
            parsed = json.loads(images) if images else []
            if not isinstance(parsed, list):
                parsed = []
        except (json.JSONDecodeError, TypeError):
            parsed = []
        history.append(
            {"role": role, "content": content, "sources": sources or "", "images": parsed}
        )
    return history


def clear_chat_messages(session_id: str) -> None:
    """Wipe just the messages of a session; the session itself survives."""
    # Yalnizca mesajlari siler, oturum kaydi (chat_sessions) korunur.
    # Boylece kullanici ayni thread'i "temizleyip" kullanmaya devam edebilir.
    with get_connection() as conn:
        conn.execute("DELETE FROM chat_history WHERE session_id = ?", (session_id,))
