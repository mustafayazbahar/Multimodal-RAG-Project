"""Browser-localStorage-backed session persistence.

Streamlit's session_state is wiped on a full page refresh. We persist the
JWT plus username and role into the browser's localStorage so F5 keeps
the user signed in until the JWT expires.

Why localStorage and not cookies (third attempt):
- The extra-streamlit-components iframe-cookie path was inconsistent in
  Chrome: the document.cookie write from inside the sandboxed iframe
  was occasionally dropped, and the F5 read-retry hit the iframe-boot
  race in some sessions.
- The interim URL-query-params fallback (token=eyJ... in the address
  bar) was a security hole — Referer headers, browser history and any
  reverse-proxy access log would leak the JWT.
- localStorage is per-origin, synchronous from the JS side, immune to
  Chrome's third-party-cookie blocking, and survives F5 as long as the
  user doesn't clear browsing data. streamlit-local-storage wraps the
  same Streamlit component pattern but the underlying browser API is
  far more deterministic than document.cookie inside a sandboxed
  iframe.

Trade-off: localStorage is readable from JavaScript on the same origin,
so the token is exposed to XSS if the app ever renders attacker-
controlled HTML. We mitigate by escaping all user-controlled output in
the Streamlit layer and by issuing short-lived JWTs (12 h default).
For a hardened public deployment, switch to backend-issued HttpOnly +
Secure cookies — out of scope here.

Public API (unchanged, callers don't need to know about the swap):
    save_cookie(token, username, role)
    load_cookie() -> dict | None
    clear_cookie()
    hydrate_from_cookie()  # call once near the top of the script
"""
from __future__ import annotations

import json
import os
import time

import streamlit as st

# Lazy import so the app still boots if the package is missing — the
# user will simply have to re-login after every F5, which is a degraded
# but not broken state.
try:
    from streamlit_local_storage import LocalStorage  # type: ignore

    _LS_AVAILABLE = True
except Exception:  # noqa: BLE001
    LocalStorage = None  # type: ignore
    _LS_AVAILABLE = False

# localStorage'da oturum verisinin saklandigi anahtar.
SESSION_KEY = "dc_session"
# JWT'nin gecerlilik suresi (saat); ortam degiskeninden okunur, varsayilan 12 saat.
JWT_TTL_HOURS = int(os.getenv("JWT_TTL_HOURS", "12"))
# Streamlit components iframe needs a couple of reruns to push the
# localStorage value back into Python on a cold page load. 12 * 0.3 s
# = 3.6 s is comfortably above the worst observed iframe boot time.
# Yeniden deneme butcesi ve denemeler arasi bekleme suresi (saniye).
RETRY_BUDGET = 12
RETRY_DELAY_S = 0.3


# Streamlit oturumu basina TEK bir LocalStorage ornegi olusturup onbellekler.
def _storage() -> "LocalStorage | None":
    """Return a single LocalStorage instance per Streamlit session.

    Caching on session_state matters: each LocalStorage() constructor
    registers a hidden iframe component, and re-registering it in the
    same script run trips the StreamlitDuplicateElementKey guard.
    """
    if not _LS_AVAILABLE:
        return None
    if "_dc_local_storage" not in st.session_state:
        st.session_state["_dc_local_storage"] = LocalStorage()
    return st.session_state["_dc_local_storage"]


def _decode(raw: object) -> dict | None:
    """Best-effort parser: handles JSON strings, plain dicts, and junk.

    The optional `session_id` and `id_token` keys are tolerated when
    present (added in the v2.5 Topics + OAuth Code flow). Older blobs
    without them still validate.
    """
    if raw is None or raw == "":
        return None
    # localStorage bazen dogrudan dict, bazen JSON string dondurebilir; her iki
    # durumu da destekleyip beklenmeyen tipleri (junk) None'a indiriyoruz.
    if isinstance(raw, dict):
        data = raw
    elif isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return None
    else:
        return None
    if not isinstance(data, dict):
        return None
    # Zorunlu alanlar eksikse blob gecersizdir; opsiyonel alanlar (session_id,
    # id_token) eski kayitlarda bulunmayabilir, onlar dogrulamada aranmaz.
    if not all(k in data for k in ("token", "username", "role")):
        return None
    return data


def load_cookie() -> dict | None:
    """Read the session blob from localStorage. Returns None if absent."""
    ls = _storage()
    if ls is None:
        return None
    try:
        raw = ls.getItem(SESSION_KEY)
    except Exception:  # noqa: BLE001
        return None
    return _decode(raw)


def save_cookie(
    token: str,
    username: str,
    role: str,
    id_token: str | None = None,
    active_session_id: str | None = None,
) -> None:
    """Persist auth + UI state as a JSON blob in localStorage.

    `id_token` is needed for a silent OAuth logout (no Keycloak confirm
    page). `active_session_id` lets F5 restore the chat thread the
    user was looking at — without it every refresh dumps them back on
    General Chat.
    """
    ls = _storage()
    if ls is None:
        return
    payload = json.dumps(
        {
            "token": token,
            "username": username,
            "role": role,
            "id_token": id_token,
            "active_session_id": active_session_id,
        }
    )
    try:
        ls.setItem(SESSION_KEY, payload)
    except Exception:  # noqa: BLE001
        # localStorage write can fail on Safari private mode. Don't
        # crash the auth flow — degrade to "session lasts until F5".
        pass


def update_active_session(active_session_id: str | None) -> None:
    """Re-write the cookie with a new active_session_id, keeping auth fields.

    Used when the user switches Topics; we need to remember which one
    so F5 lands them back on the same thread.
    """
    cookie = load_cookie()
    if not cookie:
        return
    save_cookie(
        token=cookie["token"],
        username=cookie["username"],
        role=cookie["role"],
        id_token=cookie.get("id_token"),
        active_session_id=active_session_id,
    )


def clear_cookie() -> None:
    """Remove the session blob from localStorage."""
    ls = _storage()
    if ls is None:
        return
    try:
        ls.deleteItem(SESSION_KEY)
    except Exception:  # noqa: BLE001
        pass


def hydrate_from_cookie() -> None:
    """If the session is empty but a valid blob exists in localStorage,
    restore it.

    Called once on every page load. The first run after a page refresh
    often sees None because the localStorage component hasn't pushed
    its value into Python yet — we trigger a few short reruns to give
    it a chance before falling back to the login screen. After
    RETRY_BUDGET attempts we stop trying so an empty localStorage
    (user never logged in, or signed out) doesn't loop forever.
    """
    # Oturum zaten doluysa (kullanici bu calismada giris yapmis) hicbir sey yapma.
    if st.session_state.get("token"):
        return

    cookie = load_cookie()
    # Gecerli blob bulunduysa session_state'i geri doldur ve deneme sayacini sifirla.
    if cookie:
        st.session_state.token = cookie["token"]
        st.session_state.username = cookie["username"]
        st.session_state.role = cookie["role"]
        # New (v2.5) optional fields; missing on legacy blobs.
        if cookie.get("id_token"):
            st.session_state["id_token"] = cookie["id_token"]
        if cookie.get("active_session_id"):
            st.session_state["active_session_id"] = cookie["active_session_id"]
        st.session_state["_ls_attempts"] = 0
        return

    # Blob henuz gelmediyse: iframe degerini Python'a itene kadar kisa araliklarla
    # birkac kez yeniden dene. Butce dolunca durur ki bos localStorage'da (hic
    # giris yapmamis kullanici) sonsuz dongu olusmasin.
    attempts = st.session_state.get("_ls_attempts", 0)
    if attempts < RETRY_BUDGET:
        st.session_state["_ls_attempts"] = attempts + 1
        time.sleep(RETRY_DELAY_S)
        st.rerun()
