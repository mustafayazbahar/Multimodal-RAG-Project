"""Cookie-backed session persistence.

Streamlit's session state is wiped on full page refresh. We store the JWT in
an HTTP cookie (via extra-streamlit-components) so a refresh keeps the user
logged in until the token expires.

Cookie attributes:
- max_age = JWT TTL (default 12h)
- The cookie is *not* httpOnly (extra-streamlit-components reads from JS).
  That's acceptable for this app because the backend treats the JWT as a
  bearer token; we mitigate XSS with strict CSP on the backend and by not
  rendering user-controlled HTML.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import extra_streamlit_components as stx
import streamlit as st

COOKIE_NAME = "dc_session"
COOKIE_TTL_HOURS = int(os.getenv("JWT_TTL_HOURS", "12"))


@st.cache_resource
def _cookie_manager() -> stx.CookieManager:
    return stx.CookieManager(key="dc_cookie_manager")


def load_cookie() -> dict | None:
    """Read the session cookie (returns dict with token/username/role or None)."""
    cm = _cookie_manager()
    raw = cm.get(cookie=COOKIE_NAME)
    if not raw or not isinstance(raw, dict):
        return None
    if not all(k in raw for k in ("token", "username", "role")):
        return None
    return raw


def save_cookie(token: str, username: str, role: str) -> None:
    cm = _cookie_manager()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=COOKIE_TTL_HOURS)
    cm.set(
        COOKIE_NAME,
        {"token": token, "username": username, "role": role},
        expires_at=expires_at,
        key="dc_cookie_set",
    )


def clear_cookie() -> None:
    cm = _cookie_manager()
    try:
        cm.delete(COOKIE_NAME, key="dc_cookie_delete")
    except KeyError:
        # extra-streamlit-components raises if the cookie was never set
        pass


def hydrate_from_cookie() -> None:
    """If the session is empty but a valid cookie exists, restore it.

    Called once on every page load — keeps the user logged in across refreshes.
    """
    if st.session_state.get("token"):
        return
    cookie = load_cookie()
    if not cookie:
        return
    st.session_state.token = cookie["token"]
    st.session_state.username = cookie["username"]
    st.session_state.role = cookie["role"]
