"""Cookie-backed session persistence.

Streamlit's session state is wiped on full page refresh. We store the JWT in
an HTTP cookie (via extra-streamlit-components) so a refresh keeps the user
logged in until the token expires.

Two quirks worked around here:
- The custom-component iframe needs at least one rerun after page load to
  push browser cookies into Python. The first script run reports "no
  cookie" even when one exists. hydrate_from_cookie therefore retries via
  st.rerun up to RETRY_BUDGET times before giving up.
- extra-streamlit-components serializes the cookie value with json.dumps
  but only does so for primitive scalars cleanly. Passing a dict has
  produced "[object Object]" in the past. We explicitly json.dumps on
  save and json.loads on load so the round-trip is deterministic.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

import extra_streamlit_components as stx
import streamlit as st

COOKIE_NAME = "dc_session"
COOKIE_TTL_HOURS = int(os.getenv("JWT_TTL_HOURS", "12"))
RETRY_BUDGET = 3
RETRY_DELAY_S = 0.25


def _cookie_manager() -> stx.CookieManager:
    """Return a single CookieManager instance per session."""
    if "_dc_cookie_manager" not in st.session_state:
        st.session_state["_dc_cookie_manager"] = stx.CookieManager(
            key="dc_cookie_manager"
        )
    return st.session_state["_dc_cookie_manager"]


def _decode(raw: object) -> dict | None:
    """Best-effort parser: handles json strings, plain dicts, and junk."""
    if raw is None or raw == "":
        return None
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
    if not all(k in data for k in ("token", "username", "role")):
        return None
    return data


def load_cookie() -> dict | None:
    """Read the session cookie. Returns None if missing or malformed."""
    cm = _cookie_manager()
    raw = cm.get(cookie=COOKIE_NAME)
    return _decode(raw)


def save_cookie(token: str, username: str, role: str) -> None:
    cm = _cookie_manager()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=COOKIE_TTL_HOURS)
    payload = json.dumps({"token": token, "username": username, "role": role})
    cm.set(
        COOKIE_NAME,
        payload,
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

    Called once on every page load. The first call after a page refresh
    often sees an empty cookie because the iframe hasn't pushed its
    state to Python yet — we trigger a few short reruns to give it a
    chance before falling back to the login screen.
    """
    if st.session_state.get("token"):
        return

    cookie = load_cookie()
    if cookie:
        st.session_state.token = cookie["token"]
        st.session_state.username = cookie["username"]
        st.session_state.role = cookie["role"]
        st.session_state["_cookie_attempts"] = 0
        return

    attempts = st.session_state.get("_cookie_attempts", 0)
    if attempts < RETRY_BUDGET:
        st.session_state["_cookie_attempts"] = attempts + 1
        time.sleep(RETRY_DELAY_S)
        st.rerun()
