"""Cookie-backed session persistence."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

import extra_streamlit_components as stx
import streamlit as st

COOKIE_NAME = "dc_session"
COOKIE_TTL_HOURS = int(os.getenv("JWT_TTL_HOURS", "12"))
RETRY_BUDGET = 8
RETRY_DELAY_S = 0.5


def _cookie_manager():
    if "_dc_cookie_manager" not in st.session_state:
        st.session_state["_dc_cookie_manager"] = stx.CookieManager(key="dc_cookie_manager")
    return st.session_state["_dc_cookie_manager"]


def _decode(raw):
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


def load_cookie():
    cm = _cookie_manager()
    raw = cm.get(cookie=COOKIE_NAME)
    return _decode(raw)


def save_cookie(token, username, role):
    cm = _cookie_manager()
    expires_at = datetime.now(timezone.utc) + timedelta(hours=COOKIE_TTL_HOURS)
    payload = json.dumps({"token": token, "username": username, "role": role})
    cm.set(COOKIE_NAME, payload, expires_at=expires_at, key="dc_cookie_set")


def clear_cookie():
    cm = _cookie_manager()
    try:
        cm.delete(COOKIE_NAME, key="dc_cookie_delete")
    except KeyError:
        pass


def hydrate_from_cookie():
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
