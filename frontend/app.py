from __future__ import annotations

# --- 1. STANDART KÜTÜPHANE İMPORTLARI ---
import json
import os
import re
import time

# --- 2. ÜÇÜNCÜ PARTİ KÜTÜPHANELER ---
import streamlit as st

# Tarayıcı tabanlı speech-to-text (Web Speech API wrapper). Eksikse
# voice özellikleri sessizce devre dışı kalır, uygulama yine açılır.
try:
    from streamlit_mic_recorder import speech_to_text  # type: ignore
    _STT_AVAILABLE = True
except Exception:  # noqa: BLE001
    speech_to_text = None  # type: ignore
    _STT_AVAILABLE = False

# --- 3. LOKAL PROJE İMPORTLARI ---
from frontend import api_client as api
from frontend import session as ses
from frontend.components import (
    chat_bubble_meta,
    hero,
    sidebar_section_title,
    source_cards,
    status_pill,
    timestamp_now,
    welcome_screen,
)
from frontend.styles import (
    autofocus_chat_input,
    bind_login_enter,
    inject_styles,
    scroll_to_bottom,
)

# --- 4. STREAMLIT SAYFA YAPILANDIRMASI ---
st.set_page_config(
    page_title="DeepCampus — Hybrid RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
inject_styles(st.session_state["theme"])

# Browser-facing Streamlit URL — embedded in the OAuth redirect_uri so
# Keycloak knows where to send the user back. Must match a redirect
# URI the realm accepts (dev realm uses ["*"]).
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:8501").rstrip("/")

# --- 5. GLOBAL DEĞİŞKENLER VE SABİTLER ---
# Backend already strips [GÖRSEL: ...] from the saved final_answer, but
# (a) live streamed tokens arrive raw and (b) the LLM occasionally
# improvises formats like "[IMAGE - Page 43]" / "[Figure 3]" outside
# the spec. We aggressively scrub anything that smells like an image
# citation; the actual image rendering happens via a separate channel
# (images_buffer / msg.images) so the text never needs the citation.
_IMAGE_PATTERN = re.compile(
    r"\[[^\]\n]*?(?:GÖRSEL|GORSEL|IMAGE|RESIM|RESİM|FIGÜR|FIGURE|SAYFA|PAGE)\b[^\]\n]*?\]",
    re.IGNORECASE,
)
_BARE_IMAGE_LINE = re.compile(
    r"(?im)^\s*(?:görsel|gorsel|image|resim|resi̇m|figür|figure|sayfa|page)\s*[:\-]\s*\S+.*$"
)


def _stt_lang_code(label: str) -> str:
    return "tr" if label == "Turkish" else "en"


def _tts_lang_code(label: str) -> str:
    return "tr-TR" if label == "Turkish" else "en-US"


def _speak_button(text: str, lang_tag: str, key: str) -> None:
    """Render a 'Read aloud' button next to an answer — invokes browser TTS."""
    if not text:
        return
    safe_text = json.dumps(text)
    safe_lang = json.dumps(lang_tag)
    btn_id = f"dc-tts-{key}"
    st.components.v1.html(
        f"""
        <button id="{btn_id}"
                style="background:transparent;border:1px solid #6b7280;
                       color:#d4d4d8;padding:4px 10px;border-radius:6px;
                       cursor:pointer;font-size:12px;margin-top:6px;">
            🔊 Read aloud
        </button>
        <script>
        (function() {{
            const btn = document.getElementById("{btn_id}");
            if (!btn) return;
            btn.addEventListener("click", () => {{
                try {{
                    const synth = window.parent.speechSynthesis;
                    synth.cancel();
                    const u = new SpeechSynthesisUtterance({safe_text});
                    u.lang = {safe_lang};
                    synth.speak(u);
                }} catch (e) {{
                    console.error("TTS failed:", e);
                }}
            }});
        }})();
        </script>
        """,
        height=44,
    )

# --- 6. YARDIMCI FONKSİYONLAR ---
_IMAGE_MAX_WIDTH = 420


def _render_image_blob(blob: bytes) -> None:
    st.image(blob, width=_IMAGE_MAX_WIDTH)


@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def _cached_image_bytes(token: str, img_path: str) -> bytes | None:
    return api.fetch_image_bytes(token, img_path)


def _toggle_theme() -> None:
    st.session_state["theme"] = (
        "light" if st.session_state.get("theme", "dark") == "dark" else "dark"
    )


def _render_content_with_images(text: str) -> None:
    clean_text = _IMAGE_PATTERN.sub("", text)
    clean_text = _BARE_IMAGE_LINE.sub("", clean_text).strip()
    if clean_text:
        st.markdown(clean_text)


def _render_messages() -> None:
    lang_tag = _tts_lang_code(st.session_state.get("voice_lang", "Turkish"))
    for idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        avatar = "🧑‍🎓" if role == "user" else "🎓"
        with st.chat_message(role, avatar=avatar):
            chat_bubble_meta(role, msg.get("ts", ""))

            content = msg.get("content", "")
            _render_content_with_images(content)

            if role == "assistant":
                for img_path in (msg.get("images") or []):
                    if not img_path:
                        continue
                    blob = _cached_image_bytes(st.session_state.token, img_path)
                    if blob:
                        _render_image_blob(blob)
                if msg.get("sources"):
                    with st.expander("View sources", expanded=False):
                        source_cards(msg["sources"])
                _speak_button(content, lang_tag, key=f"hist-{idx}")


# ────────────────────────────────────────────────────────────────────────────
# Session state defaults & F5 HYDRATION
# ────────────────────────────────────────────────────────────────────────────
for k, v in {
    "token": None,
    "id_token": None,
    "username": None,
    "role": None,
    "messages": [],
    "temperature": 0.3,
    "k_value": 20,
    "rerank_n": 8,
    "dense_weight": 0.6,
    "selected_model": None,
    "pullable_models": [],
    "models_initialized": False,
    "available_models": [],
    "last_query_at": None,
    "pending_query": None,
    "voice_lang": "Turkish",
    "active_session_id": None,
    "sessions": [],
    "editing_session_id": None,
}.items():
    st.session_state.setdefault(k, v)

ses.hydrate_from_cookie()


# ────────────────────────────────────────────────────────────────────────────
# OAuth Authorization Code callback — runs before any auth check.
# ────────────────────────────────────────────────────────────────────────────
def _handle_oauth_callback() -> None:
    """If the URL carries `?code=...`, trade it for a token and sign in.

    Only fires when no token is already loaded — otherwise a stale code
    in the URL (e.g. from a back-button) could overwrite a fresh
    session. The query param is cleared either way so the URL stays
    clean.
    """
    if st.session_state.get("token"):
        # Clear any leftover code param so it can't be replayed.
        if "code" in st.query_params:
            del st.query_params["code"]
        return

    code = st.query_params.get("code")
    if not code:
        return

    del st.query_params["code"]
    if "state" in st.query_params:
        del st.query_params["state"]
    if "session_state" in st.query_params:
        del st.query_params["session_state"]

    with st.spinner("Completing Keycloak sign-in..."):
        try:
            data = api.exchange_code(code, FRONTEND_URL)
        except api.ApiError as exc:
            st.error(f"Keycloak sign-in failed: {exc}")
            return
    _post_auth_success(data)


# Eski URL fallback'inden artakalmış paramları temizle (token/user/role
# eskiden URL'e yazılıyordu; artık localStorage kullanıyoruz).
if any(k in st.query_params for k in ("token", "user", "role")):
    for k in ("token", "user", "role"):
        if k in st.query_params:
            del st.query_params[k]


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _logged_in() -> bool:
    return bool(st.session_state.token)


def _refresh_sessions() -> None:
    """Pull the user's session list; sync active_session_id if it's gone stale."""
    try:
        sessions = api.list_sessions(st.session_state.token)
    except api.ApiError:
        sessions = []
    st.session_state.sessions = sessions

    valid_ids = {s["session_id"] for s in sessions}
    if st.session_state.active_session_id not in valid_ids:
        # Fall back to General Chat (always first per backend ordering).
        default_session = next((s for s in sessions if s.get("is_default")), None)
        st.session_state.active_session_id = (
            default_session["session_id"] if default_session else None
        )
        ses.update_active_session(st.session_state.active_session_id)


def _refresh_history() -> None:
    """Load the active session's messages into st.session_state.messages."""
    if not st.session_state.token:
        return
    try:
        messages, resolved = api.get_history(
            st.session_state.token,
            st.session_state.active_session_id,
        )
    except api.ApiError:
        messages, resolved = [], st.session_state.active_session_id
    st.session_state.messages = messages
    # Backend may have resolved a stale session_id back to General Chat;
    # mirror that so the UI doesn't keep pointing at a missing thread.
    if resolved and resolved != st.session_state.active_session_id:
        st.session_state.active_session_id = resolved
        ses.update_active_session(resolved)


def _refresh_models() -> None:
    try:
        info = api.list_models(st.session_state.token)
        st.session_state.available_models = info.get("available", [])
        st.session_state.pullable_models = info.get("pullable", [])
        if not st.session_state.selected_model:
            st.session_state.selected_model = info.get("default")
    except api.ApiError:
        pass


def _switch_session(session_id: str) -> None:
    """Make `session_id` the active topic; reload its history."""
    st.session_state.active_session_id = session_id
    ses.update_active_session(session_id)
    _refresh_history()


def _handle_login(username: str, password: str) -> None:
    """Password-grant login via the backend's Keycloak proxy."""
    try:
        data = api.login(username, password)
        _post_auth_success(data)
    except api.ApiError as exc:
        st.error(str(exc))


def _handle_register(
    username: str,
    password: str,
    email: str,
    first_name: str,
    last_name: str,
) -> None:
    try:
        data = api.register(username, password, email, first_name, last_name)
        _post_auth_success(data)
    except api.ApiError as exc:
        st.error(str(exc))


def _redirect_top_window(url: str) -> None:
    """Navigate the browser's top window — escapes Streamlit's iframe.

    `<meta http-equiv="refresh">` inside `st.markdown` only refreshes
    Streamlit's inner iframe, not the parent page; a Keycloak login or
    end-session URL has to land in the top window for the cookie flow
    to take effect. We inject a tiny components iframe whose script
    bumps `window.top.location.href` — same-origin (both iframes live
    on the Streamlit host) so the navigation is allowed.
    """
    safe_url = json.dumps(url)
    st.components.v1.html(
        f"""
        <script>
            (function() {{
                try {{
                    window.top.location.href = {safe_url};
                }} catch (e) {{
                    // Fallback for any cross-origin edge case.
                    window.location.href = {safe_url};
                }}
            }})();
        </script>
        """,
        height=0,
    )
    st.stop()


def _start_keycloak_login() -> None:
    """Redirect the browser to Keycloak's authorize page (OAuth Code flow)."""
    try:
        login_url = api.get_login_url(FRONTEND_URL)
    except api.ApiError as exc:
        st.error(f"Could not reach the backend: {exc}")
        return
    _redirect_top_window(login_url)


def _post_auth_success(data: dict) -> None:
    st.session_state.token = data["access_token"]
    st.session_state.role = data["role"]
    st.session_state.username = data["username"]
    st.session_state.id_token = data.get("id_token")

    # Hydrate sessions + history immediately so the post-rerun render
    # already shows the right thread.
    _refresh_sessions()
    _refresh_history()
    _refresh_models()

    ses.save_cookie(
        token=data["access_token"],
        username=data["username"],
        role=data["role"],
        id_token=data.get("id_token"),
        active_session_id=st.session_state.active_session_id,
    )

    # Give the localStorage iframe a moment to flush before rerun, so
    # an immediate F5 finds the freshly-written blob.
    time.sleep(0.4)
    st.rerun()


def _logout() -> None:
    """Local logout + Keycloak end-session redirect.

    Order matters:
    1. Clear localStorage so the next page load can't auto-restore.
    2. Wipe auth-related session_state keys *selectively* — a wholesale
       `st.session_state.clear()` also drops Streamlit-internal widget
       state and the theme, which can leave the page in a half-broken
       state for the brief instant before the redirect lands.
    3. Sleep ~0.3 s so the localStorage iframe finishes its delete
       postMessage before the navigation kicks in. Without this nudge,
       Keycloak occasionally redirects us back to Streamlit so fast
       that the cookie is still around and `hydrate_from_cookie()`
       signs us right back in.
    4. Force a top-window navigation to the Keycloak end-session URL
       (see _redirect_top_window — meta-refresh inside Streamlit's
       inner iframe doesn't escape to the parent).
    """
    id_token = st.session_state.get("id_token")
    try:
        logout_url = api.get_logout_url(FRONTEND_URL, id_token)
    except api.ApiError:
        # If we can't reach the backend, at least bounce the user back
        # to the login screen locally.
        logout_url = FRONTEND_URL

    ses.clear_cookie()

    for k in (
        "token", "id_token", "username", "role", "messages",
        "active_session_id", "sessions", "editing_session_id",
        "models_initialized", "available_models", "pullable_models",
        "selected_model",
    ):
        st.session_state.pop(k, None)
    st.query_params.clear()

    time.sleep(0.3)
    _redirect_top_window(logout_url)


# ────────────────────────────────────────────────────────────────────────────
# Auth screen (login / register)
# ────────────────────────────────────────────────────────────────────────────
_handle_oauth_callback()

if not _logged_in():
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        hero(
            "DeepCampus",
            "Hybrid Retrieval-Augmented Generation over your academic PDFs. "
            "Sign in via Keycloak to start asking questions.",
        )

        # Primary path is browser-redirected OAuth Code flow (password
        # never touches Streamlit). The form-based password-grant path
        # is kept under a fallback expander for headless / API-style
        # access and for environments where the Keycloak host isn't
        # browser-reachable.
        st.button(
            "🔐 Sign in with Keycloak",
            type="primary",
            use_container_width=True,
            on_click=_start_keycloak_login,
            help="Opens the Keycloak login page in this tab.",
        )
        st.caption(
            "Recommended. Your password is entered on Keycloak's own login "
            "page — it never flows through DeepCampus."
        )

        with st.expander("Other sign-in options", expanded=False):
            mode = st.radio(
                "Mode",
                options=["Email / password", "Create account"],
                horizontal=True,
                label_visibility="collapsed",
                key="auth_mode",
            )

            if mode == "Email / password":
                with st.form("login_form", clear_on_submit=False):
                    u = st.text_input("Username", key="login_user")
                    p = st.text_input("Password", type="password", key="login_pw")
                    if st.form_submit_button(
                        "Sign in", type="primary", use_container_width=True
                    ):
                        _handle_login(u, p)
                st.caption(
                    "Default admin in the seeded realm: `admin / admin123` — "
                    "change it from the Keycloak admin console for production."
                )
            else:
                with st.form("register_form", clear_on_submit=False):
                    ru = st.text_input("Username *", key="reg_user")
                    rmail = st.text_input("Email *", key="reg_email")
                    cols = st.columns(2)
                    rfirst = cols[0].text_input("First name (optional)", key="reg_first")
                    rlast = cols[1].text_input("Last name (optional)", key="reg_last")
                    rp = st.text_input("Password *", type="password", key="reg_pw")
                    rp2 = st.text_input(
                        "Password (confirm) *", type="password", key="reg_pw2"
                    )
                    if st.form_submit_button(
                        "Create account", type="primary", use_container_width=True
                    ):
                        if not ru or not rmail or not rp:
                            st.error("Username, email and password are required.")
                        elif rp != rp2:
                            st.error("Passwords do not match.")
                        else:
                            _handle_register(ru, rp, rmail, rfirst, rlast)

        bind_login_enter()

    if not _logged_in():
        st.stop()


# ────────────────────────────────────────────────────────────────────────────
# Post-login bootstrap: sessions, history and models
# ────────────────────────────────────────────────────────────────────────────
if not st.session_state.get("models_initialized"):
    _refresh_sessions()
    _refresh_history()
    _refresh_models()
    st.session_state["models_initialized"] = True


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
def _handle_create_topic() -> None:
    title = (st.session_state.get("new_topic_title") or "").strip()
    if not title:
        st.toast("Topic name is required.", icon="⚠️")
        return
    try:
        created = api.create_session(st.session_state.token, title)
    except api.ApiError as exc:
        st.error(str(exc))
        return
    st.session_state["new_topic_title"] = ""
    _refresh_sessions()
    _switch_session(created["session_id"])


def _handle_rename_topic(session_id: str) -> None:
    new_title = (st.session_state.get(f"edit_input_{session_id}") or "").strip()
    if not new_title:
        st.toast("New title required.", icon="⚠️")
        return
    try:
        api.rename_session(st.session_state.token, session_id, new_title)
    except api.ApiError as exc:
        st.error(str(exc))
        return
    st.session_state.editing_session_id = None
    _refresh_sessions()


def _handle_delete_topic(session_id: str) -> None:
    try:
        api.delete_session(st.session_state.token, session_id)
    except api.ApiError as exc:
        st.error(str(exc))
        return
    if st.session_state.active_session_id == session_id:
        st.session_state.active_session_id = None
    _refresh_sessions()
    _refresh_history()


with st.sidebar:
    # Theme toggle — top of the sidebar, small and unobtrusive.
    theme_label = (
        "☀️ Light theme" if st.session_state.get("theme", "dark") == "dark"
        else "🌙 Dark theme"
    )
    st.button(
        theme_label,
        key="theme_toggle",
        on_click=_toggle_theme,
        use_container_width=True,
        help="Switch between light and dark theme.",
    )

    st.markdown("### DeepCampus")
    st.markdown(
        status_pill(f"{st.session_state.username} · {st.session_state.role}"),
        unsafe_allow_html=True,
    )

    if st.button("Logout", use_container_width=True):
        _logout()

    st.divider()
    sidebar_section_title("📚 My Topics")

    with st.expander("➕ Create new topic", expanded=False):
        st.text_input(
            "Topic name",
            key="new_topic_title",
            placeholder="e.g. Computer Networks – Chapter 4",
        )
        st.button(
            "Create",
            type="primary",
            on_click=_handle_create_topic,
            use_container_width=True,
            key="create_topic_btn",
        )

    sessions = st.session_state.sessions or []
    for session in sessions:
        sid = session["session_id"]
        is_active = (sid == st.session_state.active_session_id)
        is_default = session.get("is_default", False)
        icon = "🟢" if is_active else "💬"

        # Inline rename mode (only for non-default topics).
        if st.session_state.editing_session_id == sid:
            col1, col2, col3 = st.columns([5, 1, 1])
            with col1:
                st.text_input(
                    "Rename",
                    value=session["title"],
                    key=f"edit_input_{sid}",
                    label_visibility="collapsed",
                )
            with col2:
                if st.button("✅", key=f"save_{sid}", help="Save"):
                    _handle_rename_topic(sid)
                    st.rerun()
            with col3:
                if st.button("❌", key=f"cancel_{sid}", help="Cancel"):
                    st.session_state.editing_session_id = None
                    st.rerun()
            continue

        if is_default:
            # General Chat — no rename / delete buttons.
            if st.button(
                f"{icon} {session['title']}",
                key=f"sel_{sid}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                _switch_session(sid)
                st.rerun()
        else:
            col1, col2, col3 = st.columns([5, 1, 1])
            with col1:
                if st.button(
                    f"{icon} {session['title']}",
                    key=f"sel_{sid}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    _switch_session(sid)
                    st.rerun()
            with col2:
                if st.button("✏️", key=f"edit_{sid}", help="Rename"):
                    st.session_state.editing_session_id = sid
                    st.rerun()
            with col3:
                if st.button("🗑️", key=f"del_{sid}", help="Delete"):
                    _handle_delete_topic(sid)
                    st.rerun()

    st.divider()
    if st.button("Clear current topic", use_container_width=True, key="clear_chat_btn"):
        try:
            api.clear_history(st.session_state.token, st.session_state.active_session_id)
            st.session_state.messages = []
            st.rerun()
        except api.ApiError as exc:
            st.error(str(exc))

    st.divider()
    sidebar_section_title("Voice")
    st.session_state.voice_lang = st.radio(
        "Voice language",
        options=["Turkish", "English"],
        index=0 if st.session_state.get("voice_lang", "Turkish") == "Turkish" else 1,
        horizontal=True,
        key="voice_lang_radio",
        help="Language used for voice input and read-aloud.",
    )

    st.divider()
    sidebar_section_title("Model")
    pulled = st.session_state.available_models or []
    pullable = st.session_state.get("pullable_models") or []

    if not pulled and not pullable:
        if st.button("Refresh model list", key="refresh_models", use_container_width=True):
            _refresh_models()
            st.rerun()

    if pulled:
        current = st.session_state.selected_model or pulled[0]
        idx = pulled.index(current) if current in pulled else 0
        st.session_state.selected_model = st.selectbox(
            "Active LLM",
            pulled,
            index=idx,
            help="Models actually loaded in Ollama. Switching may take 2–3 s.",
        )
    else:
        st.warning("No LLM is pulled yet. Pick one below and click Download.")

    if pullable and st.session_state.role == "instructor":
        with st.expander("Download more models", expanded=not pulled):
            to_pull = st.selectbox(
                "Model to download",
                pullable,
                key="pull_choice",
            )
            if st.button("Download", key="pull_btn", use_container_width=True):
                status = st.status(f"Downloading {to_pull}…", expanded=True)
                progress_text = st.empty()
                last_status = ""
                try:
                    for evt in api.pull_model(st.session_state.token, to_pull):
                        msg = evt.get("status") or evt.get("error") or ""
                        if msg and msg != last_status:
                            progress_text.markdown(f"`{msg}`")
                            last_status = msg
                        if "error" in evt:
                            status.update(label=f"Failed: {evt['error']}", state="error")
                            break
                    else:
                        status.update(label=f"Downloaded {to_pull}", state="complete", expanded=False)
                        _refresh_models()
                        st.rerun()
                except api.ApiError as exc:
                    status.update(label=f"Failed: {exc}", state="error")

    sidebar_section_title("Retrieval")
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
    st.session_state.k_value = st.slider("Top-k retrieval", 4, 40, st.session_state.k_value, 1)
    st.session_state.dense_weight = st.slider("Dense ↔ Sparse weight", 0.0, 1.0, st.session_state.dense_weight, 0.05)

    if st.session_state.role == "instructor":
        st.divider()
        sidebar_section_title("Knowledge base")
        try:
            status_info = api.ingest_status(st.session_state.token)
            st.markdown(
                status_pill(
                    f"{len(status_info['indexed_sources'])} indexed · "
                    f"{status_info['documents_in_state']} tracked"
                ),
                unsafe_allow_html=True,
            )
            with st.expander("Files in docs/", expanded=False):
                for f in status_info["files"]:
                    st.text(f"• {f}")
        except api.ApiError as exc:
            st.warning(str(exc))

        with st.expander("Image summaries (Moondream)", expanded=False):
            try:
                summaries = api.get_image_summaries(st.session_state.token)
            except api.ApiError as exc:
                summaries = []
                st.warning(str(exc))
            if not summaries:
                st.caption("No ingestion has run yet, or no images were extracted.")
            else:
                st.caption(f"{len(summaries)} images · stored in `data/image_summaries.json`")
                for item in summaries:
                    page = (item.get("page") or 0) + 1
                    st.markdown(
                        f"**{item.get('source', '?')}** · page {page}"
                    )
                    st.text(item.get("image_path", ""))
                    st.markdown(
                        f"<div style='color:var(--text-muted);font-size:0.85rem;"
                        f"padding:6px 8px;background:var(--surface-2);"
                        f"border:1px solid var(--border);border-radius:6px;"
                        f"margin:4px 0 12px 0'>{item.get('summary', '')}</div>",
                        unsafe_allow_html=True,
                    )

        uploaded = st.file_uploader("Upload PDF", type="pdf")
        if uploaded is not None:
            try:
                result = api.upload_pdf(
                    st.session_state.token,
                    uploaded.name,
                    uploaded.getbuffer().tobytes(),
                )
                st.success(f"Saved: {result['saved_as']}")
            except api.ApiError as exc:
                st.error(str(exc))

        if st.button("Process & update database", type="primary", use_container_width=True):
            with st.spinner("Running ingestion pipeline..."):
                try:
                    result = api.run_ingest(st.session_state.token)
                except api.ApiError as exc:
                    st.error(str(exc))
                    result = None
            if result is not None:
                summary = (
                    f"Processed: **{result.get('processed', 0)}** · "
                    f"Duplicates: **{result.get('duplicates', 0)}** · "
                    f"Errors: **{result.get('errors', 0)}**"
                )
                if result.get('chunks'):
                    summary += f" · Chunks: **{result['chunks']}**"
                st.success(summary)
                dup_items = [d for d in (result.get("details") or []) if d.get("status") == "duplicate"]
                if dup_items:
                    with st.expander("Duplicate files skipped", expanded=True):
                        for d in dup_items:
                            st.info(f"**{d.get('file', '?')}** — already indexed.")
                err_items = [d for d in (result.get("details") or []) if d.get("status") == "error"]
                if err_items:
                    with st.expander("Errors", expanded=True):
                        for d in err_items:
                            st.error(f"**{d.get('file', '?')}** — {d.get('reason', 'error')}")

        with st.expander("Danger zone", expanded=False):
            confirm = st.checkbox("I understand this wipes the vector store.", key="reset_confirm")
            if st.button("Reset knowledge base", key="reset_kb_btn", use_container_width=True, disabled=not confirm):
                try:
                    api.reset_knowledge_base(st.session_state.token)
                    st.success("Knowledge base cleared.")
                    st.rerun()
                except api.ApiError as exc:
                    st.error(str(exc))

    st.divider()
    st.caption("DeepCampus v2 · BGE-M3 hybrid + Qdrant · Local Ollama LLMs")


# ────────────────────────────────────────────────────────────────────────────
# Main screen
# ────────────────────────────────────────────────────────────────────────────
SUGGESTIONS = [
    "Summarize the main contributions of the indexed papers.",
    "Compare the proposed methods across the documents.",
    "Which papers discuss evaluation metrics, and how do they differ?",
    "Explain a figure or chart from the most relevant document.",
]

active_session = next(
    (s for s in (st.session_state.sessions or [])
     if s["session_id"] == st.session_state.active_session_id),
    None,
)
active_title = active_session["title"] if active_session else "General Chat"

if not st.session_state.messages:
    picked = welcome_screen(st.session_state.username, SUGGESTIONS)
    if picked:
        st.session_state.pending_query = picked
else:
    hero(
        f"DeepCampus · {active_title}",
        "Hybrid retrieval (BGE-M3 dense + sparse, RRF-fused) with multimodal "
        "PDF context. Ask anything — every claim is grounded.",
    )
    _render_messages()

# Microphone
if _STT_AVAILABLE:
    voice_col, _spacer = st.columns([1, 4])
    with voice_col:
        stt_lang = _stt_lang_code(st.session_state.get("voice_lang", "Turkish"))
        spoken = speech_to_text(
            language=stt_lang,
            start_prompt="🎤 Speak",
            stop_prompt="⏹️ Stop",
            just_once=True,
            use_container_width=True,
            key="stt_widget",
        )
        if spoken:
            st.session_state.pending_query = spoken

# Message submission
typed = st.chat_input(f"Ask a question in '{active_title}'...")
user_query = st.session_state.pending_query or typed
st.session_state.pending_query = None

if user_query:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
            "sources": "",
            "images": [],
            "ts": timestamp_now(),
        }
    )
    with st.chat_message("user", avatar="🧑‍🎓"):
        chat_bubble_meta("user", timestamp_now())
        st.markdown(user_query)

    with st.chat_message("assistant", avatar="🎓"):
        chat_bubble_meta("assistant", timestamp_now())
        status_box = st.status("Searching knowledge base...", expanded=True)
        placeholder = st.empty()
        sources_buffer = ""
        images_buffer: list[str] = []
        tokens: list[str] = []
        start = time.perf_counter()
        ttft: float | None = None
        resolved_session_id: str | None = None

        try:
            for event in api.stream_query(
                st.session_state.token,
                user_query,
                model=st.session_state.selected_model,
                temperature=st.session_state.temperature,
                top_k=st.session_state.k_value,
                session_id=st.session_state.active_session_id,
            ):
                etype = event.get("event")
                if etype == "session":
                    resolved_session_id = event.get("data")
                elif etype == "sources":
                    sources_buffer += (event.get("data") or "")
                    status_box.update(label="Generating answer...", state="running")
                elif etype == "token":
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    tokens.append(event.get("data", ""))
                    placeholder.markdown("".join(tokens) + "▌")
                elif etype == "images":
                    images_buffer = event.get("data") or []
                elif etype == "error":
                    raise api.ApiError(event.get("data", "Stream error"))
                elif etype == "done":
                    break

            elapsed = time.perf_counter() - start
            final_answer = "".join(tokens).strip()
            placeholder.empty()
            _render_content_with_images(final_answer)
            ttft_str = f"{ttft:.2f}s" if ttft else "—"
            status_box.update(
                label=f"Done · {elapsed:.1f}s total · first token in {ttft_str}",
                state="complete",
                expanded=False,
            )

            for img_path in images_buffer:
                if not img_path:
                    continue
                blob = _cached_image_bytes(st.session_state.token, img_path)
                if blob:
                    _render_image_blob(blob)

            if sources_buffer:
                with st.expander("View sources", expanded=False):
                    source_cards(sources_buffer)

            _speak_button(
                final_answer,
                _tts_lang_code(st.session_state.get("voice_lang", "Turkish")),
                key="live-answer",
            )

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "sources": sources_buffer,
                    "images": images_buffer,
                    "ts": timestamp_now(),
                }
            )

            # Backend may have promoted the request to General Chat
            # (stale session_id from cookie). Mirror the resolution so
            # the next message goes to the same place.
            if resolved_session_id and resolved_session_id != st.session_state.active_session_id:
                st.session_state.active_session_id = resolved_session_id
                ses.update_active_session(resolved_session_id)
                _refresh_sessions()
        except api.ApiError as exc:
            status_box.update(label="Error", state="error")
            st.error(str(exc))

    scroll_to_bottom()
    autofocus_chat_input()
