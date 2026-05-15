"""DeepCampus — modern Streamlit frontend.

Talks to the FastAPI backend over HTTP. Visual + UX upgrades:
- Amber-on-dark theme with system-aware light override.
- Cookie-persisted session (no re-login on refresh).
- Welcome screen with example prompts.
- Avatar chat bubbles with timestamps.
- Source cards (PDF + page) instead of plain text.
- Sidebar live status (model, k, last ingest).
- Sliders with explanatory tooltips.
- Enter-to-submit login form + auto-scroll on new message.
"""
from __future__ import annotations

import os
import time

import streamlit as st

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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

st.set_page_config(
    page_title="DeepCampus — Hybrid RAG Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_styles()

# ────────────────────────────────────────────────────────────────────────────
# Session state defaults
# ────────────────────────────────────────────────────────────────────────────
for k, v in {
    "token": None,
    "username": None,
    "role": None,
    "messages": [],
    "temperature": 0.3,
    "k_value": 20,
    "rerank_n": 8,
    "dense_weight": 0.6,
    "selected_model": None,
    "pullable_models": [],
    "available_models": [],
    "last_query_at": None,
    "pending_query": None,
}.items():
    st.session_state.setdefault(k, v)

ses.hydrate_from_cookie()


# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _logged_in() -> bool:
    return bool(st.session_state.token)


def _refresh_models_and_history() -> None:
    """Pull chat history and model list from the backend into session state."""
    try:
        st.session_state.messages = api.get_history(st.session_state.token)
    except api.ApiError:
        st.session_state.messages = []
    try:
        info = api.list_models(st.session_state.token)
        st.session_state.available_models = info.get("available", [])
        st.session_state.pullable_models = info.get("pullable", [])
        if not st.session_state.selected_model:
            st.session_state.selected_model = info.get("default")
    except api.ApiError:
        pass


def _handle_auth(form_name: str, username: str, password: str) -> None:
    try:
        action = api.register if form_name == "Register" else api.login
        data = action(username, password)
        st.session_state.token = data["access_token"]
        st.session_state.role = data["role"]
        st.session_state.username = data["username"]
        ses.save_cookie(data["access_token"], data["username"], data["role"])
        _refresh_models_and_history()
        st.rerun()
    except api.ApiError as exc:
        st.error(str(exc))


def _logout() -> None:
    ses.clear_cookie()
    for key in ("token", "username", "role", "messages", "available_models",
                "selected_model"):
        st.session_state[key] = [] if key in ("messages", "available_models") else None
    st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# Auth screen
# ────────────────────────────────────────────────────────────────────────────
if not _logged_in():
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        hero(
            "DeepCampus",
            "Hybrid Retrieval-Augmented Generation over your academic PDFs. "
            "Sign in to start asking questions.",
        )

        # Mode toggle lives OUTSIDE the form. st.form inside st.tabs in
        # Streamlit 1.57 sometimes swallows the Enter keystroke; placing
        # the form at top level fixes Enter-to-submit reliably.
        mode = st.radio(
            "Mode",
            options=["Sign in", "Create account"],
            horizontal=True,
            label_visibility="collapsed",
            key="auth_mode",
        )

        with st.form("auth_form", clear_on_submit=False):
            u = st.text_input("Username", key="auth_user")
            p = st.text_input("Password", type="password", key="auth_pw")
            label = "Sign in" if mode == "Sign in" else "Create account"
            if st.form_submit_button(label, type="primary", use_container_width=True):
                _handle_auth("Login" if mode == "Sign in" else "Register", u, p)

        if mode == "Sign in":
            st.caption("Default admin: `admin / admin123` — change in `.env` before production.")
        bind_login_enter()
    st.stop()


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### DeepCampus")
    st.markdown(
        status_pill(f"{st.session_state.username} · {st.session_state.role}"),
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Logout", use_container_width=True):
            _logout()
    with col_b:
        if st.button("Clear chat", use_container_width=True):
            try:
                api.clear_history(st.session_state.token)
                st.session_state.messages = []
                st.rerun()
            except api.ApiError as exc:
                st.error(str(exc))

    st.divider()
    sidebar_section_title("Model")
    if not st.session_state.available_models and not st.session_state.get("pullable_models"):
        _refresh_models_and_history()

    pulled = st.session_state.available_models or []
    pullable = st.session_state.get("pullable_models") or []

    if pulled:
        current = st.session_state.selected_model or pulled[0]
        idx = pulled.index(current) if current in pulled else 0
        st.session_state.selected_model = st.selectbox(
            "Active LLM",
            pulled,
            index=idx,
            help=(
                "Models actually loaded in Ollama. Switching may take 2–3 s "
                "while Ollama evicts the previous one."
            ),
        )
    else:
        st.warning("No LLM is pulled yet. Pick one below and click Download.")

    if pullable and st.session_state.role == "instructor":
        with st.expander("Download more models", expanded=not pulled):
            to_pull = st.selectbox(
                "Model to download",
                pullable,
                key="pull_choice",
                help="Models configured in AVAILABLE_LLMS but not yet in Ollama.",
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
                        status.update(
                            label=f"Downloaded {to_pull}",
                            state="complete",
                            expanded=False,
                        )
                        _refresh_models_and_history()
                        st.rerun()
                except api.ApiError as exc:
                    status.update(label=f"Failed: {exc}", state="error")

    sidebar_section_title("Retrieval")
    st.session_state.temperature = st.slider(
        "Temperature",
        0.0,
        1.0,
        st.session_state.temperature,
        0.05,
        help=(
            "How creative the LLM is. Low values (0.0–0.3) keep the answer close "
            "to the retrieved context — recommended for academic Q&A. Higher "
            "values introduce more variation but risk hallucination."
        ),
    )
    st.session_state.k_value = st.slider(
        "Top-k retrieval",
        4,
        40,
        st.session_state.k_value,
        1,
        help=(
            "How many candidate chunks the hybrid search fetches *before* "
            "fusion. Larger k → broader recall but more LLM context. "
            "After RRF fusion the top 8 chunks are sent to the model."
        ),
    )
    st.session_state.dense_weight = st.slider(
        "Dense ↔ Sparse weight",
        0.0,
        1.0,
        st.session_state.dense_weight,
        0.05,
        help=(
            "Tilt of the hybrid fusion. 1.0 = pure semantic (BGE-M3 dense); "
            "0.0 = pure lexical (sparse, BM25-like). Default 0.6 balances "
            "concept matching with exact-keyword fidelity."
        ),
    )

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

        uploaded = st.file_uploader(
            "Upload PDF",
            type="pdf",
            help="Drops the file into docs/. Run 'Process' afterwards to embed it.",
        )
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
                    api.run_ingest(st.session_state.token)
                    st.success("Vector store updated.")
                    st.rerun()
                except api.ApiError as exc:
                    st.error(str(exc))

    st.divider()
    st.caption(
        "DeepCampus v2 · BGE-M3 hybrid + Qdrant · Local Ollama LLMs · "
        "All inference stays on your machine."
    )


# ────────────────────────────────────────────────────────────────────────────
# Main pane
# ────────────────────────────────────────────────────────────────────────────
def _render_messages() -> None:
    for msg in st.session_state.messages:
        role = msg["role"]
        avatar = "🧑‍🎓" if role == "user" else "🎓"
        with st.chat_message(role, avatar=avatar):
            chat_bubble_meta(role, msg.get("ts", ""))
            st.markdown(msg.get("content", ""))
            if role == "assistant":
                if msg.get("sources"):
                    with st.expander("View sources", expanded=False):
                        source_cards(msg["sources"])
                for img_path in (msg.get("images") or []):
                    if img_path:
                        st.image(api.fetch_image_url(img_path), use_container_width=True)


SUGGESTIONS = [
    "Summarize the main contributions of the indexed papers.",
    "Compare the proposed methods across the documents.",
    "Which papers discuss evaluation metrics, and how do they differ?",
    "Explain a figure or chart from the most relevant document.",
]


if not st.session_state.messages:
    picked = welcome_screen(st.session_state.username, SUGGESTIONS)
    if picked:
        st.session_state.pending_query = picked
else:
    hero(
        "DeepCampus",
        "Hybrid retrieval (BGE-M3 dense + sparse, RRF-fused) with multimodal "
        "PDF context. Ask anything — every claim is grounded.",
    )
    _render_messages()


# ────────────────────────────────────────────────────────────────────────────
# Chat input + streaming response
# ────────────────────────────────────────────────────────────────────────────
typed = st.chat_input("Ask a question about the documents...")
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

        try:
            for event in api.stream_query(
                st.session_state.token,
                user_query,
                model=st.session_state.selected_model,
                temperature=st.session_state.temperature,
                top_k=st.session_state.k_value,
            ):
                etype = event.get("event")
                if etype == "sources":
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
            placeholder.markdown(final_answer)
            ttft_str = f"{ttft:.2f}s" if ttft else "—"
            status_box.update(
                label=f"Done · {elapsed:.1f}s total · first token in {ttft_str}",
                state="complete",
                expanded=False,
            )

            if sources_buffer:
                with st.expander("View sources", expanded=False):
                    source_cards(sources_buffer)
            for img_path in images_buffer:
                if img_path:
                    st.image(api.fetch_image_url(img_path), use_container_width=True)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "sources": sources_buffer,
                    "images": images_buffer,
                    "ts": timestamp_now(),
                }
            )
        except api.ApiError as exc:
            status_box.update(label="Error", state="error")
            st.error(str(exc))

    scroll_to_bottom()
    autofocus_chat_input()
