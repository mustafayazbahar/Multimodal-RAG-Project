"""DeepCampus Streamlit frontend — talks to the FastAPI backend over HTTP."""
from __future__ import annotations

import os

import streamlit as st

from frontend import api_client as api

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

st.set_page_config(page_title="DeepCampus", layout="wide", initial_sidebar_state="expanded")

# --- session state ----------------------------------------------------------
for k, v in {
    "token": None,
    "username": None,
    "role": None,
    "messages": [],
    "temperature": 0.3,
    "k_value": 20,
    "selected_model": None,
    "available_models": [],
}.items():
    st.session_state.setdefault(k, v)


def _logged_in() -> bool:
    return bool(st.session_state.token)


# --- header -----------------------------------------------------------------
col1, col2 = st.columns([1, 6])
with col1:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png",
        width=60,
    )
with col2:
    st.title("DeepCampus: Intelligent Research Assistant")
    st.caption("Hybrid RAG (BGE-M3 + Qdrant) | Local Ollama LLMs | Multi-model")

st.divider()

# --- sidebar ----------------------------------------------------------------
with st.sidebar:
    st.header("User Authentication")
    if not _logged_in():
        menu = st.radio("Select Option", ["Login", "Register"])
        with st.form("auth_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Create Account" if menu == "Register" else "Login")
        if submitted:
            try:
                action = api.register if menu == "Register" else api.login
                data = action(username, password)
                st.session_state.token = data["access_token"]
                st.session_state.role = data["role"]
                st.session_state.username = data["username"]
                st.session_state.messages = api.get_history(st.session_state.token)
                try:
                    info = api.list_models(st.session_state.token)
                    st.session_state.available_models = info.get("available", [])
                    st.session_state.selected_model = info.get("default")
                except api.ApiError:
                    pass
                st.rerun()
            except api.ApiError as exc:
                st.error(str(exc))
    else:
        st.success(f"Logged in as {st.session_state.username} ({st.session_state.role})")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Logout", use_container_width=True):
                for key in ("token", "username", "role", "messages"):
                    st.session_state[key] = None if key != "messages" else []
                st.rerun()
        with c2:
            if st.button("Clear Chat", use_container_width=True):
                try:
                    api.clear_history(st.session_state.token)
                    st.session_state.messages = []
                    st.rerun()
                except api.ApiError as exc:
                    st.error(str(exc))

        st.divider()
        st.subheader("Model")
        if st.session_state.available_models:
            current = st.session_state.selected_model or st.session_state.available_models[0]
            idx = st.session_state.available_models.index(current) if current in st.session_state.available_models else 0
            st.session_state.selected_model = st.selectbox(
                "LLM", st.session_state.available_models, index=idx
            )

        if st.session_state.role == "instructor":
            st.subheader("RAG Parameters")
            st.session_state.temperature = st.slider(
                "Creativity (Temperature)", 0.0, 1.0, st.session_state.temperature, 0.1
            )
            st.session_state.k_value = st.slider(
                "Top-k Retrieval", 1, 40, st.session_state.k_value, 1
            )

            st.divider()
            st.subheader("Knowledge Base")
            try:
                status_info = api.ingest_status(st.session_state.token)
                st.text("Files in docs/:")
                for f in status_info["files"]:
                    st.text(f"- {f}")
                st.caption(
                    f"Indexed sources: {len(status_info['indexed_sources'])} | "
                    f"State entries: {status_info['documents_in_state']}"
                )
            except api.ApiError as exc:
                st.warning(str(exc))

            uploaded = st.file_uploader("Upload PDF", type="pdf")
            if uploaded is not None:
                try:
                    result = api.upload_pdf(st.session_state.token, uploaded.name, uploaded.getbuffer().tobytes())
                    st.success(f"Saved: {result['saved_as']}")
                except api.ApiError as exc:
                    st.error(str(exc))

            if st.button("Process & Update Database", type="primary"):
                with st.spinner("Running ingestion..."):
                    try:
                        api.run_ingest(st.session_state.token)
                        st.success("Vector store updated.")
                        st.rerun()
                    except api.ApiError as exc:
                        st.error(str(exc))

# --- main chat --------------------------------------------------------------
if not _logged_in():
    st.warning("Please login to use the system. (Default admin -> admin / admin123)")
    st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("sources"):
                with st.expander("View Sources"):
                    st.info(message["sources"])
            for img_path in message.get("images", []) or []:
                if img_path:
                    st.image(api.fetch_image_url(img_path), use_container_width=True)

if user_query := st.chat_input("Ask a question about the documents..."):
    st.session_state.messages.append(
        {"role": "user", "content": user_query, "sources": "", "images": []}
    )
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        status_box = st.status("Processing...", expanded=True)
        placeholder = st.empty()
        sources_buffer = ""
        images_buffer: list[str] = []
        tokens: list[str] = []
        try:
            status_box.write("Searching knowledge base...")
            for event in api.stream_query(
                st.session_state.token,
                user_query,
                model=st.session_state.selected_model,
                temperature=st.session_state.temperature,
                top_k=st.session_state.k_value,
            ):
                etype = event.get("event")
                if etype == "sources":
                    sources_buffer = event.get("data", "")
                elif etype == "token":
                    tokens.append(event.get("data", ""))
                    placeholder.markdown("".join(tokens) + "▌")
                elif etype == "images":
                    images_buffer = event.get("data") or []
                elif etype == "error":
                    raise api.ApiError(event.get("data", "Stream error"))
                elif etype == "done":
                    break

            final_answer = "".join(tokens).strip()
            placeholder.markdown(final_answer)
            status_box.update(label="Complete.", state="complete", expanded=False)
            if sources_buffer:
                with st.expander("View Sources"):
                    st.info(sources_buffer)
            for img_path in images_buffer:
                if img_path:
                    st.image(api.fetch_image_url(img_path), use_container_width=True)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": final_answer,
                    "sources": sources_buffer,
                    "images": images_buffer,
                }
            )
        except api.ApiError as exc:
            status_box.update(label="Error", state="error")
            st.error(str(exc))
