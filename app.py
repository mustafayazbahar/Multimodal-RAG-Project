import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests
import streamlit as st
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

from auth import (
    clear_chat_history,
    create_chat_table,
    create_default_admin,
    create_users_table,
    load_chat_history,
    login_user,
    register_user,
    save_message,
)
from config import settings
from logging_config import get_logger
from rag import retrieve

os.environ["TOKENIZERS_PARALLELISM"] = "false"

log = get_logger(__name__)

st.set_page_config(
    page_title="DeepCampus",
    layout="wide",
    initial_sidebar_state="expanded",
)

create_users_table()
create_default_admin()
create_chat_table()

PROMPT_PATH = settings.paths.prompts / "rag_answer.txt"
SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(name: str) -> str:
    """Strip directory components and dangerous characters from an uploaded filename."""
    base = os.path.basename(name or "").strip()
    base = base.lstrip(".") or "upload.pdf"
    cleaned = SAFE_FILENAME_RE.sub("_", base)
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned[:255]


col1, col2 = st.columns([1, 6])
with col1:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png",
        width=60,
    )
with col2:
    st.title("DeepCampus: Intelligent Research Assistant")
    st.caption("Powered by RAG Architecture | Local Llama 3.1 8-Bit | Real-Time Sync")

st.divider()


@st.cache_resource
def initialize_system():
    embeddings = HuggingFaceEmbeddings(model_name=settings.models.embedding_model)
    db_path = settings.paths.chroma_db
    if not db_path.exists():
        return None, embeddings
    vectorstore = Chroma(
        persist_directory=str(db_path),
        embedding_function=embeddings,
    )
    return vectorstore, embeddings


@st.cache_resource
def load_prompt_template() -> ChatPromptTemplate:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    return ChatPromptTemplate.from_template(text)


def normalize_messages(messages):
    normalized = []
    for msg in messages:
        if isinstance(msg, dict):
            normalized.append(
                {
                    "role": msg.get("role", "assistant"),
                    "content": msg.get("content", ""),
                    "images": msg.get("images", []) or [],
                    "sources": msg.get("sources", "") or "",
                }
            )
        elif isinstance(msg, (list, tuple)):
            role = msg[0] if len(msg) > 0 else "assistant"
            content = msg[1] if len(msg) > 1 else ""
            normalized.append({"role": role, "content": content, "images": [], "sources": ""})
    return normalized


vectorstore, embeddings = initialize_system()

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.username = None

if "messages" not in st.session_state:
    st.session_state.messages = []
else:
    st.session_state.messages = normalize_messages(st.session_state.messages)

if "temperature" not in st.session_state:
    st.session_state.temperature = settings.rag.temperature
if "k_value" not in st.session_state:
    st.session_state.k_value = settings.rag.top_k

# --- SIDEBAR ---
with st.sidebar:
    st.header("User Authentication")

    if not st.session_state.logged_in:
        menu = st.radio("Select Option", ["Login", "Register"])
        with st.form("auth_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Create Account" if menu == "Register" else "Login")

        if submitted:
            if menu == "Register":
                if register_user(username, password):
                    st.success("Account created! You can now login.")
                else:
                    st.error("Username already exists or invalid input.")
            else:
                success, role = login_user(username, password)
                if success:
                    st.session_state.logged_in = True
                    st.session_state.role = role
                    st.session_state.username = (username or "").strip()
                    history = load_chat_history(st.session_state.username)
                    st.session_state.messages = normalize_messages(history)
                    st.rerun()
                else:
                    st.error("Invalid credentials")
    else:
        st.success(f"Logged in as {st.session_state.username} ({st.session_state.role})")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Logout", use_container_width=True):
                st.session_state.logged_in = False
                st.session_state.role = None
                st.session_state.username = None
                st.session_state.messages = []
                st.rerun()
        with c2:
            if st.button("Clear Chat", use_container_width=True):
                clear_chat_history(st.session_state.username)
                st.session_state.messages = []
                st.rerun()

        st.divider()

        if st.session_state.role == "instructor":
            st.subheader("Model Parameters")
            st.session_state.temperature = st.slider(
                "Creativity (Temperature)", 0.0, 1.0, st.session_state.temperature, 0.1
            )
            st.session_state.k_value = st.slider(
                "Top-k Retrieval", 1, 25, st.session_state.k_value, 1
            )

            st.divider()
            st.subheader("Knowledge Base Management")

            docs_dir = settings.paths.docs
            docs_dir.mkdir(parents=True, exist_ok=True)
            files = sorted(p.name for p in docs_dir.iterdir() if p.is_file())
            if files:
                st.text("Current Files:")
                for f in files:
                    st.text(f"- {f}")
            else:
                st.warning("Folder is empty.")

            st.markdown("---")

            uploaded_file = st.file_uploader("Upload New PDF", type="pdf")
            if uploaded_file is not None:
                safe_name = sanitize_filename(uploaded_file.name)
                save_path = docs_dir / safe_name
                try:
                    save_path.resolve().relative_to(docs_dir.resolve())
                except ValueError:
                    st.error("Invalid filename.")
                else:
                    save_path.write_bytes(uploaded_file.getbuffer())
                    st.success(f"Saved: {safe_name}")

            if st.button("Process & Update Database", type="primary"):
                with st.spinner("Evicting VRAM and processing PDFs..."):
                    try:
                        requests.post(
                            f"{settings.models.ollama_host}/api/generate",
                            json={"model": settings.models.llm_model, "keep_alive": 0},
                            timeout=3,
                        )
                        log.info("Ollama VRAM evict request sent.")
                    except requests.RequestException as exc:
                        log.warning("Ollama eviction request failed: %s", exc)

                    try:
                        result = subprocess.run(
                            [sys.executable, "ingest.py"],
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if result.returncode == 0:
                            st.success("Database updated successfully.")
                            time.sleep(1)
                            initialize_system.clear()
                            st.rerun()
                        else:
                            st.error("Error during ingestion.")
                            st.code(result.stderr)
                    except OSError as exc:
                        st.error(f"Failed to run ingestion script: {exc}")
        else:
            st.session_state.temperature = settings.rag.temperature
            st.session_state.k_value = settings.rag.top_k
            st.info("Student View Active. Advanced settings are locked.")

# --- MAIN CHAT ---
st.session_state.messages = normalize_messages(st.session_state.messages)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("sources"):
                with st.expander("View Sources"):
                    st.info(f"References: {message['sources']}")
            existing_images = [
                img for img in message.get("images", [])
                if isinstance(img, str) and os.path.exists(img)
            ]
            if existing_images:
                st.markdown("### Related Images")
                for img_path in existing_images:
                    st.image(img_path, use_container_width=True)

if not st.session_state.logged_in:
    st.warning("Please login to use the system. (Default admin -> user: admin | pass: admin123)")
    st.stop()


def build_context(docs):
    context_text = ""
    found_images: list[str] = []
    sources_list: list[str] = []
    for d in docs:
        source_name = d.metadata.get("source", "Unknown Source")
        page_num = int(d.metadata.get("page", 0)) + 1
        doc_type = d.metadata.get("type", "text")
        if doc_type == "image":
            sources_list.append(f"[IMAGE] {source_name} (Page {page_num})")
            img_path = d.metadata.get("image_path")
            if img_path and os.path.exists(img_path):
                found_images.append(img_path)
                context_text += f"[IMAGE SUMMARY - ID: {img_path}]: {d.page_content}\n\n"
        else:
            sources_list.append(f"[TEXT] {source_name} (Page {page_num})")
            context_text += f"[TEXT - Page {page_num}]: {d.page_content}\n\n"
    unique_images = list(dict.fromkeys(found_images))
    unique_sources = sorted(set(sources_list))
    return context_text, unique_images, ", ".join(unique_sources)


if user_query := st.chat_input("Ask a question about the documents..."):
    user_message = {"role": "user", "content": user_query, "images": [], "sources": ""}
    st.session_state.messages.append(user_message)
    st.session_state.messages = normalize_messages(st.session_state.messages)
    save_message(st.session_state.username, "user", user_query, sources="", images=[])

    with st.chat_message("user"):
        st.markdown(user_query)

    if vectorstore is None:
        st.error("Knowledge base is empty. Please login as Instructor and upload documents first.")
    else:
        with st.chat_message("assistant"):
            status_container = st.status("Processing...", expanded=True)
            try:
                status_container.write("Searching knowledge base & images...")
                docs = retrieve(vectorstore, user_query, k=st.session_state.k_value)

                status_container.write("Synthesizing context...")
                context_text, unique_images, sources_formatted = build_context(docs)

                llm = ChatOllama(
                    model=settings.models.llm_model,
                    temperature=st.session_state.temperature,
                )

                history_window = settings.rag.history_window
                chat_history_text = ""
                history_slice = st.session_state.messages[-(history_window + 1):-1]
                for msg in history_slice:
                    role_name = "Student" if msg["role"] == "user" else "Assistant"
                    chat_history_text += f"{role_name}: {msg.get('content', '')}\n"

                prompt = load_prompt_template()
                chain = prompt | llm | StrOutputParser()

                status_container.update(label="Generating answer...", state="running")
                answer_placeholder = st.empty()
                streamed_chunks: list[str] = []
                for chunk in chain.stream(
                    {
                        "context": context_text,
                        "history": chat_history_text,
                        "question": user_query,
                    }
                ):
                    streamed_chunks.append(chunk)
                    visible = re.sub(
                        r"\[(?:GÖRSEL|IMAGE|RESIM|RESİM):\s*.*?\]",
                        "",
                        "".join(streamed_chunks),
                    )
                    answer_placeholder.markdown(visible + "▌")

                raw_answer = "".join(streamed_chunks)
                status_container.update(label="Complete.", state="complete", expanded=False)

                cited_images = re.findall(
                    r"\[(?:GÖRSEL|IMAGE|RESIM|RESİM):\s*(.*?)\]",
                    raw_answer,
                    re.IGNORECASE,
                )
                cited_set = {c.strip() for c in cited_images}
                final_display_images = [img for img in unique_images if img in cited_set]
                final_answer = re.sub(
                    r"\[(?:GÖRSEL|IMAGE|RESIM|RESİM):\s*.*?\]",
                    "",
                    raw_answer,
                    flags=re.IGNORECASE,
                ).strip()

                answer_placeholder.markdown(final_answer)

                if sources_formatted:
                    with st.expander("View Sources"):
                        st.info(f"References: {sources_formatted}")

                existing_images = [
                    img for img in final_display_images
                    if isinstance(img, str) and os.path.exists(img)
                ]
                if existing_images:
                    st.markdown("### Related Images")
                    for img in existing_images:
                        st.image(img, use_container_width=True)

                assistant_message = {
                    "role": "assistant",
                    "content": final_answer,
                    "sources": sources_formatted,
                    "images": existing_images,
                }
                st.session_state.messages.append(assistant_message)
                st.session_state.messages = normalize_messages(st.session_state.messages)
                save_message(
                    st.session_state.username,
                    "assistant",
                    final_answer,
                    sources=sources_formatted,
                    images=existing_images,
                )

            except requests.RequestException as exc:
                status_container.update(label="LLM connection error", state="error")
                st.error(f"Could not reach the LLM backend: {exc}")
            except FileNotFoundError as exc:
                status_container.update(label="Missing resource", state="error")
                st.error(f"Required file missing: {exc}")
            except Exception as exc:  # noqa: BLE001 - last-resort UI guard
                status_container.update(label="Error", state="error")
                log.exception("Unexpected error during chat handling")
                st.error(f"An error occurred: {exc}")
