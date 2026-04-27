import streamlit as st
import os
import time
import subprocess
import sys
import re
import requests
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Terminal Uyarısını Susturucu
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Güvenlik ve Veritabanı (auth.py'dan)
from auth import (
    create_users_table,
    create_default_admin,
    create_chat_table,
    save_message,
    load_chat_history,
    register_user,
    login_user
)

st.set_page_config(
    page_title="DeepCampus",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Sistem İlk Kurulum
create_users_table()
create_default_admin()
create_chat_table()

col1, col2 = st.columns([1, 6])
with col1:
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c3/Python-logo-notext.svg/1200px-Python-logo-notext.svg.png",
        width=60
    )
with col2:
    st.title("DeepCampus: Intelligent Research Assistant")
    st.caption("Powered by RAG Architecture | Local Llama 3.1 8-Bit | Real-Time Sync 🚀")

st.divider()

@st.cache_resource
def initialize_system():
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    if not os.path.exists("./chroma_db"):
        return None, embeddings
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )
    return vectorstore, embeddings

def normalize_messages(messages):
    normalized = []
    for msg in messages:
        if isinstance(msg, dict):
            normalized.append({
                "role": msg.get("role", "assistant"),
                "content": msg.get("content", ""),
                "images": msg.get("images", []) or [],
                "sources": msg.get("sources", "") or ""
            })
        elif isinstance(msg, (list, tuple)):
            role = msg[0] if len(msg) > 0 else "assistant"
            content = msg[1] if len(msg) > 1 else ""
            normalized.append({
                "role": role,
                "content": content,
                "images": [],
                "sources": ""
            })
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
    st.session_state.temperature = 0.3

if "k_value" not in st.session_state:
    st.session_state.k_value = 15

# --- SOL PANEL (GİRİŞ EKRANI VE KONTROLLER) ---
with st.sidebar:
    st.header("User Authentication")

    if not st.session_state.logged_in:
        menu = st.radio("Select Option", ["Login", "Register"])
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if menu == "Register":
            if st.button("Create Account"):
                success = register_user(username, password)
                if success:
                    st.success("Account created! You can now login.")
                else:
                    st.error("Username already exists or invalid input.")
        else:
            if st.button("Login"):
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

        if st.button("Logout"):
            st.session_state.logged_in = False
            st.session_state.role = None
            st.session_state.username = None
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

            if os.path.exists("docs"):
                files = os.listdir("docs")
                if files:
                    st.text("Current Files:")
                    for f in files:
                        st.text(f"- {f}")
                else:
                    st.warning("Folder is empty.")
            else:
                os.makedirs("docs")

            st.markdown("---")

            uploaded_file = st.file_uploader("Upload New PDF", type="pdf")
            if uploaded_file is not None:
                save_path = os.path.join("docs", uploaded_file.name)
                with open(save_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"Saved: {uploaded_file.name}")

            if st.button("Process & Update Database", type="primary"):
                with st.spinner("VRAM boşaltılıyor ve PDF'ler işleniyor..."):
                    # 🚀 MİMARIN DOKUNUŞU: OLLAMA'YI VRAM'DEN KOV
                    try:
                        requests.post(
                            "http://localhost:11434/api/generate", 
                            json={"model": "llama3.1:8b-instruct-q8_0", "keep_alive": 0}, 
                            timeout=3
                        )
                        print("[BILGI] Ollama VRAM tahliyesi başarılı.")
                    except:
                        pass

                    try:
                        result = subprocess.run(
                            [sys.executable, "ingest.py"],
                            capture_output=True,
                            text=True
                        )
                        if result.returncode == 0:
                            st.success("Database Updated Successfully.")
                            time.sleep(1)
                            # 🚀 MİMARIN DOKUNUŞU: Önbelleği temizle ve anlık güncelle!
                            initialize_system.clear()
                            st.rerun()
                        else:
                            st.error("Error during ingestion.")
                            st.code(result.stderr)
                    except Exception as e:
                        st.error(f"Failed to run ingestion script: {e}")
        else:
            st.session_state.temperature = 0.3
            st.session_state.k_value = 15
            st.info("Student View Active. Advanced settings are locked.")

# --- ANA EKRAN (SOHBET SİSTEMİ) ---
st.session_state.messages = normalize_messages(st.session_state.messages)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            if message.get("sources"):
                with st.expander("View Sources"):
                    st.info(f"References: {message['sources']}")

            if message.get("images"):
                existing_images = [
                    img for img in message["images"]
                    if isinstance(img, str) and os.path.exists(img)
                ]
                if existing_images:
                    st.markdown("### 🖼️ İlgili Görsel")
                    for img_path in existing_images:
                        st.image(img_path, use_container_width=True)

if not st.session_state.logged_in:
    st.warning("Please login to use the system. (Default Admin -> user: admin | pass: admin123)")
    st.stop()

if user_query := st.chat_input("Ask a question about the documents..."):
    user_message = {
        "role": "user",
        "content": user_query,
        "images": [],
        "sources": ""
    }
    st.session_state.messages.append(user_message)
    st.session_state.messages = normalize_messages(st.session_state.messages)

    save_message(
        st.session_state.username,
        "user",
        user_query,
        sources="",
        images=[]
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    if vectorstore is None:
        st.error("Knowledge Base is empty. Please login as Instructor and upload documents first.")
    else:
        with st.chat_message("assistant"):
            status_container = st.status("Processing...", expanded=True)

            try:
                status_container.write("Searching knowledge base & images...")
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": st.session_state.k_value}
                )
                docs = retriever.invoke(f"query: {user_query}")

                status_container.write("Synthesizing and filtering images...")
                context_text = ""
                found_images = []
                sources_list = []

                for d in docs:
                    source_name = d.metadata.get("source", "Unknown Source")
                    page_num = int(d.metadata.get("page", 0)) + 1
                    doc_type = d.metadata.get("type", "text")

                    sources_list.append(f"{source_name} (Page {page_num})")

                    if doc_type == "image" or doc_type == "table":
                        img_path = d.metadata.get("source_path")
                        if img_path and os.path.exists(img_path):
                            found_images.append(img_path)
                            context_text += f"[{doc_type.upper()} SUMMARY - ID: {img_path}]: {d.page_content}\n\n"
                    else:
                        context_text += f"[TEXT - Page {page_num}]: {d.page_content}\n\n"

                unique_images = list(dict.fromkeys(found_images))
                unique_sources = sorted(list(set(sources_list)))
                sources_formatted = ", ".join(unique_sources)

                # 🚀 YEREL 8-BİT LLM BAĞLANTISI
                llm = ChatOllama(
                    model="llama3.1:8b-instruct-q8_0",
                    temperature=st.session_state.temperature
                )

                # 🚀 YENİ ESNEK VE YARDIMSEVER PROMPT
                template = """You are a helpful and expert academic assistant. Answer based on the provided context.
Rules:
1. Use the provided context to answer the question in detail. If the context only partially answers the question, share what you can find. 
2. If the context is completely unrelated, politely state that the documents don't contain the exact answer, but share any relevant clues you found. Do not use the raw "NO_INFO_FOUND" code anymore.
3. IMPORTANT: Answer in the EXACT SAME LANGUAGE as the user's question.
4. IMAGE RULE: Be generous with images. If there is an image, chart, or table in the context that is even slightly relevant or helpful to your explanation, you MUST cite it!
5. MAXIMUM LIMIT: Cite MAXIMUM 1 IMAGE. Format: [GÖRSEL: filepath].

Context:
{context}

Question: {question}
"""

                prompt = ChatPromptTemplate.from_template(template)
                chain = prompt | llm | StrOutputParser()

                raw_answer = chain.invoke({
                    "context": context_text,
                    "question": user_query
                })

                status_container.update(label="Complete.", state="complete", expanded=False)

                # 🚀 MİMARIN DOKUNUŞU: Esnek Format Yakalayıcı (Regex)
                cited_images = re.findall(r'\[(?:GÖRSEL|IMAGE|RESIM|RESİM):\s*(.*?)\]', raw_answer, re.IGNORECASE)
                final_display_images = [img for img in unique_images if img in cited_images][:1]
                final_answer = re.sub(r'\[(?:GÖRSEL|IMAGE|RESIM|RESİM):\s*.*?\]', '', raw_answer, flags=re.IGNORECASE).strip()
                sources_to_save = sources_formatted

                st.markdown(final_answer)

                if sources_to_save:
                    with st.expander("View Sources"):
                        st.info(f"References: {sources_to_save}")

                existing_images = [
                    img for img in final_display_images
                    if isinstance(img, str) and os.path.exists(img)
                ]
                if existing_images:
                    st.markdown("### 🖼️ İlgili Görsel")
                    for img in existing_images:
                        st.image(img, use_container_width=True)
                    final_display_images = existing_images
                else:
                    final_display_images = []

                assistant_message = {
                    "role": "assistant",
                    "content": final_answer,
                    "sources": sources_to_save,
                    "images": final_display_images
                }

                st.session_state.messages.append(assistant_message)
                st.session_state.messages = normalize_messages(st.session_state.messages)

                save_message(
                    st.session_state.username,
                    "assistant",
                    final_answer,
                    sources=sources_to_save,
                    images=final_display_images
                )

            except Exception as e:
                status_container.update(label="Error", state="error")
                st.error(f"An error occurred: {e}")