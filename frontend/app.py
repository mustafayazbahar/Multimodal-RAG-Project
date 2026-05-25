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

# Çevre değişkenleri ve stil enjeksiyonu
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
inject_styles()

# --- 5. GLOBAL DEĞİŞKENLER VE SABİTLER ---
_IMAGE_PATTERN = re.compile(r"\[IMAGE SUMMARY - ID:\s*(.*?)\]")

# Voice dil eşlemeleri.
def _stt_lang_code(label: str) -> str:
    return "tr" if label == "Türkçe" else "en"

def _tts_lang_code(label: str) -> str:
    return "tr-TR" if label == "Türkçe" else "en-US"

def _speak_button(text: str, lang_tag: str, key: str) -> None:
    """Cevabın yanına 'Sesli oku' butonu basar — tarayıcı TTS'i çağırır."""
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
            🔊 Sesli oku
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
@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def _cached_image_bytes(token: str, img_path: str) -> bytes | None:
    return api.fetch_image_bytes(token, img_path)

def _render_content_with_images(text: str) -> None:
    """LLM'den gelen ham metni sanitize eder ve resimleri renderlar."""
    clean_text = _IMAGE_PATTERN.sub("", text).strip()
    if clean_text:
        st.markdown(clean_text)
    
    for match in _IMAGE_PATTERN.finditer(text):
        img_path = match.group(1).strip()
        blob = _cached_image_bytes(st.session_state.token, img_path)
        if blob:
            st.image(blob, use_container_width=True)
        else:
            st.error(f"Görsel API'den çekilemedi veya backend'de yok: {img_path}")

def _render_messages() -> None:
    lang_tag = _tts_lang_code(st.session_state.get("voice_lang", "Türkçe"))
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
                        st.image(blob, use_container_width=True)
                if msg.get("sources"):
                    with st.expander("View sources", expanded=False):
                        source_cards(msg["sources"])
                _speak_button(content, lang_tag, key=f"hist-{idx}")

# ────────────────────────────────────────────────────────────────────────────
# Session state defaults & F5 HYDRATION (Kritik Bölge)
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
    "models_initialized": False,
    "available_models": [],
    "last_query_at": None,
    "pending_query": None,
    "voice_lang": "Türkçe",
}.items():
    st.session_state.setdefault(k, v)

# F5 koruması: localStorage'a yazılmış {token, username, role} blob'unu
# tarayıcıdan geri okur. URL'de hiçbir kimlik bilgisi tutulmuyor — eski
# query-params yaklaşımı JWT'yi access log + Referer header + tarayıcı
# geçmişine sızdırıyordu.
ses.hydrate_from_cookie()

# Eski URL fallback'inden artakalmış paramları temizle. Bir önceki sürümde
# login token=eyJ... şeklinde URL'e yazılıyordu; kullanıcı pull edip
# ilk girişini yapana kadar adres çubuğu kirli kalmasın diye yutuyoruz.
if any(k in st.query_params for k in ("token", "user", "role")):
    for k in ("token", "user", "role"):
        if k in st.query_params:
            del st.query_params[k]

# ────────────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────────────
def _logged_in() -> bool:
    return bool(st.session_state.token)

def _refresh_models_and_history() -> None:
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
    """Create a Keycloak user and auto-login on success."""
    try:
        data = api.register(username, password, email, first_name, last_name)
        _post_auth_success(data)
    except api.ApiError as exc:
        st.error(str(exc))


def _post_auth_success(data: dict) -> None:
    # 1. RAM'e kaydet
    st.session_state.token = data["access_token"]
    st.session_state.role = data["role"]
    st.session_state.username = data["username"]

    # 2. Tarayıcı localStorage'ına kaydet — F5'te tekrar girmek yerine
    #    session restore edilecek. URL'e hiçbir şey yazmıyoruz.
    ses.save_cookie(data["access_token"], data["username"], data["role"])

    _refresh_models_and_history()
    # localStorage iframe async; rerun'dan önce kısa bir yield bırakıyoruz
    # ki postMessage tarayıcıya ulaşsın ve F5'te bulunabilsin.
    time.sleep(0.4)
    st.rerun()

def _logout() -> None:
    # 1. localStorage'tan token blob'unu sil — yoksa F5 tekrar oturum
    #    açacak.
    ses.clear_cookie()

    # 2. RAM'i temizle
    for key in ("token", "username", "role", "messages", "available_models", "selected_model"):
        st.session_state[key] = [] if key in ("messages", "available_models") else None

    # 3. Eski URL fallback'inden artakalmış paramları da süpür.
    st.query_params.clear()
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
            "Sign in via Keycloak to start asking questions.",
        )

        # st.form inside st.tabs would swallow Enter on Streamlit 1.57,
        # so the mode toggle is a radio at the top level instead.
        mode = st.radio(
            "Mode",
            options=["Sign in", "Create account"],
            horizontal=True,
            label_visibility="collapsed",
            key="auth_mode",
        )

        if mode == "Sign in":
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
    sidebar_section_title("Voice")
    st.session_state.voice_lang = st.radio(
        "Konuşma dili",
        options=["Türkçe", "English"],
        index=0 if st.session_state.get("voice_lang", "Türkçe") == "Türkçe" else 1,
        horizontal=True,
        key="voice_lang_radio",
        help="Mikrofonla sorma ve cevabı sesli okumada kullanılacak dil.",
    )

    st.divider()
    sidebar_section_title("Model")
    if not st.session_state.get("models_initialized"):
        _refresh_models_and_history()
        st.session_state["models_initialized"] = True

    pulled = st.session_state.available_models or []
    pullable = st.session_state.get("pullable_models") or []

    if not pulled and not pullable:
        if st.button("Refresh model list", key="refresh_models", use_container_width=True):
            _refresh_models_and_history()
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
                        _refresh_models_and_history()
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

        # Moondream-generated image captions, persisted by ingestion.
        # Lets the instructor verify the VLM's understanding of each
        # figure before relying on the chat citations.
        with st.expander("Image summaries (Moondream)", expanded=False):
            try:
                summaries = api.get_image_summaries(st.session_state.token)
            except api.ApiError as exc:
                summaries = []
                st.warning(str(exc))
            if not summaries:
                st.caption("Henüz ingest çalışmadı veya hiçbir görsel çıkarılmadı.")
            else:
                st.caption(f"{len(summaries)} görsel · `data/image_summaries.json` dosyasında")
                for item in summaries:
                    page = (item.get("page") or 0) + 1
                    st.markdown(
                        f"**{item.get('source', '?')}** · page {page}"
                    )
                    st.text(item.get("image_path", ""))
                    st.markdown(
                        f"<div style='color:#a1a1aa;font-size:0.85rem;"
                        f"padding:6px 8px;background:#1e1e26;border-radius:6px;"
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
# Ana Ekran
# ────────────────────────────────────────────────────────────────────────────
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

# Mikrofon
if _STT_AVAILABLE:
    voice_col, _spacer = st.columns([1, 4])
    with voice_col:
        stt_lang = _stt_lang_code(st.session_state.get("voice_lang", "Türkçe"))
        spoken = speech_to_text(
            language=stt_lang,
            start_prompt="🎤 Konuş",
            stop_prompt="⏹️ Durdur",
            just_once=True,
            use_container_width=True,
            key="stt_widget",
        )
        if spoken:
            st.session_state.pending_query = spoken

# Mesaj Gönderme
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
                    st.image(blob, use_container_width=True)

            if sources_buffer:
                with st.expander("View sources", expanded=False):
                    source_cards(sources_buffer)

            _speak_button(
                final_answer,
                _tts_lang_code(st.session_state.get("voice_lang", "Türkçe")),
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
        except api.ApiError as exc:
            status_box.update(label="Error", state="error")
            st.error(str(exc))

    scroll_to_bottom()
    autofocus_chat_input()