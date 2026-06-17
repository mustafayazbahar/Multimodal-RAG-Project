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


# Web Speech API'nin konusma-tanima (STT) icin bekledigi kisa dil kodunu dondurur.
# Arayuzdeki "Turkish"/"English" etiketini "tr"/"en" formatina cevirir.
def _stt_lang_code(label: str) -> str:
    return "tr" if label == "Turkish" else "en"


# Sesli okuma (TTS) icin tam BCP-47 dil etiketini dondurur (orn. tr-TR / en-US).
# STT'den ayri tutulur cunku TTS daha spesifik bolge kodu ister.
def _tts_lang_code(label: str) -> str:
    return "tr-TR" if label == "Turkish" else "en-US"


# Cevabin yanina "Sesli oku" butonu ekler; tiklaninca tarayicinin TTS
# (metin-okuma) motoru ilgili dilde metni seslendirir.
def _speak_button(text: str, lang_tag: str, key: str) -> None:
    """Render a 'Read aloud' button next to an answer — invokes browser TTS."""
    if not text:
        return
    # Metni ve dil etiketini JS icine guvenli sekilde gomebilmek icin JSON'a
    # kacisliyoruz (tirnak/yeni satir gibi karakterler script'i bozmasin diye).
    safe_text = json.dumps(text)
    safe_lang = json.dumps(lang_tag)
    btn_id = f"dc-tts-{key}"
    # Buton + tarayici TTS scripti dogrudan HTML olarak gomulur. speechSynthesis
    # window.parent uzerinden cagrilir cunku bu HTML, Streamlit'in iframe'i
    # icinde calisir; ses ana pencere baglaminda uretilmelidir.
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
# Gorseller asiri buyumesin diye sabit bir maksimum genislik (px) belirliyoruz.
_IMAGE_MAX_WIDTH = 420


# Backend'den gelen ham gorsel byte'larini sabit genislikte ekrana basar.
def _render_image_blob(blob: bytes) -> None:
    st.image(blob, width=_IMAGE_MAX_WIDTH)


# Gorsel byte'larini backend'den ceker ve cache'ler; ayni gorsel her rerun'da
# tekrar indirilmesin diye 1 saatlik TTL ile onbellege alinir (performans).
@st.cache_data(show_spinner=False, ttl=3600, max_entries=64)
def _cached_image_bytes(token: str, img_path: str) -> bytes | None:
    return api.fetch_image_bytes(token, img_path)


# Tema durumunu dark <-> light arasinda degistirir (sidebar'daki tema butonu
# bunu on_click ile cagirir).
def _toggle_theme() -> None:
    st.session_state["theme"] = (
        "light" if st.session_state.get("theme", "dark") == "dark" else "dark"
    )


# Cevap metnini ekrana basmadan once icindeki [GORSEL: ...] gibi gorsel
# etiketlerini regex ile temizler; gercek gorseller ayri kanaldan render edilir,
# bu yuzden metinde etiket gorunmesine gerek yok.
def _render_content_with_images(text: str) -> None:
    clean_text = _IMAGE_PATTERN.sub("", text)
    clean_text = _BARE_IMAGE_LINE.sub("", clean_text).strip()
    if clean_text:
        st.markdown(clean_text)


# session_state.messages icindeki tum gecmis mesajlari (kullanici + asistan)
# sirayla sohbet balonlari halinde ekrana cizer. Her rerun'da bastan calisir.
def _render_messages() -> None:
    lang_tag = _tts_lang_code(st.session_state.get("voice_lang", "Turkish"))
    for idx, msg in enumerate(st.session_state.messages):
        role = msg["role"]
        # Role gore avatar sec: kullanici icin ogrenci, asistan icin mezuniyet ikonu.
        avatar = "🧑‍🎓" if role == "user" else "🎓"
        with st.chat_message(role, avatar=avatar):
            chat_bubble_meta(role, msg.get("ts", ""))

            content = msg.get("content", "")
            _render_content_with_images(content)

            # Yalnizca asistan mesajlari icin ek ogeler: gorseller, kaynaklar ve
            # sesli okuma butonu. Kullanici mesajlarinda bunlar bulunmaz.
            if role == "assistant":
                # Mesaja bagli gorselleri (varsa) cache uzerinden cekip goster.
                for img_path in (msg.get("images") or []):
                    if not img_path:
                        continue
                    blob = _cached_image_bytes(st.session_state.token, img_path)
                    if blob:
                        _render_image_blob(blob)
                # Cevabin dayandigi kaynaklari katlanabilir bir panelde sun.
                if msg.get("sources"):
                    with st.expander("View sources", expanded=False):
                        source_cards(msg["sources"])
                _speak_button(content, lang_tag, key=f"hist-{idx}")


# ────────────────────────────────────────────────────────────────────────────
# Session state defaults & F5 HYDRATION
# ────────────────────────────────────────────────────────────────────────────
# Tum session_state anahtarlarini varsayilan degerleriyle bir kez baslatiyoruz.
# setdefault kullaniliyor ki mevcut bir deger varsa ezilmesin (rerun guvenli).
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

# F5/sayfa yenileme korumasi: token ve aktif oturum localStorage'a yazildigi icin
# burada geri yukleniyor. Boylece kullanici sayfayi yenileyince oturumu dusmez.
ses.hydrate_from_cookie()


# ────────────────────────────────────────────────────────────────────────────
# OAuth Authorization Code callback — runs before any auth check.
# ────────────────────────────────────────────────────────────────────────────
# OAuth geri donus isleyicisi: URL'de ?code=... varsa Keycloak'tan donmusuz
# demektir. Bu yetki kodunu backend uzerinden token'a cevirip oturum aciyoruz.
# Sadece henuz token yokken calisir ki geri-tusu kaynakli bayat bir kod taze
# oturumu ezmesin; kod paramini her durumda URL'den temizleriz.
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

    # Kodu okuduktan sonra URL'i temizle: code + Keycloak'in eklediği state ve
    # session_state paramlarini kaldir ki adres cubugu temiz kalsin ve kod
    # yeniden kullanilamasin.
    del st.query_params["code"]
    if "state" in st.query_params:
        del st.query_params["state"]
    if "session_state" in st.query_params:
        del st.query_params["session_state"]

    # Yetki kodunu backend araciligiyla erisim token'ina cevir (token exchange).
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
# Oturum acik mi? Tek olcut: gecerli bir erisim token'inin bulunmasi.
def _logged_in() -> bool:
    return bool(st.session_state.token)


# Kullanicinin oturum (konu) listesini backend'den ceker. Aktif oturum id'si
# artik gecersizse, varsayilan konuya (General Chat) geri duser.
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


# Aktif oturumun mesaj gecmisini backend'den cekip session_state.messages'a yukler.
# Backend bayat bir id'yi General Chat'e cozmusse, aktif id'yi de buna gore gunceller.
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


# Backend'den LLM listesini ceker: yuklu (available) ve indirilebilir (pullable)
# modelleri ayri ayri saklar. Henuz model secilmemisse varsayilani secer.
def _refresh_models() -> None:
    try:
        info = api.list_models(st.session_state.token)
        st.session_state.available_models = info.get("available", [])
        st.session_state.pullable_models = info.get("pullable", [])
        if not st.session_state.selected_model:
            st.session_state.selected_model = info.get("default")
    except api.ApiError:
        # Model listesi alinamazsa sessizce gec; arayuz yine de acilabilmeli.
        pass


# Verilen oturumu aktif konu yapar, localStorage'a yazar ve gecmisini yeniden yukler.
def _switch_session(session_id: str) -> None:
    """Make `session_id` the active topic; reload its history."""
    st.session_state.active_session_id = session_id
    ses.update_active_session(session_id)
    _refresh_history()


# Klasik kullanici adi/parola girisi (fallback). Backend, Keycloak proxy'si
# uzerinden parola-grant akisiyla token alir.
def _handle_login(username: str, password: str) -> None:
    """Password-grant login via the backend's Keycloak proxy."""
    try:
        data = api.login(username, password)
        _post_auth_success(data)
    except api.ApiError as exc:
        st.error(str(exc))


# Yeni kullanici kaydi olusturur ve basariliysa otomatik oturum acar.
# Backend, Keycloak'ta kullaniciyi yaratip token doner; hata ApiError ile gelir.
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


# Sayfayi tamamen meta-refresh yonlendirmesi + gorunur yedek link ile degistirir.
# Logout akisinda kullanilir: Streamlit'in component iframe'i sandbox'li oldugu
# icin JS ile ust pencere yonlendirmesi engellenir; bu yuzden meta-refresh, ana
# Streamlit belgesine (st.markdown ile) gomulerek ust pencere dogrudan yonlendirilir.
def _render_redirect_screen(url: str, heading: str) -> None:
    """Replace the page with a meta-refresh + visible fallback link.

    Used by the logout flow. Two reasons not to use a JS redirect via
    `st.components.v1.html`:
    - Streamlit's component iframe is sandboxed without
      `allow-top-navigation`, so `window.top.location = ...` is
      silently blocked by Chromium and Safari.
    - The components iframe is its own browsing context — a meta
      refresh inside it only refreshes the iframe, not the top window.

    `st.markdown` renders straight into the main Streamlit document
    (no nested iframe), so a `<meta http-equiv="refresh">` inserted
    there hits the top window directly. Chrome/Firefox process meta
    refresh elements added to the body dynamically. If a browser
    ignores it, the visible "click here" anchor is the user's manual
    fallback.
    """
    safe_url = url.replace('"', '&quot;')
    st.info(heading)
    st.markdown(
        f"""
        <meta http-equiv="refresh" content="0; url={safe_url}">
        <p style="margin-top:14px;font-size:0.95rem;">
            If you are not redirected automatically,
            <a href="{safe_url}" target="_self"
               style="color:#F59E0B;font-weight:600;">click here</a>.
        </p>
        """,
        unsafe_allow_html=True,
    )
    st.stop()


# Her basarili giris (OAuth, parola veya kayit) sonrasi calisan ortak adim:
# token/kullanici bilgilerini session'a yazar, oturum-gecmis-model verisini
# tazeler, durumu localStorage'a kaydeder ve sayfayi yeniden cizdirir.
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


# Cikis akisi (iki asamali): once yerel oturumu/localStorage'i temizler ve
# Keycloak cikis URL'sini session_state'e koyup rerun tetikler; bir sonraki
# calismada ust kisimdaki isleyici bu URL'yi gorup yonlendirme ekranini cizer.
def _logout() -> None:
    """Local logout + queue a Keycloak end-session redirect.

    Two-phase design (was a single-shot JS redirect before — see the
    fix commit, sandbox blocked it):

    1. Right now: clear localStorage, drop auth state from session,
       wipe URL params, stash the Keycloak logout URL in session_state,
       trigger a rerun.
    2. On the next run, the top-of-script handler sees the stashed URL
       and renders a meta-refresh + visible click-through link instead
       of the normal page. The redirect fires; if the browser ignores
       the dynamic meta refresh, the user clicks the link manually.
    """
    id_token = st.session_state.get("id_token")
    try:
        logout_url = api.get_logout_url(FRONTEND_URL, id_token)
    except api.ApiError:
        # If we can't reach the backend, at least bounce the user back
        # to the login screen locally — they're already logged out of
        # DeepCampus by the cookie/state wipe below.
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

    # localStorage delete is iframe-async (postMessage round-trip).
    # If we rerun immediately the cookie can still be readable when
    # hydrate_from_cookie() fires on the next run and we re-login
    # ourselves. ~0.3 s is the empirical minimum.
    time.sleep(0.3)

    st.session_state["_pending_logout_redirect"] = logout_url
    st.rerun()


# ────────────────────────────────────────────────────────────────────────────
# Auth screen (login / register)
# ────────────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────────
# Pending Keycloak end-session redirect.
# _logout() stashes the logout URL into session_state and triggers a rerun;
# we render the redirect screen here on the *next* run instead of inside the
# callback, so the page has actually been redrawn before the navigation kicks
# in (otherwise the user sees a half-blanked page while the redirect lands).
# Pop-style read so this only fires once.
# ────────────────────────────────────────────────────────────────────────────
# Beklemede bir Keycloak cikis yonlendirmesi varsa, normal sayfa yerine
# yonlendirme ekranini cizeriz. pop ile okunur ki bu blok yalnizca bir kez calissin.
_pending_logout_redirect = st.session_state.pop("_pending_logout_redirect", None)
if _pending_logout_redirect:
    _render_redirect_screen(_pending_logout_redirect, "Logging out of Keycloak…")

# OAuth geri donusunu (URL'deki ?code=) her auth kontrolunden ONCE isle ki
# Keycloak'tan donen kullanici dogrudan oturum acmis sayilsin.
_handle_oauth_callback()

# Oturum yoksa giris ekranini goster. Sayfa, ortadaki sutuna ortalanir.
if not _logged_in():
    left, mid, right = st.columns([1, 2, 1])
    with mid:
        hero(
            "DeepCampus",
            "Hybrid Retrieval-Augmented Generation over your academic PDFs. "
            "Sign in via Keycloak to start asking questions.",
        )

        # Primary path is browser-redirected OAuth Code flow. The
        # button is rendered as a real <a href="..."> anchor styled
        # like a Streamlit button — NOT as st.button + JS redirect.
        # Reasoning: Streamlit's component iframes are sandboxed
        # without `allow-top-navigation`, so JS-driven navigation
        # (window.top.location = ...) gets silently blocked. A plain
        # anchor click is a native browser navigation, no sandbox
        # involved, and it Just Works on every browser.
        #
        # We pre-compute the URL at render time so the link is alive
        # immediately — one click, no roundtrip through a button
        # callback.
        try:
            kc_login_url = api.get_login_url(FRONTEND_URL)
        except api.ApiError as exc:
            kc_login_url = None
            st.error(f"Could not reach the backend to build the Keycloak login URL: {exc}")

        if kc_login_url:
            st.markdown(
                f"""
                <a href="{kc_login_url}" target="_self"
                   style="display:block;
                          padding:12px 16px;
                          background:linear-gradient(180deg,#F59E0B 0%,#D97706 100%);
                          color:#111827;
                          font-weight:600;
                          text-align:center;
                          text-decoration:none;
                          border-radius:8px;
                          font-size:1rem;
                          margin:6px 0 4px 0;
                          border:1px solid rgba(0,0,0,0.18);
                          box-shadow:0 1px 2px rgba(0,0,0,0.15);">
                    🔐 Sign in with Keycloak
                </a>
                """,
                unsafe_allow_html=True,
            )

        st.caption(
            "Recommended. Your password is entered on Keycloak's own login "
            "page — it never flows through DeepCampus."
        )

        # Yedek (fallback) giris yollari: eski parola-grant girisi ve kayit formu.
        # Onerilen yol yukaridaki Keycloak yonlendirmesidir; bunlar geri planda kalir.
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

        # Enter tusuna basinca giris formunu gondermeyi saglayan JS baglamasi.
        bind_login_enter()

    # Hala giris yapilmadiysa scripti burada durdur; alttaki uygulama hic cizilmez.
    if not _logged_in():
        st.stop()


# ────────────────────────────────────────────────────────────────────────────
# Post-login bootstrap: sessions, history and models
# ────────────────────────────────────────────────────────────────────────────
# Giristen sonra ilk kez calisirken oturum listesi, gecmis ve model bilgisini
# bir defa yukleriz. Bayrak (models_initialized) sayesinde her rerun'da
# tekrar tekrar API cagrisi yapilmaz.
if not st.session_state.get("models_initialized"):
    _refresh_sessions()
    _refresh_history()
    _refresh_models()
    st.session_state["models_initialized"] = True


# ────────────────────────────────────────────────────────────────────────────
# Sidebar
# ────────────────────────────────────────────────────────────────────────────
# Yeni bir konu (sohbet) olusturur. Bos baslik kabul edilmez; basariliysa
# girdi temizlenir, oturum listesi tazelenir ve yeni konuya gecilir.
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


# Bir konunun basligini degistirir. Basariliysa duzenleme modundan cikar
# (editing_session_id sifirlanir) ve liste yeniden yuklenir.
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


# Bir konuyu siler. Silinen konu o an aktifse, aktif secimi bosaltiriz ki
# liste tazelendiginde varsayilan (General Chat) konuya geri dusulsun.
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

    # Giris yapan kullanicinin adi ve rolu kucuk bir rozet (pill) olarak gosterilir.
    st.markdown("### DeepCampus")
    st.markdown(
        status_pill(f"{st.session_state.username} · {st.session_state.role}"),
        unsafe_allow_html=True,
    )

    # Cikis butonu: _logout, yerel oturumu temizleyip Keycloak cikisina yonlendirir.
    if st.button("Logout", use_container_width=True):
        _logout()

    st.divider()
    # "My Topics" bolumu: kullanicinin tum sohbet konularinin listelendigi alan.
    sidebar_section_title("📚 My Topics")

    # Yeni konu olusturma formu (katlanabilir panel icinde gizli durur).
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

    # Konu listesini tek tek cizeriz. Her konu icin aktif mi, varsayilan mi
    # (General Chat) oldugunu belirleyip ona gore ikon ve butonlari uretiriz.
    sessions = st.session_state.sessions or []
    for session in sessions:
        sid = session["session_id"]
        is_active = (sid == st.session_state.active_session_id)
        is_default = session.get("is_default", False)
        # Aktif konuyu yesil nokta ile, digerlerini sohbet balonu ile isaretle.
        icon = "🟢" if is_active else "💬"

        # Inline rename mode (only for non-default topics).
        # Bu konu su an duzenleniyorsa, baslik yerine satir-ici yeniden adlandirma
        # girisini (kaydet/iptal butonlariyla) goster ve dongunun gerisini atla.
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
            # Varsayilan konu (General Chat) silinemez/yeniden adlandirilamaz;
            # sadece secim butonu olarak gosterilir, hep listenin basinda durur.
            if st.button(
                f"{icon} {session['title']}",
                key=f"sel_{sid}",
                type="primary" if is_active else "secondary",
                use_container_width=True,
            ):
                _switch_session(sid)
                st.rerun()
        else:
            # Normal konular: secim butonu + yeniden adlandir + sil butonlari.
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
                # Kalem butonu: bu konuyu duzenleme moduna alir (rerun ile satir-ici
                # girise doner).
                if st.button("✏️", key=f"edit_{sid}", help="Rename"):
                    st.session_state.editing_session_id = sid
                    st.rerun()
            with col3:
                # Cop kutusu butonu: konuyu kalici olarak siler.
                if st.button("🗑️", key=f"del_{sid}", help="Delete"):
                    _handle_delete_topic(sid)
                    st.rerun()

    st.divider()
    # Aktif konunun mesaj gecmisini backend'de ve ekranda temizler (konuyu silmez).
    if st.button("Clear current topic", use_container_width=True, key="clear_chat_btn"):
        try:
            api.clear_history(st.session_state.token, st.session_state.active_session_id)
            st.session_state.messages = []
            st.rerun()
        except api.ApiError as exc:
            st.error(str(exc))

    st.divider()
    # Ses bolumu: sesli giris ve sesli okumada kullanilacak dil secimi.
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
    # Model bolumu: yuklu (pulled) ve indirilebilir (pullable) LLM'leri yonetir.
    sidebar_section_title("Model")
    pulled = st.session_state.available_models or []
    pullable = st.session_state.get("pullable_models") or []

    # Hicbir model bilgisi yoksa (backend yeni acilmis olabilir) elle yenileme sun.
    if not pulled and not pullable:
        if st.button("Refresh model list", key="refresh_models", use_container_width=True):
            _refresh_models()
            st.rerun()

    # Yuklu model varsa aktif LLM'i sectirir; mevcut secimi listede konumlandirir.
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

    # Yeni model indirme yalnizca egitmen (instructor) rolune aciktir.
    if pullable and st.session_state.role == "instructor":
        with st.expander("Download more models", expanded=not pulled):
            to_pull = st.selectbox(
                "Model to download",
                pullable,
                key="pull_choice",
            )
            if st.button("Download", key="pull_btn", use_container_width=True):
                # Ollama indirme ilerlemesi akis (stream) halinde gelir; her yeni
                # durum mesajini canli olarak ekrana basariz.
                status = st.status(f"Downloading {to_pull}…", expanded=True)
                progress_text = st.empty()
                last_status = ""
                try:
                    for evt in api.pull_model(st.session_state.token, to_pull):
                        msg = evt.get("status") or evt.get("error") or ""
                        # Ayni mesaji tekrar yazmamak icin yalnizca degisince guncelle.
                        if msg and msg != last_status:
                            progress_text.markdown(f"`{msg}`")
                            last_status = msg
                        if "error" in evt:
                            status.update(label=f"Failed: {evt['error']}", state="error")
                            break
                    else:
                        # for-else: dongu hatasiz bitti => indirme tamam, listeyi tazele.
                        status.update(label=f"Downloaded {to_pull}", state="complete", expanded=False)
                        _refresh_models()
                        st.rerun()
                except api.ApiError as exc:
                    status.update(label=f"Failed: {exc}", state="error")

    # Getirme (retrieval) ayarlari: yaraticilik (temperature), getirilecek belge
    # sayisi (top-k) ve dense/sparse arama agirligi kullaniciya birakilir.
    sidebar_section_title("Retrieval")
    st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, 0.05)
    st.session_state.k_value = st.slider("Top-k retrieval", 4, 40, st.session_state.k_value, 1)
    st.session_state.dense_weight = st.slider("Dense ↔ Sparse weight", 0.0, 1.0, st.session_state.dense_weight, 0.05)

    # Bilgi tabani yonetimi (PDF yukleme, indeksleme, sifirlama) yalnizca
    # egitmen rolune gosterilir; ogrenciler bu araclari goremez.
    if st.session_state.role == "instructor":
        st.divider()
        sidebar_section_title("Knowledge base")
        # Indeksleme durumu: kac kaynak indekslendi ve docs/ altinda hangi
        # dosyalar var bilgisini backend'den cekip ozet rozet olarak gosterir.
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

        # Moondream gorsel-ozetleri: ingest sirasinda cikarilan gorseller icin
        # uretilen aciklamalar burada listelenir (kaynak, sayfa, ozet).
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

        # PDF yukleme: secilen dosya bayt olarak backend'e gonderilip docs/
        # altina kaydedilir. Kaydetme indeksleme degildir; asil islem asagidaki
        # "Process & update database" adimiyla yapilir.
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

        # Indeksleme hattini (ingestion pipeline) calistirir: docs/ altindaki yeni
        # PDF'leri isler, parcalara boler ve vektor veritabanina ekler.
        if st.button("Process & update database", type="primary", use_container_width=True):
            with st.spinner("Running ingestion pipeline..."):
                try:
                    result = api.run_ingest(st.session_state.token)
                except api.ApiError as exc:
                    st.error(str(exc))
                    result = None
            # Islem sonucunu ozetle: islenen/tekrar eden/hatali dosya sayilari.
            if result is not None:
                summary = (
                    f"Processed: **{result.get('processed', 0)}** · "
                    f"Duplicates: **{result.get('duplicates', 0)}** · "
                    f"Errors: **{result.get('errors', 0)}**"
                )
                if result.get('chunks'):
                    summary += f" · Chunks: **{result['chunks']}**"
                st.success(summary)
                # Tekrar eden (zaten indeksli) dosyalari ayri bir panelde listele.
                dup_items = [d for d in (result.get("details") or []) if d.get("status") == "duplicate"]
                if dup_items:
                    with st.expander("Duplicate files skipped", expanded=True):
                        for d in dup_items:
                            st.info(f"**{d.get('file', '?')}** — already indexed.")
                # Isleme sirasinda hata alan dosyalari nedenleriyle birlikte goster.
                err_items = [d for d in (result.get("details") or []) if d.get("status") == "error"]
                if err_items:
                    with st.expander("Errors", expanded=True):
                        for d in err_items:
                            st.error(f"**{d.get('file', '?')}** — {d.get('reason', 'error')}")

        # Tehlikeli bolge: vektor veritabanini tamamen sifirlar. Yanlislikla
        # tetiklenmesin diye onay kutusu isaretlenmeden buton etkinlesmez.
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
# Karsilama ekraninda gosterilen ornek sorular; kullaniciya hizli baslangic sunar.
SUGGESTIONS = [
    "Summarize the main contributions of the indexed papers.",
    "Compare the proposed methods across the documents.",
    "Which papers discuss evaluation metrics, and how do they differ?",
    "Explain a figure or chart from the most relevant document.",
]

# Aktif oturum nesnesini listede bul; basligini ust kisimda gostermek icin
# cikar. Bulunamazsa varsayilan olarak "General Chat" kullanilir.
active_session = next(
    (s for s in (st.session_state.sessions or [])
     if s["session_id"] == st.session_state.active_session_id),
    None,
)
active_title = active_session["title"] if active_session else "General Chat"

# Konuda hic mesaj yoksa karsilama ekranini goster; oneri kartlarindan birine
# tiklanirsa onu bekleyen sorgu (pending_query) olarak kuyruga al.
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

# Mikrofon: STT kutuphanesi yuklu ise sesli giris butonu gosterilir.
# Konusma metne cevrilince, yazili soruyla ayni yola (pending_query) aktarilir.
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

# Mesaj gonderimi: girdi iki kaynaktan gelebilir; bekleyen sorgu (sesli giris
# veya oneri karti) varsa o onceliklidir, yoksa kullanicinin yazdigi metin alinir.
# Okuduktan hemen sonra pending_query temizlenir ki bir sonraki rerun'da yinelenmesin.
typed = st.chat_input(f"Ask a question in '{active_title}'...")
user_query = st.session_state.pending_query or typed
st.session_state.pending_query = None

# Gecerli bir kullanici sorusu varsa cevap uretme akisini baslat.
if user_query:
    # Once kullanici mesajini gecmise ekle ve hemen ekranda goster.
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

    # Asistan balonu: cevap akarken token'lar buraya canli olarak yazilir.
    with st.chat_message("assistant", avatar="🎓"):
        chat_bubble_meta("assistant", timestamp_now())
        status_box = st.status("Searching knowledge base...", expanded=True)
        placeholder = st.empty()
        # Akis boyunca biriktirilen tamponlar: kaynaklar, gorseller ve token'lar.
        sources_buffer = ""
        images_buffer: list[str] = []
        tokens: list[str] = []
        # Toplam sure ve ilk-token-suresi (TTFT) olcumu icin baslangic zamani.
        start = time.perf_counter()
        ttft: float | None = None
        resolved_session_id: str | None = None

        try:
            # Backend'den NDJSON akisi gelir; her olay turune (event) gore islenir:
            # session (cozulen oturum id'si), sources (kaynaklar), token (cevap
            # parcasi), images (gorseller), error (hata), done (akis bitti).
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
                    # Backend hangi oturuma yazdigini bildirir (bayat id duzeltilebilir).
                    resolved_session_id = event.get("data")
                elif etype == "sources":
                    # Kaynak metnini tamponla ve durumu "cevap uretiliyor"a cek.
                    sources_buffer += (event.get("data") or "")
                    status_box.update(label="Generating answer...", state="running")
                elif etype == "token":
                    # Ilk token geldiginde TTFT'yi olc, sonra token'i ekle ve
                    # imlec (▌) ile birlikte canli olarak ekrana bas.
                    if ttft is None:
                        ttft = time.perf_counter() - start
                    tokens.append(event.get("data", ""))
                    placeholder.markdown("".join(tokens) + "▌")
                elif etype == "images":
                    images_buffer = event.get("data") or []
                elif etype == "error":
                    # Akis icinde gelen hata, ApiError'a cevrilip asagida yakalanir.
                    raise api.ApiError(event.get("data", "Stream error"))
                elif etype == "done":
                    break

            # Akis bitti: imleci kaldirip nihai cevabi (gorsel etiketleri
            # temizlenmis halde) son kez basariz ve durum kutusunu tamamlariz.
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

            # Cevaba ait gorselleri (varsa) cache uzerinden cekip goster.
            for img_path in images_buffer:
                if not img_path:
                    continue
                blob = _cached_image_bytes(st.session_state.token, img_path)
                if blob:
                    _render_image_blob(blob)

            # Cevabin dayandigi kaynaklari katlanabilir panelde sun.
            if sources_buffer:
                with st.expander("View sources", expanded=False):
                    source_cards(sources_buffer)

            # Bu cevap icin sesli okuma butonu ekle.
            _speak_button(
                final_answer,
                _tts_lang_code(st.session_state.get("voice_lang", "Turkish")),
                key="live-answer",
            )

            # Tamamlanan asistan cevabini (metin + kaynak + gorsel) gecmise yaz ki
            # sonraki rerun'larda _render_messages tarafindan tekrar cizilebilsin.
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
            # Akis sirasinda olusan herhangi bir hatada durumu hata olarak isaretle
            # ve kullaniciya mesaji goster.
            status_box.update(label="Error", state="error")
            st.error(str(exc))

    # Cevap basildiktan sonra sayfayi en alta kaydir ve girdi kutusuna odaklan ki
    # kullanici dogrudan bir sonraki soruyu yazabilsin.
    scroll_to_bottom()
    autofocus_chat_input()
