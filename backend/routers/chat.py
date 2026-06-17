"""Chat endpoints: sessions ("topics") + history + streaming query."""
from __future__ import annotations

import json
import re
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse

from backend.schemas import (
    BenchmarkRequest,
    BenchmarkResponse,
    ChatHistoryResponse,
    ChatMessage,
    ChatQueryRequest,
    ChatSessionInfo,
    ChatSessionListResponse,
    CreateSessionRequest,
    ModelListResponse,
    PullModelRequest,
    RenameSessionRequest,
)
from backend.security import CurrentUser, get_current_user
from services.auth import (
    clear_chat_messages,
    create_session,
    delete_session,
    ensure_general_chat,
    list_sessions,
    load_chat_history,
    resolve_session,
    save_message,
    update_session_title,
)
from services.config import settings
from services.llm import (
    benchmark_models,
    list_available_models,
    list_pulled_models,
    pull_model,
    render_prompt,
    stream_chat,
)
from services.logging_config import get_logger
from services.retriever import build_context, hybrid_search

log = get_logger(__name__)

# Bu modulun tum endpoint'leri /chat on eki altinda toplanir.
router = APIRouter(prefix="/chat", tags=["chat"])

# LLM cevabindaki gorsel atiflarini yakalamak icin regex. Model, ilgili figurleri
# [GÖRSEL: yol] gibi etiketlerle isaretler; Turkce/Ingilizce ve i/I varyasyonlari
# desteklenir. Yakalanan yollar daha sonra metinden ayiklanip frontend'e gonderilir.
_IMAGE_TAG_RE = re.compile(r"\[(?:GÖRSEL|IMAGE|RESIM|RESİM):\s*(.*?)\]", re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────
# Sessions (chat "topics")
# ─────────────────────────────────────────────────────────────────────────
# Kullanicinin sohbet oturumlarini listeler. Ilk cagrida varsayilan "General
# Chat" oturumu yoksa otomatik olusturulur (kullanicinin daima bir oturumu olur).
@router.get("/sessions", response_model=ChatSessionListResponse)
def get_sessions(
    user: Annotated[CurrentUser, Depends(get_current_user)],
) -> ChatSessionListResponse:
    """List the user's chat sessions; auto-creates General Chat on first call."""
    ensure_general_chat(user.username)
    sessions = list_sessions(user.username)
    return ChatSessionListResponse(
        sessions=[ChatSessionInfo(**s) for s in sessions]
    )


# Verilen baslikla yeni bir sohbet oturumu olusturur.
@router.post("/sessions", response_model=ChatSessionInfo, status_code=status.HTTP_201_CREATED)
def post_session(
    payload: CreateSessionRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
) -> ChatSessionInfo:
    s = create_session(user.username, payload.title)
    return ChatSessionInfo(**s)


# Bir oturumu yeniden adlandirir. Islem sadece oturum bulunur, kullaniciya ait
# ve varsayilan degilse basarili olur; aksi halde 400 doner (General Chat ismi
# degistirilemez).
@router.patch("/sessions/{session_id}", response_model=ChatSessionInfo)
def patch_session(
    session_id: str,
    payload: RenameSessionRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
) -> ChatSessionInfo:
    if not update_session_title(user.username, session_id, payload.title):
        raise HTTPException(
            status_code=400,
            detail="Cannot rename this session (not found, not owned, or default).",
        )
    return ChatSessionInfo(
        session_id=session_id,
        title=payload.title.strip(),
        is_default=False,
    )


# Bir oturumu siler. Yine sadece bulunur, kullaniciya ait ve varsayilan degilse
# silinebilir; aksi halde 400 doner (varsayilan General Chat silinemez).
@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_session(
    session_id: str,
    user: Annotated[CurrentUser, Depends(get_current_user)],
) -> None:
    if not delete_session(user.username, session_id):
        raise HTTPException(
            status_code=400,
            detail="Cannot delete this session (not found, not owned, or default).",
        )


# ─────────────────────────────────────────────────────────────────────────
# History
# ─────────────────────────────────────────────────────────────────────────
# Bir oturumun mesaj gecmisini doner. session_id verilmez veya gecersizse
# resolve_session kullaniciyi General Chat'e dusurur, bu yuzden cevapta hangi
# oturumun cozuldugu (resolved) da geri gonderilir.
@router.get("/history", response_model=ChatHistoryResponse)
def get_history(
    user: Annotated[CurrentUser, Depends(get_current_user)],
    session_id: str | None = Query(None),
) -> ChatHistoryResponse:
    resolved = resolve_session(user.username, session_id)
    messages = [ChatMessage(**m) for m in load_chat_history(resolved)]
    return ChatHistoryResponse(messages=messages, session_id=resolved)


# Tek bir oturumun mesajlarini siler (oturumun kendisini degil). session_id
# verilmezse General Chat hedef alinir.
@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
def delete_history(
    user: Annotated[CurrentUser, Depends(get_current_user)],
    session_id: str | None = Query(None),
) -> None:
    """Wipe a single session's messages. Defaults to General Chat."""
    resolved = resolve_session(user.username, session_id)
    clear_chat_messages(resolved)


# ─────────────────────────────────────────────────────────────────────────
# Models (LLM choice)
# ─────────────────────────────────────────────────────────────────────────
# Bir modelin indirilmis olup olmadigini etiketten (tag) bagimsiz kontrol eder.
# Ollama model isimlerinde "ad:tag" bicimi vardir; bu yuzden 'llama3' ile
# 'llama3:latest' ayni model sayilmali. Once tam eslesme, sonra taban ada (":"
# oncesi) bakilarak iki yonlu karsilastirma yapilir.
def _model_is_pulled(target: str, pulled: list[str]) -> bool:
    """Tag-agnostic membership: 'llama3' matches 'llama3:latest' and vice versa."""
    if target in pulled:
        return True
    target_base = target.split(":", 1)[0]
    for p in pulled:
        if p == target_base:
            return True
        if p.split(":", 1)[0] == target_base:
            return True
    return False


# Hangi LLM'lerin hazir (pulled) oldugunu, hangilerinin tanimli ama indirilmemis
# (pullable) oldugunu doner. Varsayilan model indirilmemisse ilk hazir modele
# duser ki frontend her zaman kullanilabilir bir varsayilan gosterebilsin.
@router.get("/models", response_model=ModelListResponse)
def get_models(_: Annotated[CurrentUser, Depends(get_current_user)]) -> ModelListResponse:
    """Return which LLMs are pulled vs. configured-but-not-yet-pulled."""
    pulled = list_pulled_models()
    configured = list_available_models()
    # Tanimli modellerden henuz indirilmemis olanlar pullable listesini olusturur.
    pullable = [m for m in configured if not _model_is_pulled(m, pulled)]
    # Yapilandirilmis varsayilan hazirsa onu kullan; degilse ilk hazir modele dus.
    default = settings.models.llm_model if _model_is_pulled(settings.models.llm_model, pulled) \
        else (pulled[0] if pulled else settings.models.llm_model)
    return ModelListResponse(available=pulled, pullable=pullable, default=default)


# Istenen modeli Ollama'dan indirir ve ilerlemeyi satir satir (NDJSON) stream eder.
# Indirme uzun surebildigi icin tek seferde beklemek yerine akis kullanilir.
# Yalnizca instructor yetkilidir.
@router.post("/models/pull")
def trigger_pull(
    payload: PullModelRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
):
    """Stream Ollama pull progress for the requested model (instructor only)."""
    if user.role != "instructor":
        raise HTTPException(status_code=403, detail="Instructor role required")

    # Ic uretici: pull_model'in urettigi her ilerleme olayini JSON satiri olarak
    # akitir. Hata olursa akis kesilmez; hata da bir JSON satiri olarak gonderilir
    # ki istemci durumu gorebilsin.
    def event_stream():
        try:
            for evt in pull_model(payload.model):
                yield json.dumps(evt) + "\n"
        except Exception as exc:  # noqa: BLE001
            log.exception("Pull failed for %s", payload.model)
            yield json.dumps({"error": str(exc)}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# ─────────────────────────────────────────────────────────────────────────
# Streaming RAG query
# ─────────────────────────────────────────────────────────────────────────
# Hibrit RAG akisinin ana endpoint'i. Kullanici sorusunu alir; once ilgili
# chunk'lari arar, baglam ve sohbet gecmisiyle bir prompt kurar, sonra LLM
# cevabini token token Server-Sent Events olarak akitir. session_id verilmez
# veya gecersizse kullanicinin General Chat'ine yazilir.
@router.post("/query")
def query_chat(
    payload: ChatQueryRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
):
    """Stream a hybrid-RAG answer as Server-Sent Events.

    Stream protocol (one JSON object per line):
      {"event":"session","data":"<resolved-session-uuid>"}
      {"event":"sources","data":"..."}
      {"event":"token","data":"..."}
      {"event":"images","data":["path1","path2"]}
      {"event":"done"}
    """
    # Hedef oturum cozulur ve kullanicinin sorusu daha akis baslamadan once
    # gecmise kaydedilir (boylece soru kaybolmaz). Ardindan guncel gecmis okunur.
    session_id = resolve_session(user.username, payload.session_id)
    save_message(session_id, user.username, "user", payload.query)
    history = load_chat_history(session_id)

    # Hibrit arama (vektor + anahtar kelime) ile en alakali chunk'lar getirilir.
    # Arama coker veya Qdrant erisilemezse istek 500 ile sonlanir.
    try:
        chunks = hybrid_search(payload.query, top_k=payload.top_k)
    except Exception as exc:  # noqa: BLE001
        log.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    # Getirilen chunk'lardan modele verilecek baglam metni, tekil gorsel listesi
    # ve kaynak satiri olusturulur.
    context_text, unique_images, sources_line = build_context(chunks)

    # Son birkac mesaj prompt'a kisa bir sohbet gecmisi olarak eklenir. Dilim
    # [-(window+1):-1] ile en son mesaj (yani simdi kaydettigimiz kullanici sorusu)
    # haric tutulur; cunku soru prompt'a ayrica "question" olarak veriliyor.
    window = settings.rag.history_window
    chat_history_text = ""
    for msg in history[-(window + 1):-1]:
        # Rol etiketleri prompt sablonuyla tutarli olacak sekilde Ingilizce yazilir.
        role_name = "Student" if msg["role"] == "user" else "Assistant"
        chat_history_text += f"{role_name}: {msg.get('content','')}\n"

    # Sablon (rag_answer) gecmis + baglam + soru ile doldurulup nihai prompt elde edilir.
    prompt_text = render_prompt(
        "rag_answer",
        history=chat_history_text,
        context=context_text,
        question=payload.query,
    )
    # Istekte model belirtilmemisse yapilandirmadaki varsayilan model kullanilir.
    model_name = payload.model or settings.models.llm_model

    # Ic uretici: cevabi NDJSON olayları halinde akitir. Once oturum ve kaynaklar
    # gonderilir, sonra LLM token'lari geldikce iletilir ve ayrica biriktirilir.
    def event_stream():
        yield json.dumps({"event": "session", "data": session_id}) + "\n"
        yield json.dumps({"event": "sources", "data": sources_line}) + "\n"
        # collected: token'lar burada toplanir ki akis bitince tam cevap uzerinde
        # gorsel ayiklama yapilip mesaj kaydedilebilsin.
        collected: list[str] = []
        # LLM akisi ortada koparsa hatayi bir olay olarak gonderip uretimi
        # durduruyoruz (yarim cevap kaydedilmesin diye return ile cikiliyor).
        try:
            for token in stream_chat(model_name, prompt_text, temperature=payload.temperature):
                collected.append(token)
                yield json.dumps({"event": "token", "data": token}) + "\n"
        except Exception as exc:  # noqa: BLE001
            log.exception("LLM streaming failed")
            yield json.dumps({"event": "error", "data": str(exc)}) + "\n"
            return

        # Akis bitti: token'lar birlestirilip tam cevap olusturulur.
        raw_answer = "".join(collected)
        # Modelin gercekten atifta bulundugu gorseller [GÖRSEL: yol] etiketlerinden
        # cikarilir; yalnizca hem getirilen hem de atif yapilan gorseller gosterilir
        # (model uydurmus olabilir, bu yuzden unique_images ile kesisim alinir).
        cited = {c.strip() for c in _IMAGE_TAG_RE.findall(raw_answer)}
        final_images = [img for img in unique_images if img in cited]
        # Etiketler kullaniciya gosterilecek metinden temizlenir (sadece gorsel olarak render edilecekler).
        final_answer = _IMAGE_TAG_RE.sub("", raw_answer).strip()

        # Nihai cevap, kaynaklar ve gorsellerle birlikte gecmise kaydedilir.
        save_message(
            session_id,
            user.username,
            "assistant",
            final_answer,
            sources=sources_line,
            images=final_images,
        )
        # Son olarak gorseller ve "done" olayi gonderilerek akis kapatilir.
        yield json.dumps({"event": "images", "data": final_images}) + "\n"
        yield json.dumps({"event": "done"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


# Ayni prompt'u birden fazla model uzerinde calistirip sonuclari karsilastirir
# (model secimi/degerlendirmesi icin). models verilmezse mevcut tum modeller
# denenir. Yalnizca instructor yetkilidir.
@router.post("/benchmark", response_model=BenchmarkResponse)
def benchmark(
    payload: BenchmarkRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
) -> BenchmarkResponse:
    if user.role != "instructor":
        raise HTTPException(status_code=403, detail="Instructor role required")
    models_to_test = payload.models or list_available_models()
    results = benchmark_models(models_to_test, payload.prompt, temperature=payload.temperature)
    return BenchmarkResponse(results=results)
