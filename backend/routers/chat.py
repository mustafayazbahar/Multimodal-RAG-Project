"""Chat endpoints: history, clear, streaming query."""
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
    ModelListResponse,
)
from backend.security import CurrentUser, get_current_user
from services.auth import clear_chat_history, load_chat_history, save_message
from services.config import settings
from services.llm import benchmark_models, list_available_models, render_prompt, stream_chat
from services.logging_config import get_logger
from services.retriever import build_context, hybrid_search

log = get_logger(__name__)

router = APIRouter(prefix="/chat", tags=["chat"])

_IMAGE_TAG_RE = re.compile(r"\[(?:GÖRSEL|IMAGE|RESIM|RESİM):\s*(.*?)\]", re.IGNORECASE)


@router.get("/history", response_model=ChatHistoryResponse)
def get_history(user: Annotated[CurrentUser, Depends(get_current_user)]) -> ChatHistoryResponse:
    messages = [ChatMessage(**m) for m in load_chat_history(user.username)]
    return ChatHistoryResponse(messages=messages)


@router.delete("/history", status_code=status.HTTP_204_NO_CONTENT)
def delete_history(user: Annotated[CurrentUser, Depends(get_current_user)]) -> None:
    clear_chat_history(user.username)


@router.get("/models", response_model=ModelListResponse)
def get_models(_: Annotated[CurrentUser, Depends(get_current_user)]) -> ModelListResponse:
    return ModelListResponse(
        available=list_available_models(),
        default=settings.models.llm_model,
    )


@router.post("/query")
def query_chat(
    payload: ChatQueryRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
):
    """Stream a hybrid-RAG answer as Server-Sent Events.

    Stream protocol (one JSON object per line):
      {"event":"sources","data":"..."}
      {"event":"token","data":"..."}
      {"event":"images","data":["path1","path2"]}
      {"event":"done"}
    """
    save_message(user.username, "user", payload.query)
    history = load_chat_history(user.username)

    try:
        chunks = hybrid_search(payload.query, top_k=payload.top_k)
    except Exception as exc:  # noqa: BLE001
        log.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {exc}") from exc

    context_text, unique_images, sources_line = build_context(chunks)

    window = settings.rag.history_window
    chat_history_text = ""
    for msg in history[-(window + 1):-1]:
        role_name = "Student" if msg["role"] == "user" else "Assistant"
        chat_history_text += f"{role_name}: {msg.get('content','')}\n"

    prompt_text = render_prompt(
        "rag_answer",
        history=chat_history_text,
        context=context_text,
        question=payload.query,
    )
    model_name = payload.model or settings.models.llm_model

    def event_stream():
        yield json.dumps({"event": "sources", "data": sources_line}) + "\n"
        collected: list[str] = []
        try:
            for token in stream_chat(model_name, prompt_text, temperature=payload.temperature):
                collected.append(token)
                yield json.dumps({"event": "token", "data": token}) + "\n"
        except Exception as exc:  # noqa: BLE001
            log.exception("LLM streaming failed")
            yield json.dumps({"event": "error", "data": str(exc)}) + "\n"
            return

        raw_answer = "".join(collected)
        cited = {c.strip() for c in _IMAGE_TAG_RE.findall(raw_answer)}
        final_images = [img for img in unique_images if img in cited]
        final_answer = _IMAGE_TAG_RE.sub("", raw_answer).strip()

        save_message(
            user.username,
            "assistant",
            final_answer,
            sources=sources_line,
            images=final_images,
        )
        yield json.dumps({"event": "images", "data": final_images}) + "\n"
        yield json.dumps({"event": "done"}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")


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
