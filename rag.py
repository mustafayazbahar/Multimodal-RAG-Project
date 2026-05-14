"""Retrieval helpers: query-prefixing for E5 + optional cross-encoder reranking."""
from __future__ import annotations

from functools import lru_cache
from typing import List

from langchain_core.documents import Document

from config import settings
from logging_config import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_reranker():
    """Lazy-load the cross-encoder reranker. Returns None on failure."""
    try:
        from sentence_transformers import CrossEncoder

        log.info("Loading reranker '%s'...", settings.models.reranker_model)
        return CrossEncoder(settings.models.reranker_model)
    except Exception as exc:  # noqa: BLE001 - reranker is optional
        log.warning("Reranker unavailable, falling back to plain retrieval: %s", exc)
        return None


def retrieve(vectorstore, query: str, k: int, rerank_top_n: int | None = None) -> List[Document]:
    """Retrieve docs from the vectorstore and optionally rerank with a cross-encoder.

    E5 requires the "query: " prefix for queries (and "passage: " for stored chunks).
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    docs = retriever.invoke(f"query: {query}")

    if not settings.models.reranker_enabled or not docs:
        return docs

    reranker = _load_reranker()
    if reranker is None:
        return docs

    pairs = [(query, d.page_content) for d in docs]
    try:
        scores = reranker.predict(pairs)
    except Exception as exc:  # noqa: BLE001
        log.warning("Reranker scoring failed, returning unranked docs: %s", exc)
        return docs

    ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
    top_n = rerank_top_n or settings.rag.rerank_top_n
    return [doc for doc, _ in ranked[:top_n]]
