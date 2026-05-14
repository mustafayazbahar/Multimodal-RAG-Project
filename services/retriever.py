"""Hybrid retrieval: dense (BGE-M3) + sparse (BM25-like) with RRF fusion."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from services.config import settings
from services.embeddings import embed_query
from services.fusion import reciprocal_rank_fusion
from services.logging_config import get_logger
from services.vectorstore import search_dense, search_sparse

__all__ = ["RetrievedChunk", "hybrid_search", "build_context", "reciprocal_rank_fusion"]

log = get_logger(__name__)


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    type: str
    image_path: Optional[str]
    fingerprint: Optional[str]
    score: float

    @classmethod
    def from_point(cls, point, score: float) -> "RetrievedChunk":
        p = point.payload or {}
        return cls(
            text=p.get("text", ""),
            source=p.get("source", "Unknown"),
            page=int(p.get("page", 0)),
            type=p.get("type", "text"),
            image_path=p.get("image_path"),
            fingerprint=p.get("fingerprint"),
            score=score,
        )


def hybrid_search(query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
    """Run dense + sparse search in BGE-M3, fuse with RRF, return top-N chunks."""
    k = top_k or settings.rag.top_k
    dense_vec, sparse_vec = embed_query(query)

    over_fetch = max(k * 2, 30)
    dense_hits = search_dense(dense_vec, limit=over_fetch)
    sparse_hits = search_sparse(sparse_vec, limit=over_fetch)

    fused = reciprocal_rank_fusion(
        dense_hits,
        sparse_hits,
        dense_w=settings.rag.dense_weight,
        sparse_w=settings.rag.sparse_weight,
        k=settings.rag.rrf_k,
    )

    rerank_n = settings.rag.rerank_top_n
    return [RetrievedChunk.from_point(p, s) for p, s in fused[:rerank_n]]


def build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[str], str]:
    """Format chunks into LLM context, collect unique image paths and sources line."""
    context_text = ""
    found_images: list[str] = []
    sources_list: list[str] = []
    for c in chunks:
        page_label = c.page + 1
        if c.type == "image":
            sources_list.append(f"[IMAGE] {c.source} (Page {page_label})")
            if c.image_path:
                found_images.append(c.image_path)
                context_text += f"[IMAGE SUMMARY - ID: {c.image_path}]: {c.text}\n\n"
        else:
            sources_list.append(f"[TEXT] {c.source} (Page {page_label})")
            context_text += f"[TEXT - Page {page_label}]: {c.text}\n\n"
    return (
        context_text,
        list(dict.fromkeys(found_images)),
        ", ".join(sorted(set(sources_list))),
    )
