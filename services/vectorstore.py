"""Qdrant vector store with hybrid (dense + sparse) collection support.

Schema:
- Named dense vector "dense": 1024-d cosine (BGE-M3)
- Named sparse vector "sparse": BGE-M3 lexical weights
- Payload: source, page, type, text (for display), image_path, fingerprint
"""
from __future__ import annotations

import uuid
from typing import Iterable, Optional

from qdrant_client import QdrantClient, models
from qdrant_client.http.exceptions import UnexpectedResponse

from services.config import settings
from services.embeddings import DENSE_DIM
from services.logging_config import get_logger

log = get_logger(__name__)


def get_client() -> QdrantClient:
    qcfg = settings.qdrant
    # Default HTTP timeout (5 s) yetmiyor: büyük PDF'lerde upsert tek bir
    # istekte uçtuğunda kolayca aşıyor. 120 s sentetik tavanı zaten
    # batched upsert ile birlikte güvenli kalmaya yetiyor.
    if qcfg.use_grpc:
        return QdrantClient(
            host=qcfg.host,
            grpc_port=qcfg.grpc_port,
            prefer_grpc=True,
            api_key=qcfg.api_key or None,
            timeout=120,
        )
    return QdrantClient(
        host=qcfg.host,
        port=qcfg.port,
        api_key=qcfg.api_key or None,
        timeout=120,
    )


def ensure_collection(client: Optional[QdrantClient] = None) -> None:
    """Create the collection with named dense+sparse vectors if missing."""
    client = client or get_client()
    name = settings.qdrant.collection
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return

    log.info("Creating Qdrant collection '%s'", name)
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),
            ),
        },
    )
    # Useful payload indexes for filtering/dedup queries.
    for field_name in ("source", "fingerprint", "type"):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse:
            pass


def reset_collection() -> None:
    """Drop and recreate the collection (used for fresh-start migrations)."""
    client = get_client()
    name = settings.qdrant.collection
    try:
        client.delete_collection(collection_name=name)
        log.info("Dropped collection '%s'", name)
    except (UnexpectedResponse, ValueError):
        pass
    ensure_collection(client)


def fingerprint_exists(fingerprint: str) -> bool:
    """Return True if any point with this content fingerprint already exists."""
    client = get_client()
    ensure_collection(client)
    try:
        result, _ = client.scroll(
            collection_name=settings.qdrant.collection,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="fingerprint",
                        match=models.MatchValue(value=fingerprint),
                    )
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
    except UnexpectedResponse as exc:
        log.warning("fingerprint_exists scroll failed: %s", exc)
        return False
    return bool(result)


def upsert_chunks(
    dense_vecs: list[list[float]],
    sparse_vecs: list[dict[int, float]],
    payloads: list[dict],
    batch_size: int = 64,
) -> None:
    """Upsert chunks with dense + sparse vectors, batched to avoid HTTP timeouts.

    Tek-istek upsert büyük PDF'lerde (>500 chunk) Qdrant tarafında
    `ResponseHandlingException: timed out` veriyor. 64'erli batch'ler hem
    payload boyutunu makul tutuyor hem de hata anında ne kadar verinin
    yazıldığını görünür kılıyor.
    """
    if not dense_vecs:
        return
    client = get_client()
    ensure_collection(client)
    collection = settings.qdrant.collection
    total = len(dense_vecs)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_points = []
        for dense, sparse, payload in zip(
            dense_vecs[start:end], sparse_vecs[start:end], payloads[start:end]
        ):
            indices = list(sparse.keys())
            values = [sparse[i] for i in indices]
            batch_points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),
                    vector={
                        "dense": dense,
                        "sparse": models.SparseVector(indices=indices, values=values),
                    },
                    payload=payload,
                )
            )
        client.upsert(collection_name=collection, points=batch_points, wait=True)
        log.info("Upserted %d/%d points to '%s'", end, total, collection)


def search_dense(query_vec: list[float], limit: int) -> list[models.ScoredPoint]:
    client = get_client()
    return client.query_points(
        collection_name=settings.qdrant.collection,
        query=query_vec,
        using="dense",
        limit=limit,
        with_payload=True,
    ).points


def search_sparse(sparse: dict[int, float], limit: int) -> list[models.ScoredPoint]:
    if not sparse:
        return []
    indices = list(sparse.keys())
    values = [sparse[i] for i in indices]
    client = get_client()
    return client.query_points(
        collection_name=settings.qdrant.collection,
        query=models.SparseVector(indices=indices, values=values),
        using="sparse",
        limit=limit,
        with_payload=True,
    ).points


def list_sources() -> list[str]:
    """Return distinct source PDF names currently indexed."""
    client = get_client()
    ensure_collection(client)
    seen: set[str] = set()
    next_page = None
    while True:
        result, next_page = client.scroll(
            collection_name=settings.qdrant.collection,
            limit=256,
            with_payload=["source"],
            with_vectors=False,
            offset=next_page,
        )
        for point in result:
            src = (point.payload or {}).get("source")
            if src:
                seen.add(src)
        if next_page is None:
            break
    return sorted(seen)
