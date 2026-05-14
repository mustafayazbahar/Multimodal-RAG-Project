"""BGE-M3 embedding service: dense (1024-dim) + sparse (lexical) outputs.

Uses the official FlagEmbedding package which produces both representations
in a single forward pass. The sparse output is a {token_id: weight} dict that
maps directly to Qdrant's SparseVector format.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Iterable

from services.config import settings
from services.logging_config import get_logger

log = get_logger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def get_embedder():
    """Lazy-load BGE-M3. Returns a FlagEmbedding BGEM3FlagModel instance."""
    from FlagEmbedding import BGEM3FlagModel

    device = _resolve_device(settings.models.embedding_device)
    log.info(
        "Loading BGE-M3 (%s) on %s, fp16=%s",
        settings.models.embedding_model,
        device,
        settings.models.embedding_use_fp16,
    )
    model = BGEM3FlagModel(
        settings.models.embedding_model,
        use_fp16=settings.models.embedding_use_fp16 and device != "cpu",
        device=device,
    )
    log.info("BGE-M3 ready.")
    return model


def _to_sparse_dict(lexical_weights) -> dict[int, float]:
    """Normalize FlagEmbedding's lexical output into {token_id: weight}."""
    if not lexical_weights:
        return {}
    return {int(token_id): float(weight) for token_id, weight in lexical_weights.items()}


def embed_passages(texts: Iterable[str], batch_size: int = 8) -> tuple[list[list[float]], list[dict[int, float]]]:
    """Embed documents (passages) into dense + sparse representations."""
    texts = list(texts)
    if not texts:
        return [], []
    model = get_embedder()
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=settings.rag.chunk_size + 64,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = [vec.tolist() for vec in out["dense_vecs"]]
    sparse = [_to_sparse_dict(w) for w in out["lexical_weights"]]
    return dense, sparse


def embed_query(text: str) -> tuple[list[float], dict[int, float]]:
    """Embed a single query string. Returns (dense_vec, sparse_dict)."""
    model = get_embedder()
    out = model.encode(
        [text],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    dense = out["dense_vecs"][0].tolist()
    sparse = _to_sparse_dict(out["lexical_weights"][0])
    return dense, sparse


DENSE_DIM = 1024  # BGE-M3 fixed dimensionality
