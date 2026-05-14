"""Tests for the hybrid RRF fusion math (no Qdrant needed)."""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class FakePoint:
    id: str
    payload: dict | None = None


def _load_fuse():
    import importlib

    sys.modules.pop("services.fusion", None)
    return importlib.import_module("services.fusion").reciprocal_rank_fusion


def test_rrf_combines_two_lists():
    fuse = _load_fuse()
    dense = [FakePoint("a"), FakePoint("b"), FakePoint("c")]
    sparse = [FakePoint("b"), FakePoint("d"), FakePoint("a")]
    result = fuse(dense, sparse, dense_w=0.5, sparse_w=0.5, k=60)
    ids = [p.id for p, _ in result]
    # 'b' appears at rank 0 of sparse + rank 1 of dense → should beat 'a'
    # (rank 0 of dense + rank 2 of sparse).
    assert ids[0] in {"a", "b"}
    assert "d" in ids


def test_rrf_weights_bias_results():
    fuse = _load_fuse()
    dense = [FakePoint("only_dense")]
    sparse = [FakePoint("only_sparse")]
    # Boost dense → "only_dense" should outrank "only_sparse"
    result = fuse(dense, sparse, dense_w=0.9, sparse_w=0.1, k=60)
    assert result[0][0].id == "only_dense"
    # Flip the bias
    result_rev = fuse(dense, sparse, dense_w=0.1, sparse_w=0.9, k=60)
    assert result_rev[0][0].id == "only_sparse"


def test_rrf_handles_empty_lists():
    fuse = _load_fuse()
    assert fuse([], [], dense_w=0.5, sparse_w=0.5, k=60) == []
    only_dense = fuse([FakePoint("x")], [], dense_w=0.5, sparse_w=0.5, k=60)
    assert only_dense[0][0].id == "x"
