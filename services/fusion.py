"""Pure-Python rank fusion utilities (no Qdrant dependency).

Kept separate so it can be imported in tests without dragging in the vector
store client.
"""
from __future__ import annotations

from typing import Iterable


def reciprocal_rank_fusion(
    dense_hits: Iterable,
    sparse_hits: Iterable,
    dense_w: float,
    sparse_w: float,
    k: int,
) -> list[tuple[object, float]]:
    """Combine two ranked lists via weighted Reciprocal Rank Fusion.

    score(d) = sum_over_lists( weight_i / (k + rank_i(d)) )
    Items not present in a list contribute 0 from that list.
    """
    scores: dict[str, float] = {}
    point_by_id: dict[str, object] = {}

    for rank, point in enumerate(dense_hits):
        pid = str(point.id)
        scores[pid] = scores.get(pid, 0.0) + dense_w / (k + rank + 1)
        point_by_id[pid] = point

    for rank, point in enumerate(sparse_hits):
        pid = str(point.id)
        scores[pid] = scores.get(pid, 0.0) + sparse_w / (k + rank + 1)
        point_by_id.setdefault(pid, point)

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(point_by_id[pid], score) for pid, score in ranked]
