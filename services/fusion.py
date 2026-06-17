"""Pure-Python rank fusion utilities (no Qdrant dependency).

Kept separate so it can be imported in tests without dragging in the vector
store client.
"""
from __future__ import annotations

from typing import Iterable


# Agirlikli Reciprocal Rank Fusion (RRF): iki sirali listeyi (dense ve sparse)
# tek birlesik siralamaya indirger. Her aday icin, gectigi her listede
# agirlik / (k + rank) puani toplanir. Mutlak benzerlik skorlari yerine SIRA
# (rank) kullanildigindan, olcekleri farkli iki yontem adil bicimde birlesir.
# k sabiti ust siralardaki kucuk rank farklarini yumusatir (buyuk k -> daha duz dagilim).
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
    # scores: aday kimligi -> birikmis RRF puani
    # point_by_id: aday kimligi -> ham point nesnesi (sonucta geri dondurmek icin)
    scores: dict[str, float] = {}
    point_by_id: dict[str, object] = {}

    # Dense listesi: enumerate ile 0-tabanli sira alinir; formulde "+1" ile
    # 1-tabanli rank'a cevrilir (ilk sira rank=1 olur, payda hicbir zaman k+0 olmaz).
    for rank, point in enumerate(dense_hits):
        pid = str(point.id)
        # Mevcut puana dense katkisini EKLE; aday her iki listede de varsa katkilar toplanir.
        scores[pid] = scores.get(pid, 0.0) + dense_w / (k + rank + 1)
        point_by_id[pid] = point

    # Sparse listesi icin ayni RRF katkisi, bu kez sparse agirligi ile eklenir.
    for rank, point in enumerate(sparse_hits):
        pid = str(point.id)
        scores[pid] = scores.get(pid, 0.0) + sparse_w / (k + rank + 1)
        # setdefault: nesne dense'te zaten kaydedildiyse uzerine yazma (ayni point).
        point_by_id.setdefault(pid, point)

    # Adaylari birikmis puana gore azalan sirada sirala (en yuksek puan en ustte).
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    # Kimlikleri tekrar ham point nesnelerine eslestirerek (point, skor) listesi dondur.
    return [(point_by_id[pid], score) for pid, score in ranked]
