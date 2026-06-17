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


# Aramadan donen tek bir sonucu (chunk) temsil eden veri sinifi.
# Qdrant payload alanlarini (metin, kaynak, sayfa, tur, gorsel yolu vb.) ve
# fusion sonrasi nihai skoru bir arada tasiyarak ust katmanlara dusuk-bagli
# (decoupled) temiz bir arayuz sunar.
@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int
    type: str
    image_path: Optional[str]
    fingerprint: Optional[str]
    score: float

    # Qdrant'tan donen ham bir "point" + skor degerini RetrievedChunk'a cevirir.
    # Payload eksik/None olabilecegi icin tum alanlar guvenli varsayilanlarla okunur.
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


# Hibrit arama: sorguyu embed eder, dense ve sparse aramalarini paralel calistirir,
# sonuclari RRF ile tek siralamaya birlestirir ve en iyi N chunk'i doner.
def hybrid_search(query: str, top_k: Optional[int] = None) -> list[RetrievedChunk]:
    """Run dense + sparse search in BGE-M3, fuse with RRF, return top-N chunks."""
    # top_k verilmediyse ayardaki varsayilan deger kullanilir.
    k = top_k or settings.rag.top_k
    # Sorguyu tek geciste hem dense hem sparse temsile cevir.
    dense_vec, sparse_vec = embed_query(query)

    # Over-fetch: fusion'in adillik kazanmasi icin her listeden ihtiyactan FAZLA
    # aday cekilir (en az 30, ya da k'nin 2 kati). Boylece yalnizca tek yontemde
    # ust siralarda cikan iyi adaylar fusion oncesi elenmez.
    over_fetch = max(k * 2, 30)
    dense_hits = search_dense(dense_vec, limit=over_fetch)
    sparse_hits = search_sparse(sparse_vec, limit=over_fetch)

    # Iki sirali listeyi agirlikli RRF ile birlestir; agirliklar ve k sabiti ayardan gelir.
    fused = reciprocal_rank_fusion(
        dense_hits,
        sparse_hits,
        dense_w=settings.rag.dense_weight,
        sparse_w=settings.rag.sparse_weight,
        k=settings.rag.rrf_k,
    )

    # Birlesik siralamadan yalnizca en iyi rerank_top_n kaydi al ve RetrievedChunk'a cevir.
    rerank_n = settings.rag.rerank_top_n
    return [RetrievedChunk.from_point(p, s) for p, s in fused[:rerank_n]]


# Bulunan chunk'lari LLM'e verilecek tek bir baglam metnine donusturur; ayrica
# benzersiz gorsel yollarini ve kullaniciya gosterilecek kaynak satirini toplar.
# Donus: (baglam_metni, gorsel_yollari, kaynaklar_satiri)
def build_context(chunks: list[RetrievedChunk]) -> tuple[str, list[str], str]:
    """Format chunks into LLM context, collect unique image paths and sources line."""
    context_text = ""
    found_images: list[str] = []
    sources_list: list[str] = []
    for c in chunks:
        # Sayfa numaralari 0-tabanli saklandigi icin kullanici gosteriminde +1 yapilir.
        page_label = c.page + 1
        # Gorsel chunk'lar ile metin chunk'lari baglamda farkli etiketlerle isaretlenir;
        # boylece LLM bir ozetin gorselden mi yoksa metinden mi geldigini ayirt edebilir.
        if c.type == "image":
            sources_list.append(f"[IMAGE] {c.source} (Page {page_label})")
            # Gorsel yolu varsa hem gosterilecek listeye ekle hem de ozetini baglama koy.
            if c.image_path:
                found_images.append(c.image_path)
                context_text += f"[IMAGE SUMMARY - ID: {c.image_path}]: {c.text}\n\n"
        else:
            sources_list.append(f"[TEXT] {c.source} (Page {page_label})")
            context_text += f"[TEXT - Page {page_label}]: {c.text}\n\n"
    return (
        context_text,
        # dict.fromkeys: sirayi koruyarak yinelenen gorsel yollarini eler.
        list(dict.fromkeys(found_images)),
        # Kaynaklari tekillestirip alfabetik sirala, virgulle birlestirip tek satir yap.
        ", ".join(sorted(set(sources_list))),
    )
