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


# Yapilandirilmis ayarlara gore bir Qdrant istemcisi olusturup doner.
# gRPC veya HTTP secimi ayardan gelir; her iki yolda da uzun timeout kullanilir.
def get_client() -> QdrantClient:
    qcfg = settings.qdrant
    # Varsayılan 5 sn timeout büyük PDF'lerde yetmiyor: ders kitabı binlerce
    # chunk üretiyor, tek istekteki upsert kolayca aşıyor. 120 sn tavanı
    # batched upsert ile birlikte güvenli kalmaya yetiyor.
    # gRPC daha dusuk gecikme ve buyuk yuklerde daha iyi verim saglar.
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


# Koleksiyon yoksa, isimli dense + sparse vektorlerle birlikte olusturur.
# Idempotent calisir: koleksiyon zaten varsa hicbir sey yapmadan doner, bu yuzden
# her indeksleme/arama oncesi guvenle cagrilabilir.
def ensure_collection(client: Optional[QdrantClient] = None) -> None:
    """Create the collection with named dense+sparse vectors if missing."""
    client = client or get_client()
    name = settings.qdrant.collection
    # Mevcut koleksiyon adlarini kume olarak topla ve hedef adi kontrol et.
    existing = {c.name for c in client.get_collections().collections}
    if name in existing:
        return

    log.info("Creating Qdrant collection '%s'", name)
    # Hibrit arama icin iki isimli vektor tanimlanir:
    #  - "dense": 1024 boyutlu, cosine benzerligi (BGE-M3 anlamsal vektor)
    #  - "sparse": BGE-M3 lexical agirliklari (BM25 benzeri sozcuksel eslesme)
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=False),  # Seyrek indeks RAM'de tutulur (daha hizli arama).
            ),
        },
    )
    # Useful payload indexes for filtering/dedup queries.
    # Bu alanlar uzerinde payload indeksi olusturmak, filtreli aramalari ve
    # dedup (fingerprint) sorgularini hizlandirir. Index varsa Qdrant hata
    # dondurebilir; bu durumda gormezden gelinir (idempotentlik icin).
    for field_name in ("source", "fingerprint", "type"):
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except UnexpectedResponse:
            pass


# Koleksiyonu silip yeniden olusturur. Sema degisikligi ya da sifirdan yeniden
# indeksleme (fresh-start migration) gerektiginde tum veriyi temizlemek icin kullanilir.
def reset_collection() -> None:
    """Drop and recreate the collection (used for fresh-start migrations)."""
    client = get_client()
    name = settings.qdrant.collection
    try:
        client.delete_collection(collection_name=name)
        log.info("Dropped collection '%s'", name)
    except (UnexpectedResponse, ValueError):
        # Koleksiyon zaten yoksa silme hata verebilir; yeni olusturulacagi icin onemsiz.
        pass
    ensure_collection(client)


# Verilen icerik parmak izine (fingerprint) sahip bir kayit zaten var mi diye bakar.
# Ayni chunk'in tekrar tekrar indekslenmesini onlemek (dedup) icin kullanilir.
def fingerprint_exists(fingerprint: str) -> bool:
    """Return True if any point with this content fingerprint already exists."""
    client = get_client()
    ensure_collection(client)
    try:
        # scroll ile filtreye uyan kayitlari tara; limit=1 yeterli cunku tek bir
        # eslesme bile "var" demek icin kafidir (varlik kontrolu).
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
            with_payload=False,   # Sadece varlik kontrolu; payload/vektor cekmeye gerek yok.
            with_vectors=False,
        )
    except UnexpectedResponse as exc:
        # Hata durumunda "yok" varsayilir; bu, en kotu ihtimalle tekrar indekslemeye
        # yol acar ama yanlislikla veri atlamaktan daha guvenlidir.
        log.warning("fingerprint_exists scroll failed: %s", exc)
        return False
    # result bos degilse en az bir eslesme var demektir.
    return bool(result)


# Chunk'lari (dense + sparse vektor + payload) Qdrant'a yazar/gunceller (upsert).
# Buyuk PDF'lerde HTTP timeout'unu onlemek icin kayitlar batch_size'lik gruplara
# bolunerek gonderilir.
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
    yazıldığını log'da görünür kılıyor.
    """
    # Yazilacak veri yoksa erken cik.
    if not dense_vecs:
        return
    client = get_client()
    ensure_collection(client)
    collection = settings.qdrant.collection
    total = len(dense_vecs)

    # Veriyi [start, end) araliklarina bolerek batch_size'lik gruplar halinde isle.
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)  # Son grup tasmasin diye total ile sinirla.
        batch_points = []
        # Bu gruptaki dense vektor, sparse sozluk ve payload'lari hizalayarak gez.
        for dense, sparse, payload in zip(
            dense_vecs[start:end], sparse_vecs[start:end], payloads[start:end]
        ):
            # Sparse sozlugu Qdrant'in bekledigi paralel indeks/deger dizilerine ayir.
            # indices[i] token kimligini, values[i] o token'in agirligini tutar.
            indices = list(sparse.keys())
            values = [sparse[i] for i in indices]
            batch_points.append(
                models.PointStruct(
                    id=str(uuid.uuid4()),  # Her kayda benzersiz UUID; cakismasiz ekleme saglar.
                    vector={
                        # Tek kayitta hem dense hem sparse vektor birlikte saklanir (hibrit).
                        "dense": dense,
                        "sparse": models.SparseVector(indices=indices, values=values),
                    },
                    payload=payload,
                )
            )
        # wait=True: yazma diske islenene kadar bekle; boylece hemen ardindan
        # yapilan aramalar yeni eklenen kayitlari gorebilir (tutarlilik).
        client.upsert(collection_name=collection, points=batch_points, wait=True)
        # Ilerlemeyi logla: hata halinde ne kadarinin yazildigi gorunur olur.
        log.info("Upserted %d/%d points to '%s'", end, total, collection)


# Dense (anlamsal) vektor uzerinden en yakin "limit" kadar kaydi getirir.
# "using=dense" ile koleksiyondaki isimli dense vektor alani hedeflenir.
def search_dense(query_vec: list[float], limit: int) -> list[models.ScoredPoint]:
    client = get_client()
    return client.query_points(
        collection_name=settings.qdrant.collection,
        query=query_vec,
        using="dense",
        limit=limit,
        with_payload=True,  # Sonuc metni/kaynagi icin payload da dondur.
    ).points


# Sparse (sozcuksel/BM25 benzeri) vektor uzerinden en uygun "limit" kadar kaydi getirir.
# Sorgunun seyrek temsili bostan donerse arama yapmadan bos liste doner.
def search_sparse(sparse: dict[int, float], limit: int) -> list[models.ScoredPoint]:
    if not sparse:
        return []
    # Sozlugu Qdrant SparseVector icin paralel indeks/deger dizilerine cevir.
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


# Su an indekslenmis olan benzersiz kaynak (PDF) adlarini doner.
# Koleksiyon buyuk olabileceginden tum kayitlar sayfalama (scroll) ile gezilir.
def list_sources() -> list[str]:
    """Return distinct source PDF names currently indexed."""
    client = get_client()
    ensure_collection(client)
    seen: set[str] = set()
    next_page = None  # Sayfalama imleci; Qdrant her cagrida bir sonraki ofseti doner.
    # Tum koleksiyonu 256'sarlik sayfalarla gez; next_page None olunca biter.
    while True:
        result, next_page = client.scroll(
            collection_name=settings.qdrant.collection,
            limit=256,
            with_payload=["source"],  # Yalnizca "source" alanini cek (gereksiz veri tasinmasin).
            with_vectors=False,
            offset=next_page,
        )
        # Bu sayfadaki kayitlarin kaynak adlarini kumeye ekle (otomatik tekilleştirme).
        for point in result:
            src = (point.payload or {}).get("source")
            if src:
                seen.add(src)
        # Daha fazla sayfa yoksa donguden cik.
        if next_page is None:
            break
    return sorted(seen)
