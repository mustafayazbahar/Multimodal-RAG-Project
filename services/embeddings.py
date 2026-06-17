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


# Modelin calisacagi donanimi (cihazi) belirler. "auto" verilirse sirasiyla
# CUDA (NVIDIA GPU) ve MPS (Apple Silicon) denenir, hicbiri yoksa CPU'ya duser.
def _resolve_device(device: str) -> str:
    # Kullanici acikca bir cihaz belirttiyse otomatik tespite gerek yok.
    if device != "auto":
        return device
    try:
        import torch  # noqa: WPS433

        # GPU varsa onceligi GPU'ya ver: embedding cikarimi GPU'da cok daha hizli.
        if torch.cuda.is_available():
            return "cuda"
        # Apple Silicon (M serisi) makinelerde MPS hizlandirmasini kullan.
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        # torch kurulu degilse sessizce CPU'ya geri don.
        pass
    return "cpu"


# BGE-M3 modelini tembel (lazy) yukler ve tekil (singleton) olarak onbellekler.
# lru_cache(maxsize=1) sayesinde model surec boyunca yalnizca BIR KEZ yuklenir;
# sonraki cagrilar ayni model nesnesini doner. Model yuklemesi pahali oldugundan
# bu tekrari onlemek bellek ve sure acisindan kritiktir.
@lru_cache(maxsize=1)
def get_embedder():
    """Lazy-load BGE-M3. Returns a FlagEmbedding BGEM3FlagModel instance."""
    from FlagEmbedding import BGEM3FlagModel

    # Hedef cihazi (cuda/mps/cpu) ayardan cozumle.
    device = _resolve_device(settings.models.embedding_device)
    log.info(
        "Loading BGE-M3 (%s) on %s, fp16=%s",
        settings.models.embedding_model,
        device,
        settings.models.embedding_use_fp16,
    )
    # fp16 (yarim hassasiyet) yalnizca GPU/MPS uzerinde aciliyor: CPU'da fp16
    # genelde desteklenmez veya yavaslatir, bu yuzden device == "cpu" ise kapatilir.
    model = BGEM3FlagModel(
        settings.models.embedding_model,
        use_fp16=settings.models.embedding_use_fp16 and device != "cpu",
        device=device,
    )
    log.info("BGE-M3 ready.")
    return model


# FlagEmbedding'in lexical (seyrek) ciktisini standart {token_id: agirlik}
# sozlugune cevirir. Anahtar/degerler kesin tiplere (int/float) zorlanir; cunku
# Qdrant SparseVector formati saf int indeks ve float deger bekler.
def _to_sparse_dict(lexical_weights) -> dict[int, float]:
    """Normalize FlagEmbedding's lexical output into {token_id: weight}."""
    # Bos/None cikti gelirse bos sozluk don (seyrek vektor yok demektir).
    if not lexical_weights:
        return {}
    return {int(token_id): float(weight) for token_id, weight in lexical_weights.items()}


# Belge parcalarini (chunk/passage) toplu olarak embed eder. Indeksleme
# asamasinda kullanilir: her chunk icin hem dense (anlamsal) hem sparse (sozcuksel)
# temsil tek bir model.encode geçisinde uretilir.
def embed_passages(texts: Iterable[str], batch_size: int = 8) -> tuple[list[list[float]], list[dict[int, float]]]:
    """Embed documents (passages) into dense + sparse representations."""
    # Generator gelebilecegi icin listeye sabitliyoruz (iki kez gezecegiz / len alacagiz).
    texts = list(texts)
    # Bos giris -> bos sonuc; gereksiz model yuklemesini de onler.
    if not texts:
        return [], []
    model = get_embedder()
    # max_length = chunk_size + 64: chunk metnine ek olarak ozel token'lar ve
    # ufak tasmalar icin pay birakir, boylece icerik kirpilmaz.
    out = model.encode(
        texts,
        batch_size=batch_size,
        max_length=settings.rag.chunk_size + 64,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,  # ColBERT cok-vektor ciktisi gerekmiyor (bellek tasarrufu).
    )
    # numpy dizilerini, JSON/Qdrant'a yazilabilir saf Python listelerine cevir.
    dense = [vec.tolist() for vec in out["dense_vecs"]]
    sparse = [_to_sparse_dict(w) for w in out["lexical_weights"]]
    return dense, sparse


# Tek bir kullanici sorusunu embed eder ve (dense_vektor, sparse_sozluk) doner.
# Arama (retrieval) asamasinda cagrilir; cikan vektorler Qdrant'ta dense ve
# sparse aramalarini beslemek icin kullanilir.
def embed_query(text: str) -> tuple[list[float], dict[int, float]]:
    """Embed a single query string. Returns (dense_vec, sparse_dict)."""
    model = get_embedder()
    # Tek metin oldugundan [text] olarak listelenir; sorgular kisa oldugu icin
    # max_length=512 yeterli (chunk'lardan farkli olarak buyuk pay gerekmez).
    out = model.encode(
        [text],
        batch_size=1,
        max_length=512,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    # Tek elemanli ciktidan ilk (ve tek) sonucu al.
    dense = out["dense_vecs"][0].tolist()
    sparse = _to_sparse_dict(out["lexical_weights"][0])
    return dense, sparse


DENSE_DIM = 1024  # BGE-M3'un sabit dense vektor boyutu (koleksiyon semasinda kullanilir).
