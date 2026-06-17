"""Centralized configuration for DeepCampus (v2: Qdrant + BGE-M3 + hybrid + multi-LLM).

All tunable parameters and secrets live here. Override defaults via env vars
(see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# .env dosyasini surece yukler; boylece asagidaki _env_* yardimcilari
# os.getenv ile bu degerlere ulasabilir. Modul import edilirken bir kez calisir.
load_dotenv()


# Asagidaki _env_* yardimcilari ortam degiskenlerini tip-guvenli okumak icindir.
# Ortak desen: degisken yoksa ya da gecersizse sessizce varsayilana duser ki
# hatali bir .env girisi uygulamayi cokertmesin.
def _env_str(key: str, default: str) -> str:
    # Metin ayar: env yoksa varsayilani dondur.
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    # Tamsayi ayar. Bos string'i de gecersiz sayariz (not raw), boylece
    # KEY= seklindeki bos bir tanim varsayilani ezmez.
    raw = os.getenv(key)
    if not raw:
        return default
    # Sayiya cevrilemezse (ornegin "abc") varsayilana duser, ValueError yutulur.
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    # Ondalik ayar; _env_int ile ayni bos/gecersiz koruma mantigi.
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    # Boolean ayar. Burada "raw is None" kontrolu onemli: bos string ("")
    # gecerli bir deger sayilir ve False'a esitlenir (kasitli "kapat" anlami).
    raw = os.getenv(key)
    if raw is None:
        return default
    # Yaygin dogruluk ifadelerini kabul ediyoruz; buyuk/kucuk harf ve bosluk fark etmez.
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(key: str, default: Path) -> Path:
    # Dosya yolu ayari: env varsa Path nesnesine sarar, yoksa varsayilani kullanir.
    raw = os.getenv(key)
    return Path(raw) if raw else default


# Proje kok dizini: bu dosya services/ altinda oldugu icin iki seviye yukari cikariz.
# Tum varsayilan yollar (docs, data, prompts) buna gore turetilir.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Uygulamanin kullandigi dosya/dizin yollarini tek yerde toplar.
# frozen=True: ayarlar olusturulduktan sonra degistirilemez (immutable),
# boylece calisma aninda yanlislikla ezilmez ve guvenle paylasilabilir.
@dataclass(frozen=True)
class Paths:
    # field(default_factory=...) kullaniyoruz cunku varsayilan deger her ornek
    # icin env okunarak uretilmeli; sinif tanim aninda sabitlenmemeli.
    docs: Path = field(default_factory=lambda: _env_path("DOCS_PATH", PROJECT_ROOT / "docs"))
    docs_images: Path = field(default_factory=lambda: _env_path("DOCS_IMAGES_PATH", PROJECT_ROOT / "docs_images"))
    state_file: Path = field(default_factory=lambda: _env_path("INGEST_STATE_PATH", PROJECT_ROOT / "data" / "ingest_state.json"))
    user_db: Path = field(default_factory=lambda: _env_path("USER_DB_PATH", PROJECT_ROOT / "data" / "user.db"))
    prompts: Path = field(default_factory=lambda: PROJECT_ROOT / "prompts")
    embed_cache: Path = field(default_factory=lambda: _env_path("EMBED_CACHE_PATH", PROJECT_ROOT / "data" / "embed_cache"))


# RAG hattindaki tum modellerin (LLM, embedding, gorsel/VLM) ayarlari.
# Ollama uzerinden cevap ureten LLM, BGE-M3 embedding ve Moondream2 gorsel modeli.
@dataclass(frozen=True)
class ModelSettings:
    # LLM (default model — overridable per-request via API)
    # Varsayilan sohbet modeli; API istegi basina farkli model secilebilir.
    llm_model: str = field(default_factory=lambda: _env_str("LLM_MODEL", "llama3.1:8b-instruct-q8_0"))
    # Kullanici arayuzunde secilebilecek modellerin listesi. Tek env'i virgulle
    # ayirip tuple'a ceviriyoruz; boylece coklu model tek satirda tanimlanabilir.
    available_llms: tuple = field(
        default_factory=lambda: tuple(
            _env_str(
                "AVAILABLE_LLMS",
                "llama3.1:8b-instruct-q8_0,qwen2.5:14b-instruct-q4_K_M,gemma2:9b-instruct-q4_K_M",
            ).split(",")
        )
    )
    # Ollama servisinin adresi (Docker ag adi). LLM cagrilari buraya gider.
    ollama_host: str = field(default_factory=lambda: _env_str("OLLAMA_HOST", "http://ollama:11434"))

    # Embedding (BGE-M3 produces 1024-dim dense + sparse + multi-vec; we use dense+sparse)
    # BGE-M3 yogun (dense) + seyrek (sparse) vektor uretir; hibrit aramada ikisini de kullaniyoruz.
    embedding_model: str = field(default_factory=lambda: _env_str("EMBEDDING_MODEL", "BAAI/bge-m3"))
    # "auto": GPU varsa CUDA, yoksa CPU otomatik secilir.
    embedding_device: str = field(default_factory=lambda: _env_str("EMBEDDING_DEVICE", "auto"))
    # fp16 yari hassasiyet: GPU'da hizi artirir ve bellegi azaltir.
    embedding_use_fp16: bool = field(default_factory=lambda: _env_bool("EMBEDDING_USE_FP16", True))

    # VLM (image summaries)
    # Gorsel-dil modeli: dokumanlardaki gorsellerin metin ozetini cikarir.
    vlm_model: str = field(default_factory=lambda: _env_str("VLM_MODEL", "vikhyatk/moondream2"))
    # Model surumunu sabitliyoruz ki ust surum degisikligi ciktiyi beklenmedik sekilde bozmasin.
    vlm_revision: str = field(default_factory=lambda: _env_str("VLM_REVISION", "2024-08-26"))


# Vektor veritabani (Qdrant) baglanti ayarlari. Embedding'ler burada saklanir ve aranir.
@dataclass(frozen=True)
class QdrantSettings:
    host: str = field(default_factory=lambda: _env_str("QDRANT_HOST", "qdrant"))
    # REST API portu (HTTP).
    port: int = field(default_factory=lambda: _env_int("QDRANT_PORT", 6333))
    # gRPC portu; use_grpc=True iken daha dusuk gecikme icin kullanilir.
    grpc_port: int = field(default_factory=lambda: _env_int("QDRANT_GRPC_PORT", 6334))
    # Vektorlerin tutuldugu koleksiyon (tablo karsiligi) adi.
    collection: str = field(default_factory=lambda: _env_str("QDRANT_COLLECTION", "deepcampus"))
    use_grpc: bool = field(default_factory=lambda: _env_bool("QDRANT_USE_GRPC", False))
    # Bos birakilirsa kimlik dogrulamasiz baglanir (yerel/dev kurulum).
    api_key: str = field(default_factory=lambda: _env_str("QDRANT_API_KEY", ""))


# RAG hattinin davranisini belirleyen ayar/esik degerleri:
# parcalama (chunking), getirme (retrieval), hibrit fuzyon ve uretim parametreleri.
@dataclass(frozen=True)
class RAGSettings:
    # BGE-M3 has 8192 token context; 1024 is the sweet spot for retrieval quality.
    # Parca boyutu (token). 1024 getirme kalitesi ile baglam butunlugu arasinda denge noktasi.
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1024))
    # Komsu parcalar arasi ortusme; cumlelerin parca sinirinda kesilip baglam kaybini onler.
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 128))
    # Ilk getirmede vektor aramasindan donecek aday sayisi (genis ag).
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 20))
    # Yeniden siralama (rerank) sonrasi LLM'e verilecek en iyi parca sayisi (daraltilmis kume).
    rerank_top_n: int = field(default_factory=lambda: _env_int("RERANK_TOP_N", 8))
    # Hybrid search weights (dense vs sparse) for RRF fusion.
    # Hibrit arama agirliklari: dense anlamsal benzerligi, sparse anahtar kelime eslesmesini temsil eder.
    # Ikisinin RRF ile birlesimi, toplami 1.0 olacak sekilde ayarlanir (0.6 + 0.4).
    dense_weight: float = field(default_factory=lambda: _env_float("DENSE_WEIGHT", 0.6))
    sparse_weight: float = field(default_factory=lambda: _env_float("SPARSE_WEIGHT", 0.4))
    # RRF (Reciprocal Rank Fusion) sabiti; buyuk deger ust siralarin etkisini yumusatir.
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    # Bu esigin altindaki gorseller (byte) ikon/cizgi gibi anlamsiz sayilip atlanir.
    min_image_bytes: int = field(default_factory=lambda: _env_int("MIN_IMAGE_BYTES", 15000))
    # Bu kadar karakterden kisa metin parcalari indekslenmez (gurultu temizligi).
    min_text_chars: int = field(default_factory=lambda: _env_int("MIN_TEXT_CHARS", 15))
    # LLM ureteme sicakligi; dusuk deger (0.3) daha tutarli/olgusal cevaplar verir.
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.3))
    # Modele baglam olarak tasinacak son mesaj cifti sayisi (sohbet hafizasi penceresi).
    history_window: int = field(default_factory=lambda: _env_int("HISTORY_WINDOW", 4))
    # OCR icin desteklenen diller: Turkce ve Ingilizce.
    ocr_languages: tuple = ("tr", "en")


@dataclass(frozen=True)
class AuthSettings:
    """Legacy auth knobs.

    Kept for backwards compatibility with chat_history bootstrap; the
    actual auth path runs through Keycloak now (see KeycloakSettings).
    bcrypt / JWT_SECRET fields are unused but documented in .env.example
    so downstream tooling that still reads them doesn't break.
    """
    # NOT: Bu sinif eski (legacy) auth icindir. Gercek kimlik dogrulama artik
    # Keycloak'ta yapiliyor (bkz. KeycloakSettings). Asagidaki bcrypt/jwt alanlari
    # kullanilmiyor; yalnizca eski araclarla geriye donuk uyumluluk icin duruyor.
    default_admin_username: str = field(default_factory=lambda: _env_str("ADMIN_USERNAME", "admin"))
    default_admin_password: str = field(default_factory=lambda: _env_str("ADMIN_PASSWORD", "admin123"))
    bcrypt_rounds: int = field(default_factory=lambda: _env_int("BCRYPT_ROUNDS", 12))
    jwt_secret: str = field(default_factory=lambda: _env_str("JWT_SECRET", "change-me-in-production-please"))
    jwt_ttl_hours: int = field(default_factory=lambda: _env_int("JWT_TTL_HOURS", 12))


@dataclass(frozen=True)
class KeycloakSettings:
    """Keycloak OIDC client configuration.

    `url` is the in-Docker hostname for backend-to-Keycloak calls
    (token exchange, JWKS fetch, Admin API). `public_url` is the
    browser-facing hostname embedded in Authorization Code redirects
    — the user's browser has to reach Keycloak directly, so this must
    be a host the browser can resolve (typically localhost:8080 in
    dev). `admin_*` are master-realm credentials used by the Admin
    API path for registering new users.
    """
    # KRITIK AYRIM: iki ayri Keycloak adresi var.
    # url: Docker ag ici adres; backend->Keycloak cagrilari (token, JWKS, Admin API) bunu kullanir.
    url: str = field(default_factory=lambda: _env_str("KEYCLOAK_URL", "http://keycloak:8080"))
    # public_url: tarayiciya yonlendirilen adres. Kullanici tarayicisi Docker ag adini
    # cozemeyecegi icin OAuth yonlendirmelerinde mutlaka bu (orn. localhost:8080) kullanilmali.
    public_url: str = field(default_factory=lambda: _env_str("KEYCLOAK_PUBLIC_URL", "http://localhost:8080"))
    realm: str = field(default_factory=lambda: _env_str("KEYCLOAK_REALM", "deepcampus"))
    client_id: str = field(default_factory=lambda: _env_str("KEYCLOAK_CLIENT_ID", "streamlit-app"))
    # admin_*: master realm kimlik bilgileri; yeni kullanici olusturmak icin Admin API'de kullanilir.
    admin_user: str = field(default_factory=lambda: _env_str("KEYCLOAK_ADMIN", "adminn"))
    admin_password: str = field(default_factory=lambda: _env_str("KEYCLOAK_ADMIN_PASSWORD", "admin123"))


# FastAPI backend servisinin ag ayarlari.
@dataclass(frozen=True)
class BackendSettings:
    # 0.0.0.0: konteyner icinde tum arayuzlerden gelen baglantilari dinle.
    host: str = field(default_factory=lambda: _env_str("BACKEND_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("BACKEND_PORT", 8000))
    # Frontend'in backend'e ulasmak icin kullandigi Docker ag ici adres.
    url: str = field(default_factory=lambda: _env_str("BACKEND_URL", "http://backend:8000"))


@dataclass(frozen=True)
class FrontendSettings:
    """Frontend host configuration.

    `url` is the browser-facing Streamlit address used as the OAuth
    Authorization Code redirect_uri. Must match a redirect URI accepted
    by the Keycloak `streamlit-app` client (the seeded realm uses
    `["*"]` for dev, tighten this for production).
    """
    # Tarayici tarafindaki Streamlit adresi; OAuth redirect_uri olarak kullanilir.
    # Keycloak streamlit-app istemcisinin kabul ettigi redirect URI ile eslesmeli.
    url: str = field(default_factory=lambda: _env_str("FRONTEND_URL", "http://localhost:8501"))


# Tum ayar gruplarini tek bir koke toplayan ust sinif.
# Uygulama genelinde "settings.rag.top_k" gibi tek noktadan erisim saglar.
@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    models: ModelSettings = field(default_factory=ModelSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    rag: RAGSettings = field(default_factory=RAGSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
    keycloak: KeycloakSettings = field(default_factory=KeycloakSettings)
    backend: BackendSettings = field(default_factory=BackendSettings)
    frontend: FrontendSettings = field(default_factory=FrontendSettings)


# Modul seviyesinde tekil (singleton) ayar nesnesi. Diger moduller
# "from services.config import settings" ile bu hazir ornegi paylasir.
settings = Settings()
