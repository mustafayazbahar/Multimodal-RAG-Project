"""Centralized configuration for DeepCampus (v2: Qdrant + BGE-M3 + hybrid + multi-LLM).

All tunable parameters and secrets live here. Override defaults via env vars
(see .env.example).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_str(key: str, default: str) -> str:
    return os.getenv(key, default)


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_bool(key: str, default: bool) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key)
    return Path(raw) if raw else default


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Paths:
    docs: Path = field(default_factory=lambda: _env_path("DOCS_PATH", PROJECT_ROOT / "docs"))
    docs_images: Path = field(default_factory=lambda: _env_path("DOCS_IMAGES_PATH", PROJECT_ROOT / "docs_images"))
    state_file: Path = field(default_factory=lambda: _env_path("INGEST_STATE_PATH", PROJECT_ROOT / "data" / "ingest_state.json"))
    user_db: Path = field(default_factory=lambda: _env_path("USER_DB_PATH", PROJECT_ROOT / "data" / "user.db"))
    prompts: Path = field(default_factory=lambda: PROJECT_ROOT / "prompts")
    embed_cache: Path = field(default_factory=lambda: _env_path("EMBED_CACHE_PATH", PROJECT_ROOT / "data" / "embed_cache"))


@dataclass(frozen=True)
class ModelSettings:
    # LLM (default model — overridable per-request via API)
    llm_model: str = field(default_factory=lambda: _env_str("LLM_MODEL", "llama3.1:8b-instruct-q8_0"))
    available_llms: tuple = field(
        default_factory=lambda: tuple(
            _env_str(
                "AVAILABLE_LLMS",
                "llama3.1:8b-instruct-q8_0,qwen2.5:14b-instruct-q4_K_M,gemma2:9b-instruct-q4_K_M",
            ).split(",")
        )
    )
    ollama_host: str = field(default_factory=lambda: _env_str("OLLAMA_HOST", "http://ollama:11434"))

    # Embedding (BGE-M3 produces 1024-dim dense + sparse + multi-vec; we use dense+sparse)
    embedding_model: str = field(default_factory=lambda: _env_str("EMBEDDING_MODEL", "BAAI/bge-m3"))
    embedding_device: str = field(default_factory=lambda: _env_str("EMBEDDING_DEVICE", "auto"))
    embedding_use_fp16: bool = field(default_factory=lambda: _env_bool("EMBEDDING_USE_FP16", True))

    # VLM (image summaries)
    vlm_model: str = field(default_factory=lambda: _env_str("VLM_MODEL", "vikhyatk/moondream2"))
    vlm_revision: str = field(default_factory=lambda: _env_str("VLM_REVISION", "2024-08-26"))


@dataclass(frozen=True)
class QdrantSettings:
    host: str = field(default_factory=lambda: _env_str("QDRANT_HOST", "qdrant"))
    port: int = field(default_factory=lambda: _env_int("QDRANT_PORT", 6333))
    grpc_port: int = field(default_factory=lambda: _env_int("QDRANT_GRPC_PORT", 6334))
    collection: str = field(default_factory=lambda: _env_str("QDRANT_COLLECTION", "deepcampus"))
    use_grpc: bool = field(default_factory=lambda: _env_bool("QDRANT_USE_GRPC", False))
    api_key: str = field(default_factory=lambda: _env_str("QDRANT_API_KEY", ""))


@dataclass(frozen=True)
class RAGSettings:
    # BGE-M3 has 8192 token context; 1024 is the sweet spot for retrieval quality.
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 1024))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 128))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 20))
    rerank_top_n: int = field(default_factory=lambda: _env_int("RERANK_TOP_N", 8))
    # Hybrid search weights (dense vs sparse) for RRF fusion.
    dense_weight: float = field(default_factory=lambda: _env_float("DENSE_WEIGHT", 0.6))
    sparse_weight: float = field(default_factory=lambda: _env_float("SPARSE_WEIGHT", 0.4))
    rrf_k: int = field(default_factory=lambda: _env_int("RRF_K", 60))
    # 15 KB filtresi ders kitaplarındaki ufak diyagramları topluca eliyordu;
    # 2 KB ile ikon/dekor süzülürken anlamlı diyagramlar yakalanıyor.
    min_image_bytes: int = field(default_factory=lambda: _env_int("MIN_IMAGE_BYTES", 2000))
    min_text_chars: int = field(default_factory=lambda: _env_int("MIN_TEXT_CHARS", 15))
    # Bir sayfada bu kadar vektör çizim primitif'i varsa ve sayfadan raster
    # resim çıkmadıysa, sayfayı PNG olarak rasterize edip "figür" gibi
    # işliyoruz — Kurose & Ross gibi kitaplardaki vektör diyagramları
    # PyMuPDF'in get_images() yöntemi göremiyor.
    page_render_drawing_threshold: int = field(
        default_factory=lambda: _env_int("PAGE_RENDER_DRAWING_THRESHOLD", 20)
    )
    page_render_dpi: int = field(default_factory=lambda: _env_int("PAGE_RENDER_DPI", 150))
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.3))
    history_window: int = field(default_factory=lambda: _env_int("HISTORY_WINDOW", 4))
    ocr_languages: tuple = ("tr", "en")


@dataclass(frozen=True)
class AuthSettings:
    """Legacy auth knobs.

    Kept for backwards compatibility with chat_history bootstrap; the
    actual auth path runs through Keycloak now (see KeycloakSettings).
    bcrypt / JWT_SECRET fields are unused but documented in .env.example
    so downstream tooling that still reads them doesn't break.
    """
    default_admin_username: str = field(default_factory=lambda: _env_str("ADMIN_USERNAME", "admin"))
    default_admin_password: str = field(default_factory=lambda: _env_str("ADMIN_PASSWORD", "admin123"))
    bcrypt_rounds: int = field(default_factory=lambda: _env_int("BCRYPT_ROUNDS", 12))
    jwt_secret: str = field(default_factory=lambda: _env_str("JWT_SECRET", "change-me-in-production-please"))
    jwt_ttl_hours: int = field(default_factory=lambda: _env_int("JWT_TTL_HOURS", 12))


@dataclass(frozen=True)
class KeycloakSettings:
    """Keycloak OIDC client configuration.

    `url` is the in-Docker hostname for backend-to-Keycloak calls; the
    browser talks to localhost:8080 directly. `admin_*` are master-realm
    credentials used by the Admin API path for registering new users.
    """
    url: str = field(default_factory=lambda: _env_str("KEYCLOAK_URL", "http://keycloak:8080"))
    realm: str = field(default_factory=lambda: _env_str("KEYCLOAK_REALM", "deepcampus"))
    client_id: str = field(default_factory=lambda: _env_str("KEYCLOAK_CLIENT_ID", "streamlit-app"))
    admin_user: str = field(default_factory=lambda: _env_str("KEYCLOAK_ADMIN", "adminn"))
    admin_password: str = field(default_factory=lambda: _env_str("KEYCLOAK_ADMIN_PASSWORD", "admin123"))


@dataclass(frozen=True)
class BackendSettings:
    host: str = field(default_factory=lambda: _env_str("BACKEND_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("BACKEND_PORT", 8000))
    url: str = field(default_factory=lambda: _env_str("BACKEND_URL", "http://backend:8000"))


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    models: ModelSettings = field(default_factory=ModelSettings)
    qdrant: QdrantSettings = field(default_factory=QdrantSettings)
    rag: RAGSettings = field(default_factory=RAGSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)
    keycloak: KeycloakSettings = field(default_factory=KeycloakSettings)
    backend: BackendSettings = field(default_factory=BackendSettings)


settings = Settings()
