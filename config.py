"""Centralized configuration for DeepCampus.

All tunable parameters and environment-driven secrets live here.
Override any default via environment variables (see .env.example).
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
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or raw == "":
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


PROJECT_ROOT = Path(__file__).resolve().parent


def _env_path(key: str, default: Path) -> Path:
    raw = os.getenv(key)
    return Path(raw) if raw else default


@dataclass(frozen=True)
class Paths:
    docs: Path = field(default_factory=lambda: _env_path("DOCS_PATH", PROJECT_ROOT / "docs"))
    chroma_db: Path = field(default_factory=lambda: _env_path("CHROMA_DB_PATH", PROJECT_ROOT / "chroma_db"))
    docs_images: Path = field(default_factory=lambda: _env_path("DOCS_IMAGES_PATH", PROJECT_ROOT / "docs_images"))
    state_file: Path = field(default_factory=lambda: _env_path("INGEST_STATE_PATH", PROJECT_ROOT / "ingest_state.json"))
    user_db: Path = field(default_factory=lambda: _env_path("USER_DB_PATH", PROJECT_ROOT / "user.db"))
    prompts: Path = field(default_factory=lambda: PROJECT_ROOT / "prompts")


@dataclass(frozen=True)
class ModelSettings:
    llm_model: str = field(default_factory=lambda: _env_str("LLM_MODEL", "llama3.1:8b-instruct-q8_0"))
    ollama_host: str = field(default_factory=lambda: _env_str("OLLAMA_HOST", "http://ollama:11434"))
    embedding_model: str = field(default_factory=lambda: _env_str("EMBEDDING_MODEL", "intfloat/multilingual-e5-base"))
    vlm_model: str = field(default_factory=lambda: _env_str("VLM_MODEL", "vikhyatk/moondream2"))
    vlm_revision: str = field(default_factory=lambda: _env_str("VLM_REVISION", "2024-08-26"))
    reranker_model: str = field(default_factory=lambda: _env_str("RERANKER_MODEL", "BAAI/bge-reranker-base"))
    reranker_enabled: bool = field(default_factory=lambda: _env_bool("RERANKER_ENABLED", True))


@dataclass(frozen=True)
class RAGSettings:
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 300))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 50))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 15))
    rerank_top_n: int = field(default_factory=lambda: _env_int("RERANK_TOP_N", 6))
    min_image_bytes: int = field(default_factory=lambda: _env_int("MIN_IMAGE_BYTES", 15000))
    min_text_chars: int = field(default_factory=lambda: _env_int("MIN_TEXT_CHARS", 15))
    temperature: float = field(default_factory=lambda: _env_float("TEMPERATURE", 0.3))
    history_window: int = field(default_factory=lambda: _env_int("HISTORY_WINDOW", 4))
    ocr_languages: tuple = ("tr", "en")


@dataclass(frozen=True)
class AuthSettings:
    default_admin_username: str = field(default_factory=lambda: _env_str("ADMIN_USERNAME", "admin"))
    default_admin_password: str = field(default_factory=lambda: _env_str("ADMIN_PASSWORD", "admin123"))
    bcrypt_rounds: int = field(default_factory=lambda: _env_int("BCRYPT_ROUNDS", 12))


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    models: ModelSettings = field(default_factory=ModelSettings)
    rag: RAGSettings = field(default_factory=RAGSettings)
    auth: AuthSettings = field(default_factory=AuthSettings)


settings = Settings()
