"""Pydantic schemas for the backend API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(LoginRequest):
    pass


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


class ChatMessage(BaseModel):
    role: str
    content: str
    sources: str = ""
    images: list[str] = []


class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessage]


class ChatQueryRequest(BaseModel):
    query: str
    model: Optional[str] = None
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=1, le=50)


class IngestStatusResponse(BaseModel):
    files: list[str]
    indexed_sources: list[str]
    documents_in_state: int


class IngestRunResponse(BaseModel):
    processed: int
    skipped: int
    duplicates: int
    errors: int
    chunks: Optional[int] = None
    details: list[dict]


class ModelListResponse(BaseModel):
    available: list[str]   # actually pulled in Ollama (ready to use)
    pullable: list[str]    # configured in AVAILABLE_LLMS but not pulled yet
    default: str


class PullModelRequest(BaseModel):
    model: str


class BenchmarkRequest(BaseModel):
    prompt: str
    models: Optional[list[str]] = None
    temperature: float = 0.3


class BenchmarkResponse(BaseModel):
    results: list[dict]
