"""Pydantic schemas for the backend API."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class LoginRequest(BaseModel):
    username: str
    password: str


class RegisterRequest(BaseModel):
    """Keycloak Admin API requires an email; firstName/lastName are
    optional but populated when the UI collects them."""
    username: str
    password: str
    email: str
    first_name: Optional[str] = ""
    last_name: Optional[str] = ""


class TokenResponse(BaseModel):
    access_token: str
    # id_token is only present on browser-initiated logins (the OAuth
    # Code flow); password-grant responses leave it None. We need it
    # to drive a silent Keycloak logout — without it the IdP shows a
    # "do you want to log out?" confirm screen.
    id_token: Optional[str] = None
    token_type: str = "bearer"
    role: str
    username: str


class ExchangeCodeRequest(BaseModel):
    """Browser hands back the `?code=` from a Keycloak redirect plus the
    redirect_uri that was used when building the original auth URL —
    Keycloak validates the two match."""
    code: str
    redirect_uri: str


class OauthUrlResponse(BaseModel):
    url: str


class ChatMessage(BaseModel):
    role: str
    content: str
    sources: str = ""
    images: list[str] = []


class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessage]
    session_id: str


class ChatSessionInfo(BaseModel):
    session_id: str
    title: str
    is_default: bool = False
    created_at: Optional[str] = None


class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionInfo]


class CreateSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


class ChatQueryRequest(BaseModel):
    query: str
    # Optional — server falls back to the user's General Chat when
    # omitted or unknown.
    session_id: Optional[str] = None
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
