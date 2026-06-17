"""Pydantic schemas for the backend API."""
# Backend API'sinin istek (request) ve cevap (response) govdelerini tanimlayan
# Pydantic semalari. FastAPI bu siniflarla otomatik dogrulama ve serlestirme yapar.
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# Parola ile giris (password grant) icin gonderilen kullanici adi + parola.
class LoginRequest(BaseModel):
    username: str
    password: str


# Yeni kullanici kaydi istegi. Keycloak Admin API email zorunlu kildigi icin
# email burada da zorunlu; ad/soyad istege bagli.
class RegisterRequest(BaseModel):
    """Keycloak Admin API requires an email; firstName/lastName are
    optional but populated when the UI collects them."""
    username: str
    password: str
    email: str
    first_name: Optional[str] = ""
    last_name: Optional[str] = ""


# Basarili giris/kayit sonrasi frontend'e donen token paketi (rol ve kullanici
# adi ile birlikte). Tum giris yollari ayni sekli dondurur.
class TokenResponse(BaseModel):
    access_token: str
    # id_token is only present on browser-initiated logins (the OAuth
    # Code flow); password-grant responses leave it None. We need it
    # to drive a silent Keycloak logout — without it the IdP shows a
    # "do you want to log out?" confirm screen.
    # id_token yalnizca tarayici uzerinden OAuth Code akisinda gelir; parola
    # grant'inda None kalir. Sessiz (onaysiz) Keycloak cikisini tetiklemek icin
    # gerekir; olmazsa IdP "cikis yapmak istiyor musun?" onay ekrani gosterir.
    id_token: Optional[str] = None
    token_type: str = "bearer"
    role: str
    username: str


# OAuth Code akisinda tarayicinin geri dondurdugu yetki kodu (code) ve ilk auth
# URL'i olusturulurken kullanilan redirect_uri. Keycloak ikisinin eslesmesini dogrular.
class ExchangeCodeRequest(BaseModel):
    """Browser hands back the `?code=` from a Keycloak redirect plus the
    redirect_uri that was used when building the original auth URL —
    Keycloak validates the two match."""
    code: str
    redirect_uri: str


# Frontend'in yonlendirilecegi Keycloak URL'ini (login/logout) tasiyan cevap.
class OauthUrlResponse(BaseModel):
    url: str


# Tek bir sohbet mesaji. sources: cevabin dayandigi kaynaklar; images: cevapta
# atifta bulunulan gorsel yollari (varsa).
class ChatMessage(BaseModel):
    role: str
    content: str
    sources: str = ""
    images: list[str] = []


# Belirli bir oturumun gecmisi: mesaj listesi + ait oldugu session_id.
class ChatHistoryResponse(BaseModel):
    messages: list[ChatMessage]
    session_id: str


# Bir sohbet oturumunun ("topic") ozet bilgisi. is_default: General Chat mi?
class ChatSessionInfo(BaseModel):
    session_id: str
    title: str
    is_default: bool = False
    created_at: Optional[str] = None


# Kullanicinin tum sohbet oturumlarini listeleyen cevap.
class ChatSessionListResponse(BaseModel):
    sessions: list[ChatSessionInfo]


# Yeni oturum olusturma istegi. Baslik bos olamaz, en fazla 120 karakter.
class CreateSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


# Var olan oturumu yeniden adlandirma istegi (ayni baslik kisitlari gecerli).
class RenameSessionRequest(BaseModel):
    title: str = Field(min_length=1, max_length=120)


# RAG sorgu istegi. session_id, model gibi alanlar istege bagli; temperature ve
# top_k Field ile gecerli araliga (yaraticilik ve getirilecek chunk sayisi) sinirlanir.
class ChatQueryRequest(BaseModel):
    query: str
    # Optional — server falls back to the user's General Chat when
    # omitted or unknown.
    # Istege bagli; verilmezse veya tanimsizsa sunucu kullanicinin General Chat'ine duser.
    session_id: Optional[str] = None
    model: Optional[str] = None
    temperature: float = Field(default=0.3, ge=0.0, le=1.0)
    top_k: int = Field(default=20, ge=1, le=50)


# Ingestion durum cevabi: docs/ klasorundeki dosyalar, indekslenmis kaynaklar ve
# durum dosyasindaki dokuman sayisi.
class IngestStatusResponse(BaseModel):
    files: list[str]
    indexed_sources: list[str]
    documents_in_state: int


# Ingestion calistirma sonucu sayaclari (islenen, atlanan, tekrar eden, hatali +
# uretilen chunk sayisi ve dosya bazli detaylar).
class IngestRunResponse(BaseModel):
    processed: int
    skipped: int
    duplicates: int
    errors: int
    chunks: Optional[int] = None
    details: list[dict]


# Model listesi cevabi: Ollama'da hazir (pulled) modeller, tanimli ama henuz
# indirilmemis (pullable) modeller ve varsayilan model.
class ModelListResponse(BaseModel):
    available: list[str]   # actually pulled in Ollama (ready to use)
    pullable: list[str]    # configured in AVAILABLE_LLMS but not pulled yet
    default: str


# Ollama'dan belirli bir modeli indirme (pull) istegi.
class PullModelRequest(BaseModel):
    model: str


# Birden fazla modeli ayni prompt ile karsilastirma (benchmark) istegi.
# models verilmezse mevcut tum modeller test edilir.
class BenchmarkRequest(BaseModel):
    prompt: str
    models: Optional[list[str]] = None
    temperature: float = 0.3


# Benchmark sonuclari: her model icin olcum verileri liste halinde.
class BenchmarkResponse(BaseModel):
    results: list[dict]
