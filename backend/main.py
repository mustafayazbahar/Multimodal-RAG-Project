"""FastAPI entrypoint for DeepCampus backend."""
# DeepCampus backend'inin FastAPI giris noktasi. Router'lari (auth/chat/ingest)
# uygulamaya baglar ve uygulama acilirken gerekli baslangic islerini yapar.
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.routers import auth as auth_router
from backend.routers import chat as chat_router
from backend.routers import ingest as ingest_router
from services.auth import create_chat_table
from services.logging_config import get_logger
from services.vectorstore import ensure_collection

log = get_logger(__name__)


# Uygulama yasam dongusu (lifespan): yield'den onceki kisim acilista, sonrasi
# kapanista calisir. Burada sadece acilis hazirligi yapilir.
@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Only the chat_history table is local now — user accounts live in
    # Keycloak (see services/keycloak_auth.py). No SQLite users to seed
    # at boot anymore.
    # Sadece chat_history tablosu yerelde tutuluyor; kullanici hesaplari
    # Keycloak'ta yasiyor. Bu yuzden acilista yalnizca chat tablosu kuruluyor.
    create_chat_table()
    # Qdrant koleksiyonunun varligini garanti et. Hata yutuluyor cunku ilk
    # acilista Qdrant servisi henuz ayakta olmayabilir; bu durumda backend
    # cokmemeli, sadece uyari loglanmali (koleksiyon sonradan da olusabilir).
    try:
        ensure_collection()
    except Exception as exc:  # noqa: BLE001 - Qdrant may not be up yet on first boot
        log.warning("Qdrant collection check failed at startup: %s", exc)
    log.info("DeepCampus backend ready.")
    yield


# Uygulama nesnesi olusturuluyor; lifespan ile acilis/kapanis kancalari baglaniyor.
app = FastAPI(title="DeepCampus Backend", version="2.0.0", lifespan=lifespan)

# CORS ayari: frontend (Streamlit) farkli bir origin'den istek attigi icin
# tarayicinin engellememesi adina tum origin/method/header'lara izin veriliyor.
# Token Authorization header'i ile gonderildiginden cookie tabanli kimlik (credentials)
# kapali tutuluyor.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


# Saglik kontrolu endpoint'i: servisin ayakta olup olmadigini anlamak icin kullanilir.
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


# Modullere ayrilmis router'lar tek tek uygulamaya kaydediliyor.
app.include_router(auth_router.router)
app.include_router(chat_router.router)
app.include_router(ingest_router.router)
