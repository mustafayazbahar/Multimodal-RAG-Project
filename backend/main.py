"""FastAPI entrypoint for DeepCampus backend."""
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


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Only the chat_history table is local now — user accounts live in
    # Keycloak (see services/keycloak_auth.py). No SQLite users to seed
    # at boot anymore.
    create_chat_table()
    try:
        ensure_collection()
    except Exception as exc:  # noqa: BLE001 - Qdrant may not be up yet on first boot
        log.warning("Qdrant collection check failed at startup: %s", exc)
    log.info("DeepCampus backend ready.")
    yield


app = FastAPI(title="DeepCampus Backend", version="2.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


app.include_router(auth_router.router)
app.include_router(chat_router.router)
app.include_router(ingest_router.router)
