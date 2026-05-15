"""Ingest endpoints: upload PDF, run pipeline, status, image fetch."""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse

from backend.schemas import IngestRunResponse, IngestStatusResponse
from backend.security import CurrentUser, get_current_user, require_instructor
from services.config import settings
from services.llm import evict_model
from services.logging_config import get_logger
from services.vectorstore import list_sources

log = get_logger(__name__)

router = APIRouter(prefix="/ingest", tags=["ingest"])

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize_filename(name: str) -> str:
    base = os.path.basename(name or "").strip().lstrip(".")
    if not base:
        base = "upload.pdf"
    cleaned = _SAFE_FILENAME_RE.sub("_", base)
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned[:255]


@router.get("/status", response_model=IngestStatusResponse)
def status_endpoint(_: Annotated[CurrentUser, Depends(require_instructor)]) -> IngestStatusResponse:
    docs_dir = settings.paths.docs
    docs_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in docs_dir.iterdir() if p.is_file())
    state_path = settings.paths.state_file
    doc_count = 0
    if state_path.exists():
        try:
            doc_count = len(json.loads(state_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, OSError):
            doc_count = 0
    return IngestStatusResponse(
        files=files,
        indexed_sources=list_sources(),
        documents_in_state=doc_count,
    )


@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_pdf(
    file: UploadFile = File(...),
    _: CurrentUser = Depends(require_instructor),
) -> dict:
    docs_dir = settings.paths.docs
    docs_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(file.filename or "")
    save_path = docs_dir / safe_name
    try:
        save_path.resolve().relative_to(docs_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid filename") from exc

    contents = file.file.read()
    save_path.write_bytes(contents)
    return {"saved_as": safe_name, "bytes": len(contents)}


_INGEST_RESULT_MARKER = "INGESTION_RESULT:"


def _parse_ingest_summary(stdout):
    for line in reversed(stdout.splitlines()):
        idx = line.find(_INGEST_RESULT_MARKER)
        if idx == -1:
            continue
        payload = line[idx + len(_INGEST_RESULT_MARKER):].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return None


@router.post("/run", response_model=IngestRunResponse)
def run_ingest(_: Annotated[CurrentUser, Depends(require_instructor)]) -> IngestRunResponse:
    """Run the ingestion pipeline as a subprocess."""
    evict_model(settings.models.llm_model)
    try:
        result = subprocess.run(
            [sys.executable, "-m", "services.ingestion"],
            capture_output=True,
            text=True,
            check=False,
            cwd=str(Path(__file__).resolve().parents[2]),
        )
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to launch ingestion: {exc}") from exc

    if result.returncode != 0:
        log.error("Ingestion subprocess failed: %s", result.stderr[-2000:])
        raise HTTPException(status_code=500, detail=result.stderr[-1000:] or "Ingestion failed")

    summary = _parse_ingest_summary(result.stdout) or {}
    return IngestRunResponse(
        processed=summary.get("processed", 0),
        skipped=summary.get("skipped", 0),
        duplicates=summary.get("duplicates", 0),
        errors=summary.get("errors", 0),
        chunks=summary.get("chunks"),
        details=summary.get("details", []),
    )


@router.get("/image")
def get_image(path: str, _: CurrentUser = Depends(get_current_user)) -> FileResponse:
    """Serve an extracted image, restricted to the docs_images dir."""
    abs_path = Path(path).resolve()
    base = settings.paths.docs_images.resolve()
    try:
        abs_path.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path outside images dir") from exc
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(abs_path)