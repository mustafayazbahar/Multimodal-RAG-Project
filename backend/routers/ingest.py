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
from services.vectorstore import list_sources, reset_collection

log = get_logger(__name__)

# Bu modulun tum endpoint'leri /ingest on eki altinda toplanir.
router = APIRouter(prefix="/ingest", tags=["ingest"])

# Dosya adinda izin verilmeyen (harf/rakam/.-_ disindaki) karakterleri yakalar.
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


# Yuklenen dosya adini guvenli hale getirir. Path-traversal'a (../ gibi) karsi
# sadece taban ad alinir, bastaki noktalar atilir, tehlikeli karakterler "_" ile
# degistirilir, .pdf uzantisi garanti edilir ve ad 255 karakterle sinirlandirilir.
def _sanitize_filename(name: str) -> str:
    # basename + lstrip(".") gizli/dizin kacisi denemelerini etkisiz birakir.
    base = os.path.basename(name or "").strip().lstrip(".")
    # Ad tamamen bossa makul bir varsayilan kullanilir.
    if not base:
        base = "upload.pdf"
    cleaned = _SAFE_FILENAME_RE.sub("_", base)
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned[:255]


# Ingestion durumunu doner: docs/ klasorundeki dosyalar, Qdrant'ta indekslenmis
# kaynaklar ve durum dosyasindaki dokuman sayisi. Yalnizca instructor erisebilir.
@router.get("/status", response_model=IngestStatusResponse)
def status_endpoint(_: Annotated[CurrentUser, Depends(require_instructor)]) -> IngestStatusResponse:
    docs_dir = settings.paths.docs
    # Klasor henuz yoksa olusturulur ki ilk cagrida hata vermesin.
    docs_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in docs_dir.iterdir() if p.is_file())
    state_path = settings.paths.state_file
    # Durum dosyasi bozuk/okunamaz olsa bile endpoint cokmemeli; sayac 0 kalir.
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


# Bir PDF dosyasini docs/ klasorune yukler. Yalnizca instructor yetkilidir.
@router.post("/upload", status_code=status.HTTP_201_CREATED)
def upload_pdf(
    file: UploadFile = File(...),
    _: CurrentUser = Depends(require_instructor),
) -> dict:
    docs_dir = settings.paths.docs
    docs_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _sanitize_filename(file.filename or "")
    save_path = docs_dir / safe_name
    # Ikinci bir guvenlik katmani: temizlenmis ad bile cozumlendiginde docs/
    # disina cikiyorsa (path-traversal) istek reddedilir.
    try:
        save_path.resolve().relative_to(docs_dir.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid filename") from exc

    # Dosya icerigi okunup diske yazilir; kaydedilen ad ve boyut geri bildirilir.
    contents = file.file.read()
    save_path.write_bytes(contents)
    return {"saved_as": safe_name, "bytes": len(contents)}


# Ingestion alt sureci, ozet sonucunu stdout'a bu isaretle baslayan bir satir
# olarak yazar; ana surec bu satiri ayiklayarak sonuclari okur.
_INGEST_RESULT_MARKER = "INGESTION_RESULT:"


# Alt surecin ciktisinda en SON INGESTION_RESULT satirini bulup JSON'a cevirir.
# Sondan basa taranir cunku gecerli olan en son yazilan ozettir. Satir bozuksa
# None doner (cagiran taraf bos ozet gibi ele alir).
def _parse_ingest_summary(stdout: str) -> dict | None:
    """Find the last INGESTION_RESULT line in subprocess output and parse it."""
    for line in reversed(stdout.splitlines()):
        idx = line.find(_INGEST_RESULT_MARKER)
        if idx == -1:
            continue
        payload = line[idx + len(_INGEST_RESULT_MARKER):].strip()
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            log.warning("Could not parse ingest summary line: %s", payload[:200])
            return None
    return None


# Ingestion (PDF parse + embedding + indeksleme) hattini ayri bir alt surec
# olarak calistirir. Yalnizca instructor yetkilidir.
@router.post("/run", response_model=IngestRunResponse)
def run_ingest(_: Annotated[CurrentUser, Depends(require_instructor)]) -> IngestRunResponse:
    """Run the ingestion pipeline as a subprocess."""
    # Ingestion VLM/embedding icin bellek/GPU isteyecegi icin once LLM bellekten
    # bosaltilir (kaynak cakismasini onlemek icin).
    evict_model(settings.models.llm_model)
    # Hatti ayri bir surecte calistirmak agir/uzun isi API surecinden izole eder.
    # cwd proje koku olarak ayarlanir ki "services.ingestion" modulu bulunabilsin.
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

    # Alt surec hata koduyla dondiyse stderr'in son kismi loglanir ve istemciye
    # kisaltilmis hata mesaji 500 olarak iletilir.
    if result.returncode != 0:
        log.error("Ingestion subprocess failed: %s", result.stderr[-2000:])
        raise HTTPException(status_code=500, detail=result.stderr[-1000:] or "Ingestion failed")

    # Ozet ayiklanir; bulunamazsa bos sozluk kullanilarak sayaclar 0 dondurulur.
    summary = _parse_ingest_summary(result.stdout) or {}
    return IngestRunResponse(
        processed=summary.get("processed", 0),
        skipped=summary.get("skipped", 0),
        duplicates=summary.get("duplicates", 0),
        errors=summary.get("errors", 0),
        chunks=summary.get("chunks", 0),
        details=summary.get("details", []),
    )


# Ingestion sirasinda Moondream (VLM) tarafindan uretilip kaydedilen tum gorsel
# aciklamalarini doner. UI'da VLM'in her figur icin ne dedigini gostermek (dogrulama
# + demo) icin kullanilir. Henuz ingest calismadiysa veya dosya bozuksa bos liste doner.
@router.get("/image-summaries")
def get_image_summaries(
    _: CurrentUser = Depends(get_current_user),
) -> list[dict]:
    """Return every Moondream-generated caption persisted by ingestion.

    Surfaced so the UI can show what the VLM actually said about each
    figure (sanity check + demo). The file is created on each
    /ingest/run; if no ingest has run yet, returns an empty list.
    """
    path = settings.paths.state_file.parent / "image_summaries.json"
    if not path.exists():
        return []
    # Dosya okunamaz/bozuksa endpoint cokmemeli; bos liste donulur.
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    # Beklenen tip liste degilse (dosya beklenmedik bicimde) yine bos liste donulur.
    if not isinstance(data, list):
        return []
    return data


# Bilgi tabanini sifirlar: Qdrant koleksiyonunu siler VE ingest_state.json
# durum dosyasini temizler. Boylece bir sonraki ingest tum PDF'leri yeni sayar.
# Diskten silinen bir dosyayi yeniden indekslemek icin gerekir; cunku tekillik
# (dedup) icin asil kaynak Qdrant'tir, sadece dosyayi silmek yeniden saymaya yetmez.
# Yalnizca instructor yetkilidir.
@router.post("/reset", status_code=status.HTTP_200_OK)
def reset_knowledge_base(
    _: Annotated[CurrentUser, Depends(require_instructor)],
) -> dict:
    """Drop the Qdrant collection AND wipe ingest_state.json.

    After this call the next ingest treats every PDF in docs/ as new.
    Use when you want to re-index after deleting a file from disk —
    Qdrant is the source of truth for dedup, so simply removing the
    file is not enough to make a re-upload count as fresh.
    """
    reset_collection()
    # Durum dosyasi silinemese bile islem kritik degildir; sadece uyari loglanir
    # (koleksiyon zaten dusurulmus durumda).
    state_path = settings.paths.state_file
    if state_path.exists():
        try:
            state_path.unlink()
        except OSError as exc:
            log.warning("Could not delete state file %s: %s", state_path, exc)
    log.info("Knowledge base reset: collection dropped + state file cleared")
    return {"status": "reset", "collection": settings.qdrant.collection}


# Ingestion'in cikardigi bir gorseli servis eder; erisim docs_images klasoruyle
# sinirlidir. Yetki modeli: kimligi dogrulanmis HER kullanici (instructor veya
# student) gorselleri gorebilir. Bu bilincli bir tercihtir; cunku chat cevaplari
# figurleri [GÖRSEL: yol] etiketiyle gosterir ve ogrencilerin makaleyi okumak
# icin bu atiflari render etmesi gerekir. Path-traversal yine docs_images'a
# sabitlendiginden alakasiz dosya okumalari engellenir.
@router.get("/image")
def get_image(
    path: str,
    _: CurrentUser = Depends(get_current_user),
) -> FileResponse:
    """Serve an extracted image, restricted to the docs_images dir.

    Auth model rationale (PR #8 feedback):
    Any authenticated user — instructor OR student — may view images.
    This is intentional: chat answers cite figures via [GÖRSEL: path]
    tags, and students need to render those citations to read the
    paper. Students already have read access to the indexed content
    via the chat endpoint, so the image endpoint inherits the same
    trust boundary. Path-traversal is still pinned to docs_images so
    no unrelated filesystem reads are possible.
    """
    # Istenen yol cozumlenip docs_images'in altinda mi diye dogrulanir; degilse
    # (klasor disi erisim denemesi) 400 verilir.
    abs_path = Path(path).resolve()
    base = settings.paths.docs_images.resolve()
    try:
        abs_path.relative_to(base)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path outside images dir") from exc
    # Yol gecerli ama dosya yoksa 404 doner.
    if not abs_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(abs_path)