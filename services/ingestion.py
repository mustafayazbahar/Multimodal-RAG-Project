"""PDF ingestion pipeline (Qdrant + BGE-M3 + multi-layer dedup).

Pipeline per PDF:
1. Compute multi-layer fingerprint (file / content / metadata).
2. Check ingest_state.json + Qdrant `fingerprint` payload for duplicates.
3. Extract per-page text (PyMuPDF) with EasyOCR fallback for scans.
4. Extract embedded images, dedup by MD5, summarize via Moondream2 (VLM).
5. Chunk text with a HuggingFace tokenizer (BGE-M3 tokenizer at chunk_size).
6. Embed chunks with BGE-M3 (dense + sparse) and upsert into Qdrant.
"""
from __future__ import annotations

import gc
import hashlib
import io
import json
import re
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber
import torch
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from services.config import settings
from services.embeddings import embed_passages
from services.logging_config import get_logger
from services.pdf_fingerprint import compute_fingerprint
from services.vectorstore import ensure_collection, fingerprint_exists, upsert_chunks

log = get_logger(__name__)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_state() -> dict:
    path = settings.paths.state_file
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        log.warning("Corrupt state file at %s — resetting.", path)
        return {}


def _save_state(state: dict) -> None:
    path = settings.paths.state_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _persist_image_summaries(documents: list[Document]) -> Path:
    """Write a human-readable JSON of every Moondream-generated image
    summary into data/image_summaries.json.

    Pulled out of the pipeline because users want to *see* what the VLM
    actually said about each figure (debugging, sanity-checking, demo).
    The file is merged with any prior entries keyed by image_path so
    repeat ingests don't lose history.
    """
    summaries: list[dict] = []
    for doc in documents:
        meta = doc.metadata
        if meta.get("type") != "image":
            continue
        img_path = meta.get("image_path")
        if not img_path:
            continue
        summary_text = doc.page_content
        if summary_text.startswith("[IMAGE SUMMARY]: "):
            summary_text = summary_text[len("[IMAGE SUMMARY]: "):]
        summaries.append(
            {
                "source": meta.get("source"),
                "page": meta.get("page", 0),
                "image_path": img_path,
                "summary": summary_text,
                "fingerprint": meta.get("fingerprint"),
            }
        )

    out_path = settings.paths.state_file.parent / "image_summaries.json"
    existing: list[dict] = []
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except (json.JSONDecodeError, OSError):
            existing = []

    # Dedupe by image_path; new entries win.
    merged: dict[str, dict] = {item.get("image_path"): item for item in existing if item.get("image_path")}
    for item in summaries:
        merged[item["image_path"]] = item

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(list(merged.values()), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info("Wrote %d image summaries to %s", len(merged), out_path)
    return out_path


def _is_duplicate_against_state(fp, state: dict) -> tuple[bool, str]:
    """Check the local state file for any matching prior fingerprint.

    Only file_hash (exact byte match) and content_hash (first 3 pages of
    text + metadata) trigger a duplicate verdict. metadata_hash alone is
    intentionally NOT a dedup signal (PR #8 review): generic titles like
    "Progress Report" or "Lecture Notes" collide between unrelated
    documents, and rejecting a fresh upload because two files happen to
    share a title is worse than re-indexing the same PDF.
    """
    for filename, entry in state.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("file_hash") == fp.file_hash:
            return True, f"identical file (matches '{filename}')"
        if fp.content_hash and entry.get("content_hash") == fp.content_hash:
            return True, f"same content as '{filename}' (different filename or PDF stamp)"
    return False, ""


def _load_vlm(device: str):
    model_id = settings.models.vlm_model
    revision = settings.models.vlm_revision
    log.info("Loading VLM '%s' on %s", model_id, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, revision=revision)
    if device in ("cuda", "mps"):
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def _load_ocr(device: str):
    import easyocr

    return easyocr.Reader(list(settings.rag.ocr_languages), gpu=device in ("cuda", "mps"), verbose=False)


# PDF'lerin içinde sıkça JPEG 2000 (.jp2 / .jpx) gömülü oluyor — özellikle
# taranmış kitaplarda. Tarayıcılar bu formatı render etmiyor (Chrome dahil
# desteklemiyor), bu yüzden chat'te resim bozuk görünüyor. Pillow ile
# açabildiğimiz her şeyi PNG'ye çeviriyoruz; açamazsak orijinal byte'ı
# yazıp en azından dedup/summary akışına sokuyoruz.
_BROWSER_SAFE_EXTS = {"png", "jpg", "jpeg", "gif", "webp", "bmp"}

# Sayfa metninde figür altyazısı arayan regex. "Figure 4.1:", "Fig. 12",
# "Şekil 3.2", "Şekil 7:" gibi varyantları yakalar. Bu en güvenilir
# diyagram göstergesi — eşik tabanlı yaklaşım referans sayfalarını yanlış
# tarıyordu; figür altyazısı yalnızca gerçek figür sayfalarında bulunur.
_FIGURE_CAPTION_RE = re.compile(
    r"(?im)\b(?:figure|fig\.?|şekil|sekil)\s*\d+(?:[\.\:\-]\s*\d+)*\b"
)


def _save_browser_safe_image(image_bytes: bytes, dest_no_ext: Path, fallback_ext: str) -> Path:
    """Image byte'larını tarayıcıda gösterilebilen bir formatta diske yazar.

    JPX/JP2 gibi PDF'e gömülü ama browser'ın anlamadığı formatları PNG'ye
    çevirir. Pillow image'i açamazsa orijinal byte'ı orijinal uzantıyla
    yazıp dosya yolunu döner — VLM çoğu formatı yine de işliyor, sadece
    chat'te görünmüyor.
    """
    ext = fallback_ext.lower().lstrip(".")
    if ext in _BROWSER_SAFE_EXTS:
        out = dest_no_ext.with_suffix(f".{ext}")
        out.write_bytes(image_bytes)
        return out
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGBA" if "A" in img.mode else "RGB")
            out = dest_no_ext.with_suffix(".png")
            img.save(out, format="PNG", optimize=True)
            return out
    except (OSError, ValueError) as exc:
        log.warning(
            "Could not re-encode image to PNG (%s); writing original .%s. "
            "Browser may not render it.",
            exc, ext,
        )
        out = dest_no_ext.with_suffix(f".{ext}")
        out.write_bytes(image_bytes)
        return out


def _summarize_image(model, tokenizer, image_path: Path) -> str:
    try:
        image = Image.open(image_path)
        enc = model.encode_image(image)
        prompt = (
            "Describe this image, chart, or table briefly for a search engine. "
            "Extract all visible text, column names, and data. Be concise and technical."
        )
        return model.answer_question(enc, prompt, tokenizer)
    except (OSError, RuntimeError) as exc:
        log.warning("VLM failed on %s: %s", image_path.name, exc)
        return "Image content could not be analyzed."


def _plumber_page_text(plumber_doc, page_num: int) -> str:
    """pdfplumber ile tek bir sayfanın metnini çek; varsa tabloları da
    Markdown benzeri satır biçimine ekle.

    PyMuPDF tablolarda zayıf; pdfplumber satır/kolon hizalamasını daha iyi
    çözüyor. Tablolar `| col | col |` satırları olarak ekleniyor ki
    chunker ve BGE-M3 anlamlı tokenlar görsün.
    """
    if plumber_doc is None or page_num >= len(plumber_doc.pages):
        return ""
    page = plumber_doc.pages[page_num]
    parts: list[str] = []
    page_text = (page.extract_text() or "").strip()
    if page_text:
        parts.append(page_text)
    try:
        for table in page.extract_tables() or []:
            rows = [
                "| " + " | ".join((cell or "").strip() for cell in row) + " |"
                for row in table
                if any((cell or "").strip() for cell in row)
            ]
            if rows:
                parts.append("\n".join(rows))
    except (ValueError, IndexError) as exc:
        log.debug("pdfplumber table extraction skipped on page %d: %s", page_num, exc)
    return "\n\n".join(parts).strip()


def _extract_pdf(
    pdf_path: Path,
    fingerprint_hash: str,
    img_root: Path,
    vlm_model,
    vlm_tokenizer,
    ocr_reader,
) -> list[Document]:
    docs: list[Document] = []
    img_folder = img_root / pdf_path.stem
    img_folder.mkdir(parents=True, exist_ok=True)

    # pdfplumber'ı tüm PDF için bir kere aç — sayfa başına yeniden açmak
    # pahalı. Açılamazsa (bozuk PDF, şifreli, vb.) sessizce None bırak ve
    # fallback'leri devre dışı say.
    try:
        plumber_doc = pdfplumber.open(pdf_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("pdfplumber could not open %s: %s — falling back to PyMuPDF/OCR only.",
                    pdf_path.name, exc)
        plumber_doc = None

    try:
        with fitz.open(pdf_path) as pdf:
            seen_image_hashes: set[str] = set()
            for page_num in range(len(pdf)):
                page = pdf[page_num]
                text = page.get_text("text").strip()

                # Katmanlı fallback:
                #   1. PyMuPDF metni yeterliyse onu kullan.
                #   2. Yetersizse pdfplumber'a sor (tablolar dahil).
                #   3. O da yetersizse OCR'a düş.
                if len(text) < settings.rag.min_text_chars:
                    plumber_text = _plumber_page_text(plumber_doc, page_num)
                    if len(plumber_text) >= settings.rag.min_text_chars:
                        text = plumber_text
                    else:
                        pix = page.get_pixmap()
                        ocr_results = ocr_reader.readtext(
                            pix.tobytes("png"), detail=0, paragraph=True
                        )
                        text = "\n".join(ocr_results).strip()
                        # OCR de boş döndüyse en azından pdfplumber'ın
                        # kısa çıktısını kaybetmeyelim.
                        if not text and plumber_text:
                            text = plumber_text

                if text:
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": pdf_path.name,
                                "page": page_num,
                                "type": "text",
                                "fingerprint": fingerprint_hash,
                            },
                        )
                    )

                raster_count_this_page = 0
                for img_index, img in enumerate(page.get_images(full=True)):
                    xref = img[0]
                    base_image = pdf.extract_image(xref)
                    image_bytes = base_image["image"]
                    if len(image_bytes) < settings.rag.min_image_bytes:
                        continue
                    img_hash = hashlib.md5(image_bytes).hexdigest()
                    if img_hash in seen_image_hashes:
                        continue
                    seen_image_hashes.add(img_hash)

                    # Uzantıyı verirken yazılan yolu (PNG'ye çevrilmiş
                    # olabilir) geri al; metadata'ya o yolu koy.
                    image_stem = img_folder / f"page_{page_num + 1}_img_{img_index + 1}"
                    image_path = _save_browser_safe_image(
                        image_bytes, image_stem, base_image["ext"]
                    )
                    raster_count_this_page += 1

                    summary = _summarize_image(vlm_model, vlm_tokenizer, image_path)
                    docs.append(
                        Document(
                            page_content=f"[IMAGE SUMMARY]: {summary}",
                            metadata={
                                "source": pdf_path.name,
                                "page": page_num,
                                "type": "image",
                                "image_path": str(image_path),
                                "fingerprint": fingerprint_hash,
                            },
                        )
                    )

                # Vektör diyagram kurtarma: yalnızca sayfa metninde figür
                # altyazısı VARSA (Figure 4.1, Şekil 7, vb.) ve sayfadan
                # raster resim çıkmadıysa sayfayı PNG'ye render et.
                # Çizim sayısına bakmak güvensizdi — tablo/dipnot kenarları
                # da çizim sayılıyor, referans sayfaları yanlış tarıyordu.
                # Altyazı sinyali ise sadece figür içeren sayfalarda
                # bulunur, false positive vermez.
                if (
                    settings.rag.page_render_captions_enabled
                    and raster_count_this_page == 0
                    and text
                    and _FIGURE_CAPTION_RE.search(text)
                ):
                    dpi = settings.rag.page_render_dpi
                    pix = page.get_pixmap(dpi=dpi)
                    png_bytes = pix.tobytes("png")
                    page_img_path = img_folder / f"page_{page_num + 1}_rendered.png"
                    img_hash = hashlib.md5(png_bytes).hexdigest()
                    if img_hash not in seen_image_hashes:
                        seen_image_hashes.add(img_hash)
                        page_img_path.write_bytes(png_bytes)
                        summary = _summarize_image(
                            vlm_model, vlm_tokenizer, page_img_path
                        )
                        docs.append(
                            Document(
                                page_content=f"[IMAGE SUMMARY]: {summary}",
                                metadata={
                                    "source": pdf_path.name,
                                    "page": page_num,
                                    "type": "image",
                                    "image_path": str(page_img_path),
                                    "fingerprint": fingerprint_hash,
                                    "rendered_from_vectors": True,
                                },
                            )
                        )
    finally:
        if plumber_doc is not None:
            try:
                plumber_doc.close()
            except Exception:  # noqa: BLE001
                pass
    return docs


def run_ingestion() -> dict:
    """Run the full ingestion pipeline. Returns a summary dict."""
    log.info("Starting ingestion (BGE-M3 + Qdrant + dedup)...")
    settings.paths.docs.mkdir(parents=True, exist_ok=True)
    settings.paths.docs_images.mkdir(parents=True, exist_ok=True)
    ensure_collection()

    device = _detect_device()
    state = _load_state()

    pdf_files = sorted(p for p in settings.paths.docs.iterdir() if p.suffix.lower() == ".pdf")
    if not pdf_files:
        return {"processed": 0, "skipped": 0, "duplicates": 0, "errors": 0, "details": []}

    vlm_model, vlm_tokenizer = _load_vlm(device)
    ocr_reader = _load_ocr(device)

    documents: list[Document] = []
    processed = skipped = duplicates = errors = 0
    details: list[dict] = []

    for pdf_path in pdf_files:
        try:
            fp = compute_fingerprint(pdf_path)
        except OSError as exc:
            log.error("Cannot read %s: %s", pdf_path.name, exc)
            errors += 1
            details.append({"file": pdf_path.name, "status": "error", "reason": str(exc)})
            continue

        # 1) Local state file check (fast, no Qdrant round-trip).
        is_dup, reason = _is_duplicate_against_state(fp, state)
        # 2) Authoritative Qdrant check (covers state corruption / multi-host).
        if not is_dup and fingerprint_exists(fp.content_hash or fp.file_hash):
            is_dup, reason = True, "content already in Qdrant"
        if is_dup:
            log.info("[DUP] %s — %s", pdf_path.name, reason)
            duplicates += 1
            details.append({"file": pdf_path.name, "status": "duplicate", "reason": reason})
            continue

        if state.get(pdf_path.name, {}).get("file_hash") == fp.file_hash:
            log.info("[SKIP] %s unchanged", pdf_path.name)
            skipped += 1
            continue

        log.info("[PROCESS] %s", pdf_path.name)
        try:
            fp_for_payload = fp.content_hash or fp.file_hash
            new_docs = _extract_pdf(
                pdf_path,
                fp_for_payload,
                settings.paths.docs_images,
                vlm_model,
                vlm_tokenizer,
                ocr_reader,
            )
            documents.extend(new_docs)
            state[pdf_path.name] = {
                "file_hash": fp.file_hash,
                "content_hash": fp.content_hash,
                "metadata_hash": fp.metadata_hash,
                "title": fp.title,
                "author": fp.author,
            }
            processed += 1
            details.append({"file": pdf_path.name, "status": "processed"})
        except (RuntimeError, OSError, ValueError) as exc:
            log.error("Ingestion failed for %s: %s", pdf_path.name, exc)
            errors += 1
            details.append({"file": pdf_path.name, "status": "error", "reason": str(exc)})

    if not documents:
        _save_state(state)
        log.info("Nothing new to embed.")
        return {
            "processed": processed,
            "skipped": skipped,
            "duplicates": duplicates,
            "errors": errors,
            "details": details,
        }

    # Free VLM VRAM before loading BGE-M3.
    log.info("Evicting VLM from VRAM before embedding...")
    try:
        del vlm_model
        del vlm_tokenizer
    except NameError:
        pass
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    gc.collect()

    # Chunk with BGE-M3 tokenizer at the requested chunk size.
    log.info("Chunking with BGE-M3 tokenizer (size=%d, overlap=%d)...",
             settings.rag.chunk_size, settings.rag.chunk_overlap)
    bge_tokenizer = AutoTokenizer.from_pretrained(settings.models.embedding_model)
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=bge_tokenizer,
        chunk_size=settings.rag.chunk_size,
        chunk_overlap=settings.rag.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    log.info("Embedding %d chunks with BGE-M3...", len(chunks))
    texts = [c.page_content for c in chunks]
    dense_vecs, sparse_vecs = embed_passages(texts, batch_size=8)

    payloads = []
    for chunk in chunks:
        meta = chunk.metadata
        payloads.append(
            {
                "text": chunk.page_content,
                "source": meta.get("source"),
                "page": meta.get("page", 0),
                "type": meta.get("type", "text"),
                "image_path": meta.get("image_path"),
                "fingerprint": meta.get("fingerprint"),
            }
        )

    upsert_chunks(dense_vecs, sparse_vecs, payloads)
    _save_state(state)
    # Surface every Moondream caption to a JSON file so the user can
    # inspect what the VLM said about each figure. Independent from
    # Qdrant — convenience, not the retrieval source of truth.
    _persist_image_summaries(documents)

    log.info("Ingestion done. Processed=%d, dup=%d, err=%d, chunks=%d",
             processed, duplicates, errors, len(chunks))
    return {
        "processed": processed,
        "skipped": skipped,
        "duplicates": duplicates,
        "errors": errors,
        "chunks": len(chunks),
        "details": details,
    }


if __name__ == "__main__":
    summary = run_ingestion()
    print("INGESTION_RESULT:" + json.dumps(summary), flush=True)
