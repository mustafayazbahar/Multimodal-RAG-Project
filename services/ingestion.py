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
import json
from pathlib import Path

import fitz  # PyMuPDF
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

    with fitz.open(pdf_path) as pdf:
        seen_image_hashes: set[str] = set()
        for page_num in range(len(pdf)):
            page = pdf[page_num]
            text = page.get_text("text").strip()
            if len(text) < settings.rag.min_text_chars:
                pix = page.get_pixmap()
                ocr_results = ocr_reader.readtext(pix.tobytes("png"), detail=0, paragraph=True)
                text = "\n".join(ocr_results).strip()
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

                image_name = f"page_{page_num + 1}_img_{img_index + 1}.{base_image['ext']}"
                image_path = img_folder / image_name
                image_path.write_bytes(image_bytes)

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
