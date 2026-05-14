"""PDF ingestion pipeline: hybrid OCR + VLM image summarization + E5 chunking."""
from __future__ import annotations

import gc
import hashlib
import json
import ssl
import sys
from pathlib import Path

import fitz  # PyMuPDF
import torch
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import settings
from logging_config import get_logger

ssl._create_default_https_context = ssl._create_unverified_context
try:
    sys.stdout.reconfigure(encoding="utf-8")
except AttributeError:
    pass

log = get_logger(__name__)


def _detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_file_hash(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_processed_state(state_file: Path) -> dict:
    if state_file.exists():
        try:
            return json.loads(state_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            log.warning("Corrupt state file at %s — starting fresh.", state_file)
    return {}


def save_processed_state(state: dict, state_file: Path) -> None:
    state_file.write_text(json.dumps(state, indent=4), encoding="utf-8")


def load_moondream(device: str):
    model_id = settings.models.vlm_model
    revision = settings.models.vlm_revision
    log.info("Loading VLM '%s' (revision %s) on %s...", model_id, revision, device)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision
    )
    if device in ("cuda", "mps"):
        model = model.to(device=device, dtype=torch.float16)
    else:
        model = model.to(device)
    model.eval()
    log.info("VLM ready on %s.", device)
    return model, tokenizer


def load_ocr_reader(device: str):
    import easyocr

    log.info("Loading EasyOCR fallback...")
    reader = easyocr.Reader(
        list(settings.rag.ocr_languages),
        gpu=device in ("cuda", "mps"),
        verbose=False,
    )
    log.info("EasyOCR ready.")
    return reader


def summarize_image(model, tokenizer, image_path: Path) -> str:
    try:
        image = Image.open(image_path)
        enc_image = model.encode_image(image)
        prompt = (
            "Describe this image, chart, or table briefly for a search engine. "
            "Read and extract all visible text, column names, and data strictly. "
            "Be concise and technical."
        )
        return model.answer_question(enc_image, prompt, tokenizer)
    except (OSError, RuntimeError) as exc:
        log.warning("Image summarization failed for %s: %s", image_path, exc)
        return "Image content could not be analyzed."


def extract_documents_from_pdf(
    pdf_path: Path,
    img_root: Path,
    moondream_model,
    moondream_tokenizer,
    ocr_reader,
) -> list[Document]:
    documents: list[Document] = []
    pdf_img_folder = img_root / pdf_path.stem
    pdf_img_folder.mkdir(parents=True, exist_ok=True)

    with fitz.open(pdf_path) as pdf_document:
        seen_image_hashes: set[str] = set()
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]

            text = page.get_text("text").strip()
            if len(text) < settings.rag.min_text_chars:
                pix = page.get_pixmap()
                img_bytes = pix.tobytes("png")
                ocr_results = ocr_reader.readtext(img_bytes, detail=0, paragraph=True)
                text = "\n".join(ocr_results).strip()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": pdf_path.name, "page": page_num, "type": "text"},
                    )
                )

            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = pdf_document.extract_image(xref)
                image_bytes = base_image["image"]
                if len(image_bytes) < settings.rag.min_image_bytes:
                    continue
                img_hash = hashlib.md5(image_bytes).hexdigest()
                if img_hash in seen_image_hashes:
                    continue
                seen_image_hashes.add(img_hash)

                image_name = f"page_{page_num + 1}_img_{img_index + 1}.{base_image['ext']}"
                image_path = pdf_img_folder / image_name
                image_path.write_bytes(image_bytes)

                summary = summarize_image(moondream_model, moondream_tokenizer, image_path)
                documents.append(
                    Document(
                        page_content=f"[IMAGE SUMMARY]: {summary}",
                        metadata={
                            "source": pdf_path.name,
                            "page": page_num,
                            "image_path": str(image_path),
                            "type": "image",
                        },
                    )
                )
    return documents


def main() -> None:
    log.info("Starting token-based hybrid ingestion pipeline...")
    device = _detect_device()

    settings.paths.docs.mkdir(parents=True, exist_ok=True)
    settings.paths.docs_images.mkdir(parents=True, exist_ok=True)

    try:
        moondream_model, moondream_tokenizer = load_moondream(device)
    except Exception as exc:  # noqa: BLE001 - critical bootstrap failure
        log.critical("Could not load VLM: %s", exc)
        sys.exit(1)

    ocr_reader = load_ocr_reader(device)

    log.info("Loading E5 embedding model and tokenizer...")
    embeddings = HuggingFaceEmbeddings(model_name=settings.models.embedding_model)
    e5_tokenizer = AutoTokenizer.from_pretrained(settings.models.embedding_model)

    db = None
    if settings.paths.chroma_db.exists():
        try:
            db = Chroma(
                persist_directory=str(settings.paths.chroma_db),
                embedding_function=embeddings,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not open existing Chroma store, will recreate: %s", exc)

    processed_files = load_processed_state(settings.paths.state_file)
    documents: list[Document] = []

    pdf_files = sorted(p for p in settings.paths.docs.iterdir() if p.suffix.lower() == ".pdf")

    for pdf_path in pdf_files:
        try:
            current_hash = get_file_hash(pdf_path)
        except OSError as exc:
            log.warning("Could not read %s: %s", pdf_path, exc)
            continue

        if processed_files.get(pdf_path.name) == current_hash:
            log.info("[SKIP] '%s' already indexed and unchanged.", pdf_path.name)
            continue

        log.info("[PROCESS] '%s' — new or modified, vectorizing...", pdf_path.name)
        try:
            new_docs = extract_documents_from_pdf(
                pdf_path,
                settings.paths.docs_images,
                moondream_model,
                moondream_tokenizer,
                ocr_reader,
            )
            documents.extend(new_docs)
            processed_files[pdf_path.name] = current_hash
        except (RuntimeError, OSError, ValueError) as exc:
            log.error("Failed to ingest %s: %s", pdf_path.name, exc)

    if not documents:
        log.info("No new documents found. Vector store is already up-to-date.")
        return

    log.info("PDF reading done. Evicting VLM from VRAM before embedding...")
    try:
        del moondream_model
        del moondream_tokenizer
    except NameError:
        pass
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()
    gc.collect()

    log.info("Splitting documents with E5 tokenizer...")
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=e5_tokenizer,
        chunk_size=settings.rag.chunk_size,
        chunk_overlap=settings.rag.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    for chunk in chunks:
        chunk.page_content = f"passage: {chunk.page_content}"

    if db is not None:
        db.add_documents(chunks)
    else:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=str(settings.paths.chroma_db),
        )

    save_processed_state(processed_files, settings.paths.state_file)
    log.info("Vector store updated with %d new chunks.", len(chunks))


if __name__ == "__main__":
    main()
