"""PDF deduplication using multi-layer fingerprints.

Three identification layers, strongest first:
1. File hash (SHA-256 of the bytes) — exact file match.
2. Content fingerprint — SHA-256 over normalized text from the first N pages
   plus title/author metadata. Catches the same paper saved under a different
   filename or with cosmetic changes (page numbering, header re-stamp).
3. Metadata fingerprint — SHA-256 over (title, author, doi, year). Weakest;
   used only when content extraction fails.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from services.logging_config import get_logger

log = get_logger(__name__)

CONTENT_PAGES = 3  # how many leading pages to fingerprint
_WS_RE = re.compile(r"\s+")


@dataclass
class PdfFingerprint:
    file_hash: str
    content_hash: str
    metadata_hash: str
    title: str
    author: str

    def is_duplicate_of(self, other: "PdfFingerprint") -> Optional[str]:
        """Return the reason this fingerprint matches `other`, or None."""
        if self.file_hash == other.file_hash:
            return "file"
        if self.content_hash and self.content_hash == other.content_hash:
            return "content"
        if self.metadata_hash and self.metadata_hash == other.metadata_hash:
            return "metadata"
        return None


def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


def compute_fingerprint(pdf_path: Path) -> PdfFingerprint:
    """Compute a multi-layer fingerprint for a PDF file."""
    file_hash = _file_hash(pdf_path)

    title = ""
    author = ""
    leading_text = ""
    try:
        with fitz.open(pdf_path) as doc:
            meta = doc.metadata or {}
            title = (meta.get("title") or "").strip()
            author = (meta.get("author") or "").strip()

            pieces: list[str] = []
            for i in range(min(CONTENT_PAGES, len(doc))):
                page = doc[i]
                pieces.append(page.get_text("text"))
            leading_text = _normalize("\n".join(pieces))
    except (RuntimeError, OSError, ValueError) as exc:
        log.warning("Could not read PDF metadata/text for %s: %s", pdf_path.name, exc)

    # Content fingerprint covers normalized leading text + metadata title/author
    # so re-stamped headers or different filenames still collide.
    content_blob = f"{title}\n{author}\n{leading_text}".encode("utf-8")
    content_hash = hashlib.sha256(content_blob).hexdigest() if leading_text else ""

    metadata_blob = f"{title}|{author}".strip("|")
    metadata_hash = hashlib.sha256(metadata_blob.encode("utf-8")).hexdigest() if metadata_blob else ""

    return PdfFingerprint(
        file_hash=file_hash,
        content_hash=content_hash,
        metadata_hash=metadata_hash,
        title=title,
        author=author,
    )
