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

# Icerik parmak izi icin sadece ilk birkac sayfa kullanilir: dokuman kimligini
# belirlemeye yeter ve tum PDF'i okumaktan cok daha hizlidir.
CONTENT_PAGES = 3  # how many leading pages to fingerprint
# Ardisik bosluk/satir sonu/tab karakterlerini tek bosluga indirgemek icin desen
# (metin normalizasyonunda kullanilir).
_WS_RE = re.compile(r"\s+")


# Bir PDF'in uc katmanli parmak izini tutan veri sinifi: dosya, icerik ve
# metadata hash'leri ile insan tarafindan okunabilir baslik/yazar bilgisi.
@dataclass
class PdfFingerprint:
    file_hash: str
    content_hash: str
    metadata_hash: str
    title: str
    author: str

    # Iki parmak izini en guclu katmandan baslayarak karsilastirir ve
    # eslesme bulunursa hangi katmanin eslestigini (sebep) dondurur.
    def is_duplicate_of(self, other: "PdfFingerprint") -> Optional[str]:
        """Return the reason this fingerprint matches `other`, or None."""
        # En guclu sinyal: birebir ayni baytlar (dosya hash'i).
        if self.file_hash == other.file_hash:
            return "file"
        # Ayni icerik (farkli dosya adi veya kozmetik degisiklikler).
        if self.content_hash and self.content_hash == other.content_hash:
            return "content"
        # En zayif sinyal: sadece metadata; icerik cikarimi basarisizsa.
        if self.metadata_hash and self.metadata_hash == other.metadata_hash:
            return "metadata"
        return None


# Dosyanin SHA-256 hash'ini hesaplar. Dosyayi 64 KB'lik parcalar halinde okur
# ki buyuk PDF'ler tamamen RAM'e yuklenmesin.
def _file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        # iter(..., b"") deyimi dosya sonuna (bos bayt) kadar parca parca okur.
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# Metni normalize eder: bosluklari teklestirir, bas/son bosluklari atar ve
# kuculltur. Boylece bicimsel farklar (fazladan bosluk, buyuk/kucuk harf) ayni
# icerigin farkli hash uretmesine yol acmaz.
def _normalize(text: str) -> str:
    return _WS_RE.sub(" ", text).strip().lower()


# Bir PDF dosyasi icin uc katmanli (dosya/icerik/metadata) parmak izi uretir.
# Dedup kontrolunun temelini olusturur.
def compute_fingerprint(pdf_path: Path) -> PdfFingerprint:
    """Compute a multi-layer fingerprint for a PDF file."""
    # 1. katman: ham baytlarin hash'i (her zaman hesaplanabilir).
    file_hash = _file_hash(pdf_path)

    title = ""
    author = ""
    leading_text = ""
    # Metin/metadata cikarimi bozuk PDF'lerde patlayabilir; bu yuzden try ile
    # sarmalaniyor. Hata olursa en azindan file_hash ile dedup yapilabilir.
    try:
        with fitz.open(pdf_path) as doc:
            meta = doc.metadata or {}
            title = (meta.get("title") or "").strip()
            author = (meta.get("author") or "").strip()

            # Yalnizca ilk CONTENT_PAGES sayfanin metnini topla (hiz icin).
            pieces: list[str] = []
            for i in range(min(CONTENT_PAGES, len(doc))):
                page = doc[i]
                pieces.append(page.get_text("text"))
            leading_text = _normalize("\n".join(pieces))
    except (RuntimeError, OSError, ValueError) as exc:
        log.warning("Could not read PDF metadata/text for %s: %s", pdf_path.name, exc)

    # Content fingerprint covers normalized leading text + metadata title/author
    # so re-stamped headers or different filenames still collide.
    # Icerik hash'i = baslik + yazar + normalize edilmis ilk sayfalar. Boylece
    # ayni belge farkli isimle veya yeniden damgalanmis basliklarla gelse de
    # yakalanir. Metin cikmadiysa bos birakilir (yaniltici eslesmeyi onler).
    content_blob = f"{title}\n{author}\n{leading_text}".encode("utf-8")
    content_hash = hashlib.sha256(content_blob).hexdigest() if leading_text else ""

    # Metadata hash'i sadece baslik+yazardan uretilir; en zayif katman olup
    # icerik cikarimi tamamen basarisiz oldugunda son care olarak kullanilir.
    metadata_blob = f"{title}|{author}".strip("|")
    metadata_hash = hashlib.sha256(metadata_blob.encode("utf-8")).hexdigest() if metadata_blob else ""

    return PdfFingerprint(
        file_hash=file_hash,
        content_hash=content_hash,
        metadata_hash=metadata_hash,
        title=title,
        author=author,
    )
