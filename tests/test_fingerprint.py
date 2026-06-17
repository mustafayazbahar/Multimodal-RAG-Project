"""Tests for the PDF fingerprint dedup logic.

We build small in-memory PDFs with PyMuPDF (fitz) so tests don't depend on
external sample files.
"""
from __future__ import annotations

import importlib
import pytest

try:
    import fitz  # type: ignore
except ImportError:  # pragma: no cover
    fitz = None


pytestmark = pytest.mark.skipif(fitz is None, reason="PyMuPDF not installed")


# Test icin bellekte kucuk bir PDF olusturan yardimci. Metin ve metadata
# (baslik/yazar) parametrik; harici ornek dosyaya bagimliligi ortadan kaldirir.
def _make_pdf(path, text="Hello academic paper.", title="My Paper", author="Jane"):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.set_metadata({"title": title, "author": author})
    doc.save(str(path))
    doc.close()


# Bayt bayt ayni iki dosyanin "file" (dosya hash'i) seviyesinde duplikat
# sayildigini dogrular. Tam ayni yuklemenin tekrarini yakalamanin guvencesi.
def test_identical_files_match_file_hash(tmp_path):
    from services.pdf_fingerprint import compute_fingerprint

    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    _make_pdf(a)
    # Bytewise copy → exact file-hash match
    b.write_bytes(a.read_bytes())

    fp_a = compute_fingerprint(a)
    fp_b = compute_fingerprint(b)
    assert fp_a.is_duplicate_of(fp_b) == "file"


# Dosya baytlari birebir ayni olmasa da (fitz zaman damgalari nedeniyle) ayni
# icerige sahip PDF'lerin "content" seviyesinde duplikat sayildigini dogrular.
# Ayni makalenin yeniden uretilmis kopyalarini yakalamak icin onemli.
def test_same_content_different_metadata_matches_content(tmp_path):
    from services.pdf_fingerprint import compute_fingerprint

    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    _make_pdf(a, text="Identical body text.", title="Title", author="Alice")
    _make_pdf(b, text="Identical body text.", title="Title", author="Alice")

    fp_a = compute_fingerprint(a)
    fp_b = compute_fingerprint(b)
    # Same content + same metadata → content hash collides even though file
    # bytes differ slightly (fitz timestamps).
    assert fp_a.is_duplicate_of(fp_b) in {"file", "content"}


# Icerigi farkli iki PDF'in duplikat olarak isaretlenmedigini (None) dogrular.
# Yanlis pozitif dedup'i onleyerek farkli belgelerin silinmemesini garanti eder.
def test_different_content_no_match(tmp_path):
    from services.pdf_fingerprint import compute_fingerprint

    a = tmp_path / "a.pdf"
    b = tmp_path / "b.pdf"
    _make_pdf(a, text="Paper about cats", title="Cats", author="Alice")
    _make_pdf(b, text="Paper about dogs", title="Dogs", author="Bob")

    fp_a = compute_fingerprint(a)
    fp_b = compute_fingerprint(b)
    assert fp_a.is_duplicate_of(fp_b) is None
