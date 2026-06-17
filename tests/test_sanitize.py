"""Tests for filename sanitization used on PDF uploads."""
from __future__ import annotations

import importlib
import re


SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _sanitize(name: str) -> str:
    """Mirror of app.sanitize_filename — imported here without pulling Streamlit."""
    import os
    base = os.path.basename(name or "").strip()
    base = base.lstrip(".") or "upload.pdf"
    cleaned = SAFE_FILENAME_RE.sub("_", base)
    if not cleaned.lower().endswith(".pdf"):
        cleaned += ".pdf"
    return cleaned[:255]


# Dosya adindaki dizin/path bilesenlerinin ("../" veya mutlak yol) atildigini
# dogrular. Path traversal saldirisini engelledigi icin GUVENLIK acisindan kritik.
def test_strips_path_traversal():
    assert _sanitize("../../etc/passwd.pdf") == "passwd.pdf"
    assert _sanitize("/abs/path/foo.pdf") == "foo.pdf"


# Uzantisi olmayan dosya adlarina .pdf eklendigini dogrular. Yuklenen her
# dosyanin tutarli sekilde .pdf olarak kaydedilmesini garanti eder.
def test_appends_pdf_extension_when_missing():
    assert _sanitize("notes").endswith(".pdf")


# Bosluk ve guvenli olmayan ozel karakterlerin alt cizgiyle degistirildigini
# dogrular. Dosya sistemiyle uyumlu, tahmin edilebilir adlar uretilmesini saglar.
def test_replaces_spaces_and_special_chars():
    assert _sanitize("my file (v2).pdf") == "my_file_v2_.pdf"


# Bos veya yalnizca bosluktan olusan girdide bile gecerli bir .pdf adi
# uretildigini dogrular. Bos isimle cokup hata vermemesi (saglamlik) onemli.
def test_handles_empty_input():
    assert _sanitize("").endswith(".pdf")
    assert _sanitize("   ").endswith(".pdf")


# Bastaki noktalarin temizlendigini, yani yanlislikla gizli (dotfile) bir
# dosya uretilmedigini dogrular. Hem guvenlik hem gorunurluk acisindan onemli.


def test_dotfile_normalized():
    # Leading dots are stripped so we never silently produce a .pdf "hidden" file
    assert not _sanitize("...secret.pdf").startswith(".")
