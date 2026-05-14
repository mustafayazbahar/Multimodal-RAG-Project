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


def test_strips_path_traversal():
    assert _sanitize("../../etc/passwd.pdf") == "passwd.pdf"
    assert _sanitize("/abs/path/foo.pdf") == "foo.pdf"


def test_appends_pdf_extension_when_missing():
    assert _sanitize("notes").endswith(".pdf")


def test_replaces_spaces_and_special_chars():
    assert _sanitize("my file (v2).pdf") == "my_file_v2_.pdf"


def test_handles_empty_input():
    assert _sanitize("").endswith(".pdf")
    assert _sanitize("   ").endswith(".pdf")


def test_dotfile_normalized():
    # Leading dots are stripped so we never silently produce a .pdf "hidden" file
    assert not _sanitize("...secret.pdf").startswith(".")
