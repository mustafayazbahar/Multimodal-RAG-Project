"""Shared pytest fixtures."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_configure(config):  # noqa: D401 - pytest hook
    """Redirect persistent paths to a temp area so tests never touch real data."""
    tmp = Path(os.getenv("PYTEST_TMP_ROOT", str(ROOT / ".pytest_data")))
    tmp.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("USER_DB_PATH", str(tmp / "test_user.db"))
    os.environ.setdefault("INGEST_STATE_PATH", str(tmp / "test_state.json"))
    os.environ.setdefault("DOCS_PATH", str(tmp / "docs"))
    os.environ.setdefault("CHROMA_DB_PATH", str(tmp / "chroma_db"))
    os.environ.setdefault("DOCS_IMAGES_PATH", str(tmp / "docs_images"))
    os.environ.setdefault("RERANKER_ENABLED", "false")
