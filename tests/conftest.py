"""Shared pytest fixtures."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Proje kok dizinini hesaplayip sys.path'e ekliyoruz; boylece testler
# "services" gibi proje modullerini sorunsuz import edebilir.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def pytest_configure(config):  # noqa: D401 - pytest hook
    """Redirect persistent paths to a temp area so tests never touch real data."""
    # Tum kalici dosya yollarini gecici bir dizine yonlendiriyoruz.
    # Amac: testlerin gercek kullanici veritabani/dokuman dizinlerine asla
    # dokunmamasi (veri kaybi/yan etki riskini ortadan kaldirmak).
    tmp = Path(os.getenv("PYTEST_TMP_ROOT", str(ROOT / ".pytest_data")))
    tmp.mkdir(parents=True, exist_ok=True)
    # setdefault kullaniyoruz: degisken disaridan zaten verilmisse ezmiyoruz,
    # sadece set edilmemisse guvenli gecici degeri atiyoruz.
    os.environ.setdefault("USER_DB_PATH", str(tmp / "test_user.db"))
    os.environ.setdefault("INGEST_STATE_PATH", str(tmp / "test_state.json"))
    os.environ.setdefault("DOCS_PATH", str(tmp / "docs"))
    os.environ.setdefault("DOCS_IMAGES_PATH", str(tmp / "docs_images"))
    # Testlerde gercek sir yerine sabit, zararsiz bir JWT anahtari kullaniyoruz.
    os.environ.setdefault("JWT_SECRET", "test-secret")
