"""Smoke tests for config and logging modules."""
from __future__ import annotations

import sys


# config modulu degerlerini import sirasinda (env'den) okur. Her testte
# guncel env'i yansitan taze bir config almak icin modulu cache'ten dusurup
# yeniden import ediyoruz.
def _reload_config():
    import importlib

    for mod in ("services.config", "services.logging_config"):
        sys.modules.pop(mod, None)
    return importlib.import_module("services.config")


# Hicbir env degiskeni set edilmediginde config'in makul varsayilan degerleri
# yukledigini dogrular. Uygulamanin "kutudan cikar cikmaz" calismasinin guvencesi.
def test_config_loads_defaults(monkeypatch):
    for key in ("LLM_MODEL", "TEMPERATURE", "TOP_K"):
        monkeypatch.delenv(key, raising=False)
    config = _reload_config()
    assert config.settings.models.llm_model.startswith("llama3.1")
    # Sicaklik mantikli araliklarda (0-1) ve top_k pozitif olmali.
    assert 0.0 <= config.settings.rag.temperature <= 1.0
    assert config.settings.rag.top_k > 0


# Env degiskenleri set edildiginde config'in varsayilan yerine onlari okudugunu
# dogrular. Deployment'ta ayarlarin .env ile gercekten ezilebildiginin guvencesi.
def test_config_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("TEMPERATURE", "0.42")
    config = _reload_config()
    # String "7" ve "0.42" degerlerinin dogru tiplere (int/float) cast edildigine
    # de dikkat edin; env her zaman metin gelir.
    assert config.settings.models.llm_model == "test-model"
    assert config.settings.rag.top_k == 7
    assert config.settings.rag.temperature == 0.42


# get_logger'in her cagrida istenen isimde bir logger dondurdugunu dogrular.
# Loglama altyapisinin sorunsuz kurulup farkli isimleri ayirt ettiginin kontrolu.
def test_logging_configures_once():
    sys.modules.pop("services.logging_config", None)
    from services import logging_config  # noqa: WPS433
    log1 = logging_config.get_logger("a")
    log2 = logging_config.get_logger("b")
    assert log1.name == "a"
    assert log2.name == "b"
