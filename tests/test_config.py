"""Smoke tests for config and logging modules."""
from __future__ import annotations

import sys


def _reload_config():
    import importlib

    for mod in ("services.config", "services.logging_config"):
        sys.modules.pop(mod, None)
    return importlib.import_module("services.config")


def test_config_loads_defaults(monkeypatch):
    for key in ("LLM_MODEL", "TEMPERATURE", "TOP_K"):
        monkeypatch.delenv(key, raising=False)
    config = _reload_config()
    assert config.settings.models.llm_model.startswith("llama3.1")
    assert 0.0 <= config.settings.rag.temperature <= 1.0
    assert config.settings.rag.top_k > 0


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("TEMPERATURE", "0.42")
    config = _reload_config()
    assert config.settings.models.llm_model == "test-model"
    assert config.settings.rag.top_k == 7
    assert config.settings.rag.temperature == 0.42


def test_logging_configures_once():
    sys.modules.pop("services.logging_config", None)
    from services import logging_config  # noqa: WPS433
    log1 = logging_config.get_logger("a")
    log2 = logging_config.get_logger("b")
    assert log1.name == "a"
    assert log2.name == "b"
