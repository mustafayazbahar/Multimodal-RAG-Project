"""Smoke tests for config and logging modules."""
from __future__ import annotations

import importlib
import sys


def test_config_loads_defaults(monkeypatch):
    for key in ("LLM_MODEL", "TEMPERATURE", "TOP_K"):
        monkeypatch.delenv(key, raising=False)
    sys.modules.pop("config", None)
    import config  # noqa: WPS433

    assert config.settings.models.llm_model == "llama3.1:8b-instruct-q8_0"
    assert 0.0 <= config.settings.rag.temperature <= 1.0
    assert config.settings.rag.top_k > 0


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.setenv("TOP_K", "7")
    monkeypatch.setenv("TEMPERATURE", "0.42")
    sys.modules.pop("config", None)
    import config  # noqa: WPS433

    assert config.settings.models.llm_model == "test-model"
    assert config.settings.rag.top_k == 7
    assert config.settings.rag.temperature == 0.42


def test_logging_configures_once():
    sys.modules.pop("logging_config", None)
    import logging_config  # noqa: WPS433

    log1 = logging_config.get_logger("a")
    log2 = logging_config.get_logger("b")
    assert log1.name == "a"
    assert log2.name == "b"
