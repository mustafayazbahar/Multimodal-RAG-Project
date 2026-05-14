"""LLM selection + Ollama interaction (streaming) + benchmark helper."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import requests

from services.config import settings
from services.logging_config import get_logger

log = get_logger(__name__)


_PROMPT_CACHE: dict[str, str] = {}


def load_prompt(name: str = "rag_answer") -> str:
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]
    path: Path = settings.paths.prompts / f"{name}.txt"
    template = path.read_text(encoding="utf-8")
    _PROMPT_CACHE[name] = template
    return template


def list_available_models() -> list[str]:
    """Return the configured menu of LLM choices (drops blanks)."""
    return [m.strip() for m in settings.models.available_llms if m.strip()]


def evict_model(model: str, host: str | None = None) -> None:
    """Ask Ollama to drop the model from VRAM (keep_alive=0)."""
    base = (host or settings.models.ollama_host).rstrip("/")
    try:
        requests.post(
            f"{base}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=3,
        )
    except requests.RequestException as exc:
        log.warning("Could not evict model %s: %s", model, exc)


def render_prompt(template_name: str, **kwargs) -> str:
    template = load_prompt(template_name)
    return template.format(**kwargs)


def stream_chat(
    model: str,
    prompt: str,
    temperature: float = 0.3,
    host: str | None = None,
) -> Iterator[str]:
    """Stream tokens from Ollama's /api/generate endpoint."""
    import json

    base = (host or settings.models.ollama_host).rstrip("/")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature},
    }
    with requests.post(f"{base}/api/generate", json=payload, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            chunk = data.get("response", "")
            if chunk:
                yield chunk
            if data.get("done"):
                break


def generate_once(
    model: str,
    prompt: str,
    temperature: float = 0.3,
    host: str | None = None,
) -> str:
    """Non-streaming helper (used by benchmarks)."""
    base = (host or settings.models.ollama_host).rstrip("/")
    resp = requests.post(
        f"{base}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def benchmark_models(
    models: list[str],
    prompt: str,
    temperature: float = 0.3,
    host: str | None = None,
) -> list[dict]:
    """Run the same prompt through each model, capturing latency + tokens/sec."""
    results: list[dict] = []
    for model in models:
        log.info("Benchmarking %s ...", model)
        evict_model(model, host=host)  # cold-start measurement
        start = time.perf_counter()
        token_count = 0
        first_token_time: float | None = None
        text_parts: list[str] = []
        try:
            for chunk in stream_chat(model, prompt, temperature=temperature, host=host):
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                token_count += max(1, len(chunk) // 4)  # rough token estimate
                text_parts.append(chunk)
            elapsed = time.perf_counter() - start
            ttft = (first_token_time - start) if first_token_time else None
            tok_per_sec = (token_count / elapsed) if elapsed > 0 else 0.0
            results.append(
                {
                    "model": model,
                    "elapsed_s": round(elapsed, 2),
                    "time_to_first_token_s": round(ttft, 2) if ttft else None,
                    "approx_tokens": token_count,
                    "tokens_per_second": round(tok_per_sec, 2),
                    "answer": "".join(text_parts),
                    "ok": True,
                }
            )
        except requests.RequestException as exc:
            results.append({"model": model, "ok": False, "error": str(exc)})
    return results
