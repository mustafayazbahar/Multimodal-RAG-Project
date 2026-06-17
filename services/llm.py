"""LLM selection + Ollama interaction (streaming) + benchmark helper."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator

import requests

from services.config import settings
from services.logging_config import get_logger

log = get_logger(__name__)


# Prompt sablonlarinin disk okumasini tekrarlamamak icin basit bellek onbellegi:
# sablon_adi -> sablon_metni.
_PROMPT_CACHE: dict[str, str] = {}


# Adi verilen prompt sablonunu diskten okur ve onbellekler.
# Ayni sablon ikinci kez istendiginde dosya tekrar okunmaz, onbellekten doner.
def load_prompt(name: str = "rag_answer") -> str:
    # Onbellekte varsa diske hic gitme.
    if name in _PROMPT_CACHE:
        return _PROMPT_CACHE[name]
    # Sablonlar prompts klasorunde "<ad>.txt" olarak tutulur; UTF-8 ile okunur (Turkce karakterler).
    path: Path = settings.paths.prompts / f"{name}.txt"
    template = path.read_text(encoding="utf-8")
    _PROMPT_CACHE[name] = template
    return template


# Yapilandirmada tanimli LLM seceneklerinin listesini doner (bos girisler atilir).
# Kullaniciya gosterilecek model menusunu olusturur.
def list_available_models() -> list[str]:
    """Return the configured menu of LLM choices (drops blanks)."""
    return [m.strip() for m in settings.models.available_llms if m.strip()]


def list_pulled_models(host: str | None = None) -> list[str]:
    """Models actually present in the local Ollama instance.

    Returns the names exactly as Ollama reports them (tag included).
    On error returns an empty list — callers should fall back to
    list_available_models() so the UI is never blank.
    """
    # Taban URL'yi normalize et: sondaki "/" temizlenir ki "//api" gibi cift slash olusmasin.
    base = (host or settings.models.ollama_host).rstrip("/")
    try:
        # Ollama'nin /api/tags ucu yerel olarak kurulu modelleri listeler.
        resp = requests.get(f"{base}/api/tags", timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        # Baglanti/cozumleme hatasinda bos liste don; cagiran taraf available_llms'e duser
        # boylece arayuz asla bos kalmaz.
        log.warning("Could not list pulled models: %s", exc)
        return []
    # Yaniti gez ve adi olan modellerin isimlerini (tag dahil) topla.
    return [m["name"] for m in (data.get("models") or []) if m.get("name")]


# Verilen modeli Ollama'ya indirir ve /api/pull ilerleme olaylarini akis halinde uretir.
# Generator oldugundan cagiran taraf indirme yuzdesini canli gosterebilir.
def pull_model(model: str, host: str | None = None) -> Iterator[dict]:
    """Stream Ollama's /api/pull progress events for the given model."""
    import json

    base = (host or settings.models.ollama_host).rstrip("/")
    # timeout=None: model indirmesi cok uzun surebilecegi icin zaman asimi konmaz.
    with requests.post(
        f"{base}/api/pull",
        json={"model": model, "stream": True},
        stream=True,
        timeout=None,
    ) as resp:
        resp.raise_for_status()
        # Yanit govdesi satir-satir gelen JSON olaylaridir (NDJSON); her satir ayri ayri ayristirilir.
        for line in resp.iter_lines():
            if not line:
                continue  # Bos keep-alive satirlarini atla.
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Bozuk/yarim satir gelirse onu atla, akisi bozma.
                continue


# Modeli Ollama'nin GPU belleginden (VRAM) bosaltmasini ister (keep_alive=0).
# Benchmark'ta soguk-baslangic (cold-start) olcumu yapabilmek icin kullanilir.
def evict_model(model: str, host: str | None = None) -> None:
    """Ask Ollama to drop the model from VRAM (keep_alive=0)."""
    base = (host or settings.models.ollama_host).rstrip("/")
    try:
        # keep_alive=0: yanitin hemen ardindan model bellekten atilir.
        requests.post(
            f"{base}/api/generate",
            json={"model": model, "keep_alive": 0},
            timeout=3,
        )
    except requests.RequestException as exc:
        # Bosaltma kritik degildir; basarisiz olsa bile akisi durdurmadan uyari logla.
        log.warning("Could not evict model %s: %s", model, exc)


# Bir prompt sablonunu yukleyip verilen anahtar-deger argumanlariyla doldurur.
# Sablondaki {placeholder} alanlari kwargs ile str.format uzerinden yerine konur.
def render_prompt(template_name: str, **kwargs) -> str:
    template = load_prompt(template_name)
    return template.format(**kwargs)


# Ollama'dan yaniti token token (akis halinde) uretir. Cevap parcalari geldikce
# yield edildigi icin arayuz, yanitin tamamini beklemeden metni canli yazdirabilir.
def stream_chat(
    model: str,
    prompt: str,
    temperature: float = 0.3,
    host: str | None = None,
) -> Iterator[str]:
    """Stream tokens from Ollama's /api/generate endpoint."""
    import json

    base = (host or settings.models.ollama_host).rstrip("/")
    # temperature: yaratıcılık/rastgelelik derecesi (dusuk = daha kararli, olgusal yanit).
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {"temperature": temperature},
    }
    # timeout=600: uzun uretimlerde kesilmemek icin genis bir tavan.
    with requests.post(f"{base}/api/generate", json=payload, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        # Yanit NDJSON akisidir: her satir bir parca ("response") ve bitis bayragi ("done") tasir.
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                # Ayristirilamayan satiri atla, akisi surdur.
                continue
            # Bu parcadaki metni al; bos degilse cagiran tarafa yield et.
            chunk = data.get("response", "")
            if chunk:
                yield chunk
            # Model uretimi bitirdigini bildirdiyse donguden cik.
            if data.get("done"):
                break


# Akissiz (tek seferde tam yanit) uretim yardimcisi. Tum cevap hazir olunca
# tek string olarak doner; token token akis gerektirmeyen yerlerde kullanilir.
def generate_once(
    model: str,
    prompt: str,
    temperature: float = 0.3,
    host: str | None = None,
) -> str:
    """Non-streaming helper (used by benchmarks)."""
    base = (host or settings.models.ollama_host).rstrip("/")
    # stream=False: Ollama tum yaniti tek JSON govdesinde dondurur.
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


# Ayni prompt'u verilen her modelden gecirip performans olcer:
# toplam sure, ilk token'a kadar gecen sure (TTFT) ve saniyedeki yaklasik token.
# Modelleri karsilastirip secim yapmaya yardimci olur.
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
        # Olcum oncesi modeli bellekten bosalt; boylece her model adil sekilde
        # ayni soguk-baslangic (cold-start) kosulundan olculur.
        evict_model(model, host=host)  # cold-start measurement
        start = time.perf_counter()  # Yuksek cozunurluklu sayac (duvar saati degil).
        token_count = 0
        first_token_time: float | None = None  # Ilk parca gelince doldurulacak.
        text_parts: list[str] = []
        try:
            # Yaniti akis halinde tuket; her parcayi sayip biriktir.
            for chunk in stream_chat(model, prompt, temperature=temperature, host=host):
                # Ilk parcanin geldigi ani bir kez kaydet (TTFT hesabi icin).
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                # Kaba token tahmini: ~4 karakter ≈ 1 token; en az 1 sayilir ki bos kalmasin.
                token_count += max(1, len(chunk) // 4)  # rough token estimate
                text_parts.append(chunk)
            # Olcumler: toplam gecen sure, ilk token'a kadar sure ve token/sn.
            elapsed = time.perf_counter() - start
            ttft = (first_token_time - start) if first_token_time else None
            # Sifira bolmeyi onlemek icin elapsed > 0 kontrolu.
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
            # Bir model basarisiz olursa benchmark'i durdurma; hatayi kaydedip digerlerine devam et.
            results.append({"model": model, "ok": False, "error": str(exc)})
    return results
