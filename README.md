# DeepCampus — Multimodal Hybrid RAG Academic Assistant (v2)

Local-first, privacy-preserving Multimodal RAG over PDFs. v2 adds a
backend/frontend split, **BGE-M3 hybrid (dense + sparse) retrieval**, **Qdrant**
as the vector store, **multi-LLM benchmarking**, and **metadata-based PDF
deduplication**.

---

## Architecture

```
┌──────────────┐    HTTP    ┌─────────────┐
│  Frontend    │◄──────────►│  Backend    │
│ (Streamlit)  │            │ (FastAPI)   │
└──────────────┘            └──┬─────┬────┘
                               │     │
                  ┌────────────┘     └────────────┐
                  ▼                               ▼
            ┌──────────┐                    ┌──────────┐
            │  Qdrant  │                    │  Ollama  │
            │ (vectors)│                    │   (LLM)  │
            └──────────┘                    └──────────┘
```

Four services, orchestrated by Docker Compose:

| Service     | Tech                              | Port  |
|-------------|-----------------------------------|-------|
| `frontend`  | Streamlit                         | 8501  |
| `backend`   | FastAPI + ingestion (GPU)         | 8000  |
| `qdrant`    | Qdrant (dense + sparse vectors)   | 6333  |
| `ollama`    | Ollama (LLM inference, GPU)       | 11434 |

---

## Key Features

| Feature                 | Detail                                                                 |
|-------------------------|------------------------------------------------------------------------|
| Hybrid retrieval        | BGE-M3 dense (1024-d) + sparse (lexical) fused via weighted RRF        |
| Qdrant store            | Named dense + sparse vectors, payload indexes for dedup filters        |
| Multi-LLM support       | Llama 3.1 8B, Qwen2.5 14B (q4), Gemma 2 9B — pick at query time        |
| Benchmark endpoint      | `/chat/benchmark` reports latency, TTFT, approx tokens/sec per model   |
| Metadata-based dedup    | File hash + content fingerprint (first pages) + title/author hash      |
| Visual intelligence     | Moondream2 VLM summarizes charts/tables/diagrams                       |
| OCR fallback            | EasyOCR (TR + EN) for scanned pages                                    |
| JWT auth + RBAC         | bcrypt password hashing, instructor / student roles                    |
| Streaming UI            | Token-by-token responses over NDJSON stream                            |

---

## Project Structure

```
.
├── services/                # Shared pipeline + config
│   ├── config.py            # Centralized Settings (env-driven)
│   ├── logging_config.py    # Structured logging
│   ├── auth.py              # bcrypt + legacy-SHA-256 migration
│   ├── embeddings.py        # BGE-M3 dense + sparse wrapper
│   ├── vectorstore.py       # Qdrant client (named vectors)
│   ├── fusion.py            # Pure RRF math (test-friendly)
│   ├── retriever.py         # Hybrid search pipeline
│   ├── pdf_fingerprint.py   # Multi-layer dedup
│   ├── ingestion.py         # PDF → VLM → BGE-M3 → Qdrant
│   └── llm.py               # Ollama streaming + benchmarks
│
├── backend/                 # FastAPI service
│   ├── main.py              # App + lifespan hooks
│   ├── security.py          # JWT bearer auth
│   ├── schemas.py           # Pydantic request/response models
│   └── routers/
│       ├── auth.py          # /auth/register, /auth/login
│       ├── chat.py          # /chat/{query,history,models,benchmark}
│       └── ingest.py        # /ingest/{upload,run,status,image}
│
├── frontend/                # Streamlit thin client
│   ├── api_client.py        # HTTP client over the backend
│   └── app.py
│
├── prompts/                 # Externalized prompt templates
├── docker/                  # Per-service Dockerfiles
├── tests/                   # 20+ pytest cases (auth, fusion, fp, sanitize)
├── docker-compose.yml       # 4 services + healthchecks + GPU passthrough
├── requirements.backend.txt
├── requirements.frontend.txt
└── .env.example
```

---

## Quick Start

### Prerequisites

- Docker + Docker Compose
- NVIDIA GPU (≥ 8 GB recommended; 16 GB for Qwen2.5 14B q4)
- NVIDIA Container Toolkit

### 1. Configure environment

```bash
cp .env.example .env
# Edit ADMIN_PASSWORD and JWT_SECRET before going to production
```

### 2. Boot the stack

```bash
docker compose up -d --build
```

This starts Qdrant + Ollama + Backend + Frontend. First boot pulls BGE-M3
(~2.3 GB) into the HuggingFace cache volume.

### 3. Pull the LLM models you want to compare

The Ollama container is empty on first boot. Pull each model once:

```bash
docker exec -it deepcampus_ollama ollama pull llama3.1:8b-instruct-q8_0
docker exec -it deepcampus_ollama ollama pull qwen2.5:14b-instruct-q4_K_M
docker exec -it deepcampus_ollama ollama pull gemma2:9b-instruct-q4_K_M
```

Sizes (q-quantized): Llama 3.1 8B q8 ≈ 8.5 GB, Qwen2.5 14B q4 ≈ 9 GB,
Gemma 2 9B q4 ≈ 5.5 GB. With 16 GB VRAM only one fits alongside BGE-M3
(~2.3 GB) at a time — Ollama auto-evicts before swapping models.

### 4. Open the app

http://localhost:8501

Default credentials: `admin / admin123`. Change them via `.env` before
production.

### 5. Drop PDFs into `docs/` and click "Process & Update Database"

The ingestion pipeline:
1. Computes a multi-layer fingerprint for each PDF.
2. Skips duplicates already in Qdrant or `data/ingest_state.json`.
3. Extracts text (PyMuPDF) with EasyOCR fallback for scans.
4. Summarizes images with Moondream2.
5. Chunks at 1024 tokens (overlap 128) using BGE-M3's tokenizer.
6. Embeds with BGE-M3 (dense + sparse) and upserts into Qdrant.

---

## API

The backend is documented at `http://localhost:8000/docs` (Swagger UI).

| Method   | Path                      | Auth | Notes                                  |
|----------|---------------------------|------|----------------------------------------|
| POST     | `/auth/login`             | —    | Returns JWT bearer                     |
| POST     | `/auth/register`          | —    | Self-register as `student`             |
| GET      | `/chat/models`            | user | List configured LLMs                   |
| POST     | `/chat/query`             | user | Streaming NDJSON (sources/token/done)  |
| POST     | `/chat/benchmark`         | inst | Run prompt across multiple models      |
| GET/DEL  | `/chat/history`           | user | Get / clear chat history               |
| POST     | `/ingest/upload`          | inst | Upload PDF (sanitized filename)        |
| POST     | `/ingest/run`             | inst | Run ingestion pipeline                 |
| GET      | `/ingest/status`          | inst | Files + Qdrant sources + state         |
| GET      | `/ingest/image?path=...`  | inst | Serve an extracted image (safe path)   |

`inst` = instructor role.

---

## Configuration Knobs (`.env`)

| Variable           | Default                              | Effect                                            |
|--------------------|--------------------------------------|---------------------------------------------------|
| `LLM_MODEL`        | `llama3.1:8b-instruct-q8_0`          | Default model for non-selected queries            |
| `AVAILABLE_LLMS`   | comma-list                           | Menu shown in the UI                              |
| `CHUNK_SIZE`       | 1024                                 | BGE-M3 token chunk size (1024 or 2048 work best)  |
| `CHUNK_OVERLAP`    | 128                                  |                                                   |
| `TOP_K`            | 20                                   | Over-fetch size before RRF                        |
| `RERANK_TOP_N`     | 8                                    | Final chunks fed to LLM                           |
| `DENSE_WEIGHT`     | 0.6                                  | RRF weight for dense (semantic)                   |
| `SPARSE_WEIGHT`    | 0.4                                  | RRF weight for sparse (lexical / BM25-like)       |
| `RRF_K`            | 60                                   | Standard RRF constant                             |
| `TEMPERATURE`      | 0.3                                  |                                                   |
| `HISTORY_WINDOW`   | 4                                    | Turns of prior history sent to LLM                |

---

## Multi-LLM Benchmarking

Call `POST /chat/benchmark` with a JSON body:

```json
{
  "prompt": "Explain attention in transformers.",
  "models": ["llama3.1:8b-instruct-q8_0", "qwen2.5:14b-instruct-q4_K_M", "gemma2:9b-instruct-q4_K_M"],
  "temperature": 0.3
}
```

Response per model:
- `elapsed_s` — wall time
- `time_to_first_token_s`
- `approx_tokens`
- `tokens_per_second`
- `answer`

On an RTX 4080 + 32 GB RAM you should see roughly: Llama 3.1 8B (q8) ≈ 35 tok/s,
Qwen2.5 14B (q4) ≈ 25 tok/s, Gemma 2 9B (q4) ≈ 40 tok/s. Numbers vary with
prompt length and KV-cache state.

---

## Testing

```bash
pip install -r requirements.backend.txt pytest
pytest tests/ -v
```

22 tests cover bcrypt + legacy migration, JWT, config overrides, RRF fusion
math, PDF fingerprint dedup, and filename sanitization.

---

## License

MIT — see [LICENSE](LICENSE).
