# 🎓 DeepCampus — Multimodal Hybrid RAG Academic Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi" />
  <img src="https://img.shields.io/badge/Streamlit-1.36+-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/LLM-Llama_3.1_•_Qwen2.5_•_Gemma2-purple?style=for-the-badge&logo=meta" />
  <img src="https://img.shields.io/badge/VectorDB-Qdrant-DC382D?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Embeddings-BGE--M3_(dense+sparse)-F59E0B?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Voice-TR_•_EN-9333EA?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Docker-GPU_Ready-2496ED?style=for-the-badge&logo=docker" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

> **DeepCampus** is a **local-first, privacy-preserving** Multimodal RAG system designed for academic research. It reads, understands, and answers questions about complex PDF documents — including charts, tables, and scanned pages — entirely on your own hardware. No cloud, no API keys, no data leakage.

> **v2.2 update** — voice I/O (TR + EN), localStorage-backed F5-proof sessions, reset-knowledge-base button, tag-aware model matching, robust image rendering, and the conservative dedup model (file + content hash; metadata-only dedup removed to kill false positives).

> **v2.1 update** — backend/frontend split, **hybrid retrieval** (BGE-M3 dense + sparse with RRF fusion), **Qdrant** vector store, **multi-LLM** benchmarking, **smart PDF dedup**, and a fully redesigned **amber-on-dark UI**.

---

## ✨ Key Features

| Feature | Technology | Details |
|---|---|---|
| 🧠 **Multi-LLM** | Llama 3.1 8B • Qwen2.5 14B • Gemma 2 9B | Switch live; benchmark all three from `/chat/benchmark` |
| 🔀 **Hybrid Retrieval** | BGE-M3 (dense + sparse) + RRF | Semantic similarity AND lexical exact-match in one query |
| 🗄️ **Qdrant Vector DB** | Named dense (1024-d) + sparse vectors | Fast filtering, payload indexes for dedup, persistent storage |
| 👁️ **Visual Intelligence** | Moondream2 VLM | Summarizes charts, tables, diagrams inside PDFs |
| 📑 **Smart PDF Dedup** | File hash + content fingerprint | Catches same paper saved under a different filename or re-stamped headers, without false positives on shared titles |
| 📄 **Hybrid OCR** | PyMuPDF + EasyOCR | Digital text extraction with scanned-page fallback (TR + EN) |
| 🎙️ **Voice I/O** | Browser Web Speech API | Mic-to-text **and** "Sesli oku" TTS in TR or EN — audio never leaves the device |
| 🔐 **JWT + RBAC** | bcrypt + JWT bearer | Instructor / Student roles, legacy SHA-256 migration on login |
| 💾 **Refresh-proof Login** | Browser localStorage | F5 keeps you signed in; no JWT in the URL, no third-party cookie blocking |
| 🧹 **Reset Knowledge Base** | One-click sidebar action | Drops the Qdrant collection + state file so the next ingest is fresh |
| 🎨 **Modern UI** | Amber-on-dark, system-aware | Avatar chat bubbles, source cards, welcome screen, sliders w/ tooltips |
| 🐳 **4-Service Stack** | Docker Compose | qdrant + ollama + backend (FastAPI) + frontend (Streamlit) |

---

## 🏛️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                            DeepCampus v2                               │
│                                                                        │
│   ┌──────────────┐    HTTP / JSON     ┌─────────────────────────────┐  │
│   │  Frontend    │ ◄─────────────────► │   Backend (FastAPI :8000)   │  │
│   │ (Streamlit)  │   NDJSON stream     │                             │  │
│   │              │                     │  /auth   /chat   /ingest    │  │
│   │  • Local-    │                     │  JWT bearer  •  RBAC        │  │
│   │    Storage   │                     └──────┬───────────────┬──────┘  │
│   │    JWT       │                            │               │         │
│   │  • Voice TR/ │                            ▼               ▼         │
│   │    EN (Web   │                  ┌───────────────┐  ┌────────────┐   │
│   │    Speech)   │                  │  Qdrant :6333 │  │ Ollama     │   │
│   │  • Avatars,  │                  │  named dense  │  │   :11434   │   │
│   │    sources,  │                  │  + sparse vec │  │  multi-LLM │   │
│   │    images    │                  └───────────────┘  └────────────┘   │
│   └──────────────┘                                                     │
│                                                                        │
│   Ingestion (subprocess on /ingest/run):                               │
│                                                                        │
│      PDF → fingerprint (file hash + content hash) → skip if dup        │
│            ↓                                                           │
│            text  → PyMuPDF / EasyOCR (OCR fallback)                    │
│            ↓                                                           │
│            images → Moondream2 (VLM summaries)                         │
│            ↓                                                           │
│            chunks → BGE-M3 (dense + sparse) → Qdrant upsert            │
└────────────────────────────────────────────────────────────────────────┘
```

### VRAM Management Strategy

Three large models share a single GPU through a **sequential handoff protocol**:

1. **Ingestion phase** — Moondream2 (VLM) loads → captions all images → unloads. BGE-M3 then loads → embeds all chunks → unloads.
2. **Query phase** — Ollama keeps the active LLM warm; switching models triggers a 2–3 s eviction.
3. **Before ingestion** the backend hits Ollama with `keep_alive=0` so the LLM releases VRAM first.

This sequential approach prevents OOM on 16 GB consumer GPUs (tested on RTX 4080).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit ≥ 1.36 + `streamlit-local-storage` (session) + `streamlit-mic-recorder` (STT) |
| Backend | FastAPI 0.110+ • Uvicorn • PyJWT |
| LLM Inference | Ollama + `{Llama 3.1 8B q8 • Qwen2.5 14B q4 • Gemma 2 9B q4}` |
| Visual Language Model | Moondream2 (2024-08-26 revision) |
| Embeddings | `BAAI/bge-m3` — 1024-d dense + lexical sparse |
| Vector Store | **Qdrant ≥ 1.11** (named dense + sparse, payload indexes) |
| Hybrid Fusion | Weighted Reciprocal Rank Fusion (RRF) |
| PDF Parsing | PyMuPDF (fitz) |
| OCR Fallback | EasyOCR (TR + EN) |
| Chunking | BGE-M3 tokenizer (1024 tok / 128 overlap) |
| Auth | bcrypt + JWT • SQLite for users/history |
| Session | Browser `localStorage` (JWT TTL 12 h default) |
| Voice I/O | Browser Web Speech API (SpeechRecognition + SpeechSynthesis) |
| Containerization | Docker + Docker Compose (NVIDIA GPU passthrough) |

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose installed
- NVIDIA GPU with CUDA support (recommended: **16 GB+ VRAM** for Qwen2.5 14B; 8 GB minimum for Llama 3.1 8B alone)
- NVIDIA Container Toolkit installed ([guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))
- A Chromium-based browser (Chrome / Edge) for full voice support — Web Speech APIs work best there.

### 1. Clone the Repository

```bash
git clone https://github.com/mustafayazbahar/Multimodal-RAG-Project.git
cd Multimodal-RAG-Project
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env — at minimum change ADMIN_PASSWORD and JWT_SECRET
```

### 3. Launch the Stack

```bash
docker compose up -d --build
```

First build takes 10–15 min (downloads BGE-M3 ≈ 2.3 GB and base images). After that, watch logs with `docker compose logs -f`.

### 4. Pull the LLM Models (first run only)

The Ollama container starts empty. Pull each model you want to use:

```bash
docker exec -it deepcampus_ollama ollama pull llama3.1:8b-instruct-q8_0
docker exec -it deepcampus_ollama ollama pull qwen2.5:14b-instruct-q4_K_M
docker exec -it deepcampus_ollama ollama pull gemma2:9b-instruct-q4_K_M
```

| Model | Quant | VRAM | Approx. tok/s (RTX 4080) | Notes |
|---|---|---|---|---|
| **Llama 3.1 8B** | q8_0 | ~8.5 GB | ~35 | Most balanced baseline |
| **Qwen2.5 14B** | q4_K_M | ~9 GB | ~25 | Strongest on Turkish + technical content |
| **Gemma 2 9B** | q4_K_M | ~5.5 GB | ~40 | Fastest; good for academic English |

> On 16 GB VRAM only one LLM fits alongside BGE-M3 (~2.3 GB) at a time. Ollama auto-evicts when you switch models — expect a 2-3 s pause on first query after a switch. The model dropdown only lists actually-pulled models; the **Download more models** expander lets instructors pull missing ones with live progress.

### 5. Open DeepCampus

| URL | What |
|---|---|
| http://localhost:8501 | **Frontend (Streamlit)** |
| http://localhost:8000/docs | Backend Swagger UI |
| http://localhost:6333/dashboard | Qdrant collections dashboard |

### Default Admin Credentials

```
Username: admin
Password: admin123
```

> ⚠️ Change these in `.env` immediately for production.

### Pausing & Resuming the Stack

```bash
docker compose stop    # halts containers, keeps volumes + ingested data
docker compose start   # brings everything back exactly as it was
```

`docker compose down` is reserved for a full teardown — it removes containers (volumes survive unless you add `-v`).

---

## 📂 Project Structure

```
Multimodal-RAG-Project/
│
├── services/                  # Shared pipeline + config (used by backend & ingestion)
│   ├── config.py              # Centralized Settings (env-driven, frozen dataclass)
│   ├── logging_config.py      # Structured logging
│   ├── auth.py                # bcrypt + legacy SHA-256 migration
│   ├── embeddings.py          # BGE-M3 wrapper (dense + sparse)
│   ├── vectorstore.py         # Qdrant client (named vectors)
│   ├── fusion.py              # Pure RRF math (test-friendly, no Qdrant import)
│   ├── retriever.py           # Hybrid search pipeline
│   ├── pdf_fingerprint.py     # Multi-layer PDF dedup
│   ├── ingestion.py           # PDF → VLM → BGE-M3 → Qdrant
│   └── llm.py                 # Ollama streaming + benchmark
│
├── backend/                   # FastAPI service
│   ├── main.py                # App + lifespan (DB tables, Qdrant collection)
│   ├── security.py            # JWT bearer + role dependencies
│   ├── schemas.py             # Pydantic request/response models
│   └── routers/
│       ├── auth.py            # /auth/{login,register}
│       ├── chat.py            # /chat/{query,history,models,benchmark}
│       └── ingest.py          # /ingest/{upload,run,status,image,reset}
│
├── frontend/                  # Streamlit thin client
│   ├── app.py                 # Main UI (welcome screen, chat, sidebar, voice)
│   ├── api_client.py          # HTTP client over backend
│   ├── session.py             # localStorage-backed JWT persistence
│   ├── styles.py              # CSS injection (amber-on-dark + light override)
│   ├── components.py          # Hero, source cards, status pills, etc.
│   └── .streamlit/config.toml # Native theme config
│
├── prompts/                   # Externalized prompt templates
├── docker/                    # Per-service Dockerfiles
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── tests/                     # pytest cases (auth, fusion, fp, sanitize, JWT, retriever)
├── docs/                      # 📁 Drop your PDF files here
├── docs_images/               # 📁 Auto-extracted images from PDFs
├── data/
│   ├── user.db                # 📁 SQLite users + chat history
│   └── ingest_state.json      # 📁 Per-file fingerprint registry
│
├── docker-compose.yml         # 4-service orchestration + healthchecks
├── requirements.backend.txt
├── requirements.frontend.txt
└── .env.example
```

---

## 🔑 Role-Based Access Control (RBAC)

| Capability | Student | Instructor |
|---|:---:|:---:|
| Ask questions | ✅ | ✅ |
| View chat history | ✅ | ✅ |
| Clear own chat | ✅ | ✅ |
| Voice input / read-aloud | ✅ | ✅ |
| Switch LLM at query time | ✅ | ✅ |
| Adjust temperature / top-k / hybrid weight | ✅ | ✅ |
| Upload PDF documents | ❌ | ✅ |
| Trigger database update | ❌ | ✅ |
| Pull new LLMs from Ollama | ❌ | ✅ |
| **Reset knowledge base** | ❌ | ✅ |
| Run multi-LLM benchmark | ❌ | ✅ |

---

## ⚙️ Configuration

All knobs are env-driven (see `.env.example`). The most impactful ones:

| Variable | Default | Description |
|---|---|---|
| `LLM_MODEL` | `llama3.1:8b-instruct-q8_0` | Default model when none is selected |
| `AVAILABLE_LLMS` | 3-model list | Comma-separated menu shown in the UI |
| `CHUNK_SIZE` | 1024 | BGE-M3 tokens per chunk (1024 or 2048 are best) |
| `CHUNK_OVERLAP` | 128 | Token overlap between adjacent chunks |
| `TOP_K` | 20 | Over-fetch size before RRF fusion |
| `RERANK_TOP_N` | 8 | Final chunks fed to the LLM |
| `DENSE_WEIGHT` | 0.6 | RRF weight for dense (semantic) channel |
| `SPARSE_WEIGHT` | 0.4 | RRF weight for sparse (lexical) channel |
| `RRF_K` | 60 | Standard RRF constant |
| `TEMPERATURE` | 0.3 | LLM creativity (low = grounded) |
| `HISTORY_WINDOW` | 4 | Conversation turns sent back to LLM |
| `JWT_SECRET` | change-me | **Override for production** |
| `JWT_TTL_HOURS` | 12 | Session token lifetime |

---

## 🔄 How Ingestion Works

```
1. Scan docs/ for PDF files
2. For each PDF, compute a multi-layer fingerprint:
   • file hash         (SHA-256 of bytes)
   • content hash      (normalized text from first 3 pages + title/author)
3. Check ingest_state.json AND Qdrant payload — skip if either layer matches
4. For new/changed files:
   a. Extract page text with PyMuPDF
   b. If page text < 15 chars → EasyOCR fallback (TR + EN)
   c. Extract embedded images (> 15 KB, MD5-deduplicated)
   d. Caption each image with Moondream2
5. Free VLM VRAM, load BGE-M3 tokenizer + model
6. Chunk all docs at 1024 tokens / 128 overlap using BGE-M3 tokenizer
7. Embed each chunk → (dense 1024-d, sparse {token: weight})
8. Upsert into Qdrant with payload {source, page, type, fingerprint}
9. Update ingest_state.json
```

> 🧹 **Re-ingesting after deletion:** Removing a file from `docs/` is not enough — its fingerprint still lives in Qdrant + `ingest_state.json`, so re-uploading the same content trips the dedup check. Use the sidebar's **Danger zone → Reset knowledge base** button (instructor only) to wipe the Qdrant collection and the state file in one click before a fresh ingest. The reset is gated behind a confirm checkbox so a stray click can't nuke an indexed corpus.

> 🧩 **Why not metadata-only dedup?** A previous version also rejected uploads whose `title + author` matched an existing entry. Real-world PDFs share titles ("Lecture Notes", "Progress Report") between unrelated documents — that check produced false positives, so it was removed (PR #8 review feedback). File-hash + content-hash now cover same-bytes / re-stamped-header cases without the collateral damage.

---

## 🔀 How Hybrid Retrieval Works

```
Query: "transformer attention mechanism"

      ┌────────── BGE-M3 (one forward pass) ──────────┐
      │                                               │
      ▼                                               ▼
  Dense vector (1024-d)                  Sparse {token_id: weight}
      │                                               │
      ▼                                               ▼
  Qdrant ANN search (named="dense")     Qdrant inverted index (named="sparse")
      │                                               │
      └──────────────┬─────────────────┬──────────────┘
                     │                 │
                     ▼                 ▼
                  Reciprocal Rank Fusion (weighted)
                     score = Σ ( weight_i / (k + rank_i) )
                                  │
                                  ▼
                          Top-N chunks → LLM
```

The retriever fetches `TOP_K` candidates from each channel, fuses ranks with RRF (default weights 0.6 dense / 0.4 sparse), and passes the top `RERANK_TOP_N` chunks to the LLM. Image summaries and text chunks share the same retrieval pipeline.

---

## 🖼️ Image Retrieval (X-Ray View)

When a user asks a question:
1. The retriever fetches both text chunks AND image summaries from Qdrant
2. The LLM is instructed to cite relevant images with `[GÖRSEL: filepath]` tags
3. The backend strips citation tags from the streamed answer and emits the cited paths in a dedicated `images` stream event
4. The frontend pulls each cited image **through the backend with the JWT bearer** (`/ingest/image?path=...`) and hands the bytes to `st.image` — the browser never has to reach an internal Docker hostname or expose the JWT to a `<img src>` GET
5. Images are cached in-process via `@st.cache_data` so a re-render doesn't re-download every figure

> 🔐 The image endpoint requires any authenticated user (not just instructors). Students see the same figures the LLM cited in their own answers; the path-traversal guard still pins reads to `docs_images/`.

---

## 🎙️ Voice I/O (Türkçe + English)

DeepCampus speaks. Both directions, both languages, all in the browser:

- **Mic-to-text**: the chat pane shows a **🎤 Konuş / Stop** button (powered by `streamlit-mic-recorder` wrapping the browser's `SpeechRecognition` API). Click, speak, the transcript fires as a regular question — the streaming pipeline behind it is unchanged.
- **Read-aloud**: every assistant message gets a **🔊 Sesli oku** button that invokes `window.speechSynthesis` with the selected language locale. Hit it twice and the second click cancels the first one.
- **Language toggle**: a sidebar radio (`Türkçe` / `English`) drives both recognition (`tr` / `en`) and synthesis (`tr-TR` / `en-US`).

Audio never leaves the user's machine — no whisper round-trip, no backend audio storage. If `streamlit-mic-recorder` is missing in the image (e.g. Safari private mode), voice degrades silently and the chat text input still works.

---

## 🔒 Session Persistence (F5 fix)

`v2.2` ships the third (and final) take on "stay signed in after a refresh":

1. **Attempt 1**: HTTP cookie via `extra-streamlit-components` iframe → unreliable in Chrome, the iframe boot raced with the cookie read on F5.
2. **Attempt 2**: stuff `{token, username, role}` into `st.query_params` → it worked, but leaked the JWT through Referer headers, reverse-proxy access logs, and the user's browser history. Textbook OWASP "token in URL" anti-pattern.
3. **Attempt 3 (current)**: persist the same blob to the browser's **`localStorage`** via `streamlit-local-storage`. localStorage is same-origin, immune to third-party cookie blocking, and survives F5 reliably. The URL stays clean — no JWT in sight.

> 🛡️ **Trade-off.** localStorage is JavaScript-readable on the same origin, so a markup-injection bug would leak the token. We mitigate by escaping all user-controlled output in the Streamlit layer and by keeping JWTs short-lived (12 h default). For a hardened deployment, switch to backend-issued **HttpOnly + Secure** cookies (out of scope for this academic project).

Implementation lives in `frontend/session.py` — `save_cookie / load_cookie / clear_cookie / hydrate_from_cookie` is a stable API; the underlying storage backend swapped under the hood without touching `app.py`'s call sites.

---

## 📊 Multi-LLM Benchmarking

Instructors can run a single prompt across all three models and compare:

```bash
curl -X POST http://localhost:8000/chat/benchmark \
  -H "Authorization: Bearer <jwt>" \
  -H "Content-Type: application/json" \
  -d '{ "prompt": "Explain attention in transformers.", "temperature": 0.3 }'
```

Returns per model:
- `elapsed_s` — wall-clock latency
- `time_to_first_token_s`
- `approx_tokens`
- `tokens_per_second`
- `answer` — full generated text

Useful for deciding which model is best suited to your domain / language / VRAM budget.

---

## 🧠 Smart Model Dropdown

The sidebar's **Active LLM** picker only lists models that Ollama has actually pulled. Anything in `AVAILABLE_LLMS` that's still missing appears under a **Download more models** expander (instructor only) — pick it, hit Download, and a live progress display streams Ollama's pull events back from `/chat/models/pull`. As soon as the pull completes, the dropdown refreshes.

The match is **tag-aware**: a configured `llama3` is treated as the same model as Ollama's default `llama3:latest`, but size/version variants stay distinct (`llama3:8b` and `llama3:70b` do not collide).

---

## 🧪 Testing

```bash
pip install -r requirements.backend.txt
pytest tests/ -v
```

Test cases cover:
- bcrypt auth + legacy SHA-256 migration
- JWT token roundtrip + expiry
- RRF fusion math (weights, empty inputs, ranking)
- PDF fingerprint dedup (file / content layers)
- Filename sanitization (path traversal, dotfiles, extension fixing)
- Retriever fusion + context assembly
- Config env-var overrides

---

## 🎨 UI Highlights (v2.2)

- **Modern dark theme** with amber accent (`#F59E0B`)
- **System-aware**: a `@media (prefers-color-scheme: light)` block remaps surfaces if your OS is in light mode
- **Avatar chat bubbles** with role label + timestamp
- **Source cards**: `PDF` / `IMG` badge, filename, page number
- **Welcome screen** with clickable example prompts
- **Sidebar live status** — indexed docs, model dropdown, TTFT + elapsed after each answer
- **Sliders with help tooltips** — temperature, top-k, dense/sparse weight
- **Voice section** — TR/EN radio that drives both mic and read-aloud
- **Danger zone** — confirm-gated "Reset knowledge base" button
- **localStorage session** — refresh-proof login (12 h TTL), URL stays clean
- **Enter to submit login**, **auto-scroll** to bottom on new answer
- **Streaming responses** with token-by-token rendering

---

## 🤝 Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/streaming-support`)
3. Commit your changes (`git commit -m 'feat: add streaming response support'`)
4. Push to the branch (`git push origin feature/streaming-support`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Mustafa Yazbahar**
Computer Engineering Student

[![GitHub](https://img.shields.io/badge/GitHub-mustafayazbahar-181717?style=flat&logo=github)](https://github.com/mustafayazbahar/Multimodal-RAG-Project)

---

<p align="center">Built with 🔥 on local hardware. No cloud. No compromises.</p>
