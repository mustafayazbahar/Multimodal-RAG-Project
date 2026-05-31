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

> **v2.5 update** — **Multi-session "Topics"** (per-user named chat threads, rename / delete, a sticky General Chat) plus the **OAuth Authorization Code flow** as the default login path (the user's password is entered on Keycloak's own login page — it never flows through Streamlit). The legacy password-grant form is preserved under an "Other sign-in options" expander for API and headless use. `chat_history` gets a `session_id` column and orphan rows from the pre-Topics schema are auto-migrated into each user's General Chat on first boot. localStorage now remembers the active topic so F5 lands you on the same thread. New env vars: `KEYCLOAK_PUBLIC_URL` (browser-facing Keycloak host) and `FRONTEND_URL` (OAuth redirect_uri).

> **v2.4 update** — **Docling** (IBM) layered into PDF ingestion: layout-aware text + TableFormer-based table extraction + figure cropping, PyMuPDF+EasyOCR kept as automatic fallback. UI is now English-only with a one-click **light / dark theme toggle** at the top of the sidebar; chat-rendered images are capped at 420 px wide; streamed answers are scrubbed of leftover `[GÖRSEL: …]` / `[Figure …]` citation tags. Streamlit is pinned `<1.57` until the four `st.components.v1.html` call sites migrate to `st.html` (deprecated for removal after 2026-06-01). Qdrant upserts are batched (64 points / 120 s client timeout) so large textbooks finish without `ResponseHandlingException: timed out`.

> **v2.3 update** — **Keycloak** identity provider replaces the legacy SQLite+bcrypt auth; OAuth2 password-grant flow, realm bootstrap on first start, role assignment via Admin API, Moondream image captions persisted to a human-readable JSON file.

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
| 📄 **Layout-aware PDF parsing** | **Docling** (TableFormer) + PyMuPDF + EasyOCR fallback | Docling handles reading order, OCR on scans, table-structure recognition, and figure cropping; PyMuPDF+EasyOCR kicks in if Docling can't open a PDF |
| 🎙️ **Voice I/O** | Browser Web Speech API | Mic-to-text **and** "Sesli oku" TTS in TR or EN — audio never leaves the device |
| 🔐 **Keycloak Auth (OIDC)** | OAuth Code flow (default) + password grant (API) | Instructor / Student roles, Admin API user creation, RS256 JWTs verified against the realm JWKS; password never touches Streamlit on the Code-flow path |
| 💬 **Multi-session "Topics"** | SQLite `chat_sessions` + sidebar UI | Per-user named chat threads; rename / delete with a sticky **General Chat** that can't be removed |
| 💾 **Refresh-proof Login** | Browser localStorage | F5 keeps you signed in; no JWT in the URL, no third-party cookie blocking |
| 🧹 **Reset Knowledge Base** | One-click sidebar action | Drops the Qdrant collection + state file so the next ingest is fresh |
| 🎨 **Modern UI** | Amber accent, **explicit light/dark toggle** | English-only labels, avatar chat bubbles, source cards, welcome screen, sliders w/ tooltips, chat images capped at 420 px |
| 🖼️ **Moondream Caption Log** | `data/image_summaries.json` + UI expander | Every VLM-generated figure description is persisted so you can verify what the model "saw" |
| 🐳 **5-Service Stack** | Docker Compose | keycloak + qdrant + ollama + backend (FastAPI) + frontend (Streamlit) |

---

## 🏛️ Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                            DeepCampus v2                                   │
│                                                                            │
│   ┌──────────────┐    HTTP / JSON     ┌─────────────────────────────────┐  │
│   │  Frontend    │ ◄─────────────────► │   Backend (FastAPI :8000)       │  │
│   │ (Streamlit)  │   NDJSON stream     │                                 │  │
│   │              │                     │  /auth  /chat  /ingest          │  │
│   │  • Local-    │                     │  RS256 JWT verify  •  RBAC      │  │
│   │    Storage   │                     └──────┬─────────┬─────────┬─────┘  │
│   │    JWT       │                            │         │         │        │
│   │  • Voice TR/ │                            ▼         ▼         ▼        │
│   │    EN (Web   │                  ┌────────────┐ ┌────────┐ ┌─────────┐  │
│   │    Speech)   │                  │  Keycloak  │ │ Qdrant │ │ Ollama  │  │
│   │  • Avatars,  │   token grant ◄──┤   :8080    │ │  :6333 │ │  :11434 │  │
│   │    sources,  │   JWKS verify    │  OIDC IdP  │ │ vectors│ │  LLMs   │  │
│   │    images    │                  └────────────┘ └────────┘ └─────────┘  │
│   └──────────────┘                                                         │
│                                                                            │
│   Ingestion (subprocess on /ingest/run):                                   │
│                                                                            │
│      PDF → fingerprint (file hash + content hash) → skip if dup            │
│            ↓                                                               │
│            text  → PyMuPDF / EasyOCR (OCR fallback)                        │
│            ↓                                                               │
│            images → Moondream2 (VLM summaries) → image_summaries.json      │
│            ↓                                                               │
│            chunks → BGE-M3 (dense + sparse) → Qdrant upsert                │
└────────────────────────────────────────────────────────────────────────────┘
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
| Frontend | Streamlit ≥ 1.36, **<1.57** (deprecation cap) + `streamlit-local-storage` (session) + `streamlit-mic-recorder` (STT) |
| Backend | FastAPI 0.110+ • Uvicorn • PyJWT |
| LLM Inference | Ollama + `{Llama 3.1 8B q8 • Qwen2.5 14B q4 • Gemma 2 9B q4}` |
| Visual Language Model | Moondream2 (2024-08-26 revision) |
| Embeddings | `BAAI/bge-m3` — 1024-d dense + lexical sparse |
| Vector Store | **Qdrant ≥ 1.11** (named dense + sparse, payload indexes) — batched upserts (64 / 120 s) |
| Hybrid Fusion | Weighted Reciprocal Rank Fusion (RRF) |
| PDF Parsing | **Docling 2.x** (primary: layout + TableFormer + figure crops) → PyMuPDF fallback |
| OCR | Docling's built-in OCR; EasyOCR (TR + EN) on the PyMuPDF fallback path |
| Chunking | BGE-M3 tokenizer (1024 tok / 128 overlap) |
| Auth | Keycloak 24 (OAuth2 / OIDC) — RS256 JWTs, realm bootstrap via `--import-realm` |
| Chat history | SQLite (`data/user.db`) — keyed by Keycloak username |
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

First build takes 15–25 min on a residential connection (downloads PyTorch CUDA ≈ 2.5 GB, BGE-M3 ≈ 2.3 GB, and base images). The Dockerfiles now use **BuildKit pip cache mounts** (`# syntax=docker/dockerfile:1.4` + `--mount=type=cache,target=/root/.cache/pip`), so any subsequent rebuild reuses the cached wheels and finishes in 2–5 min. Watch live progress with:

```bash
docker compose --progress=plain up -d --build
```

> Docling pulls its layout + TableFormer models from HuggingFace on the first ingestion run (~1–2 GB). They cache in the `hf_cache` volume, so re-ingests skip the download.

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
| http://localhost:8080 | **Keycloak admin console** (master realm) |

### Default Credentials

| Surface | Username | Password | Source |
|---|---|---|---|
| **App login** (Streamlit) | `admin` | `admin123` | `docker/keycloak/realm-deepcampus.json` (instructor role) |
| **Keycloak admin** (Keycloak UI) | `adminn` | `admin123` | `.env` → `KEYCLOAK_ADMIN` / `KEYCLOAK_ADMIN_PASSWORD` |

> ⚠️ Change both before opening the stack to anything beyond your laptop. The Keycloak admin manages the realm (creates users, edits roles); the app login is a regular realm user that exists only inside the `deepcampus` realm.

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
│   ├── auth.py                # SQLite chat_history persistence
│   ├── keycloak_auth.py       # OIDC client + Admin API wrapper
│   ├── embeddings.py          # BGE-M3 wrapper (dense + sparse)
│   ├── vectorstore.py         # Qdrant client (named vectors)
│   ├── fusion.py              # Pure RRF math (test-friendly, no Qdrant import)
│   ├── retriever.py           # Hybrid search pipeline
│   ├── pdf_fingerprint.py     # Multi-layer PDF dedup
│   ├── ingestion.py           # PDF → VLM → BGE-M3 → Qdrant + image_summaries.json
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
├── docker/                    # Per-service Dockerfiles + Keycloak realm
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── keycloak/
│       └── realm-deepcampus.json   # Imported on first Keycloak boot
├── tests/                     # pytest cases (auth, fusion, fp, sanitize, JWT, retriever)
├── docs/                      # 📁 Drop your PDF files here
├── docs_images/               # 📁 Auto-extracted images from PDFs
├── data/
│   ├── user.db                # 📁 SQLite chat history (users live in Keycloak)
│   ├── ingest_state.json      # 📁 Per-file fingerprint registry
│   └── image_summaries.json   # 📁 Human-readable log of every VLM caption
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
4. For new/changed files, primary path = Docling:
   a. Layout-aware text extraction (reading order, headings, paragraphs)
   b. Built-in OCR on scanned pages
   c. TableFormer detects tables → crop each as a PNG
   d. Layout detector crops every picture/figure as a PNG
   e. Caption each cropped image with Moondream2
   • If Docling can't open the PDF (corrupted / unsupported), fall back to
     the legacy PyMuPDF + EasyOCR path so the file still indexes.
5. Free VLM VRAM, load BGE-M3 tokenizer + model
6. Chunk all docs at 1024 tokens / 128 overlap using BGE-M3 tokenizer
7. Embed each chunk → (dense 1024-d, sparse {token: weight})
8. Upsert into Qdrant in 64-point batches with payload
   {source, page, type, image_kind, fingerprint}
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

> 📝 **VLM caption log.** Each ingestion run also writes `data/image_summaries.json` — a human-readable list of every figure with its source PDF, page number, image path, and the Moondream-generated caption. The sidebar's **Image summaries (Moondream)** expander surfaces the same data inside the app. Use it to sanity-check what the VLM "saw" before trusting the chat citations.

---

## 🎙️ Voice I/O (Türkçe + English)

DeepCampus speaks. Both directions, both languages, all in the browser:

- **Mic-to-text**: the chat pane shows a **🎤 Konuş / Stop** button (powered by `streamlit-mic-recorder` wrapping the browser's `SpeechRecognition` API). Click, speak, the transcript fires as a regular question — the streaming pipeline behind it is unchanged.
- **Read-aloud**: every assistant message gets a **🔊 Sesli oku** button that invokes `window.speechSynthesis` with the selected language locale. Hit it twice and the second click cancels the first one.
- **Language toggle**: a sidebar radio (`Türkçe` / `English`) drives both recognition (`tr` / `en`) and synthesis (`tr-TR` / `en-US`).

Audio never leaves the user's machine — no whisper round-trip, no backend audio storage. If `streamlit-mic-recorder` is missing in the image (e.g. Safari private mode), voice degrades silently and the chat text input still works.

---

## 🔐 Keycloak Auth (v2.3)

Identity and user management live in a dedicated Keycloak container; the legacy SQLite + bcrypt + locally-signed JWT path is gone. The backend now validates **RS256-signed JWTs** issued by the `deepcampus` realm.

### What runs where

| Component | Role |
|---|---|
| **Keycloak container** (`:8080`) | OIDC identity provider. Hosts the `deepcampus` realm, the `streamlit-app` public client, and the `instructor` / `student` realm roles. Realm is auto-imported from `docker/keycloak/realm-deepcampus.json` on first start. |
| **Backend `/auth/login`** | Password-grant proxy. Calls Keycloak's token endpoint, returns the raw `access_token` (a Keycloak JWT) to the frontend. |
| **Backend `/auth/register`** | Admin-API proxy. Uses master-realm admin credentials (`KEYCLOAK_ADMIN`) to create a user inside `deepcampus`, then auto-logs them in. |
| **Backend `get_current_user`** | RS256 signature verification against the realm's JWKS (`/realms/deepcampus/protocol/openid-connect/certs`). Role pulled from `realm_access.roles` claim. |
| **Frontend** | Same UX as before — username + password form, registration form, localStorage session storage. The Keycloak swap is invisible to users. |

### Default credentials

The realm import seeds one app user with the `instructor` role:

```
Username: admin
Password: admin123
Email:    admin@deepcampus.local
```

> ⚠️ This is hard-coded in `docker/keycloak/realm-deepcampus.json` for the first boot. After Keycloak's volume (`keycloak_data`) is populated, the file is **not** re-applied; rotate credentials from the Keycloak admin console at http://localhost:8080.

### Managing users via the Keycloak admin UI

1. Browse to **http://localhost:8080** → Administration Console → login with `KEYCLOAK_ADMIN` (default `adminn` / `admin123`)
2. In the realm selector (top-left), pick **`deepcampus`**
3. **Users → Add user** to create accounts manually, or let users sign up through the app's Create-account tab
4. **Users → \<name\> → Role mapping → Assign role → instructor** to promote a student

### Why RS256 (asymmetric) instead of HS256?

Anyone with the symmetric secret can forge tokens. By moving to RS256, our backend only ever needs the realm's *public* key (cached from JWKS); the private key never leaves Keycloak. If the backend image is ever leaked, no tokens can be forged from its filesystem alone.

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

## 🎨 UI Highlights (v2.4)

- **Modern theme** with amber accent (`#F59E0B`) — dark by default
- **Explicit light/dark toggle** at the top of the sidebar (☀️ Light theme / 🌙 Dark theme). Theme choice is held in `session_state` and the CSS variables are baked into `:root` per theme; no more half-broken `prefers-color-scheme` fallback
- **English-only interface** — every label, button, help text, radio option and placeholder is English; voice input/output still supports both Turkish (`tr-TR`) and English (`en-US`) via the Voice section radio
- **Chat images capped at 420 px** wide so figure crops don't dominate the chat column
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
