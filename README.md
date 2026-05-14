# 🎓 DeepCampus — Multimodal RAG Academic Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-1.32+-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/LLM-Llama_3.1_8B-purple?style=for-the-badge&logo=meta" />
  <img src="https://img.shields.io/badge/VectorDB-ChromaDB-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Docker-GPU_Ready-2496ED?style=for-the-badge&logo=docker" />
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" />
</p>

> **DeepCampus** is a **local-first, privacy-preserving** Multimodal RAG system designed for academic research. It can read, understand, and answer questions about complex PDF documents — including charts, tables, and scanned pages — entirely on your own hardware. No cloud, no API keys, no data leakage.

---

## ✨ Key Features

| Feature | Technology | Details |
|---|---|---|
| 🧠 **Local LLM** | Llama 3.1 8B (q8_0) | Runs via Ollama, zero internet dependency |
| 👁️ **Visual Intelligence** | Moondream2 VLM | Understands charts, tables, and diagrams inside PDFs |
| 🌍 **Multilingual** | multilingual-E5-base | Semantic search across 100+ languages |
| 📄 **Hybrid OCR** | PyMuPDF + EasyOCR | Digital text extraction with scanned-page fallback |
| ⚡ **Smart Ingestion** | SHA-256 Hash Diffing | Only re-processes new or changed files |
| 🔐 **RBAC Security** | SQLite + SHA-256 | Instructor / Student role separation |
| 🖼️ **Image Retrieval** | X-Ray Referencing | Finds and displays the exact image relevant to the query |
| 🐳 **Containerized** | Docker Compose | One-command deployment with full CUDA/GPU support |

---

## 🏛️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        DeepCampus                            │
│                                                              │
│  ┌─────────────┐    ┌──────────────────────────────────┐     │
│  │  Streamlit  │    │         Ingestion Pipeline       │     │
│  │  Frontend   │    │                                  │     │
│  │             │    │  PDF → PyMuPDF / EasyOCR (OCR)   │     │
│  │  RBAC Auth  │    │       ↓                          │     │
│  │  Chat UI    │    │  Images → Moondream2 (VLM)       │     │
│  └──────┬──────┘    │       ↓                          │     │
│         │           │  Chunks → E5 Embeddings          │     │
│         ▼           │       ↓                          │     │
│  ┌─────────────┐    │  ChromaDB (Persistent Storage)   │     │
│  │  ChromaDB   │◄───┤                                  │     │
│  │  Retriever  │    └──────────────────────────────────┘     │
│  └──────┬──────┘                                             │
│         │                                                    │
│         ▼                                                    │
│  ┌─────────────┐    ┌──────────────┐                         │
│  │  Llama 3.1  │◄───│  RAG Chain   │                         │
│  │  via Ollama │    │  + History   │                         │
│  └─────────────┘    └──────────────┘                         │
└──────────────────────────────────────────────────────────────┘
```

### VRAM Management Strategy

Moondream2 (VLM) and Llama 3.1 (LLM) share the GPU through a **sequential handoff protocol**:
1. Moondream2 loads → processes all PDF images → unloads and flushes CUDA cache
2. Ollama evicts Llama from VRAM before ingestion begins
3. Llama reloads on next user query

This prevents OOM crashes on consumer GPUs (tested on RTX 4080 16GB).

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit ≥ 1.32 |
| LLM Inference | Ollama + Llama 3.1 8B (q8\_0 quantization) |
| Visual Language Model | Moondream2 (2024-08-26 revision) |
| Embeddings | `intfloat/multilingual-e5-base` (768-dim) |
| Vector Store | ChromaDB ≥ 0.5 (persistent) |
| PDF Parsing | PyMuPDF (fitz) + pdfplumber |
| OCR Fallback | EasyOCR (TR + EN) |
| Chunking | Token-based via E5 tokenizer (300 tok / 50 overlap) |
| Auth & History | SQLite3 + SHA-256 password hashing |
| Containerization | Docker + Docker Compose (NVIDIA GPU passthrough) |

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose installed
- NVIDIA GPU with CUDA support (recommended: 8GB+ VRAM)
- NVIDIA Container Toolkit installed ([guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html))

### 1. Clone the Repository

```bash
git clone https://github.com/mustafayazbahar/Multimodal-RAG-Project.git
cd Multimodal-RAG-Project
```

### 2. Pull the Llama Model (first run only)

```bash
docker compose up ollama -d
docker exec -it deepcampus_ollama ollama pull llama3.1:8b-instruct-q8_0
```

### 3. Launch DeepCampus

```bash
docker compose up --build
```

Open your browser at **http://localhost:8501**

### Default Admin Credentials

```
Username: admin
Password: admin123
```

> ⚠️ Change the default password immediately after first login in a production environment.

---

## 📂 Project Structure

```
DeepCampus/
│
├── app.py               # Streamlit frontend & RAG query pipeline
├── ingest.py            # PDF ingestion pipeline (VLM + OCR + Embeddings)
├── auth.py              # SQLite RBAC authentication & chat history
│
├── requirements.txt     # Python dependencies
├── Dockerfile           # App container definition
├── docker-compose.yml   # Multi-service orchestration (app + ollama)
│
├── docs/                # 📁 Drop your PDF files here
├── chroma_db/           # 📁 Auto-generated vector store (persistent)
├── docs_images/         # 📁 Auto-extracted images from PDFs
├── ingest_state.json    # 📁 Hash registry for incremental updates
└── user.db              # 📁 SQLite user & chat history database
```

---

## 🔑 Role-Based Access Control (RBAC)

| Capability | Student | Instructor |
|---|:---:|:---:|
| Ask questions | ✅ | ✅ |
| View chat history | ✅ | ✅ |
| Clear own chat | ✅ | ✅ |
| Upload PDF documents | ❌ | ✅ |
| Trigger database update | ❌ | ✅ |
| Adjust model temperature | ❌ | ✅ |
| Adjust top-k retrieval | ❌ | ✅ |

---

## ⚙️ Configuration

All tunable parameters are exposed in the Instructor sidebar at runtime. No config file edits needed.

| Parameter | Default | Description |
|---|---|---|
| Temperature | 0.3 | LLM creativity (0 = deterministic, 1 = creative) |
| Top-k Retrieval | 15 | Number of chunks retrieved from ChromaDB per query |
| Chunk Size | 300 tokens | Document chunk size during ingestion |
| Chunk Overlap | 50 tokens | Overlap between adjacent chunks |
| Min Image Size | 15 KB | Images smaller than this are skipped during ingestion |

---

## 🔄 How Ingestion Works

```
1. Scan docs/ folder for PDF files
2. Compute SHA-256 hash of each file
3. Compare with ingest_state.json → skip unchanged files
4. For new/modified files:
   a. Extract text via PyMuPDF
   b. If page text < 15 chars → fallback to EasyOCR
   c. Extract embedded images (> 15KB, deduplicated via MD5)
   d. Run each image through Moondream2 → generate text summary
5. Unload Moondream2, flush VRAM
6. Tokenize & chunk all documents with E5 tokenizer
7. Prefix chunks with "passage: " (E5 requirement)
8. Embed with multilingual-E5-base → store in ChromaDB
9. Save updated hash state to ingest_state.json
```

---

## 🖼️ Image Retrieval (X-Ray View)

When a user asks a question:
1. The retriever fetches both text chunks AND image summaries from ChromaDB
2. The LLM is instructed to cite relevant images using `[GÖRSEL: filepath]` tags
3. A regex parser extracts cited filepaths from the raw LLM output
4. Only cited images are rendered — no hallucinated or irrelevant visuals

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
