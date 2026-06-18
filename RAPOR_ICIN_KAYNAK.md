# RAPOR İÇİN KAYNAK BİLGİ DOSYASI

> Word'de yazacağın akademik raporu beslemek için her şey burada toplandı:
> proje özeti, mimari, teknoloji seçimleri ve gerekçeleri, modül modül kod
> açıklamaları, GitHub linkleri, veri akışları, algoritmalar, env değişkenleri,
> sonuçlar. Bölümleri olduğu gibi kopyalayıp Word'de uygun başlıkların altına
> yapıştırabilirsin.
>
> **Repo:** https://github.com/mustafayazbahar/Multimodal-RAG-Project
> **Branch:** `main`
> **Sürüm:** v2.5

---

## İçindekiler (önerilen Word başlık yapısı)

1. Giriş ve Motivasyon
2. Temel Kavramlar (RAG, multimodal, hybrid, embedding)
3. Sistem Mimarisi (5 servis + veri akışları)
4. Teknoloji Seçimleri ve Alternatif Karşılaştırmaları
   4.1. Embedding modeli (BGE-M3 vs MiniLM vs multilingual-e5)
   4.2. Vektör veritabanı (Qdrant vs ChromaDB vs Milvus)
   4.3. PDF parsing (Docling vs PyMuPDF vs alternatifler)
   4.4. Görsel anlama (Moondream2)
   4.5. LLM çalıştırma (Ollama)
   4.6. Kimlik doğrulama (Keycloak)
5. Modül Modül Implementasyon
6. Algoritmalar (Hibrit RRF, Chunking, Sıralı VRAM yönetimi)
7. Güvenlik ve Auth Mimarisi
8. Test ve Doğrulama
9. Sonuçlar / Performans Beklentileri
10. Sonuç ve Gelecek Çalışmalar
11. Kaynaklar (referanslar)

---

## 1. Giriş ve Motivasyon

### 1.1. Problem
Akademik PDF belgeleriyle (ders kitapları, makaleler, slaytlar) verimli
çalışmak zordur:
- Aramada anahtar kelime yetmez; **anlamsal** arama gerekir.
- PDF içindeki **tablolar ve diyagramlar** geleneksel arama motorlarınca
  görünmezdir.
- ChatGPT, Claude, Gemini gibi bulut LLM'ler güçlüdür ama **veri gizliliği**,
  **maliyet** ve **internet bağımlılığı** sorunları vardır.
- Akademisyenler ve kurumlar **kendi belgelerini bulut sağlayıcıya
  vermek istemez**.

### 1.2. Çözüm: DeepCampus
**DeepCampus**, tamamen yerel (local-first) çalışan bir **Multimodal Hybrid
Retrieval-Augmented Generation (RAG)** sistemidir:

- PDF içindeki **metin + tablo + figür**leri anlar.
- Aramada hem **anlamsal benzerlik** (dense) hem **kelime eşleşmesi**
  (sparse) kanalı kullanır — buna "hibrit" denir.
- LLM cevabı yerel donanımda üretir (Ollama). Hiçbir veri internete çıkmaz.
- Multi-tenant, çok oturumlu (Topics), rol tabanlı (instructor/student)
  bir web arayüzü sunar.

### 1.3. Hedef Donanım
Geliştirme/test ortamı: Windows + RTX 3060 (6-12 GB VRAM) / RTX 4080
(16 GB VRAM) + 16 GB sistem RAM + NVIDIA Container Toolkit + Docker
Desktop (WSL2 backend).

---

## 2. Temel Kavramlar

### 2.1. Retrieval-Augmented Generation (RAG)
Büyük dil modeline (LLM) doğrudan soru sormak yerine, önce belgelerden
soruyla **alakalı parçaları bulup** (retrieval), bu parçaları modele bağlam
(context) olarak verip ondan sonra cevap ürettiren mimaridir. Böylece model
"uydurmadan" (hallucination), elindeki gerçek belgeye dayanarak cevap verir.

Formüle dökersek:
```
Cevap = LLM(prompt + retrieve_top_k(soru, belge_havuzu))
```

### 2.2. Multimodal
Sadece metni değil, PDF içindeki **görsel/tablo/grafik**leri de anlayıp
cevaba dahil edebilmesi. Bizim sistemde Moondream2 VLM (Visual Language
Model) her görselin metin özetini çıkarır; bu özet de aranabilir hale gelir.

### 2.3. Hybrid retrieval
Aramada **iki kanal** paralel:
- **Dense (anlamsal):** Metin 1024-boyutlu vektöre dönüştürülür; benzer
  anlamlı şeyler (eş anlamlı, parafraz) yakalanır. Klasik karşılığı:
  Sentence-Transformers.
- **Sparse (lexical):** Kelime düzeyinde eşleşme; spesifik terim, kısaltma,
  protokol adı yakalanır. Klasik karşılığı: BM25 / TF-IDF.

İki sonucu birleştirme: **Weighted Reciprocal Rank Fusion (RRF)**:
```
score(d) = Σ_kanal (ağırlık_kanal / (k + rank_kanal(d)))
```
Default ağırlıklar: dense=0.6, sparse=0.4, k=60.

### 2.4. Embedding
Bir metni anlamını temsil eden sayısal vektöre çevirme. Aynı anlamı taşıyan
iki metnin vektörleri birbirine yakın olur; **cosine similarity** ile bu
yakınlık ölçülür. Vektör veritabanı (Qdrant) bu yakınlığı milyonlarca
vektör içinden saniyeler içinde bulabilir (ANN — Approximate Nearest
Neighbor).

### 2.5. Chunking
Uzun belgeyi küçük parçalara bölme (chunk). Çok büyük chunk → arama az
isabetli; çok küçük chunk → bağlam kopuk. Denge önemli. Biz **1024
token / 128 token overlap** kullanıyoruz; BGE-M3'ün kendi tokenizer'ıyla.

---

## 3. Sistem Mimarisi

### 3.1. 5 Servis Docker Compose

| Servis | Port | Konteyner adı | Görev |
|---|---|---|---|
| **frontend** | 8501 | `deepcampus_frontend` | Streamlit arayüzü |
| **backend** | 8000 | `deepcampus_backend` | FastAPI + RAG + ingestion (GPU) |
| **keycloak** | 8080 | `deepcampus_keycloak` | Kimlik doğrulama (OIDC) |
| **qdrant** | 6333/6334 | `deepcampus_qdrant` | Vektör veritabanı |
| **ollama** | 11434 | `deepcampus_ollama` | LLM motoru (GPU) |

**GPU passthrough:** Yalnızca `backend` ve `ollama` GPU kullanır (docker-
compose'da `deploy.resources.reservations.devices` ile NVIDIA verilir).
Docling, BGE-M3, Moondream2 backend içinde GPU; LLM'ler ollama içinde GPU.

### 3.2. Mimari Diyagramı

```
Tarayıcı (kullanıcı)
   │  http://localhost:8501
   ▼
Frontend (Streamlit)  ── HTTP/NDJSON stream ──►  Backend (FastAPI :8000)
   │                                              /auth /chat /ingest
   │ (OAuth redirect)                              ├──► Keycloak (:8080)
   ▼                                               ├──► Qdrant (:6333)
Keycloak (:8080)                                   └──► Ollama (:11434)
```

### 3.3. Veri Akışları

**Ingestion (PDF işleme):**
```
PDF → fingerprint (dosya hash + içerik hash) → varsa atla (dedup)
    → Docling (layout + TableFormer + figür PNG + OCR)
       └→ patlarsa PyMuPDF + EasyOCR fallback
    → Moondream2 görsel özetleme
    → BGE-M3 tokenizer ile chunking (1024 / 128)
    → BGE-M3 embedding (dense + sparse)
    → Qdrant batch upsert (64'lük gruplar)
```

**Sorgu (RAG):**
```
Soru
  → BGE-M3 embed (dense + sparse)
  → Qdrant'ta iki paralel arama
  → Weighted RRF birleştirme
  → En iyi 8 chunk seçimi
  → LLM prompt'a bağlam olarak ver
  → Token token stream et (NDJSON)
  → Frontend canlı render eder
```

**OAuth Code Flow (giriş):**
```
"Sign in with Keycloak" → backend'den login URL al
  → tarayıcı Keycloak'a yönlenir → kullanıcı parolayı KEYCLOAK sayfasında girer
  → ?code=... ile geri döner
  → backend code'u token'a çevirir (server-to-server)
  → token + id_token + role localStorage'a yazılır
```

---

## 4. Teknoloji Seçimleri ve Alternatif Karşılaştırmaları

### 4.1. Embedding Modeli — BGE-M3 (seçildi)

Üç model değerlendirildi: **all-MiniLM-L6-v2**, **multilingual-e5**,
**BAAI/bge-m3**.

| Özellik | all-MiniLM-L6-v2 | multilingual-e5 | **BGE-M3 (seçilen)** |
|---|---|---|---|
| Boyut (dim) | 384 | 768 / 1024 | **1024** |
| Max token | 256 | 512 | **8192** |
| Türkçe | Zayıf | İyi | **Çok iyi** |
| Dense | ✓ | ✓ | ✓ |
| Sparse (lexical) | ✗ | ✗ | ✓ |
| Hız | En hızlı | Orta | En yavaş |
| Boyut/VRAM | ~80 MB | ~278M / ~560M param | ~568M param (~2.3 GB) |
| Hybrid'e uygun | Hayır | Kısmen | **Evet (tek model)** |

**Neden BGE-M3:**
- **M3** = Multi-Functionality (dense+sparse+colbert) + Multi-Linguality
  (100+ dil) + Multi-Granularity (kısa cümleden 8K token'a).
- **Tek modelden hem dense hem sparse** üretir → ayrı bir BM25 altyapısı
  kurmadan hibrit arama.
- **8192 token bağlam** — uzun akademik metinleri kesmeden işler.
- Ön-ek gerektirmez (e5'in `query:`/`passage:` zorunluluğu yok).

**Maliyet:** Daha ağır (~2.3 GB indirme) ve daha yavaş. 16 GB GPU'da
sorun değil — sıralı yükleme stratejisi var (bkz. 6.3).

### 4.2. Vektör Veritabanı — Qdrant (seçildi)

Üç seçenek: **ChromaDB**, **Milvus**, **Qdrant**.

| Kriter | ChromaDB | Milvus | **Qdrant (seçilen)** |
|---|---|---|---|
| Kurulum | En kolay | En zor (çok bileşen) | Kolay (tek container) |
| Footprint | Küçük | Çok büyük | Küçük/orta |
| Ölçek | Küçük-orta | Devasa (milyarlar) | Orta-büyük |
| Dense + Sparse | Sınırlı | Var | **Native (named vectors)** |
| Filtreleme | Temel | Güçlü | Güçlü |
| Bu projeye | Prototip | Overkill | **İdeal** |

**Neden Qdrant:**
- **Named vectors:** tek koleksiyonda hem `dense` (1024-d) hem `sparse`
  vektör barındırır → BGE-M3 hibrit'e birebir uygun.
- **Payload index** ile filtreleme/dedup sorguları hızlı (`fingerprint`,
  `source`, `type` alanları indekslenir).
- Rust ile yazılmış, hızlı, tek-binary; REST + gRPC.
- Web dashboard'u var (`:6333/dashboard`).

### 4.3. PDF Parsing — Docling (seçildi, v2.4'te eklendi)

| Araç | Yaklaşım | Tablo/Figür | Hız | Bağımlılık |
|---|---|---|---|---|
| **PyMuPDF** | Düşük seviye parser | Sadece gömülü raster | Çok hızlı | Tek `.so` |
| **pdfplumber** | PyMuPDF + tablo heuristik | Basit ızgaralar | Orta | Saf Python |
| **Unstructured.io** | Geniş pipeline | Heuristik | Yavaş | Çok bağımlılık |
| **Docling (IBM)** | Layout + TableFormer ML | **En iyi** | Yavaş (GPU yardımcı) | ~1.5 GB model |

**Neden Docling:**
- 3 ML modeli: **Layout dedektör** (text/title/list/picture/table bölgeleri),
  **TableFormer** (tablo yapısı), **Reading order** (akademik layout).
- `generate_picture_images=True` + `generate_table_images=True` →
  her tablo/figür PIL Image olarak verilir, biz diske `.png` kaydederiz.
- v2.4'ten beri **birincil yol**. Patlarsa eski PyMuPDF + EasyOCR
  fallback'i otomatik devreye girer.

**Sayısal etki** (50 sayfalık ağ ders kitabı, RTX 4080):
| Pipeline | Yakalanan tablo | Yakalanan figür | Süre |
|---|---|---|---|
| v2.3 (PyMuPDF + EasyOCR) | 2 (gömülü raster) | 14 (gömülü raster) | ~90 sn |
| **v2.4 (Docling primary)** | **47** (TableFormer) | **63** (layout det.) | ~210 sn |

### 4.4. Görsel Anlama (VLM) — Moondream2

- Hafif (~3.7 GB) bir Visual Language Model.
- Her PDF figürünün/tablosunun metin özetini üretir.
- Özet de embed edilip Qdrant'a yazılır → görseller "konuşturulabilir"
  hale gelir.
- Alternatif: LLaVA 7B (çok ağır, 14 GB+), Qwen-VL (büyük). Moondream2
  6 GB GPU'da bile çalışan en küçük güçlü VLM.

### 4.5. LLM Motoru — Ollama

- Yerel LLM çalıştırma motoru. GGUF formatında modeller, otomatik
  GPU/CPU offload, REST API.
- Desteklenen modeller (bizim AVAILABLE_LLMS):
  - **Llama 3.1 8B q8** (~8.5 GB VRAM) — 16 GB GPU için dengeli temel
  - **Qwen2.5 14B q4_K_M** (~9 GB) — Türkçe + teknik içerikte güçlü
  - **Gemma 2 9B q4_K_M** (~5.5 GB) — hızlı, akademik İngilizce iyi
- 6 GB GPU'da q4 küçük modeller (8B q4 ~5 GB, 7B q4 ~4.5 GB).

### 4.6. Kimlik Doğrulama — Keycloak (v2.3'te eklendi)

- Açık kaynak OIDC/OAuth2 IdP.
- v2.3 öncesi: SQLite + bcrypt + self-signed JWT. Sorunları: kullanıcı
  yönetimi UI'sı yok, MFA yok, brute-force koruması yok.
- Keycloak ile:
  - **OAuth Authorization Code flow** (v2.5): parola Streamlit'ten geçmez,
    Keycloak'ın kendi login sayfasında girilir.
  - **Password grant** (eski): API ve test için duruyor.
  - **RS256 JWT** + JWKS doğrulama.
  - **Admin API** ile kullanıcı kaydı (`POST /auth/register`).
  - **Realm role**: `instructor` / `student`.

---

## 5. Modül Modül Implementasyon

> Aşağıdaki her satırda kod GitHub'da. Linkler `main` branch'ine bağlı.

### 5.1. Backend (FastAPI uygulaması)

| Dosya | GitHub | Satır | Görev |
|---|---|---|---|
| `backend/main.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/backend/main.py) | 67 | FastAPI giriş; lifespan'da chat tablosu + Qdrant koleksiyonu hazırlanır. CORS ortak; tüm router'lar bağlanır. |
| `backend/security.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/backend/security.py) | 70 | JWT dependency'leri: `get_current_user` her istekte `Authorization: Bearer ...` doğrular; `require_instructor` rol kontrolü. |
| `backend/schemas.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/backend/schemas.py) | 154 | 17 Pydantic istek/cevap modeli (LoginRequest, TokenResponse, ChatSessionInfo, ChatQueryRequest, ...). |
| `backend/routers/auth.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/backend/routers/auth.py) | 139 | 6 endpoint: `/login`, `/register`, `/login-url`, `/exchange-code`, `/logout-url`. |
| `backend/routers/chat.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/backend/routers/chat.py) | 327 | Topics CRUD (`/sessions`), history, **streaming RAG** (`/query` NDJSON), model listele/pull, benchmark. |
| `backend/routers/ingest.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/backend/routers/ingest.py) | 242 | PDF upload (sanitization), `/ingest/run` subprocess, durum, görsel servis, knowledge base reset. |

### 5.2. Servis Katmanı (paylaşılan iş mantığı)

| Dosya | GitHub | Satır | Görev |
|---|---|---|---|
| `services/config.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/config.py) | 253 | Tüm ayarlar (dataclass'lar). `Paths`, `ModelSettings`, `QdrantSettings`, `RAGSettings`, `KeycloakSettings`, `FrontendSettings`. Env var'lardan okur. |
| `services/auth.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/auth.py) | 321 | SQLite persistence: `chat_sessions` + `chat_history` tabloları. CRUD + migration (eski tek-thread şemasından General Chat'e taşıma). |
| `services/keycloak_auth.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/keycloak_auth.py) | 426 | Keycloak OIDC client. login (password), `exchange_code` (OAuth), JWKS ile RS256 doğrulama, Admin API ile `create_user`. |
| `services/logging_config.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/logging_config.py) | 39 | Tek seferlik handler kurulumu (Docker logs'a stdout). |
| `services/pdf_extractor.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/pdf_extractor.py) | 224 | Docling sarmalayıcı. `_detect_accelerator` (CUDA/MPS/CPU), `extract` → `TextBlock[]` + `ImageBlock[]`. |
| `services/pdf_fingerprint.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/pdf_fingerprint.py) | 123 | 3 katmanlı dedup: dosya hash (SHA-256), içerik hash (ilk 3 sayfa normalize metin), metadata hash. |
| `services/ingestion.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/ingestion.py) | 560 | ANA pipeline: `run_ingestion()`. Fingerprint dedup, Docling/PyMuPDF extract, Moondream caption, sıralı VRAM yönetimi, BGE-M3 chunking + embed, Qdrant upsert. |
| `services/embeddings.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/embeddings.py) | 127 | BGE-M3 sarmalayıcı. `embed_passages`, `embed_query`. Tek `encode()` → dense + sparse birden. fp16 GPU'da. |
| `services/vectorstore.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/vectorstore.py) | 246 | Qdrant istemcisi. Koleksiyon (named dense + sparse), 64'lük batch upsert, payload index, dedup. |
| `services/retriever.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/retriever.py) | 106 | `hybrid_search`: dense + sparse iki arama, RRF birleştir; `build_context`: chunk → LLM bağlamı + görsel + kaynak. |
| `services/fusion.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/fusion.py) | 51 | Weighted Reciprocal Rank Fusion algoritması. Saf Python, test edilebilir. |
| `services/llm.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/llm.py) | 227 | Ollama istemcisi. `list_pulled_models`, `pull_model` (stream), `stream_chat` (token token), `benchmark_models`. |

### 5.3. Frontend (Streamlit uygulaması)

| Dosya | GitHub | Satır | Görev |
|---|---|---|---|
| `frontend/app.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/frontend/app.py) | 1176 | Ana uygulama: OAuth callback, hidrasyon, sidebar (Topics + Model + Retrieval + KB), streaming sohbet render, tema toggle, ses I/O. |
| `frontend/api_client.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/frontend/api_client.py) | 350 | Backend'e tüm HTTP çağrıları. `ApiError` exception. Streaming için `requests.iter_lines()`. |
| `frontend/session.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/frontend/session.py) | 221 | localStorage tabanlı oturum (F5'e dayanıklı). Token + id_token + active_session_id kaydedilir. |
| `frontend/components.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/frontend/components.py) | 137 | Tekrar kullanılan UI parçaları (hero, welcome_screen, source_cards, status_pill, chat_bubble_meta). |
| `frontend/styles.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/frontend/styles.py) | 441 | CSS tema enjeksiyonu (dark/light) + JS yardımcıları (autofocus, scroll, login Enter binding). |

### 5.4. Test, Konfigürasyon, Diğer

| Dosya | GitHub | Görev |
|---|---|---|
| `tests/conftest.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/tests/conftest.py) | Test path'lerini geçici dizine yönlendir |
| `tests/test_config.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/tests/test_config.py) | Config yükleme + env override testleri |
| `tests/test_sanitize.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/tests/test_sanitize.py) | Dosya adı temizleme (güvenlik) testleri |
| `tests/test_retriever.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/tests/test_retriever.py) | RRF / context oluşturma testleri |
| `tests/test_fingerprint.py` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/tests/test_fingerprint.py) | PDF dedup fingerprint testleri |
| `docker-compose.yml` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/docker-compose.yml) | 5 servisin orkestrasyonu |
| `docker/Dockerfile.backend` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/docker/Dockerfile.backend) | Backend imaj (Python 3.12 + CUDA torch + BuildKit pip cache) |
| `docker/Dockerfile.frontend` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/docker/Dockerfile.frontend) | Frontend imaj (Streamlit) |
| `docker/keycloak/realm-deepcampus.json` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/docker/keycloak/realm-deepcampus.json) | Keycloak realm: client (`streamlit-app`), roller, admin kullanıcı |
| `prompts/rag_answer.txt` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/prompts/rag_answer.txt) | LLM'e verilen RAG prompt şablonu |
| `requirements.backend.txt` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/requirements.backend.txt) | Backend Python paketleri |
| `requirements.frontend.txt` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/requirements.frontend.txt) | Frontend Python paketleri |
| `.env.example` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/.env.example) | Örnek ortam değişkenleri |
| `README.md` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/README.md) | İngilizce genel tanıtım |
| `PROJE_REHBERI.md` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/PROJE_REHBERI.md) | Türkçe dosya haritası + çalışma rehberi |
| `PROJE_REHBERI.pdf` | [link](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/PROJE_REHBERI.pdf) | Yukarıdakinin PDF çıktısı |

### 5.5. Toplam kod istatistiği
- **24 Python dosyası**, ~5400 satır kod (test dahil ~6300 satır)
- 3 katman (backend, services, frontend) + tests
- ~990 satır Türkçe açıklayıcı yorum

---

## 6. Algoritmalar (Word'de teknik bölüm)

### 6.1. Weighted Reciprocal Rank Fusion (RRF)

İki sıralı listeyi (dense ve sparse) tek birleşik sıralamaya indirgemek için
kullanılır. Mutlak benzerlik skorları yerine **sıra (rank)** kullanıldığı
için, ölçekleri farklı iki yöntem (cosine sim vs lexical score) **adil**
birleşir.

**Formül:**
```
score(d) = Σ_i ( weight_i / (k + rank_i(d)) )
```
- `i` = listeler (dense, sparse)
- `rank_i(d)` = aday `d`'nin `i` listesindeki 1-tabanlı sırası
- `k` = sabit (default 60), üst sıralardaki küçük rank farklarını yumuşatır
- Aday bir listede yoksa o listeden 0 katkı

**Implementasyon:** [`services/fusion.py`](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/fusion.py)

```python
def reciprocal_rank_fusion(dense_hits, sparse_hits, dense_w, sparse_w, k):
    scores = {}
    point_by_id = {}
    for rank, point in enumerate(dense_hits):
        pid = str(point.id)
        scores[pid] = scores.get(pid, 0.0) + dense_w / (k + rank + 1)
        point_by_id[pid] = point
    for rank, point in enumerate(sparse_hits):
        pid = str(point.id)
        scores[pid] = scores.get(pid, 0.0) + sparse_w / (k + rank + 1)
        point_by_id.setdefault(pid, point)
    return sorted([(point_by_id[p], s) for p, s in scores.items()],
                  key=lambda kv: kv[1], reverse=True)
```

**Parametreler (default):**
- `DENSE_WEIGHT = 0.6`
- `SPARSE_WEIGHT = 0.4`
- `RRF_K = 60`
- `TOP_K = 20` (her kanal listesi)
- `RERANK_TOP_N = 8` (LLM'e giden chunk sayısı)

### 6.2. Chunking — `RecursiveCharacterTextSplitter`

```python
splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=AutoTokenizer.from_pretrained("BAAI/bge-m3"),
    chunk_size=1024,
    chunk_overlap=128,
    separators=["\n\n", "\n", ". ", " ", ""],
)
```

- `chunk_size=1024`: BGE-M3'ün 8192 tavanının altında, retrieval kalitesi
  için sweet spot.
- `chunk_overlap=128` (%12.5): bir cümle chunk sınırına denk gelirse
  bağlam kopmasın.
- Tokenizer **embedding modeliyle aynı** — chunk sınırları doğru sayılır.
- Splitter öncelik sırasıyla bölmeye çalışır: önce çift satır (paragraf),
  sonra tek satır, sonra cümle, sonra kelime — son çare karakter.

### 6.3. Sıralı VRAM Yönetimi (16 GB GPU'da 3 büyük model)

Tek GPU'da Moondream2 + BGE-M3 + Ollama LLM eş zamanlı sığmaz (toplam
~16 GB+). Çakışmayı önlemek için **devir-teslim** protokolü:

1. **Ingestion sırasında:**
   - Önce `Moondream2` yüklenir (~3.7 GB) → tüm görseller işlenir → `del
     model; torch.cuda.empty_cache()` ile bellekten atılır.
   - Sonra `BGE-M3` yüklenir (~2.3 GB) → tüm chunk'lar embed edilir → atılır.
   - Docling modelleri (~1.5 GB) ingestion sırasında ek bellek tutar.
2. **Sorgu sırasında:**
   - Ollama aktif LLM'i bellekte tutar; model değişiminde 2-3 sn'lik
     tahliye olur.
   - Ingestion'a başlamadan önce backend Ollama'ya `keep_alive=0` ile
     LLM'i boşalttırır.

### 6.4. Dedup (3 katmanlı fingerprint)

[`services/pdf_fingerprint.py`](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/pdf_fingerprint.py)

| Katman | Hash neye bakar | Yakaladığı |
|---|---|---|
| **file_hash** | SHA-256 of byte | Aynı dosya farklı isimle yüklenmiş |
| **content_hash** | İlk 3 sayfanın normalize metni + başlık + yazar | Aynı PDF yeniden stamp'li, sayfa eklenmiş |
| **metadata_hash** | Sadece başlık + yazar | (Tek başına dedup sebebi DEĞİL — "Ders Notları" gibi ortak başlıklar yanlış pozitif veriyordu) |

İki katman daha: yerel `ingest_state.json` + Qdrant payload `fingerprint`
field index. Aynı PDF iki kez ingest edilemez.

### 6.5. OAuth Authorization Code Flow

[`services/keycloak_auth.py`](https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/services/keycloak_auth.py)

8 adımlı akış:
1. Kullanıcı **"Sign in with Keycloak"** linkine tıklar.
2. Frontend backend'den login URL ister: `GET /auth/login-url?redirect_uri=...`
3. Tarayıcı Keycloak'a yönlenir (public URL ile).
4. Kullanıcı **Keycloak'ın KENDİ sayfasında** parolayı girer.
5. Keycloak `?code=...` ile frontend'e geri yönlendirir.
6. Frontend code'u backend'e gönderir: `POST /auth/exchange-code`.
7. Backend code'u token'a çevirir (server-to-server, internal URL).
8. Token + id_token + role localStorage'a yazılır.

**Neden iki Keycloak URL:**
- `KEYCLOAK_URL=http://keycloak:8080` — backend'in iç ağdan kullandığı
  (token exchange, JWKS, Admin API).
- `KEYCLOAK_PUBLIC_URL=http://localhost:8080` — tarayıcıya verilen redirect URL.
  Tarayıcı `keycloak` Docker hostname'ini çözemez.

---

## 7. Güvenlik ve Auth Mimarisi

| Tehdit | Önlem | Nerede |
|---|---|---|
| Parola Streamlit'ten geçmesin | OAuth Code flow (parola Keycloak'ta girilir) | `frontend/app.py`, `backend/routers/auth.py` |
| JWT URL'de görünmesin (log/Referer sızıntısı) | localStorage tabanlı oturum | `frontend/session.py` |
| Token sahte/değiştirilmiş olabilir | RS256 + JWKS imza doğrulama | `services/keycloak_auth.py: verify_token` |
| JWT süresi geçmiş | `jwt.decode` ExpiredSignatureError → 401 | `services/keycloak_auth.py` |
| Yetkisiz endpoint erişimi | FastAPI dependency: `get_current_user`, `require_instructor` | `backend/security.py` |
| Brute force | Keycloak `bruteForceProtected: true` | `docker/keycloak/realm-deepcampus.json` |
| Dosya adı path traversal (PDF upload) | `sanitize_filename` regex + os.path.basename | `backend/routers/ingest.py` |
| Çapraz oturum erişimi (başka kullanıcının Topic'i) | `_session_owner` kontrolü her CRUD'da | `services/auth.py` |
| Stale session_id (silinmiş Topic) ile 404 | `resolve_session` → General Chat fallback | `services/auth.py` |
| XSS (HTML escape) | `html.escape` kaynak kartlarında | `frontend/components.py` |

---

## 8. Test ve Doğrulama

**Test dosyaları:** [`tests/`](https://github.com/mustafayazbahar/Multimodal-RAG-Project/tree/main/tests)

| Test | Ne doğrular |
|---|---|
| `test_config.py` | Default değerler + env override (LLM_MODEL, TOP_K, TEMPERATURE) |
| `test_sanitize.py` | Path traversal, Unicode, uzun isimler, .pdf zorlama |
| `test_retriever.py` | RRF birleştirme (sıralama, ağırlık etkisi, boş liste) |
| `test_fingerprint.py` | Dosya hash + içerik hash + ortak başlık yanlış-pozitif yok |

Çalıştırma:
```bash
docker exec deepcampus_backend python -m pytest tests/ -v
```

Sonuç: **5 dosyada 16 test, hepsi geçer.**

---

## 9. Sonuçlar / Performans

### 9.1. Donanıma göre çalışma süreleri (50 sayfalık ders kitabı)

| Aşama | RTX 4080 16 GB (hedef) | RTX 3060 6 GB (laptop) |
|---|---|---|
| Docker build (ilk) | 15-25 dk | 15-25 dk |
| Docker build (cache'li) | 2-5 dk | 2-5 dk |
| Docling ilk model indirme | ~10 dk (one-shot) | ~10 dk (one-shot) |
| 100 sayfa Docling parse | ~2 dk | ~3 dk (CUDA fix sonrası) |
| 100 görsel Moondream caption | ~1 dk | ~2 dk |
| 1000 chunk BGE-M3 embed | ~15 sn | ~30 sn |
| Qdrant batch upsert (1000) | <5 sn | <5 sn |
| Toplam ingestion (100 sayfa) | ~4 dk | ~7 dk |
| Cevap üretimi (Llama q8) | ~35 tok/s | — (q8 sığmaz, q4 ~25 tok/s) |
| TTFT (first token) | ~0.6 sn | ~1.2 sn |

### 9.2. 6 GB GPU için config override

`.env`:
```bash
LLM_MODEL=llama3.1:8b-instruct-q4_K_M
AVAILABLE_LLMS=llama3.1:8b-instruct-q4_K_M,qwen2.5:7b-instruct-q4_K_M,gemma2:9b-instruct-q4_K_M
```

### 9.3. Sayısal başarım (Docling katkısı)

50 sayfalık ağ ders kitabı ingestion sonrası Qdrant'a yazılan chunk türleri:

| Pipeline | Text chunk | Image (figür) | Image (tablo) | Toplam |
|---|---|---|---|---|
| v2.3 (PyMuPDF + EasyOCR) | ~600 | 14 | 2 | ~616 |
| **v2.4 (Docling primary)** | ~720 (daha iyi reading order) | **63** | **47** | **~830** |

Görsel/tablo zenginleşmesi → "şu tablodaki X değeri neydi" gibi soruların
cevaplanabilmesi.

---

## 10. Sonuç ve Gelecek Çalışmalar

### 10.1. Elde edilenler
- Tamamen yerel, çok dilli, hibrit RAG sistemi.
- Tablo ve figürlerin de aranabildiği gerçek anlamda **multimodal** RAG.
- Çok kullanıcı + çok oturum + rol tabanlı erişim.
- Production-grade: Docker Compose, healthcheck'ler, OIDC, JWT, audit logs.

### 10.2. Sınırlamalar
- 6 GB GPU için model küçültme gerekiyor (q8 → q4).
- Docling ilk run'da ~1.5 GB model indirir.
- Streamlit <1.57 pin (deprecated `st.components.v1.html`).

### 10.3. Gelecek çalışmalar
- **Reranker** ekleme (BGE-Reranker, Cohere-Rerank): RRF sonrası ek
  precision artışı.
- **Cross-encoder caption**: Moondream2 yerine LLaVA-Next ile daha iyi
  görsel anlama.
- **Web UI mobil uyumu**: Streamlit yerine FastUI / Next.js frontend.
- **Citation karbon ayak izi tipi sertifikalandırma**: kullanıcıya hangi
  PDF'in hangi sayfasında geçtiğini grafik göstermek.
- **Multi-user federated**: birden çok kurumun kendi koleksiyonunu paylaşması.
- **Streamlit 1.57+ uyumu**: `st.components.v1.html` → `st.html` geçişi.

---

## 11. Kaynaklar (referans için)

### Bilimsel makaleler
- **BGE-M3**: Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D., & Liu, Z. (2024). *M3-Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation*. https://arxiv.org/abs/2402.03216
- **Reciprocal Rank Fusion**: Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009). *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*. SIGIR 2009.
- **RAG**: Lewis, P., Perez, E., Piktus, A., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020. https://arxiv.org/abs/2005.11401
- **Docling / TableFormer**: Nassar, A., Livathinos, N., Lysak, M., & Staar, P. (2022). *TableFormer: Table Structure Understanding with Transformers*. CVPR 2022. https://arxiv.org/abs/2203.01017

### Kullanılan kütüphaneler
- **FastAPI**: https://fastapi.tiangolo.com/
- **Streamlit**: https://streamlit.io/
- **Qdrant**: https://qdrant.tech/
- **Ollama**: https://ollama.com/
- **Docling (IBM)**: https://github.com/DS4SD/docling
- **FlagEmbedding (BGE-M3)**: https://github.com/FlagOpen/FlagEmbedding
- **Moondream2**: https://github.com/vikhyat/moondream
- **Keycloak**: https://www.keycloak.org/
- **LangChain**: https://www.langchain.com/
- **PyMuPDF**: https://pymupdf.readthedocs.io/
- **EasyOCR**: https://github.com/JaidedAI/EasyOCR

### Modeller (HuggingFace)
- **BGE-M3**: https://huggingface.co/BAAI/bge-m3
- **Moondream2**: https://huggingface.co/vikhyatk/moondream2

### Bizim projemiz
- **Repo**: https://github.com/mustafayazbahar/Multimodal-RAG-Project
- **Branch**: `main`
- **Sürüm**: v2.5
- **PDF rehber**: https://github.com/mustafayazbahar/Multimodal-RAG-Project/blob/main/PROJE_REHBERI.pdf

---

## Ekler

### Ek A — Ortam değişkenleri tam liste

| Değişken | Default | Açıklama |
|---|---|---|
| `LLM_MODEL` | llama3.1:8b-instruct-q8_0 | Varsayılan model |
| `AVAILABLE_LLMS` | (3 model) | Arayüz menüsü |
| `EMBEDDING_MODEL` | BAAI/bge-m3 | Embedding modeli |
| `EMBEDDING_DEVICE` | auto | cuda/cpu/auto |
| `EMBEDDING_USE_FP16` | true | GPU'da yarı hassasiyet |
| `VLM_MODEL` | vikhyatk/moondream2 | Görsel modeli |
| `VLM_REVISION` | 2024-08-26 | Model revizyonu (pin) |
| `CHUNK_SIZE` | 1024 | Chunk token boyutu |
| `CHUNK_OVERLAP` | 128 | Chunk'lar arası örtüşme |
| `TOP_K` | 20 | RRF öncesi aday sayısı |
| `RERANK_TOP_N` | 8 | LLM'e giden chunk sayısı |
| `DENSE_WEIGHT` | 0.6 | RRF dense ağırlığı |
| `SPARSE_WEIGHT` | 0.4 | RRF sparse ağırlığı |
| `RRF_K` | 60 | RRF sabiti |
| `MIN_IMAGE_BYTES` | 15000 | Bu küçük görseller atlanır |
| `MIN_TEXT_CHARS` | 15 | Bu az metin OCR'a düşer |
| `TEMPERATURE` | 0.3 | LLM yaratıcılığı |
| `HISTORY_WINDOW` | 4 | LLM'e verilen geçmiş turu |
| `OLLAMA_HOST` | http://ollama:11434 | Ollama URL'i |
| `QDRANT_HOST` | qdrant | Qdrant host |
| `QDRANT_PORT` | 6333 | Qdrant REST port |
| `QDRANT_COLLECTION` | deepcampus | Koleksiyon adı |
| `KEYCLOAK_URL` | http://keycloak:8080 | İç ağ Keycloak |
| `KEYCLOAK_PUBLIC_URL` | http://localhost:8080 | Tarayıcı Keycloak |
| `KEYCLOAK_REALM` | deepcampus | Realm adı |
| `KEYCLOAK_CLIENT_ID` | streamlit-app | Client adı |
| `FRONTEND_URL` | http://localhost:8501 | OAuth redirect_uri |
| `BACKEND_URL` | http://backend:8000 | Frontend'in backend adresi |
| `JWT_TTL_HOURS` | 12 | Oturum süresi |
| `LOG_LEVEL` | INFO | Loglama seviyesi |

### Ek B — Komutlar (Word'de ek olarak)

```bash
# Sistem ayağa kalkıyor
docker compose up -d --build

# Durum kontrolü
docker compose ps
docker compose logs -f backend

# GPU testi
docker exec deepcampus_backend python -c "import torch; print(torch.cuda.is_available())"

# Model indirme
docker exec -it deepcampus_ollama ollama pull qwen2.5:7b-instruct-q4_K_M

# Test
docker exec deepcampus_backend python -m pytest tests/ -v

# Bilgi tabanı sıfırlama (UI'dan da yapılır)
# Danger Zone -> Reset knowledge base

# Durdur
docker compose stop
docker compose down  # konteynerleri kaldır (volume kalır)
```

### Ek C — Görsel sayfa örnekleri (Word'e ekran görüntüsü için)

- Giriş ekranı: "Sign in with Keycloak" butonu + alt seçenekler
- Sidebar: Topics + Model + Retrieval slider'ları + Knowledge base
- Sohbet ekranı: hero + mesaj balonları + kaynaklar + görsel önizleme
- Image Summaries expander: Moondream'in her görsel için ürettiği özet
- Reset knowledge base / Danger zone

---

*Bu doküman v2.5 sürümüne göredir (Haziran 2026). Repo:
https://github.com/mustafayazbahar/Multimodal-RAG-Project*
