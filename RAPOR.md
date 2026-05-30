# DeepCampus — Multimodal Hybrid RAG Sistemi · Detaylı Proje Raporu

> Bu rapor, projeyi tanımayan birine ve proje hakkında soru soracak kişilere
> referans olması için hazırlandı. Kullanılan teknolojiler, neden seçildikleri,
> alternatiflerle karşılaştırmaları ve sistemin sıfırdan nasıl çalıştırılacağı
> adım adım anlatılır.

---

## 1. Proje Özeti

**DeepCampus**, akademik PDF belgeleri (makaleler, ders kitapları, slaytlar)
üzerinde soru-cevap yapabilen, **tamamen yerel (local-first) çalışan** bir
**Multimodal Hybrid RAG** (Retrieval-Augmented Generation) sistemidir. Bulut
yok, API anahtarı yok, veri dışarı çıkmıyor — her şey kullanıcının kendi
donanımında (test ortamı: Windows + RTX 4080 16 GB) çalışır.

**RAG nedir?** Büyük dil modeline (LLM) doğrudan soru sormak yerine, önce
belgelerden soruyla **alakalı parçaları bulup** (retrieval), bu parçaları
modele bağlam (context) olarak verip ondan sonra cevap ürettiren mimaridir.
Böylece model "uydurmadan" (hallucination), elindeki gerçek belgeye dayanarak
cevap verir.

**Multimodal** = sadece metni değil, PDF içindeki **görselleri/tabloları/
grafikleri** de anlayıp cevaba dahil edebilmesi.

**Hybrid** = arama yaparken hem **anlamsal benzerlik** (dense/semantic) hem de
**kelime eşleşmesi** (sparse/lexical) kanalını birlikte kullanması.

---

## 2. Mimari ve Servisler

Sistem **Docker Compose** ile ayağa kalkan **5 servisten** oluşur:

| Servis | İmaj | Port | Görevi |
|---|---|---|---|
| **keycloak** | `quay.io/keycloak/keycloak:24.0` | 8080 | Kimlik doğrulama (OIDC/OAuth2), kullanıcı & rol yönetimi |
| **qdrant** | `qdrant/qdrant:v1.11.0` | 6333 / 6334 | Vektör veritabanı (dense + sparse) |
| **ollama** | `ollama/ollama:latest` | 11434 | Yerel LLM çalıştırma motoru (GPU) |
| **backend** | FastAPI (kendi Dockerfile) | 8000 | API, RAG pipeline, ingestion, JWT doğrulama |
| **frontend** | Streamlit (kendi Dockerfile) | 8501 | Web arayüzü (chat, upload, ses, tema) |

### Veri akışı (basitleştirilmiş)

```
Kullanıcı sorusu
   │
   ▼
Frontend (Streamlit) ──HTTP/NDJSON stream──► Backend (FastAPI)
                                                │
                          ┌─────────────────────┼──────────────────────┐
                          ▼                     ▼                      ▼
                       Qdrant                Ollama                Keycloak
                   (vektör arama)         (LLM cevap)           (JWT doğrulama)
```

### Ingestion (belge işleme) akışı

```
PDF  →  fingerprint (dosya + içerik hash → tekrar varsa atla)
     →  metin çıkarımı: PyMuPDF  (taranmış sayfada EasyOCR yedeği, TR+EN)
     →  görseller: Moondream2 (VLM) ile başlık/özet üret
     →  chunk'lara böl: BGE-M3 tokenizer (1024 token / 128 örtüşme)
     →  embedding: BGE-M3 (dense 1024-d + sparse)
     →  Qdrant'a yaz (upsert)
```

### VRAM yönetimi (16 GB GPU'da 3 büyük model)

Tek GPU'da çakışmayı önlemek için **sıralı devir-teslim** kullanılır:
1. Ingestion sırasında önce Moondream2 (VLM) yüklenir → görseller işlenir → bellekten atılır.
2. Sonra BGE-M3 yüklenir → tüm chunk'lar embed edilir → atılır.
3. Sorgu sırasında Ollama aktif LLM'i bellekte tutar; model değişiminde 2-3 sn'lik tahliye olur.
4. Ingestion'dan önce backend, Ollama'ya `keep_alive=0` gönderip LLM'i VRAM'den boşalttırır.

---

## 3. Embedding Modelleri — Karşılaştırma

> Embedding = bir metni, anlamını temsil eden sayı vektörüne (ör. 1024 boyutlu)
> çevirme işlemi. Benzer anlamlı metinlerin vektörleri birbirine yakın olur;
> arama bu yakınlık üzerinden yapılır.

Projede değerlendirilen üç model: **all-MiniLM-L6-v2**, **multilingual-e5**,
**BAAI/bge-m3**. **Nihai tercih: BGE-M3** (kodda `services/embeddings.py`).

### 3.1. all-MiniLM-L6-v2 (sentence-transformers)

| Özellik | Değer |
|---|---|
| Boyut (dimension) | **384** |
| Parametre | ~22 milyon (çok küçük) |
| Model boyutu | ~80 MB |
| Maksimum girdi | ~256 token |
| Dil | Ağırlıklı **İngilizce** |
| Çıktı tipi | Sadece **dense** |

**Artıları:**
- Çok küçük ve **çok hızlı**, düşük VRAM/CPU bile yeter.
- Prototip ve İngilizce semantik arama için ideal başlangıç.

**Eksileri:**
- **Türkçe zayıf** — çok dilli değil, TR sorularda kalite düşer.
- **256 token sınırı** çok kısa; uzun akademik paragraflar kesilir.
- Sadece dense — **lexical/kelime eşleşmesi yok** (terim, kod, formül adı gibi
  birebir eşleşmesi gereken durumlarda zayıf).

### 3.2. multilingual-e5 (intfloat/multilingual-e5-base / -large)

| Özellik | Değer |
|---|---|
| Boyut | base: **768**, large: **1024** |
| Parametre | base ~278M, large ~560M |
| Maksimum girdi | ~512 token |
| Dil | **100+ dil** (Türkçe dahil) |
| Çıktı tipi | Sadece **dense** |
| Özel kural | Metne `query:` / `passage:` ön-eki eklemek gerekir |

**Artıları:**
- **Güçlü çok dilli** destek — Türkçe retrieval kalitesi MiniLM'den belirgin iyi.
- Retrieval için özel eğitilmiş, kaliteli dense vektörler.

**Eksileri:**
- Yine **dense-only** — hybrid (sparse) için ayrı bir BM25/SPLADE kurmak gerekir.
- **512 token** sınırı orta seviye; çok uzun bağlamlarda BGE-M3'ün gerisinde.
- `query:`/`passage:` ön-ek zorunluluğu — unutulursa kalite düşer (operasyonel risk).

### 3.3. BAAI/bge-m3 — ✅ SEÇİLEN MODEL

| Özellik | Değer |
|---|---|
| Boyut | **1024** (dense) |
| Parametre | ~568M |
| Maksimum girdi | **8192 token** (çok uzun!) |
| Dil | **100+ dil** (Türkçe güçlü) |
| Çıktı tipi | **Dense + Sparse + ColBERT (multi-vector)** — tek geçişte |

**"M3" ne demek?** **M**ulti-Functionality (dense+sparse+colbert),
**M**ulti-Linguality (çok dilli), **M**ulti-Granularity (kısa cümleden 8K
token'lık belgeye kadar).

**Neden bu seçildi (artıları):**
- **Tek modelden hem dense hem sparse** üretir → ayrı bir BM25/SPLADE altyapısına
  gerek kalmadan **hybrid arama** kurulabilir. (Kodda: `encode(..., return_dense=True,
  return_sparse=True)` tek forward pass'te ikisini de döndürür.)
- **8192 token bağlam** — uzun akademik metinleri kesmeden işler.
- **Güçlü çok dilli** — Türkçe + İngilizce karışık korpus için ideal.
- Ön-ek (`query:`) gerektirmez — operasyonel olarak basit.

**Eksileri:**
- **Daha ağır** (~2.3 GB indirme) ve MiniLM/e5'ten **daha yavaş**.
- Daha fazla VRAM ister (ama 16 GB'de sorun değil — sıralı yükleme stratejisi var).

### 3.4. Özet Karşılaştırma Tablosu

| Kriter | all-MiniLM-L6-v2 | multilingual-e5 | **BGE-M3 (seçilen)** |
|---|---|---|---|
| Boyut (dim) | 384 | 768 / 1024 | **1024** |
| Max token | 256 | 512 | **8192** |
| Türkçe | Zayıf | İyi | **Çok iyi** |
| Dense | ✅ | ✅ | ✅ |
| Sparse (lexical) | ❌ | ❌ | ✅ |
| Hız | En hızlı | Orta | En yavaş |
| Boyut/VRAM | En küçük | Orta | En büyük |
| Hybrid'e uygun | Hayır | Kısmen | **Evet (tek model)** |

**Sonuç:** Hız/boyut öncelikse MiniLM; sadece çok dilli dense yeterliyse e5;
ama **çok dilli + uzun bağlam + tek modelde hybrid** istiyorsan kazanan
**BGE-M3**. Proje bu üçüncü ihtiyacı taşıdığı için BGE-M3 seçildi.

---

## 4. Vektör Veritabanları — Karşılaştırma

> Vektör DB = embedding vektörlerini saklayan ve "bu vektöre en yakın N vektör
> hangileri?" sorgusunu (ANN — Approximate Nearest Neighbor) çok hızlı yanıtlayan
> özel veritabanı.

Değerlendirilenler: **ChromaDB**, **Milvus (standart sürüm)**, **Qdrant**.
**Nihai tercih: Qdrant** (kodda `services/vectorstore.py`).

### 4.1. ChromaDB

**Ne:** Python-native, hafif, gömülü (embedded) çalışabilen vektör DB.

**Artıları:**
- **Kurulumu en kolay** — `pip install chromadb`, kod içinde in-process çalışır.
- Prototip / küçük projeler / notebook deneyleri için harika.
- Basit, sezgisel API.

**Eksileri:**
- **Hybrid (dense+sparse) arama** desteği geç gelen / sınırlı bir özellik;
  BGE-M3'ün sparse çıktısını verimli kullanmak için ideal değil.
- Ölçekte (milyonlarca+ vektör) ve eşzamanlı yükte daha az olgun.
- Tek-düğüm odaklı; production-grade filtreleme/index seçenekleri Qdrant/Milvus
  kadar zengin değil.

### 4.2. Milvus (standart/normal sürüm)

**Ne:** Dağıtık (distributed), bulut-doğmuş (cloud-native), kurumsal ölçekli
vektör DB.

**Artıları:**
- **Devasa ölçek** — milyarlarca vektörü kaldırır.
- Çok sayıda index tipi (IVF, HNSW, DiskANN, hatta GPU index'leri).
- Yatay ölçeklenebilir, yüksek erişilebilirlik.

**Eksileri:**
- **Ağır mimari** — standalone sürüm bile genelde **etcd + MinIO + milvus**
  bileşenlerini (birden çok container) ister. Tek laptop için **fazla iri**.
- Operasyonel karmaşıklık ve yüksek kaynak tüketimi.
- Bu projenin ölçeği (birkaç PDF / on binler mertebesinde chunk) için **gereğinden
  fazla** (overkill).

### 4.3. Qdrant — ✅ SEÇİLEN

**Ne:** Rust ile yazılmış, tek-binary, hızlı, hafif vektör DB.

**Neden bu seçildi (artıları):**
- **Named vectors** desteği: tek bir koleksiyonda hem `dense` (1024-d) hem
  `sparse` vektör barındırır → **BGE-M3 hybrid için birebir uygun**. (Kodda
  koleksiyon `dense` + `sparse` named vector ile oluşturuluyor.)
- **Payload index** ile filtreleme/dedup sorguları hızlı (örn. `fingerprint`,
  `source`, `type` alanlarında index).
- **Tek container**, basit Docker dağıtımı, REST + gRPC.
- Performans / basitlik dengesi bu ölçek için ideal; web dashboard'u (`:6333/dashboard`) var.

**Eksileri:**
- Milvus kadar "devasa dağıtık ölçek" için tasarlanmamış — ama bu proje için
  fazlasıyla yeterli.

### 4.4. Özet Karşılaştırma Tablosu

| Kriter | ChromaDB | Milvus (standart) | **Qdrant (seçilen)** |
|---|---|---|---|
| Kurulum kolaylığı | En kolay | En zor (çok bileşen) | Kolay (tek container) |
| Footprint | Küçük | Çok büyük | Küçük/orta |
| Ölçek | Küçük-orta | Devasa (milyarlar) | Orta-büyük |
| Dense + Sparse (hybrid) | Sınırlı | Var | **Native (named vectors)** |
| Filtreleme/payload index | Temel | Güçlü | **Güçlü** |
| Bu projeye uygunluk | Prototip | Overkill | **İdeal** |

**Sonuç:** Hızlı prototip → Chroma; kurumsal/milyarlarca vektör → Milvus; ama
**tek makinede, BGE-M3 hybrid'i temiz şekilde kurmak** için kazanan **Qdrant**.

---

## 5. Chunking (Parçalama) ve Token Detayları

> Chunk = uzun bir belgeyi, embedding'i alınabilir ve arama yapılabilir küçük
> parçalara bölmek. Çok büyük chunk → arama az isabetli; çok küçük chunk →
> bağlam kopuk. Denge önemli.

**Kullanılan ayarlar** (`services/config.py` ve `.env.example`):

| Parametre | Değer | Açıklama |
|---|---|---|
| `CHUNK_SIZE` | **1024 token** | Her parçanın boyutu (BGE-M3 tokenizer ile sayılır) |
| `CHUNK_OVERLAP` | **128 token** | Komşu parçalar arası örtüşme (bağlam kopmasın diye) |
| Tokenizer | **BGE-M3 tokenizer** | Embedding modeliyle birebir aynı tokenizer |
| Splitter | LangChain `RecursiveCharacterTextSplitter.from_huggingface_tokenizer` | Önce paragraf, sonra cümle, sonra kelime sınırından böler |
| Ayraç önceliği | `["\n\n", "\n", ". ", " ", ""]` | Doğal sınırlardan bölmeye çalışır |

**Neden 1024 token?**
- BGE-M3'ün 8192 token kapasitesinin altında, **retrieval kalitesi için tatlı nokta**.
- 1024 token ≈ 700-800 kelime ≈ 1.5-2 akademik paragraf — anlamlı, bütünlüklü
  bir bağlam birimi.
- 128 token örtüşme (~%12.5), bir cümlenin tam parça sınırına denk gelip
  bölünmesini engeller.

**Embedding sırasında token sınırları (`services/embeddings.py`):**
- Passage (belge chunk'ı) embed: `max_length = chunk_size + 64 = 1088` token.
- Query (kullanıcı sorusu) embed: `max_length = 512` token.

**Token nedir?** Tokenizer'ın metni böldüğü en küçük birim. Kabaca İngilizce'de
1 token ≈ 0.75 kelime; Türkçe'de ekler yüzünden bir kelime birkaç token olabilir.
Önemli olan: chunk boyutunu **karakterle değil token'la** ölçmek, çünkü model de
token'la çalışır.

---

## 6. Retrieval (Arama) ve Fusion Detayları

Akış (`services/retriever.py` + `services/fusion.py`):

1. Kullanıcı sorusu BGE-M3 ile **tek geçişte** dense + sparse vektöre çevrilir.
2. Qdrant'tan **iki ayrı arama** yapılır:
   - Dense (anlamsal benzerlik) — `named="dense"`
   - Sparse (kelime/lexical eşleşme) — `named="sparse"`
3. Her kanaldan fazladan aday çekilir: `over_fetch = max(TOP_K*2, 30)`.
4. İki sıralı liste **Weighted Reciprocal Rank Fusion (RRF)** ile birleştirilir:

   ```
   score(d) = Σ_kanal ( ağırlık_kanal / (k + rank_kanal(d)) )
   ```

   - `DENSE_WEIGHT = 0.6` (anlamsal kanal ağırlığı)
   - `SPARSE_WEIGHT = 0.4` (lexical kanal ağırlığı)
   - `RRF_K = 60` (standart RRF sabiti — yüksek sıralardaki farkı yumuşatır)
5. Birleşik listeden en iyi `RERANK_TOP_N = 8` chunk LLM'e bağlam olarak verilir.

**Neden hybrid?** Dense kanal "anlamca benzer ama farklı kelime" durumlarını
yakalar (ör. "araba" ↔ "otomobil"); sparse kanal ise "birebir terim eşleşmesi"
gereken durumları (ör. bir protokol adı, fonksiyon adı, kısaltma) yakalar. İkisini
RRF ile birleştirince tek başına her birinden daha isabetli sonuç gelir.

---

## 7. Diğer Önemli Bileşenler

- **LLM'ler (Ollama üzerinden):** Llama 3.1 8B (q8), Qwen2.5 14B (q4), Gemma 2 9B (q4).
  Arayüzden canlı seçilebilir; `/chat/benchmark` ile üçü aynı anda kıyaslanabilir
  (gecikme, ilk token süresi, token/sn).
- **VLM (görsel anlama):** Moondream2 — PDF içindeki görsel/tablo/grafiklerin
  metin özetini üretir; bu özetler de aranabilir hale gelir.
- **OCR:** PyMuPDF metni az olan (taranmış) sayfalarda EasyOCR'a düşer (TR + EN).
- **Dedup (tekrar önleme):** `services/pdf_fingerprint.py` — dosya hash'i +
  içerik hash'i (ilk 3 sayfanın normalize metni + başlık/yazar). Aynı belge farklı
  isimle yüklenirse yakalanır. (Sadece başlık/yazar eşleşmesi bilerek dedup
  sebebi sayılmaz — "Ders Notları" gibi ortak başlıklar yanlış pozitif veriyordu.)
- **Auth:** Keycloak (OIDC) — RS256 imzalı JWT, instructor/student rolleri.
  Frontend JWT'yi tarayıcı `localStorage`'ında tutar (F5'e dayanıklı, URL'de token yok).
- **Ses:** Tarayıcı Web Speech API — mikrofonla soru sorma + cevabı sesli okuma
  (TR/EN), ses cihazdan çıkmaz.
- **Tema:** Açık/koyu tema geçişi (sidebar üstünde), chat görselleri makul boyutta.

---

## 8. Adım Adım Kurulum (GitHub'dan Çekip Çalıştırma)

> Başka biri projeyi kendi bilgisayarında sıfırdan nasıl çalıştırır:

### 8.1. Ön Gereksinimler

- **Docker** ve **Docker Compose** kurulu olmalı (Windows'ta Docker Desktop).
- **NVIDIA GPU** + CUDA desteği. Önerilen **16 GB+ VRAM** (Qwen2.5 14B için);
  tek başına Llama 3.1 8B için 8 GB asgari.
- **NVIDIA Container Toolkit** (Docker'ın GPU'ya erişmesi için).
- **Chrome / Edge** (ses özellikleri en iyi Chromium tabanlı tarayıcıda çalışır).

### 8.2. Depoyu Klonla

```bash
git clone https://github.com/mustafayazbahar/Multimodal-RAG-Project.git
cd Multimodal-RAG-Project
```

### 8.3. Ortam Değişkenlerini Ayarla

```bash
cp .env.example .env
# .env dosyasını aç; en azından ADMIN_PASSWORD ve JWT_SECRET değerlerini değiştir.
```

(Windows PowerShell'de `cp` yerine `Copy-Item .env.example .env`)

### 8.4. Tüm Stack'i Başlat

```bash
docker compose up -d --build
```

- İlk build **10-15 dk** sürebilir (BGE-M3 ≈ 2.3 GB + temel imajlar iner).
- Logları izlemek için: `docker compose logs -f`

### 8.5. LLM Modellerini İndir (sadece ilk kurulumda)

Ollama container'ı boş başlar. İstediğin modelleri çek:

```bash
docker exec -it deepcampus_ollama ollama pull llama3.1:8b-instruct-q8_0
docker exec -it deepcampus_ollama ollama pull qwen2.5:14b-instruct-q4_K_M
docker exec -it deepcampus_ollama ollama pull gemma2:9b-instruct-q4_K_M
```

> Alternatif: Arayüzdeki sidebar → **Download more models** (instructor rolüyle)
> üzerinden de canlı ilerleme ile indirilebilir.

| Model | Quant | VRAM | ~tok/s (RTX 4080) | Not |
|---|---|---|---|---|
| Llama 3.1 8B | q8_0 | ~8.5 GB | ~35 | En dengeli temel |
| Qwen2.5 14B | q4_K_M | ~9 GB | ~25 | Türkçe + teknik içerikte en güçlü |
| Gemma 2 9B | q4_K_M | ~5.5 GB | ~40 | En hızlı; akademik İngilizce'de iyi |

### 8.6. Arayüzleri Aç

| URL | Ne |
|---|---|
| http://localhost:8501 | **Frontend (Streamlit) — ana uygulama** |
| http://localhost:8000/docs | Backend Swagger (API dokümantasyonu) |
| http://localhost:6333/dashboard | Qdrant koleksiyon paneli |
| http://localhost:8080 | Keycloak admin konsolu |

### 8.7. Varsayılan Giriş Bilgileri

| Yer | Kullanıcı | Şifre |
|---|---|---|
| **Uygulama girişi** (Streamlit) | `admin` | `admin123` (instructor rolü) |
| **Keycloak admin** | `adminn` | `admin123` |

> ⚠️ Laptop dışına açacaksan ikisini de mutlaka değiştir.

### 8.8. İlk Kullanım Akışı

1. http://localhost:8501 → `admin / admin123` ile giriş yap.
2. Sidebar → **Upload PDF** ile bir/birkaç PDF yükle.
3. **Process & update database** butonuna bas (ingestion çalışır: metin + görsel
   + embedding + Qdrant'a yazma).
4. Aktif LLM'i seç, chat kutusuna Türkçe veya İngilizce soru yaz.
5. Cevap stream halinde gelir; "View sources" ile hangi PDF/sayfadan geldiği,
   varsa görseller gösterilir.

### 8.9. Durdurma / Devam Ettirme

```bash
docker compose stop    # container'ları durdurur, veriyi (volume) korur
docker compose start   # her şeyi aynen geri getirir
docker compose down     # tam teardown (volume'lar '-v' eklemezsen yaşar)
```

### 8.10. Sık Karşılaşılan Durumlar

- **Kod değişti ama yansımıyor:** `backend`/`frontend` kodu imaja **build**
  sırasında kopyalanır (volume mount değil). Değişikliği görmek için:
  `docker compose up -d --build backend frontend`. Sadece `restart` yetmez.
- **Bilgi tabanını sıfırlamak:** Sidebar → Danger zone → **Reset knowledge base**
  (Qdrant koleksiyonu + state dosyasını siler). Bir PDF'i `docs/`'tan silmek
  yeterli değildir; fingerprint Qdrant'ta kalır.
- **Yeni bağımlılık ekledin:** `requirements.backend.txt`'e ekleyip
  `docker compose up -d --build backend` yap; tüm requirements yeniden kurulur
  (Docker katman cache'i değişmeyen kısımları korur).

---

## 9. Konfigürasyon Cetveli (en etkili env değişkenleri)

| Değişken | Varsayılan | Açıklama |
|---|---|---|
| `LLM_MODEL` | `llama3.1:8b-instruct-q8_0` | Seçim yapılmazsa kullanılan model |
| `AVAILABLE_LLMS` | 3 modellik liste | Arayüzdeki menü |
| `EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding modeli |
| `CHUNK_SIZE` | 1024 | Chunk başına token |
| `CHUNK_OVERLAP` | 128 | Chunk'lar arası örtüşme token'ı |
| `TOP_K` | 20 | RRF öncesi her kanaldan çekilen aday |
| `RERANK_TOP_N` | 8 | LLM'e verilen nihai chunk sayısı |
| `DENSE_WEIGHT` | 0.6 | RRF dense (anlamsal) ağırlığı |
| `SPARSE_WEIGHT` | 0.4 | RRF sparse (lexical) ağırlığı |
| `RRF_K` | 60 | RRF sabiti |
| `MIN_IMAGE_BYTES` | 15000 | Bu boyuttan küçük gömülü görseller atlanır |
| `MIN_TEXT_CHARS` | 15 | Bu kadar metin çıkmayan sayfa OCR'a düşer |
| `TEMPERATURE` | 0.3 | LLM yaratıcılığı (düşük = belgeye sadık) |
| `HISTORY_WINDOW` | 4 | Modele geri verilen sohbet turu sayısı |
| `JWT_TTL_HOURS` | 12 | Oturum token ömrü |

---

## 10. Hızlı Cevap Kartı (sık sorulara)

- **"Hangi embedding modeli, neden?"** → BGE-M3. Çünkü tek modelde dense+sparse
  (hybrid), 8192 token uzun bağlam ve güçlü Türkçe. MiniLM küçük/hızlı ama TR
  zayıf ve dense-only; e5 çok dilli iyi ama yine dense-only ve 512 token.
- **"Hangi vektör DB, neden?"** → Qdrant. Named vectors ile dense+sparse'ı tek
  koleksiyonda tutar (BGE-M3 hybrid'e birebir), tek container, hızlı. Chroma
  prototip için, Milvus ise milyarlarca vektör için (bu ölçekte overkill).
- **"Chunk boyutu?"** → 1024 token, 128 token örtüşme, BGE-M3 tokenizer ile.
- **"Hybrid arama nasıl çalışıyor?"** → Dense + sparse aramaları ayrı yapılıp
  Weighted RRF (0.6 / 0.4, k=60) ile birleştirilir, en iyi 8 chunk LLM'e gider.
- **"Veri buluta gidiyor mu?"** → Hayır. LLM (Ollama), embedding (BGE-M3), VLM
  (Moondream2), vektör DB (Qdrant), auth (Keycloak) — hepsi yerel container.
- **"Multimodal kısmı nerede?"** → Moondream2 VLM, PDF görsellerini metne özetler;
  bu özetler de aranır ve ilgili görsel cevapta gösterilir.

---

*Hazırlanma tarihi: 2026 · DeepCampus v2.3 · Yazar: Mustafa Yazbahar*
