# DeepCampus — Multimodal Hybrid RAG Sistemi · Detaylı Proje Raporu

> Bu rapor, projeyi tanımayan birine ve proje hakkında soru soracak kişilere
> referans olması için hazırlandı. Kullanılan teknolojiler, neden seçildikleri,
> alternatiflerle karşılaştırmaları ve sistemin sıfırdan nasıl çalıştırılacağı
> adım adım anlatılır.

> **Sürüm notları**
> - **v2.5** — Çoklu sohbet başlığı ("Topics"): kullanıcı başına birden çok
>   adlandırılmış konu, rename / delete, silinemeyen sabit **General Chat**.
>   Giriş için artık birincil yol **OAuth Authorization Code akışı** — parola
>   Keycloak'ın kendi sayfasında girilir, Streamlit'ten geçmez. Eski email/
>   password formu "Other sign-in options" sekmesinde duruyor. localStorage
>   aktif konuyu hatırlıyor (F5 ile aynı thread'e geri dönülür). Yeni env'ler:
>   `KEYCLOAK_PUBLIC_URL`, `FRONTEND_URL`.
> - **v2.4** — PDF ayrıştırması için birincil pipeline **Docling** (IBM):
>   layout-aware metin, TableFormer ile tablo, layout dedektörüyle figür
>   kırpma. PyMuPDF + EasyOCR ise otomatik fallback. UI **tamamen
>   İngilizce** + sidebar üstünde **açık/koyu tema toggle**'ı. Chat
>   görselleri 420 px ile sınırlı; cevaplarda kalan `[GÖRSEL: ...]` etiketleri
>   süpürülüyor. Streamlit `<1.57` pin'lendi. Qdrant upsert'leri 64'lük
>   batch'lerle yapılıyor (120 sn client timeout) — büyük ders kitabı
>   ingestion'ları timeout'a düşmüyor. Dockerfile'lar **BuildKit pip cache
>   mount** ile inşa ediliyor; tekrar build 2-5 dk.
> - **v2.3** — Keycloak (OIDC) ile kimlik doğrulama, password-grant ile
>   token + Admin API ile kullanıcı oluşturma, Moondream görsel özetlerinin
>   `data/image_summaries.json`'a kalıcı yazımı.

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

### Ingestion (belge işleme) akışı (v2.4 — Docling primary)

```
PDF  →  fingerprint (dosya + içerik hash → tekrar varsa atla)
     │
     ├─►  BİRİNCİL YOL: Docling (IBM)
     │     • Layout-aware metin (okuma sırası, başlık/paragraf hiyerarşisi)
     │     • Tarama sayfalarında dahili OCR
     │     • TableFormer ile tablo bölgeleri → PNG kırpma
     │     • Layout dedektör ile figür/diyagram → PNG kırpma
     │
     └─►  FALLBACK (Docling açamazsa): PyMuPDF + EasyOCR
           (eski v2.3 pipeline'ı — bozuk/desteklenmeyen PDF için emniyet)
     │
     ▼
   görsel kırpıkları → Moondream2 (VLM) → "[IMAGE SUMMARY]: ..." metni
                     + data/image_summaries.json'a log
     │
     ▼
   metin + görsel-özetleri → BGE-M3 tokenizer ile chunk (1024 / 128 örtüşme)
     │
     ▼
   embedding: BGE-M3 (dense 1024-d + sparse)
     │
     ▼
   Qdrant upsert (64'lük batch, 120 sn timeout)
     payload: {source, page, type, image_kind, fingerprint}
```

Docling modelleri (layout + TableFormer) ilk ingestion'da HuggingFace'ten
~1-2 GB indirilir; `hf_cache` Docker volume'unda kalır, sonrakiler offline.

### VRAM yönetimi (16 GB GPU'da 3 büyük model)

Tek GPU'da çakışmayı önlemek için **sıralı devir-teslim** kullanılır:
1. Ingestion sırasında önce Moondream2 (VLM) yüklenir → görseller işlenir → bellekten atılır.
2. Sonra BGE-M3 yüklenir → tüm chunk'lar embed edilir → atılır.
3. Sorgu sırasında Ollama aktif LLM'i bellekte tutar; model değişiminde 2-3 sn'lik tahliye olur.
4. Ingestion'dan önce backend, Ollama'ya `keep_alive=0` gönderip LLM'i VRAM'den boşalttırır.

---

## 3. PDF Ayrıştırma — Docling + PyMuPDF Fallback (v2.4)

> v2.4 öncesi PDF'lerden metin/görsel çıkarımı sadece PyMuPDF + EasyOCR ile
> yapılıyordu. Yöntem hızlıydı ama **vektör tablolarını** ve **gömülü olmayan
> figürleri** (örn. çoğu ders kitabı diyagramı) sessizce kaçırıyordu —
> embed edilebilen sadece sayfaya rasterleştirilmiş gömülü görseller oluyordu.

### 3.1. Aday yaklaşımlar

| Araç | Yaklaşım | Tablo/Figür kalitesi | Hız | Bağımlılık |
|---|---|---|---|---|
| **PyMuPDF (fitz)** | Düşük seviye PDF parser | Sadece gömülü raster; vektörleri kaçırır | Çok hızlı | Tek `.so` |
| **pdfplumber** | PyMuPDF üstüne tablo heuristik'i | Basit ızgara tablolar OK, karmaşık layout zayıf | Orta | Saf Python |
| **Unstructured.io** | Geniş "her şeye" pipeline | İyi ama heuristik ağırlıklı | Yavaş | Çok bağımlılık |
| **Docling (IBM)** | Layout dedektör + **TableFormer** ML | En iyi: tablo yapısı + figür sınırları | Yavaş (GPU yardımcı) | Ağır (model ind.) |

### 3.2. Neden Docling — ✅ SEÇİLEN BİRİNCİL YOL

Docling üç ayrı ML modelini birleştirir:
1. **Layout dedektör** — sayfadaki text/title/list/picture/table bölgelerini
   bounding-box olarak tanır.
2. **TableFormer** — bulduğu tablo bölgelerinin hücre/satır/sütun yapısını çıkarır
   (yalnızca "burada tablo var" değil, "bu hücreler bu sütunlara denk").
3. **Reading order** — çok sütunlu / akademik layout'ta okuma sırasını çıkarır.

`generate_picture_images=True` ve `generate_table_images=True` bayraklarıyla
Docling tespit ettiği her tablo/figürü **PIL Image** olarak verir → biz de
`page_N_picture_M.png` / `page_N_table_M.png` adıyla diske yazarız. Sonra
Moondream2 her birine açıklama üretir.

**Artıları:**
- Computer Networks gibi ders kitaplarında **vektör tablolar** ve **diyagramlar**
  artık yakalanıyor (PyMuPDF bunları görmüyordu).
- Layout-aware okuma sırası → chunk'lar artık paragraf bütünlüğüne uyuyor.
- Built-in OCR (RapidOCR) tarama sayfalarını otomatik ele alıyor.

**Eksileri / hafifletme stratejisi:**
- İlk çalıştırmada HuggingFace'ten ~1-2 GB model indirir (sonrası cache).
- Saf PyMuPDF'ten **belirgin yavaş** (sayfa başına ~0.5-2 sn ekstra) — ama
  bu zaten "bir kerelik ingestion" maliyeti, sorgu zamanında yok.
- Bazı PDF'lerde başarısız olabiliyor → bu durumda kod **otomatik olarak**
  eski PyMuPDF + EasyOCR yoluna düşer (`services/ingestion.py` içinde
  try/except). Yani Docling kötü olsa bile o PDF indekslenmeden kalmaz.

### 3.3. Sayısal etki

Aynı 50 sayfalık ağ ders kitabıyla:

| Pipeline | Yakalanan tablo | Yakalanan figür | Süre (RTX 4080) |
|---|---|---|---|
| v2.3 (PyMuPDF + EasyOCR) | 2 (sadece gömülü raster) | 14 (sadece gömülü raster) | ~90 sn |
| **v2.4 (Docling primary)** | **47** (TableFormer) | **63** (layout det.) | ~210 sn |

Yani ders kitaplarındaki "kayıp" tabloların / diyagramların büyük çoğunluğu
artık aranabilir hale geliyor.

---

## 4. Embedding Modelleri — Karşılaştırma

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

## 5. Vektör Veritabanları — Karşılaştırma

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

## 6. Chunking (Parçalama) ve Token Detayları

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

## 7. Retrieval (Arama) ve Fusion Detayları

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

## 8. Diğer Önemli Bileşenler

- **LLM'ler (Ollama üzerinden):** Llama 3.1 8B (q8), Qwen2.5 14B (q4), Gemma 2 9B (q4).
  Arayüzden canlı seçilebilir; `/chat/benchmark` ile üçü aynı anda kıyaslanabilir
  (gecikme, ilk token süresi, token/sn).
- **VLM (görsel anlama):** Moondream2 — PDF içindeki görsel/tablo/grafiklerin
  metin özetini üretir; bu özetler de aranabilir hale gelir.
- **PDF parsing:** Birincil **Docling** (Bölüm 3); başarısız olursa otomatik
  PyMuPDF + EasyOCR yedek pipeline'ı.
- **OCR:** Docling kendi built-in OCR'ını kullanır; fallback yolunda EasyOCR
  (TR + EN) metni az olan (taranmış) sayfalarda devreye girer.
- **Dedup (tekrar önleme):** `services/pdf_fingerprint.py` — dosya hash'i +
  içerik hash'i (ilk 3 sayfanın normalize metni + başlık/yazar). Aynı belge farklı
  isimle yüklenirse yakalanır. (Sadece başlık/yazar eşleşmesi bilerek dedup
  sebebi sayılmaz — "Ders Notları" gibi ortak başlıklar yanlış pozitif veriyordu.)
- **Auth:** Keycloak (OIDC). v2.5 ile **OAuth Authorization Code akışı**
  birincil giriş yolu — parola Keycloak'ın kendi sayfasında girilir, Streamlit'ten
  geçmez. Eski password-grant formu API ve headless erişim için duruyor.
  Token'lar RS256 imzalı; instructor/student rolleri `realm_access.roles`'ten
  okunur. Frontend JWT'yi (artık `id_token` ve `active_session_id` ile birlikte)
  tarayıcı `localStorage`'ında tutar — F5'e dayanıklı, URL'de token yok.
- **Multi-session "Topics":** v2.5 ile kullanıcı başına birden çok adlandırılmış
  chat thread'i (Bölüm 9). Sabit "General Chat" varsayılan/silinemez; diğerlerine
  rename / delete butonları. Aktif konu `active_session_id` olarak localStorage'a
  yazılır.
- **Ses:** Tarayıcı Web Speech API — mikrofonla soru sorma + cevabı sesli okuma
  (TR/EN), ses cihazdan çıkmaz.
- **Tema:** Açık/koyu tema geçişi (sidebar üstünde), chat görselleri 420 px
  genişlikle sınırlandırılmıştır.
- **Qdrant batching:** v2.4 ile upsert'ler 64'lük batch'lerle ve 120 sn client
  timeout'u ile yapılır — büyük ders kitabı ingestion'ları (binlerce chunk)
  `ResponseHandlingException: timed out` hatası yememesi için.
- **Docker build cache:** Dockerfile'lar BuildKit pip cache mount (`--mount=
  type=cache,target=/root/.cache/pip`) ile inşa edilir; sadece değişen
  bağımlılıklar tekrar indirilir.

---

## 9. Multi-session "Topics" + OAuth Code Flow (v2.5)

### 9.1. Topics — kullanıcı başına çoklu sohbet thread'i

**Sorun:** v2.4'e kadar her kullanıcının tek bir mesaj geçmişi vardı. Bir
ders için sorulan sorular ve başka bir ders için sorulanlar aynı thread'de
birikiyordu — bağlam karışıyordu, geçmiş aramak zorlaşıyordu.

**Çözüm:** İki tablo (SQLite, `services/auth.py`):

```sql
chat_sessions(session_id TEXT PK, username TEXT, title TEXT,
              is_default INTEGER, created_at TIMESTAMP)
chat_history(id INTEGER PK, session_id TEXT, username TEXT,
             role TEXT, content TEXT, sources TEXT, images TEXT)
```

- Her kullanıcının ilk girişinde otomatik bir **General Chat** session'ı
  yaratılır (`is_default=1`).
- Yeni "Topic" oluşturulduğunda `is_default=0` ile başka bir session açılır.
- General Chat **rename / delete edilemez** — backend `update_session_title`
  ve `delete_session` `is_default=1` ise reddeder.
- `resolve_session(username, session_id)`: stale ya da yabancı bir
  `session_id` geldiğinde sessizce kullanıcının General Chat'ine düşer
  (frontend localStorage'dan silinmiş bir session id ile gelirse 404 atmaz).

**Migration:** v2.4'ten upgrade'de eski `chat_history` satırlarında
`session_id IS NULL` olur. `create_chat_table()` idempotent migration yapar:

1. Her distinct orphan username için bir General Chat session yaratır.
2. O username'in tüm orphan satırlarını o session'a `UPDATE` eder.

Yani upgrade'de eski geçmiş kaybolmaz, "General Chat" altında görünür.

**API:**

| Endpoint | İş |
|---|---|
| `GET /chat/sessions` | Kullanıcının tüm session'larını listele (General Chat ilk) |
| `POST /chat/sessions` body=`{title}` | Yeni topic yarat |
| `PATCH /chat/sessions/{id}` body=`{title}` | Rename (default reddedilir) |
| `DELETE /chat/sessions/{id}` | Sil (default reddedilir) |
| `GET /chat/history?session_id=...` | O thread'in mesajları |
| `POST /chat/query` body=`{...session_id}` | Sorgu + cevap; ilk SSE eventi `{event:"session", data:<resolved-id>}` |

**UI:** Sidebar'da "📚 My Topics" bölümü, "➕ Create new topic" expander,
her topic için (default değilse) ✏️ rename + 🗑️ delete butonları, aktif
topic yeşil daire (🟢). F5'te aktif topic localStorage'dan restore edilir.

### 9.2. OAuth Authorization Code akışı

**Sorun:** v2.3-v2.4'te giriş **password grant** ile yapılıyordu — kullanıcı
Streamlit'in kendi formuna parolasını yazıyor, Streamlit bunu backend'e POST
ediyor, backend de Keycloak'a iletip token alıyordu. Bu yaklaşımın iki
zayıflığı vardı:

1. Parola bizim kodumuzdan geçiyor — Streamlit'in herhangi bir bağımlılığı
   parolayı loglayabilir veya yanlışlıkla session_state'e bırakabilir.
2. Keycloak'ın güçlü "ekstra adım" özelliklerini (MFA, social login,
   "remember me", brute-force kilidi UI) atlıyoruz.

**Çözüm — OAuth 2.0 Authorization Code Flow:**

```
1. Kullanıcı "🔐 Sign in with Keycloak" butonuna basar.
2. Frontend, backend'den login URL'ini ister:
     GET /auth/login-url?redirect_uri=http://localhost:8501
3. Backend, Keycloak'ın /auth endpoint'inin tam URL'ini döner:
     http://localhost:8080/realms/deepcampus/protocol/openid-connect/auth
       ?client_id=streamlit-app
       &response_type=code
       &redirect_uri=http://localhost:8501
       &scope=openid profile email
4. Frontend tarayıcıyı bu URL'e yönlendirir (meta refresh, iframe'ten çıkar).
5. Kullanıcı Keycloak'ın KENDİ login sayfasında parolayı girer.
6. Keycloak başarılı login'de tarayıcıyı geri yönlendirir:
     http://localhost:8501/?code=ABC123...
7. Streamlit `?code=` parametresini görür, backend'e POST eder:
     POST /auth/exchange-code  body={code, redirect_uri}
8. Backend Keycloak'a server-to-server token isteği gönderir, sonucu döner:
     {access_token, id_token, role, username}
9. Frontend bunları localStorage'a yazıp normal kullanıma başlar.
```

**Neden iki Keycloak URL'i?**
- `KEYCLOAK_URL=http://keycloak:8080` — backend'in Docker iç ağında kullandığı
  (token exchange, JWKS fetch, Admin API).
- `KEYCLOAK_PUBLIC_URL=http://localhost:8080` — kullanıcının tarayıcısı için.
  Browser `keycloak:8080`'i çözemez, port mapping ile `localhost:8080` üzerinden
  erişir. Authorization Code URL'i bu public host'la üretilir.

**Logout:** `id_token_hint` ile **silent logout** (Keycloak konfirme sormaz),
yoksa Keycloak "çıkmak istediğine emin misin?" sayfası gösterir. `id_token`
login response'unda gelir, localStorage'a yazılır, logout'ta sunucudan
alınan end-session URL'ine query param olarak gider.

**Geriye dönük uyum:** Password-grant `POST /auth/login` endpoint'i hâlâ
çalışıyor — Streamlit UI'da "Other sign-in options > Email / password"
seçeneği var, ayrıca API client'lar (otomatik testler, curl, mobil) için
duruyor. Yani v2.4'ten upgrade'de mevcut otomasyonlar bozulmaz.

**Realm config (`docker/keycloak/realm-deepcampus.json`):** Halihazırda her
iki akışı destekleyecek şekilde seedlenmişti:
- `directAccessGrantsEnabled: true` (password grant için)
- `standardFlowEnabled: true` (code flow için)
- `redirectUris: ["*"]` ve `webOrigins: ["*"]` (dev için; production'da
  daraltılmalı)

---

---

## 10. Adım Adım Kurulum (GitHub'dan Çekip Çalıştırma)

> Başka biri projeyi kendi bilgisayarında sıfırdan nasıl çalıştırır:

### 10.1. Ön Gereksinimler

- **Docker** ve **Docker Compose** kurulu olmalı (Windows'ta Docker Desktop).
- **NVIDIA GPU** + CUDA desteği. Önerilen **16 GB+ VRAM** (Qwen2.5 14B için);
  tek başına Llama 3.1 8B için 8 GB asgari.
- **NVIDIA Container Toolkit** (Docker'ın GPU'ya erişmesi için).
- **Chrome / Edge** (ses özellikleri en iyi Chromium tabanlı tarayıcıda çalışır).

### 10.2. Depoyu Klonla

```bash
git clone https://github.com/mustafayazbahar/Multimodal-RAG-Project.git
cd Multimodal-RAG-Project
```

### 10.3. Ortam Değişkenlerini Ayarla (opsiyonel)

`.env` zorunlu **değil** — `docker-compose.yml` her env için
`${VAR:-default}` fallback'i tanımlı. Yani local dev için doğrudan
çalıştırabilirsin. `.env` ancak şu durumlarda gerekir:
- Production'a deploy ediyorsan (farklı Keycloak host'u, gerçek parola).
- Default admin parolasını değiştirmek istiyorsan.
- Default LLM modelini değiştirmek istiyorsan.

```bash
cp .env.example .env
# .env dosyasını aç; en azından ADMIN_PASSWORD ve JWT_SECRET değerlerini değiştir.
```

(Windows PowerShell'de `cp` yerine `Copy-Item .env.example .env`)

**v2.5 ile gelen yeni env'ler** (default'lar dev için doğrudur):
- `KEYCLOAK_PUBLIC_URL=http://localhost:8080` — tarayıcının Keycloak'a
  ulaşacağı URL (OAuth Code akışı redirect'inde gömülüdür). Production'da
  `https://auth.example.com` gibi olur.
- `FRONTEND_URL=http://localhost:8501` — OAuth redirect_uri olarak
  Keycloak'a verilen Streamlit adresi.

### 10.4. Tüm Stack'i Başlat

```bash
docker compose up -d --build
```

- İlk build **15-25 dk** sürebilir (BGE-M3 ≈ 2.3 GB + PyTorch CUDA ≈ 2.5 GB
  + temel imajlar iner).
- v2.4'ten beri Dockerfile'lar **BuildKit pip cache mount** kullanır —
  sonraki build'lerde değişmeyen wheel'ler tekrar indirilmez, 2-5 dk'da biter.
- Logları izlemek için: `docker compose logs -f`
- Canlı progress için: `docker compose --progress=plain up -d --build`

**İlk ingestion'da ek indirme:** Docling modelleri (~1-2 GB) HuggingFace'ten
gelir. `hf_cache` Docker volume'unda kalır → sonraki ingestion'larda offline.

### 10.5. LLM Modellerini İndir (sadece ilk kurulumda)

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

### 10.6. Arayüzleri Aç

| URL | Ne |
|---|---|
| http://localhost:8501 | **Frontend (Streamlit) — ana uygulama** |
| http://localhost:8000/docs | Backend Swagger (API dokümantasyonu) |
| http://localhost:6333/dashboard | Qdrant koleksiyon paneli |
| http://localhost:8080 | Keycloak admin konsolu |

### 10.7. Varsayılan Giriş Bilgileri

| Yer | Kullanıcı | Şifre |
|---|---|---|
| **Uygulama girişi** (Streamlit) | `admin` | `admin123` (instructor rolü) |
| **Keycloak admin** | `adminn` | `admin123` |

> ⚠️ Laptop dışına açacaksan ikisini de mutlaka değiştir.

### 10.8. İlk Kullanım Akışı (v2.5)

1. http://localhost:8501 → **🔐 Sign in with Keycloak** butonuna bas.
2. Tarayıcı Keycloak'a yönlenir → `admin / admin123` ile giriş yap →
   DeepCampus'a `?code=...` ile döner → otomatik logged in.
   *(Alternatif: "Other sign-in options" expander'ı altında eski email/
   password formu da var; aynı sonuç.)*
3. Sidebar → **➕ Create new topic** ile bir konu aç (örn. "Computer
   Networks Ch.4"). Mesajlar artık aktif konuda birikir; sidebar'dan
   konular arası geçiş yapabilirsin.
4. (Instructor) Sidebar → **Upload PDF** ile bir/birkaç PDF yükle.
5. **Process & update database** butonuna bas. İlk run'da Docling
   modelleri iner (~1-2 GB), sonrasındakiler hızlı.
6. Aktif LLM'i seç, chat kutusuna Türkçe veya İngilizce soru yaz.
7. Cevap stream halinde gelir; "View sources" ile hangi PDF/sayfadan
   geldiği, varsa görseller (max 420 px) gösterilir.

### 10.9. Durdurma / Devam Ettirme

```bash
docker compose stop    # container'ları durdurur, veriyi (volume) korur
docker compose start   # her şeyi aynen geri getirir
docker compose down     # tam teardown (volume'lar '-v' eklemezsen yaşar)
```

### 10.10. Sık Karşılaşılan Durumlar

- **Kod değişti ama yansımıyor:** `backend`/`frontend` kodu imaja **build**
  sırasında kopyalanır (volume mount değil). Değişikliği görmek için:
  `docker compose up -d --build backend frontend`. Sadece `restart` yetmez.
- **Bilgi tabanını sıfırlamak:** Sidebar → Danger zone → **Reset knowledge base**
  (Qdrant koleksiyonu + state dosyasını siler). Bir PDF'i `docs/`'tan silmek
  yeterli değildir; fingerprint Qdrant'ta kalır.
- **Yeni bağımlılık ekledin:** `requirements.backend.txt`'e ekleyip
  `docker compose up -d --build backend` yap. BuildKit pip cache mount
  sayesinde sadece yeni paket indirilir, eskiler cache'ten gelir.
- **Keycloak login butonu Streamlit içinde sandboxed kalıyor:** `app.py`'daki
  `_start_keycloak_login()` meta-refresh kullanır (`st.markdown(... meta
  refresh ...)`). Tarayıcı eski sekmede açık kaldıysa yenile.
- **`?code=...` URL'de takılı kaldı:** Frontend `code` paramını işledikten
  sonra `st.query_params.clear()` ile siler. Manual yenilemede bir saniye
  görünebilir, sorun değil.
- **F5'ten sonra General Chat'e düşüyor:** localStorage yazımı için 0.4 sn
  bekleme var; çok hızlı F5 atılırsa kaçabilir. Tek bir tıklama sonrası
  refresh yeterli.

---

## 11. Konfigürasyon Cetveli (en etkili env değişkenleri)

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
| `KEYCLOAK_URL` | `http://keycloak:8080` | Backend'in iç ağdaki Keycloak adresi (JWKS, token exchange, Admin API) |
| `KEYCLOAK_PUBLIC_URL` | `http://localhost:8080` | **v2.5**: tarayıcı için Keycloak adresi (OAuth redirect'inde gömülü) |
| `FRONTEND_URL` | `http://localhost:8501` | **v2.5**: OAuth redirect_uri olarak kullanılır, tarayıcı buraya geri döner |

---

## 12. Hızlı Cevap Kartı (sık sorulara)

- **"Hangi embedding modeli, neden?"** → BGE-M3. Çünkü tek modelde dense+sparse
  (hybrid), 8192 token uzun bağlam ve güçlü Türkçe. MiniLM küçük/hızlı ama TR
  zayıf ve dense-only; e5 çok dilli iyi ama yine dense-only ve 512 token.
- **"Hangi vektör DB, neden?"** → Qdrant. Named vectors ile dense+sparse'ı tek
  koleksiyonda tutar (BGE-M3 hybrid'e birebir), tek container, hızlı. Chroma
  prototip için, Milvus ise milyarlarca vektör için (bu ölçekte overkill).
- **"PDF parsing nasıl?"** → **Birincil Docling** (IBM): layout dedektör +
  TableFormer + figür kırpma. Başarısız olursa otomatik **PyMuPDF + EasyOCR**
  fallback. v2.4'ten beri ders kitabı tabloları ve diyagramları da yakalanıyor.
- **"Chunk boyutu?"** → 1024 token, 128 token örtüşme, BGE-M3 tokenizer ile.
- **"Hybrid arama nasıl çalışıyor?"** → Dense + sparse aramaları ayrı yapılıp
  Weighted RRF (0.6 / 0.4, k=60) ile birleştirilir, en iyi 8 chunk LLM'e gider.
- **"Veri buluta gidiyor mu?"** → Hayır. LLM (Ollama), embedding (BGE-M3), VLM
  (Moondream2), PDF parse (Docling), vektör DB (Qdrant), auth (Keycloak) —
  hepsi yerel container.
- **"Multimodal kısmı nerede?"** → Docling PDF içindeki tabloları/figürleri
  kırpar; Moondream2 VLM her kırpık için metin özet üretir; bu özetler de
  embed edilip arandığında ilgili görsel cevapta gösterilir.
- **"Giriş nasıl çalışıyor?"** → v2.5'ten beri birincil yol **OAuth Code
  flow**: butona basınca tarayıcı Keycloak'a yönlenir, parola Keycloak'ın
  sayfasında girilir, `?code=` ile dönülür, backend `code`'u token'a
  çevirir. Eski email/password formu API ve fallback için duruyor.
- **"Topics nedir?"** → Kullanıcı başına birden çok adlandırılmış chat
  thread'i. Sabit "General Chat" var (silinemez), diğerleri rename / delete
  edilebilir. Aktif konu localStorage'a yazılır, F5'te aynı thread'e dönülür.
- **"Eski tek-thread geçmişim ne oldu?"** → İlk v2.5 boot'unda otomatik
  migration: tüm orphan mesajlar her kullanıcı için yaratılan General Chat'e
  reparent edildi. Kayıp yok.
- **".env şart mı?"** → Hayır. `docker-compose.yml` her env için fallback
  default tutuyor. Production'a deploy ederken veya parolaları değiştirmek
  istediğinde `.env` lazım olur.

---

*Hazırlanma tarihi: 2026 · DeepCampus v2.5 · Yazar: Mustafa Yazbahar*
