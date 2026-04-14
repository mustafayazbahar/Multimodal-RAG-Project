# Multimodal RAG Projesi
# 🚀 Multimodal RAG Project: Çok Dilli Akademik Asistan

Bu proje, karmaşık akademik belgeleri (PDF) hem dijital metin hem de görsel içerik bazında analiz edebilen, **Role-Based Access Control (RBAC)** güvenlik altyapısına sahip bir **RAG (Retrieval-Augmented Generation)** sistemidir.

## ✨ Öne Çıkan Özellikler

* **Çok Dilli (Multilingual) Destek:** E5-base vektör modeli sayesinde 100+ dilde anlamsal arama ve diller arası sorgulama.
* **Görsel Zeka (Moondream2 VLM):** Sayfalardaki grafik, tablo ve şemaları analiz ederek metin bazlı özetler çıkarır.
* **Hibrit OCR Sistemi:** Dijital PDF'ler için `PyMuPDF`, taranmış belgeler için `EasyOCR` (B Planı) kullanarak veri kaybını sıfıra indirir.
* **Gelişmiş Güvenlik:** * SHA-256 şifreleme ile yerel SQLite tabanlı kullanıcı yönetimi.
    * `Instructor` (Yönetici) ve `Student` (Öğrenci) rolleri arasında kesin yetki ayrımı.
* **GPU Hızlandırma:** RTX 4080 gibi donanımları algılayarak CUDA/MPS üzerinde yüksek performanslı işleme.

## 🛠️ Teknoloji Yığını

- **Frontend:** Streamlit
- **Vektör Veritabanı:** ChromaDB
- **LLM / VLM:** Google Gemini & Moondream2
- **Embeddings:** Multilingual-E5-Base
- **Language:** Python 3.10+

## 🚀 Kurulum

1. Depoyu klonlayın:
   ```bash
   git clone [https://github.com/mustafayazbahar/Multimodal-RAG-Project.git](https://github.com/mustafayazbahar/Multimodal-RAG-Project.git)

2- Gerekli kütüphaneleri yükleyin
   pip install -r requirements.txt

3- .env dosyanızı oluşturun ve API anahtarınızı ekleyin:
   GOOGLE_API_KEY=your_api_key_here

4- Uygulamayı başlatın:
   streamlit run app.py