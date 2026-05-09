# Moondream VLM ve stabilite için Python 3.12 kullanıyoruz
FROM python:3.12-slim

# Best Practice: İşletim sistemi seviyesindeki bağımlılıklar 
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini ayarla
WORKDIR /app

# Best Practice: Sadece requirements.txt'yi önce kopyala (Docker Layer Caching avantajı için)
COPY requirements.txt .

# Bağımlılıkları önbellek kullanmadan, temiz bir şekilde kur
RUN pip install --no-cache-dir -r requirements.txt

# Projenin geri kalan tüm dosyalarını kopyala
COPY . .

# Streamlit'in varsayılan portunu dışa aç
EXPOSE 8501

# Uygulamayı başlat
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]