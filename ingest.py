import os
import sys
import fitz  # PyMuPDF
import torch
import hashlib
import easyocr
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# Terminal UTF-8 ve SSL Sorunlarını Çözmek İçin
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
sys.stdout.reconfigure(encoding='utf-8')

DOCS_FOLDER = "docs"
DB_FOLDER = "chroma_db"
IMG_FOLDER = "docs_images"

# DONANIM TESPİTİ (CUDA & Apple MPS Desteği)
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

# ---------------------------------------------------------
# 1. MOTOR: MOONDREAM2 (GÖRSEL YAPAY ZEKA - VLM)
# ---------------------------------------------------------
model_id = "vikhyatk/moondream2"
target_revision = "2024-08-26"

print(f"\n[BILGI] Yerel Görsel Yapay Zeka (Moondream2) yükleniyor...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=target_revision, trust_remote_code=True)
    moondream_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=target_revision
    )
    
    # RTX 4080 ve Mac'ler icin Float16 hizlandirmasi (Tensor Cores)
    if device in ["cuda", "mps"]:
        moondream_model = moondream_model.to(device=device, dtype=torch.float16)
    else:
        moondream_model = moondream_model.to(device)
        
    moondream_model.eval()
    print(f"[BASARILI] Moondream2 '{device}' üzerinde aktif! 🚀")

except Exception as e:
    print(f"[KRITIK HATA] Moondream yüklenemedi: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 2. MOTOR: EASYOCR (HİBRİT FALLBACK / TARANMIŞ BELGE)
# ---------------------------------------------------------
print("\n[BILGI] EasyOCR (B Planı) yükleniyor...")
ocr_reader = easyocr.Reader(['tr', 'en'], gpu=True if device in ["cuda", "mps"] else False, verbose=False)
print(f"[BASARILI] EasyOCR aktif! 🛡️")

# ---------------------------------------------------------
# GÖRSEL ANALİZ FONKSİYONU
# ---------------------------------------------------------
def summarize_image_locally(image_path):
    try:
        image = Image.open(image_path)
        enc_image = moondream_model.encode_image(image)
        strict_prompt = "Describe this image, chart, or table briefly for a search engine. Read and extract all visible text, column names, and data strictly. Be concise and technical."
        answer = moondream_model.answer_question(enc_image, strict_prompt, tokenizer)
        return answer
    except Exception as e:
        return "Görsel içeriği teknik nedenlerle çözümlenemedi."

# ---------------------------------------------------------
# ANA VERİ BORU HATTI
# ---------------------------------------------------------
def main():
    print("\n[BILGI] Token-Bazlı Hibrit İşleme Başlatılıyor...")
    
    for folder in [DOCS_FOLDER, IMG_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    # E5 EMBEDDING VE TOKENIZER YÜKLEME
    print("[BILGI] E5 Embedding ve Tokenizer yükleniyor...")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    e5_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")

    # Veritabanı Kontrolü
    existing_sources = set()
    db = None
    if os.path.exists(DB_FOLDER):
        try:
            db = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
            existing_data = db.get(include=['metadatas'])
            for meta in existing_data.get('metadatas', []):
                if meta and 'source' in meta:
                    existing_sources.add(meta['source'])
        except: pass

    files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith('.pdf')]
    new_files = [f for f in files if f not in existing_sources]

    if not new_files:
        print("[BASARILI] Sistem güncel.")
        return

    documents = []

    for file in new_files:
        file_path = os.path.join(DOCS_FOLDER, file)
        pdf_img_folder = os.path.join(IMG_FOLDER, file.replace(".pdf", ""))
        os.makedirs(pdf_img_folder, exist_ok=True)
        
        print(f"\n[ISLENIYOR] {file}...")
        try:
            pdf_document = fitz.open(file_path)
            seen_image_hashes = set()
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # A) HİBRİT METİN AYIKLAMA (Dijital + OCR Fallback)
                text = page.get_text("text").strip()
                if len(text) < 15: # Taranmış sayfa tespiti
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    ocr_results = ocr_reader.readtext(img_bytes, detail=0, paragraph=True)
                    text = "\n".join(ocr_results).strip()

                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": file, "page": page_num, "type": "text"}
                    ))
                
                # B) GÖRSEL AYIKLAMA (Filtreli)
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    if len(image_bytes) < 15000: continue # Küçük logo/çöp filtresi
                        
                    img_hash = hashlib.md5(image_bytes).hexdigest()
                    if img_hash in seen_image_hashes: continue # Kopya filtresi
                    seen_image_hashes.add(img_hash)
                        
                    image_name = f"page_{page_num+1}_img_{img_index+1}.{base_image['ext']}"
                    image_path = os.path.join(pdf_img_folder, image_name)
                    
                    with open(image_path, "wb") as f:
                        f.write(image_bytes)
                        
                    summary = summarize_image_locally(image_path)
                    documents.append(Document(
                        page_content=f"[GÖRSEL ÖZETİ]: {summary}",
                        metadata={"source": file, "page": page_num, "image_path": image_path, "type": "image"}
                    ))

            pdf_document.close()
        except Exception as e:
            print(f"[HATA] {file}: {e}")

    # TOKEN-BAZLI PARÇALAMA VE KAYIT
    if documents:
        print("\n[BILGI] E5 Tokenizer ile parçalama yapılıyor...")
        
        # 🚀 KRİTİK: Karakter değil, Token bazlı ayırıcı
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=e5_tokenizer,
            chunk_size=300,      # 300 Token
            chunk_overlap=50,    # 50 Token örtüşme
            separators=["\n\n", "\n", ".", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        for chunk in chunks:
            chunk.page_content = f"passage: {chunk.page_content}"
        
        if db is not None:
            db.add_documents(chunks)
        else:
            Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_FOLDER)
            
        print("[BASARILI] Veritabanı token-bazlı verilerle güncellendi.")

if __name__ == "__main__":
    main()
