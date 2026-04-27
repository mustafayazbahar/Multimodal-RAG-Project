import os
import sys
import ssl
import json
import fitz  # PyMuPDF
import torch
import hashlib
import easyocr
import pandas as pd
import pdfplumber
import gc # 🚀 Çöp Toplayıcı

from io import BytesIO
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------------
# ORTAM AYARLARI
# ---------------------------------------------------------
ssl._create_default_https_context = ssl._create_unverified_context
sys.stdout.reconfigure(encoding="utf-8")

DOCS_FOLDER = "docs"
DB_FOLDER = "chroma_db"
IMG_FOLDER = "docs_images"

SAVE_PAGE_PREVIEWS = False          
SAVE_TABLE_PREVIEWS = True          
SAVE_IMAGE_FILES = True             
PAGE_PREVIEW_MAX_DIM = 1400         
MIN_TEXT_LENGTH_FOR_DIGITAL = 15    
MIN_IMAGE_BYTES = 15000             

SAVE_PREVIEW_FOR_OCR_PAGES = False   
SAVE_PREVIEW_FOR_TABLE_PAGES = True  

# 🚀 DONANIM (Windows + GTX 1080 İçin CUDA)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# 1) MOONDREAM2
# ---------------------------------------------------------
model_id = "vikhyatk/moondream2"
target_revision = "2024-08-26"

print("\n[BILGI] Yerel Görsel Yapay Zeka (Moondream2) yükleniyor...")
try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=target_revision,
        trust_remote_code=True
    )

    moondream_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        revision=target_revision
    )

    if device == "cuda" or device == "mps":
        moondream_model = moondream_model.to(device=device, dtype=torch.float16)
    else:
        moondream_model = moondream_model.to(device)

    moondream_model.eval()
    print(f"[BASARILI] Moondream2 '{device}' üzerinde aktif!")
except Exception as e:
    print(f"[KRITIK HATA] Moondream yüklenemedi: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# 2) EASYOCR
# ---------------------------------------------------------
print("\n[BILGI] EasyOCR yükleniyor...")
try:
    use_gpu_for_ocr = device == "cuda"
    ocr_reader = easyocr.Reader(["tr", "en"], gpu=use_gpu_for_ocr, verbose=False)
    print(f"[BASARILI] EasyOCR aktif! GPU={use_gpu_for_ocr}")
except Exception as e:
    print(f"[KRITIK HATA] EasyOCR yüklenemedi: {e}")
    sys.exit(1)

# ---------------------------------------------------------
# YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------
def ensure_folders():
    for folder in [DOCS_FOLDER, DB_FOLDER, IMG_FOLDER]:
        os.makedirs(folder, exist_ok=True)

def safe_json_dumps(data):
    try:
        return json.dumps(data, ensure_ascii=False)
    except Exception:
        return "{}"

def resize_image_if_needed(pil_image, max_dim=1400):
    width, height = pil_image.size
    largest = max(width, height)
    if largest <= max_dim:
        return pil_image
    scale = max_dim / largest
    new_size = (int(width * scale), int(height * scale))
    return pil_image.resize(new_size)

def save_pil_image(pil_image, out_path, optimize=True, quality=75):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pil_image.save(out_path, format="JPEG", optimize=optimize, quality=quality)

def render_page_to_pil(page, dpi=160):
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    img = Image.open(BytesIO(pix.tobytes("png")))
    return img

def render_page_to_png_bytes(page, dpi=160):
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix, alpha=False)
    return pix.tobytes("png")

def page_preview_path(pdf_img_folder, page_num):
    return os.path.join(pdf_img_folder, f"page_{page_num+1}_preview.jpg")

def table_preview_path(pdf_img_folder, page_num, table_index):
    return os.path.join(pdf_img_folder, f"page_{page_num+1}_table_{table_index+1}.jpg")

def summarize_image_locally(image_path):
    try:
        image = Image.open(image_path)
        enc_image = moondream_model.encode_image(image)
        strict_prompt = (
            "Describe this image, chart, or table briefly for a search engine. "
            "Read and extract all visible text, column names, and data strictly. "
            "Be concise and technical."
        )
        answer = moondream_model.answer_question(enc_image, strict_prompt, tokenizer)
        return answer.strip()
    except Exception as e:
        print(f"[UYARI] Görsel özetlenemedi: {e}")
        return "Görsel içeriği teknik nedenlerle çözümlenemedi."

def dataframe_to_searchable_text(df):
    try:
        df = df.fillna("")
        headers = [str(c).strip() for c in df.columns.tolist()]
        rows = []
        if any(headers):
            rows.append(" | ".join(headers))
        for _, row in df.iterrows():
            values = [str(v).strip() for v in row.tolist()]
            if any(values):
                rows.append(" | ".join(values))
        return "\n".join(rows).strip()
    except Exception:
        return ""

def normalize_table_records(df):
    try:
        df = df.fillna("")
        headers = [str(c).strip() for c in df.columns.tolist()]
        rows = [[str(v).strip() for v in row.tolist()] for _, row in df.iterrows()]
        return headers, rows
    except Exception:
        return [], []

def extract_tables_pymupdf(page):
    results = []
    try:
        tables = page.find_tables()
        for idx, table in enumerate(tables.tables):
            try:
                df = table.to_pandas()
                table_text = dataframe_to_searchable_text(df)
                headers, rows = normalize_table_records(df)
                if not table_text:
                    continue
                bbox = None
                try:
                    if hasattr(table, "bbox") and table.bbox:
                        bbox = list(table.bbox)
                except Exception:
                    bbox = None
                results.append({
                    "table_index": idx,
                    "text": f"[TABLO]\n{table_text}",
                    "headers": headers,
                    "rows": rows,
                    "bbox": bbox
                })
            except Exception:
                continue
    except Exception:
        pass
    return results

def extract_tables_pdfplumber(pdf_path, page_num):
    results = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_num]
            extracted_tables = page.extract_tables(
                table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "intersection_tolerance": 5,
                    "snap_tolerance": 3,
                    "join_tolerance": 3
                }
            )
            for idx, raw_table in enumerate(extracted_tables or []):
                if not raw_table or len(raw_table) < 2:
                    continue
                header = raw_table[0]
                rows = raw_table[1:]
                try:
                    df = pd.DataFrame(rows, columns=header)
                except Exception:
                    df = pd.DataFrame(raw_table)
                table_text = dataframe_to_searchable_text(df)
                headers, norm_rows = normalize_table_records(df)
                if not table_text:
                    continue
                results.append({
                    "table_index": idx,
                    "text": f"[TABLO - FALLBACK]\n{table_text}",
                    "headers": headers,
                    "rows": norm_rows,
                    "bbox": None
                })
    except Exception:
        pass
    return results

def crop_table_preview_from_page(page, bbox, out_path, dpi=180):
    if not bbox:
        return None
    try:
        x0, y0, x1, y1 = bbox
        rect = fitz.Rect(x0, y0, x1, y1)
        matrix = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=matrix, clip=rect, alpha=False)
        image = Image.open(BytesIO(pix.tobytes("png")))
        image = resize_image_if_needed(image, max_dim=1600)
        save_pil_image(image, out_path, quality=80)
        return out_path
    except Exception:
        return None

def get_existing_sources(db):
    sources = set()
    try:
        existing_data = db.get(include=["metadatas"])
        for meta in existing_data.get("metadatas", []):
            if meta and "source" in meta:
                sources.add(meta["source"])
    except Exception as e:
        print(f"[UYARI] Var olan metadata okunamadı: {e}")
    return sources

# ---------------------------------------------------------
# ANA BORU HATTI
# ---------------------------------------------------------
def main():
    print("\n[BILGI] Düşük depolama odaklı hibrit ingest başlatılıyor...")
    ensure_folders()

    files = [f for f in os.listdir(DOCS_FOLDER) if f.lower().endswith(".pdf")]
    if not files:
        print(f"[UYARI] '{DOCS_FOLDER}' klasöründe PDF bulunamadı.")
        return

    print("[BILGI] E5 Embedding ve Tokenizer yükleniyor...")
    embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    
    db = None
    existing_sources = set()

    if os.path.exists(DB_FOLDER):
        try:
            db = Chroma(persist_directory=DB_FOLDER, embedding_function=embeddings)
            existing_sources = get_existing_sources(db)
        except Exception as e:
            print(f"[UYARI] Mevcut Chroma açılamadı: {e}")
            db = None

    new_files = [f for f in files if f not in existing_sources]

    if not new_files:
        print("[BASARILI] Sistem güncel.")
        return

    documents = []

    for file in new_files:
        file_path = os.path.join(DOCS_FOLDER, file)
        pdf_img_folder = os.path.join(IMG_FOLDER, file.replace(".pdf", ""))
        os.makedirs(pdf_img_folder, exist_ok=True)

        print(f"\n[ISLENIYOR] {file}")

        try:
            pdf_document = fitz.open(file_path)
            seen_image_hashes = set()

            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]

                # A) HİBRİT METİN AYIKLAMA
                extraction_method = "digital_text"
                text = page.get_text("text").strip()
                page_preview_saved_path = None

                if len(text) < MIN_TEXT_LENGTH_FOR_DIGITAL:
                    extraction_method = "ocr"
                    page_png_bytes = render_page_to_png_bytes(page, dpi=170)
                    ocr_results = ocr_reader.readtext(page_png_bytes, detail=0, paragraph=True)
                    text = "\n".join(ocr_results).strip()

                    if SAVE_PAGE_PREVIEWS or SAVE_PREVIEW_FOR_OCR_PAGES:
                        try:
                            page_img = Image.open(BytesIO(page_png_bytes))
                            page_img = resize_image_if_needed(page_img, PAGE_PREVIEW_MAX_DIM)
                            page_preview_saved_path = page_preview_path(pdf_img_folder, page_num)
                            save_pil_image(page_img, page_preview_saved_path, quality=70)
                        except Exception as e:
                            print(f"[UYARI] OCR preview kaydedilemedi: {e}")

                if text:
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": file,
                            "page": page_num,
                            "type": "page",
                            "display_mode": "plain_text" if extraction_method == "digital_text" else "ocr_page",
                            "extraction_method": extraction_method,
                            "raw_text": text,
                            "source_path": page_preview_saved_path
                        }
                    ))

                # B) TABLO AYIKLAMA
                page_tables = extract_tables_pymupdf(page)
                if not page_tables:
                    page_tables = extract_tables_pdfplumber(file_path, page_num)

                if page_tables and (SAVE_PAGE_PREVIEWS or SAVE_PREVIEW_FOR_TABLE_PAGES) and not page_preview_saved_path:
                    try:
                        page_img = render_page_to_pil(page, dpi=130)
                        page_img = resize_image_if_needed(page_img, PAGE_PREVIEW_MAX_DIM)
                        page_preview_saved_path = page_preview_path(pdf_img_folder, page_num)
                        save_pil_image(page_img, page_preview_saved_path, quality=68)
                    except Exception as e:
                        print(f"[UYARI] Tablo sayfa preview kaydedilemedi: {e}")

                for table_data in page_tables:
                    t_idx = table_data["table_index"]
                    table_bbox = table_data.get("bbox")
                    table_crop_path = None

                    if SAVE_TABLE_PREVIEWS and table_bbox:
                        crop_path = table_preview_path(pdf_img_folder, page_num, t_idx)
                        table_crop_path = crop_table_preview_from_page(page, table_bbox, crop_path, dpi=180)

                    documents.append(Document(
                        page_content=table_data["text"],
                        metadata={
                            "source": file,
                            "page": page_num,
                            "type": "table",
                            "display_mode": "markdown_table",
                            "table_index": t_idx,
                            "table_headers": safe_json_dumps(table_data.get("headers", [])),
                            "table_rows": safe_json_dumps(table_data.get("rows", [])),
                            "summary": table_data["text"],
                            "source_path": table_crop_path or page_preview_saved_path
                        }
                    ))

                # C) GÖRSEL AYIKLAMA
                images = page.get_images(full=True)
                for img_index, img in enumerate(images):
                    try:
                        xref = img[0]
                        base_image = pdf_document.extract_image(xref)
                        image_bytes = base_image["image"]

                        if len(image_bytes) < MIN_IMAGE_BYTES:
                            continue

                        img_hash = hashlib.md5(image_bytes).hexdigest()
                        if img_hash in seen_image_hashes:
                            continue
                        seen_image_hashes.add(img_hash)

                        image_path = None
                        if SAVE_IMAGE_FILES:
                            image_name = f"page_{page_num+1}_img_{img_index+1}.{base_image['ext']}"
                            image_path = os.path.join(pdf_img_folder, image_name)
                            with open(image_path, "wb") as f:
                                f.write(image_bytes)

                        if not image_path:
                            image_name = f"_temp_page_{page_num+1}_img_{img_index+1}.{base_image['ext']}"
                            image_path = os.path.join(pdf_img_folder, image_name)
                            with open(image_path, "wb") as f:
                                f.write(image_bytes)

                        summary = summarize_image_locally(image_path)

                        documents.append(Document(
                            page_content=f"[GÖRSEL ÖZETİ]\n{summary}",
                            metadata={
                                "source": file,
                                "page": page_num,
                                "type": "image",
                                "display_mode": "image_preview",
                                "summary": summary,
                                "source_path": image_path
                            }
                        ))

                        if not SAVE_IMAGE_FILES and os.path.exists(image_path):
                            try:
                                os.remove(image_path)
                            except Exception:
                                pass

                    except Exception as e:
                        print(f"[UYARI] Görsel işlenemedi (sayfa {page_num+1}): {e}")

            pdf_document.close()
            print(f"[TAMAMLANDI] {file} işlendi.")

        except Exception as e:
            print(f"[HATA] {file}: {e}")

    # 🚀 MİMARIN DOKUNUŞU: VRAM TAHLİYE PROTOKOLÜ
    if documents:
        print("\n[BILGI] PDF okuma tamamlandı. Moondream VRAM'den kazınıyor...")
        global moondream_model, tokenizer
        try:
            del moondream_model
            del tokenizer
        except NameError:
            pass
            
        if device == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        print("\n[BILGI] Token-bazlı parçalama yapılıyor...")
        # E5 tokenizer tekrar initialize ediliyor (bellek yönetimi için)
        e5_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-base")
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=e5_tokenizer,
            chunk_size=300,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = text_splitter.split_documents(documents)

        for chunk in chunks:
            chunk.page_content = f"passage: {chunk.page_content}"

        if db is not None:
            db.add_documents(chunks)
        else:
            Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=DB_FOLDER
            )

        print("[BASARILI] Chroma veritabanı güncellendi.")
    else:
        print("[UYARI] İşlenecek içerik üretilemedi.")

if __name__ == "__main__":
    main()