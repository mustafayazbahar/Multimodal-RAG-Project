"""Docling-based PDF extraction.

Wraps IBM's Docling converter to extract:
- Text per page (layout-aware reading order, OCR fallback on scans)
- Picture regions as PIL images (with bounding boxes)
- Table regions as PIL images (TableFormer-detected)

Docling models download from HuggingFace on first run and cache in HF_HOME
(the backend container mounts hf_cache as a volume so this is persistent).

Returns simple Python data structures so ingestion.py can plug them into the
existing Document / Qdrant pipeline without knowing about Docling internals.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from services.logging_config import get_logger

log = get_logger(__name__)


# Docling'den cikarilan bir metin parcasini temsil eder (metin + ait oldugu sayfa).
@dataclass
class TextBlock:
    text: str
    page: int  # 0-indexed


# Docling'den cikarilan bir gorseli temsil eder: diske kaydedilen PNG yolu,
# sayfa numarasi ve turu (figur mu tablo mu).
@dataclass
class ImageBlock:
    image_path: str
    page: int  # 0-indexed
    kind: str  # "picture" or "table"


# Docling cevirici tekil (singleton) olarak tutulur; ilk kullanimda kurulup
# tekrar tekrar yeniden olusturulmaz (model indirme/yuklemesi pahalidir).
_converter = None


def _detect_accelerator():
    """Pick CUDA when PyTorch sees a GPU, else MPS (Apple), else CPU.

    Docling's layout detector + TableFormer are torch models that
    default to CPU. On a 50-page textbook with figures and tables
    that's a 20-40× slowdown vs CUDA — easy to spend half an hour on
    something the GPU finishes in two minutes.
    """
    # Eski Docling surumlerinde accelerator_options modulu yoktur; bu durumda
    # None dondururuz ve Docling kendi varsayilanina (CPU) duser.
    try:
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )
    except ImportError:
        # Older docling without accelerator_options — let it fall back
        # to its internal default (CPU).
        return None

    # Donanim tespiti torch'a bagli; torch yoksa veya tespit patlarsa guvenli
    # sekilde CPU'ya geri duseriz (asagidaki genis except bunu garantiler).
    try:
        import torch
        if torch.cuda.is_available():
            device = AcceleratorDevice.CUDA
            label = f"CUDA ({torch.cuda.get_device_name(0)})"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = AcceleratorDevice.MPS
            label = "MPS (Apple Silicon)"
        else:
            device = AcceleratorDevice.CPU
            label = "CPU"
    except Exception:  # noqa: BLE001
        device = AcceleratorDevice.CPU
        label = "CPU (torch unavailable)"

    log.info("Docling accelerator: %s", label)
    # num_threads=4: CPU'ya dusulen durumlarda is parcacigi sayisini sinirlar.
    return AcceleratorOptions(num_threads=4, device=device)


def _get_converter():
    """Lazy-init Docling converter with picture+table image generation enabled.

    First call downloads layout + TableFormer models from HuggingFace (~1-2 GB);
    subsequent calls use the cached models from the hf_cache volume.
    """
    global _converter
    # Daha once kurulmussa hazir ceviriciyi dondur (tekrar kurma maliyetini onler).
    if _converter is not None:
        return _converter

    # Docling import'lari fonksiyon icinde: modul yuku agir oldugundan sadece
    # gercekten ihtiyac duyuldugunda (ilk cagrida) yuklenir.
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    # Pipeline ayarlari: OCR (taranmis sayfalar icin), tablo yapisi tespiti ve
    # hem figur hem tablo bolgelerini PNG olarak uretmeyi aciyoruz.
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_table_images = True
    # 2.0 ≈ 144 DPI; enough for VLM captioning, keeps PNG file sizes manageable.
    # Olcek araligi: VLM ozetlemesi icin yeterli netlik, ama dosya boyutu makul.
    pipeline_options.images_scale = 2.0

    # Route Docling's torch models (layout + TableFormer + OCR) to the
    # GPU when available. Without this they default to CPU and large
    # textbooks take 10×+ longer to ingest.
    # GPU tespit edilirse Docling'in torch modellerini oraya yonlendir; aksi
    # halde Docling kendi varsayilanini (CPU) kullanir.
    accelerator = _detect_accelerator()
    if accelerator is not None:
        pipeline_options.accelerator_options = accelerator

    log.info("Initializing Docling converter (first run downloads ~1-2 GB of models)...")
    _converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
        }
    )
    log.info("Docling converter ready.")
    return _converter


# Docling'in "provenance" (kaynak konum) listesinden 0-tabanli sayfa numarasini
# cikarir. Docling 1-tabanli verir, biz 1 cikararak 0-tabana ceviriyoruz.
# Bilgi yoksa veya bicim beklenenden farkliysa guvenli sekilde 0 doner.
def _page_number(prov_list) -> int:
    """Return the 0-indexed page number for a Docling provenance list."""
    if not prov_list:
        return 0
    try:
        return int(prov_list[0].page_no) - 1
    except (AttributeError, ValueError, TypeError):
        return 0


# Docling gorsel nesnesini diske PNG olarak kaydeder; basariliysa yolu, aksi
# halde None doner. Kaydetme hatasi tum cikarimi durdurmamali (uyari verip gecer).
def _save_pil(image_obj, dest: Path) -> Optional[Path]:
    """Save a Docling image object to disk as PNG. Returns the path or None."""
    if image_obj is None:
        return None
    # Docling exposes the PIL image via .pil_image; tolerate either layout.
    # Docling kimi surumde PIL goruntusunu .pil_image altinda, kimi surumde
    # dogrudan nesnenin kendisi olarak verir; her iki duruma da uyum sagliyoruz.
    pil = getattr(image_obj, "pil_image", None) or image_obj
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        pil.save(dest, "PNG")
        return dest
    except (OSError, AttributeError, ValueError) as exc:
        log.warning("Could not save Docling image to %s: %s", dest, exc)
        return None


def extract(pdf_path: Path, img_folder: Path) -> tuple[list[TextBlock], list[ImageBlock]]:
    """Run Docling on the PDF and return text blocks + image blocks.

    Raises on conversion failure so the caller can fall back to PyMuPDF.
    """
    converter = _get_converter()
    # Asil donusturme adimi: Docling PDF'i okuyup yapilandirilmis dokumana cevirir.
    result = converter.convert(str(pdf_path))
    doc = result.document

    # Group all text-bearing items by page, preserving document order via the
    # iterate_items walk.
    # Metin iceren tum ogeleri sayfa bazinda grupla. iterate_items dokuman
    # sirasini koruyarak gezdigi icin metin parcalari dogru sirada birikir.
    page_texts: dict[int, list[str]] = {}
    for item, _level in doc.iterate_items():
        text = getattr(item, "text", None)
        if not text:
            continue
        prov = getattr(item, "prov", None)
        page = _page_number(prov)
        page_texts.setdefault(page, []).append(text.strip())

    # Her sayfanin metin parcalarini tek bir metne birlestir; tamamen bos
    # sayfalari atla (any(parts) kontrolu).
    text_blocks: list[TextBlock] = [
        TextBlock(text="\n".join(parts).strip(), page=page)
        for page, parts in sorted(page_texts.items())
        if any(parts)
    ]

    image_blocks: list[ImageBlock] = []

    # Pictures (figures, diagrams, embedded raster).
    # Figurler/diyagramlar: her birini PNG'ye kaydet, basariliysa ImageBlock ekle.
    for idx, pic in enumerate(getattr(doc, "pictures", []) or []):
        page = _page_number(getattr(pic, "prov", None))
        out = img_folder / f"page_{page + 1}_picture_{idx + 1}.png"
        if _save_pil(getattr(pic, "image", None), out):
            image_blocks.append(ImageBlock(image_path=str(out), page=page, kind="picture"))

    # Tables (TableFormer-detected regions).
    # Tablolar: TableFormer ile tespit edilen bolgeler; figurlerle ayni mantik,
    # ama kind="table" olarak isaretlenir (sonradan ayirt edilebilsin diye).
    for idx, tbl in enumerate(getattr(doc, "tables", []) or []):
        page = _page_number(getattr(tbl, "prov", None))
        out = img_folder / f"page_{page + 1}_table_{idx + 1}.png"
        if _save_pil(getattr(tbl, "image", None), out):
            image_blocks.append(ImageBlock(image_path=str(out), page=page, kind="table"))

    log.info(
        "Docling extracted %d text blocks, %d pictures, %d tables from %s",
        len(text_blocks),
        sum(1 for b in image_blocks if b.kind == "picture"),
        sum(1 for b in image_blocks if b.kind == "table"),
        pdf_path.name,
    )
    return text_blocks, image_blocks
