#!/usr/bin/env python3
"""Markdown -> PDF cevirici (Turkce karakter destekli, fpdf2 + DejaVu).

DeepCampus proje dokumanlarini (PROJE_REHBERI.md gibi) PDF'e cevirmek icin
yazildi. Tam bir markdown render motoru degil; bu projedeki dokumanlarin
kullandigi alt kume icin yeterli: basliklar, listeler, tablolar, kod
bloklari, blockquote, yatay cizgi, kalin metin.

Kullanim:
    python scripts/md_to_pdf.py PROJE_REHBERI.md PROJE_REHBERI.pdf
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from fpdf import FPDF
from fpdf.fonts import FontFace

# DejaVu fontlari Turkce karakterleri (c, s, g, u, o, i) tam destekler;
# fpdf2'nin yerlesik latin-1 fontlari desteklemez.
FONT_DIR = "/usr/share/fonts/truetype/dejavu"
REG = f"{FONT_DIR}/DejaVuSans.ttf"
BOLD = f"{FONT_DIR}/DejaVuSans-Bold.ttf"
MONO = f"{FONT_DIR}/DejaVuSansMono.ttf"

# Renk paleti (DeepCampus amber temasiyla uyumlu).
ACCENT = (217, 119, 6)       # amber/turuncu basliklar
DARK = (31, 41, 55)          # ana metin
MUTED = (107, 114, 128)      # ikincil
CODE_BG = (243, 244, 246)    # kod blogu arka plani
TABLE_HEAD_BG = (245, 158, 11)
TABLE_ROW_BG = (250, 250, 250)
RULE = (209, 213, 219)


class PDF(FPDF):
    """Ust bilgi/alt bilgi (footer) ekleyen FPDF alt sinifi."""

    def header(self) -> None:
        # Ilk sayfada baslik bandi gosterme (kapak gibi dursun).
        if self.page_no() == 1:
            return
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 8, "DeepCampus — Proje Rehberi", align="L")
        self.ln(10)

    def footer(self) -> None:
        # Alt ortada sayfa numarasi.
        self.set_y(-15)
        self.set_font("DejaVu", "", 8)
        self.set_text_color(*MUTED)
        self.cell(0, 10, f"{self.page_no()}", align="C")


def _strip_inline(text: str) -> str:
    """Inline markdown isaretlerini (kalin, kod, link) duz metne indirir.

    PDF'te kalin/kod ayrimini satir bazinda yapmiyoruz; bu yuzden **..**,
    `..`, [metin](url) gibi isaretleri ayikliyoruz ki ekranda ham sembol
    gorunmesin.
    """
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)   # **kalin**
    text = re.sub(r"`(.+?)`", r"\1", text)          # `kod`
    text = re.sub(r"\[(.+?)\]\((.+?)\)", r"\1 (\2)", text)  # [metin](url)
    return text


def _emoji_safe(text: str) -> str:
    """DejaVu'nun cizemedigi karakterleri (emoji vb.) temizler.

    Dokumanlarda baslik/komut icinde nadiren emoji gecebilir; DejaVu bunlari
    desteklemediginden tofu (bos kare) cikmasin diye BMP disi ve bilinen emoji
    araliklarini ayikliyoruz.
    """
    out = []
    for ch in text:
        cp = ord(ch)
        # Temel cizgi/ok karakterlerine izin ver, emoji bloklarini at.
        if cp in (0x2192, 0x2190, 0x2191, 0x2193):  # ok karakterleri
            out.append(ch)
        elif 0x1F000 <= cp <= 0x1FAFF or 0x2600 <= cp <= 0x27BF:
            continue  # emoji araliklari
        elif cp == 0xFE0F:
            continue  # variation selector
        else:
            out.append(ch)
    return "".join(out)


def _clean(text: str) -> str:
    return _emoji_safe(_strip_inline(text))


def render(md_path: Path, pdf_path: Path) -> None:
    """Markdown dosyasini okuyup PDF olarak yazar."""
    lines = md_path.read_text(encoding="utf-8").splitlines()

    pdf = PDF(format="A4")
    pdf.add_font("DejaVu", "", REG)
    pdf.add_font("DejaVu", "B", BOLD)
    pdf.add_font("DejaVuMono", "", MONO)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()
    epw = pdf.epw  # kullanilabilir sayfa genisligi

    i = 0
    first_h1 = True
    while i < len(lines):
        raw = lines[i]
        line = raw.rstrip()

        # --- Kod blogu (```) ---
        if line.strip().startswith("```"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # kapanis ```'i atla
            _render_code(pdf, code_lines, epw)
            continue

        # --- Tablo (| ... |) ---
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|?$", lines[i + 1].strip()):
            table_block = [line]
            i += 1
            # ayrac satiri + veri satirlari
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_block.append(lines[i].rstrip())
                i += 1
            _render_table(pdf, table_block, epw)
            continue

        # --- Yatay cizgi (---) ---
        if line.strip() in ("---", "***", "___"):
            pdf.ln(2)
            pdf.set_draw_color(*RULE)
            y = pdf.get_y()
            pdf.line(pdf.l_margin, y, pdf.l_margin + epw, y)
            pdf.ln(4)
            i += 1
            continue

        # --- Basliklar ---
        if line.startswith("# "):
            if not first_h1:
                pdf.add_page()
            first_h1 = False
            _heading(pdf, _clean(line[2:]), size=20, color=ACCENT, top=2, bottom=4)
            i += 1
            continue
        if line.startswith("## "):
            _heading(pdf, _clean(line[3:]), size=15, color=ACCENT, top=4, bottom=2)
            i += 1
            continue
        if line.startswith("### "):
            _heading(pdf, _clean(line[4:]), size=12, color=DARK, top=3, bottom=1)
            i += 1
            continue

        # --- Blockquote (>) — ardisik > satirlarini tek paragrafta topla ---
        if line.startswith(">"):
            quote_lines = []
            while i < len(lines) and lines[i].lstrip().startswith(">"):
                # ">" karakterini ve hemen ardindan opsiyonel bosluk soyup metni al.
                quote_lines.append(lines[i].lstrip().lstrip(">").lstrip())
                i += 1
            # Bos > satirlari paragraf icinde gercek satir kirigi olur.
            quote = "\n".join(quote_lines).strip()
            pdf.set_font("DejaVu", "", 9.5)
            pdf.set_text_color(*MUTED)
            pdf.set_x(pdf.l_margin + 4)
            pdf.multi_cell(epw - 4, 5.2, _clean(quote), align="L", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
            continue

        # --- Liste ogesi (bullet) — devam satirlarini maddeye dahil et ---
        m = re.match(r"^(\s*)[-*]\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            parts = [m.group(2)]
            # Maddeyi takip eden, bos olmayan ve yeni bir liste/baslik olmayan
            # indentli devam satirlarini ayni maddeye birlestir.
            j = i + 1
            while j < len(lines):
                nxt = lines[j].rstrip()
                if not nxt.strip() or nxt.startswith(("#", ">", "```", "|")):
                    break
                if re.match(r"^\s*[-*]\s+", nxt) or re.match(r"^\s*\d+\.\s+", nxt):
                    break
                # Devam satiri: baslarindaki tum bosluklari sok (indent goz ardi).
                parts.append(nxt.lstrip())
                j += 1
            text = _clean(" ".join(parts))
            pdf.set_font("DejaVu", "", 10)
            pdf.set_text_color(*DARK)
            bullet_x = pdf.l_margin + 3 + indent * 5
            pdf.set_x(bullet_x)
            pdf.multi_cell(epw - (bullet_x - pdf.l_margin), 5.4, f"•  {text}",
                           align="L", new_x="LMARGIN", new_y="NEXT")
            i = j
            continue

        # --- Numarali liste — devam satirlarini maddeye dahil et ---
        m = re.match(r"^(\s*)(\d+)\.\s+(.*)$", line)
        if m:
            indent = len(m.group(1)) // 2
            number = m.group(2)
            parts = [m.group(3)]
            j = i + 1
            while j < len(lines):
                nxt = lines[j].rstrip()
                if not nxt.strip() or nxt.startswith(("#", ">", "```", "|")):
                    break
                if re.match(r"^\s*[-*]\s+", nxt) or re.match(r"^\s*\d+\.\s+", nxt):
                    break
                parts.append(nxt.lstrip())
                j += 1
            text = _clean(" ".join(parts))
            pdf.set_font("DejaVu", "", 10)
            pdf.set_text_color(*DARK)
            x = pdf.l_margin + 3 + indent * 5
            pdf.set_x(x)
            pdf.multi_cell(epw - (x - pdf.l_margin), 5.4, f"{number}.  {text}",
                           align="L", new_x="LMARGIN", new_y="NEXT")
            i = j
            continue

        # --- Bos satir ---
        if not line.strip():
            pdf.ln(2.5)
            i += 1
            continue

        # --- Normal paragraf: ardisik dolu satirlari TEK paragrafa birlestir ---
        # Markdown'da paragrafin satir bolunmesi (boslukla ayrilmamis ardisik
        # satirlar) anlamsizdir; PDF'te bunlari tek multi_cell olarak basinca
        # fpdf2 dogal olarak sarar. Ayri ayri basinca "(local-first)" gibi
        # devam satirlari sirit gibi kayiyordu.
        para_lines = [line]
        j = i + 1
        while j < len(lines):
            nxt = lines[j].rstrip()
            # Paragrafi sonlandiran durumlar: bos satir, baslik, liste, alintidir,
            # kod blogu, tablo veya yatay cizgi.
            if (not nxt.strip()
                    or nxt.startswith(("#", ">", "```", "|"))
                    or nxt.strip() in ("---", "***", "___")
                    or re.match(r"^\s*[-*]\s+", nxt)
                    or re.match(r"^\s*\d+\.\s+", nxt)):
                break
            para_lines.append(nxt)
            j += 1
        paragraph = " ".join(para_lines)
        pdf.set_font("DejaVu", "", 10)
        pdf.set_text_color(*DARK)
        # align="L": sola hizali (justify yerine) — uzun teknik kelimeler
        # arasinda asiri bosluk olusmasini onler.
        pdf.multi_cell(epw, 5.4, _clean(paragraph), align="L", new_x="LMARGIN", new_y="NEXT")
        i = j

    pdf.output(str(pdf_path))


def _heading(pdf: PDF, text: str, size: int, color, top: float, bottom: float) -> None:
    """Belirtilen boyut/renkte bir baslik basar."""
    pdf.ln(top)
    pdf.set_font("DejaVu", "B", size)
    pdf.set_text_color(*color)
    # align="L" ile sola hizali bas — JUSTIFY uzun basliklarda kelimeleri ayirip
    # "Proje    Rehberi    ve    Kod" gibi sirit gosteriyordu.
    pdf.multi_cell(pdf.epw, size * 0.5, text, align="L", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(bottom)


def _render_code(pdf: PDF, code_lines: list[str], epw: float) -> None:
    """Kod blogunu monospace + arka plan kutusu olarak basar.

    Dosya agaci gibi 80+ karakterlik blok'lar icin font boyutu icerigin
    sigmasina gore dinamik secilir: cok uzun satir varsa kuculur.
    """
    pdf.ln(1)
    # En uzun satira gore font boyutunu sec — boylece uzun ASCII agaclar
    # sayfaya sigar, kisa kod bloklari da gereksiz kuculmez.
    max_len = max((len(l) for l in code_lines), default=0)
    if max_len > 95:
        font_size = 6.5
        line_h = 3.6
    elif max_len > 75:
        font_size = 7.5
        line_h = 4.0
    else:
        font_size = 8.0
        line_h = 4.2
    pdf.set_font("DejaVuMono", "", font_size)
    pdf.set_text_color(*DARK)
    # Once arka plan dikdortgenini cizebilmek icin yukseklik hesapla.
    # Cok uzun satirlar sarilabilir; basitlik icin sayfa sonu kontrolunu
    # multi_cell'e birakiyoruz ve her satiri ayri basiyoruz.
    pad = 2
    start_y = pdf.get_y()
    # Arka plani satir satir cizmek yerine blok halinde: once toplam yukseklik.
    total_h = line_h * max(1, len(code_lines)) + pad * 2
    # Sayfa tasmasi olursa yeni sayfa.
    if start_y + total_h > pdf.h - pdf.b_margin:
        pdf.add_page()
        start_y = pdf.get_y()
    pdf.set_fill_color(*CODE_BG)
    pdf.rect(pdf.l_margin, start_y, epw, total_h, style="F")
    pdf.set_xy(pdf.l_margin + pad, start_y + pad)
    for cl in code_lines:
        safe = _emoji_safe(cl.replace("\t", "    "))
        pdf.set_x(pdf.l_margin + pad)
        # Tasan satirlari kirpmak yerine sigdir: cok uzunsa fpdf sarar.
        pdf.cell(epw - pad * 2, line_h, safe, align="L")
        pdf.ln(line_h)
    pdf.set_y(start_y + total_h)
    pdf.ln(2)


def _render_table(pdf: PDF, block: list[str], epw: float) -> None:
    """Markdown tablosunu (| ... |) basit bir tablo olarak basar."""
    # Satirlari hucrelere ayir; 2. satir ayrac (---) oldugu icin atlanir.
    rows = []
    for idx, ln in enumerate(block):
        if idx == 1:
            continue  # ayrac satiri
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        rows.append([_clean(c) for c in cells])
    if not rows:
        return

    ncols = max(len(r) for r in rows)
    rows = [r + [""] * (ncols - len(r)) for r in rows]  # eksik hucreleri doldur
    line_h = 5.0

    pdf.ln(1)
    # Tablo hucreleri icin temel font (Turkce destekli).
    pdf.set_font("DejaVu", "", 8.5)
    pdf.set_text_color(*DARK)
    # Baslik satiri stili: amber dolgu, beyaz kalin yazi (FontFace nesnesi).
    head_style = FontFace(emphasis="BOLD", color=(255, 255, 255), fill_color=TABLE_HEAD_BG)
    # fpdf2'nin yerlesik table() API'si hucre icindeki uzun metni otomatik sarar.
    with pdf.table(
        width=epw,
        col_widths=tuple([1] * ncols),
        text_align="LEFT",
        borders_layout="SINGLE_TOP_LINE",
        line_height=line_h,
        first_row_as_headings=True,
        headings_style=head_style,
    ) as table:
        for r in rows:
            row = table.row()
            for cell in r:
                row.cell(cell)
    pdf.ln(2)


if __name__ == "__main__":
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("PROJE_REHBERI.md")
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_suffix(".pdf")
    render(src, dst)
    print(f"PDF yazildi: {dst}  ({dst.stat().st_size // 1024} KB)")
