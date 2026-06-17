"""Reusable UI building blocks for DeepCampus.

Each function emits styled HTML/Markdown that pairs with the CSS in styles.py.
Keep functions pure and side-effect-free where possible — they should just
render markup.
"""
from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Iterable

import streamlit as st

# Kaynak satirlarini ayristirmak icin duzenli ifade. Beklenen bicim:
# "[TEXT] dosya.pdf (Page 3)" veya "[IMAGE] gorsel.png (Page 7)".
# Uc grup yakalanir: tur (TEXT/IMAGE), kaynak adi ve sayfa numarasi.
_SOURCE_RE = re.compile(r"\[(TEXT|IMAGE)\]\s+(.+?)\s+\(Page\s+(\d+)\)", re.IGNORECASE)


def hero(title: str, subtitle: str) -> None:
    """Top-of-page hero block with gradient title."""
    st.markdown(
        f"""
        <div class="dc-hero">
          <h1>{html.escape(title)}</h1>
          <p>{html.escape(subtitle)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Kucuk durum rozeti (status pill) HTML'i uretir. Etiket html.escape ile
# kacislanir cunku unsafe_allow_html ile basilacak (XSS'e karsi koruma).
def status_pill(label: str) -> str:
    return f'<span class="dc-status-pill">{html.escape(label)}</span>'


# Kenar cubugundaki (sidebar) bolum basligini stilli olarak basar.
def sidebar_section_title(title: str) -> None:
    st.markdown(
        f'<div class="sidebar-section-title">{html.escape(title)}</div>',
        unsafe_allow_html=True,
    )


def _parse_sources(sources_line: str) -> list[tuple[str, str, str]]:
    """Split the comma-joined sources line back into structured entries."""
    if not sources_line:
        return []
    items: list[tuple[str, str, str]] = []
    # "seen" kumesi ayni kaynagin (tur+ad+sayfa) birden fazla kez listelenmesini
    # engeller; LLM ayni parcayi tekrar tekrar atifta bulunabiliyor.
    seen: set[tuple[str, str, str]] = set()
    for part in sources_line.split(","):
        match = _SOURCE_RE.search(part.strip())
        if not match:
            continue
        # Tur buyuk harfe normalize edilir ki "image"/"IMAGE" ayni sayilsin.
        kind, source, page = match.group(1).upper(), match.group(2), match.group(3)
        key = (kind, source, page)
        if key in seen:
            continue
        seen.add(key)
        items.append(key)
    return items


def source_cards(sources_line: str) -> None:
    """Render the sources line as a list of styled cards."""
    items = _parse_sources(sources_line)
    # Ayristirma hicbir yapili kaynak bulamadiysa: ham metni varsa duz altyazi
    # olarak goster, yoksa hicbir sey basma.
    if not items:
        if sources_line:
            st.caption(sources_line)
        return
    html_parts = []
    for kind, source, page in items:
        # Gorsel kaynaklar "IMG", metin/PDF kaynaklari "PDF" rozeti alir.
        badge = "IMG" if kind == "IMAGE" else "PDF"
        html_parts.append(
            f"""
            <div class="dc-source-card">
              <div class="dc-source-icon">{badge}</div>
              <div>
                <div>{html.escape(source)}</div>
                <div class="dc-source-meta">Page {html.escape(page)}</div>
              </div>
            </div>
            """
        )
    st.markdown("\n".join(html_parts), unsafe_allow_html=True)


def welcome_screen(username: str, suggestions: Iterable[str]) -> str | None:
    """Render the welcome screen. Returns the picked suggestion or None."""
    st.markdown(
        f"""
        <div class="dc-hero">
          <h1>Welcome back, {html.escape(username)}</h1>
          <p>Ask anything about the indexed documents. The retriever combines
             dense semantic similarity with lexical sparse matching for the
             most relevant chunks, and the LLM cites every claim.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Oneriler iki sutuna sirayla dagitilir (idx % 2). Bir butona tiklanirsa
    # ilgili oneri metni "picked" olarak doner; cagiran bunu sorgu olarak kullanir.
    picked: str | None = None
    cols = st.columns(2)
    for idx, text in enumerate(suggestions):
        with cols[idx % 2]:
            if st.button(text, key=f"suggest_{idx}", use_container_width=True):
                picked = text
    return picked


# Su anki saati "SS:DD" (orn. 14:30) biciminde doner; sohbet baloncuklarinda kullanilir.
def timestamp_now() -> str:
    return datetime.now().strftime("%H:%M")


def chat_bubble_meta(role: str, ts: str) -> None:
    """Small caption rendered above the chat bubble body."""
    # "user" rolu "You", diger her rol (assistant) "Assistant" olarak etiketlenir.
    role_label = "You" if role == "user" else "Assistant"
    st.markdown(
        f'<div style="font-size:0.72rem;color:var(--text-muted);'
        f'margin-bottom:0.2rem">{html.escape(role_label)}'
        f'<span class="dc-timestamp">{html.escape(ts)}</span></div>',
        unsafe_allow_html=True,
    )
