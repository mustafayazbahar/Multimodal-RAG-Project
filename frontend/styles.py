"""CSS injection for the DeepCampus UI.

Design language:
- Dark base, amber accent (#F59E0B) — academic / warm AI tooling.
- System-aware: an @media (prefers-color-scheme: light) block remaps surfaces
  so users on a light OS still get readable contrast.
- Subtle glassmorphism on cards, soft shadows, smooth transitions.
"""
from __future__ import annotations

import streamlit as st

_CSS = r"""
<style>
:root {
  --accent: #F59E0B;
  --accent-strong: #D97706;
  --accent-soft: rgba(245, 158, 11, 0.12);
  --accent-glow: rgba(245, 158, 11, 0.35);

  --bg: #0B0B0E;
  --surface: #16161C;
  --surface-2: #1E1E26;
  --border: #2A2A33;
  --text: #FAFAFA;
  --text-muted: #A1A1AA;
  --success: #10B981;
  --danger: #EF4444;
  --info: #38BDF8;

  --radius: 14px;
  --radius-sm: 10px;
  --shadow: 0 8px 24px rgba(0, 0, 0, 0.35);
}

@media (prefers-color-scheme: light) {
  :root {
    --bg: #FAFAF9;
    --surface: #FFFFFF;
    --surface-2: #F4F4F2;
    --border: #E7E5E0;
    --text: #18181B;
    --text-muted: #52525B;
    --shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
  }
  .stApp { background: var(--bg) !important; color: var(--text) !important; }
  section[data-testid="stSidebar"] { background: var(--surface) !important; }
}

/* ───────── Global ───────── */
html, body, .stApp {
  font-family: 'Inter', 'SF Pro Text', -apple-system, BlinkMacSystemFont,
               'Segoe UI', Roboto, sans-serif;
  letter-spacing: -0.005em;
}
.stApp { background: var(--bg); }

/* Don't touch Material Symbols / Material Icons spans — they need their
   own font-family to render the icon ligatures. */
[class*="material-symbols"], [class*="material-icons"] {
  font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

/* Hide Streamlit chrome */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ───────── Sidebar ───────── */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--surface) 0%, var(--bg) 100%);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .sidebar-section-title {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--text-muted);
  margin: 1rem 0 0.5rem 0;
}

/* ───────── Buttons ───────── */
.stButton > button, .stDownloadButton > button {
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border) !important;
  background: var(--surface-2) !important;
  color: var(--text) !important;
  font-weight: 500 !important;
  transition: all 0.18s ease !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft);
  transform: translateY(-1px);
}
.stButton > button[kind="primary"], .stFormSubmitButton > button {
  background: linear-gradient(135deg, var(--accent) 0%, var(--accent-strong) 100%) !important;
  border: 1px solid var(--accent) !important;
  color: #18181B !important;
  font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button:hover {
  box-shadow: 0 0 24px var(--accent-glow);
  transform: translateY(-1px);
}

/* ───────── Inputs ───────── */
.stTextInput input, .stTextArea textarea, .stSelectbox > div > div,
[data-baseweb="select"] > div, [data-baseweb="input"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text) !important;
  transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important;
}

/* ───────── Sliders ───────── */
.stSlider [role="slider"] {
  background: var(--accent) !important;
  border: 2px solid var(--bg) !important;
  box-shadow: 0 0 0 3px var(--accent-soft) !important;
}
.stSlider [data-baseweb="slider"] > div > div { background: var(--accent) !important; }

/* ───────── Chat ───────── */
.stChatMessage {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  padding: 1rem 1.25rem !important;
  box-shadow: var(--shadow);
  margin-bottom: 0.75rem !important;
}
.stChatMessage[data-testid="stChatMessage"] [data-testid="stChatMessageAvatar"] {
  background: var(--accent-soft) !important;
  border: 1px solid var(--accent) !important;
}

/* Chat input — sticky, glassy, glow on focus */
[data-testid="stChatInput"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow);
  transition: border-color 0.18s ease, box-shadow 0.18s ease !important;
}
[data-testid="stChatInput"]:focus-within {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 3px var(--accent-soft), var(--shadow);
}
[data-testid="stChatInput"] textarea {
  color: var(--text) !important;
  background: transparent !important;
}

/* ───────── Cards / containers ───────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  transition: border-color 0.18s ease, transform 0.18s ease;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
  border-color: var(--accent);
}

/* ───────── Expander ───────── */
.streamlit-expanderHeader, [data-testid="stExpander"] summary {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 500 !important;
}

/* ───────── Alerts ───────── */
.stAlert {
  border-radius: var(--radius-sm) !important;
  border: 1px solid var(--border) !important;
}

/* ───────── DeepCampus custom components ───────── */
.dc-hero {
  background: radial-gradient(circle at 20% 30%, var(--accent-soft) 0%, transparent 55%),
              linear-gradient(135deg, var(--surface) 0%, var(--bg) 100%);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.75rem 2rem;
  margin-bottom: 1.25rem;
}
.dc-hero h1 {
  background: linear-gradient(135deg, var(--accent) 0%, #FBBF24 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 0.35rem 0 !important;
  font-size: 2rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
}
.dc-hero p {
  color: var(--text-muted);
  margin: 0;
  font-size: 0.95rem;
}

.dc-status-pill {
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--text-muted);
  background: var(--surface-2);
  border: 1px solid var(--border);
  padding: 0.25rem 0.6rem;
  border-radius: 999px;
}
.dc-status-pill::before {
  content: '';
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 8px var(--success);
}

.dc-source-card {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  padding: 0.6rem 0.8rem;
  background: var(--surface-2);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  margin-bottom: 0.4rem;
  font-size: 0.85rem;
  transition: all 0.15s ease;
}
.dc-source-card:hover {
  border-color: var(--accent);
  background: var(--accent-soft);
}
.dc-source-card .dc-source-icon {
  width: 28px; height: 28px;
  display: flex; align-items: center; justify-content: center;
  background: var(--accent-soft);
  color: var(--accent);
  border-radius: 6px;
  font-weight: 700;
  font-size: 11px;
  flex-shrink: 0;
}
.dc-source-card .dc-source-meta { color: var(--text-muted); font-size: 0.78rem; }

.dc-suggest-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 0.75rem;
  margin-top: 1rem;
}
.dc-suggest-card {
  padding: 1rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: all 0.18s ease;
}
.dc-suggest-card:hover {
  border-color: var(--accent);
  transform: translateY(-2px);
  box-shadow: 0 0 24px var(--accent-glow);
}
.dc-suggest-card .dc-suggest-title {
  font-weight: 600;
  margin-bottom: 0.25rem;
}
.dc-suggest-card .dc-suggest-desc {
  font-size: 0.8rem;
  color: var(--text-muted);
}

.dc-timestamp {
  font-size: 0.72rem;
  color: var(--text-muted);
  margin-left: 0.5rem;
}

/* ───────── Scrollbar ───────── */
*::-webkit-scrollbar { width: 10px; height: 10px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb {
  background: var(--surface-2);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: padding-box;
}
*::-webkit-scrollbar-thumb:hover { background: var(--accent); background-clip: padding-box; }

/* ───────── Animation ───────── */
@keyframes dc-fade-in {
  from { opacity: 0; transform: translateY(6px); }
  to   { opacity: 1; transform: translateY(0); }
}
.stChatMessage, .dc-source-card, .dc-suggest-card, .dc-hero {
  animation: dc-fade-in 0.25s ease-out;
}
</style>
"""


def inject_styles() -> None:
    """Inject the project's CSS once per page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def scroll_to_bottom() -> None:
    """Force-scroll the parent page to the bottom (used after new messages)."""
    st.components.v1.html(
        """
        <script>
          const doc = window.parent.document;
          window.parent.scrollTo({ top: doc.body.scrollHeight, behavior: 'smooth' });
        </script>
        """,
        height=0,
    )


def autofocus_chat_input() -> None:
    """Move keyboard focus to the chat input so Enter posts a message."""
    st.components.v1.html(
        """
        <script>
          const doc = window.parent.document;
          const ta = doc.querySelector('[data-testid="stChatInput"] textarea');
          if (ta) ta.focus();
        </script>
        """,
        height=0,
    )


def bind_login_enter() -> None:
    """Force Enter-in-password-field to click the primary submit button.

    Streamlit 1.57's st.form sometimes drops the Enter keystroke when
    iframe components are on the page. We attach an explicit listener to
    every password input that finds the form's submit button and clicks
    it. Idempotent: the listener self-deduplicates via a data attribute.
    """
    st.components.v1.html(
        """
        <script>
          const attach = () => {
            const doc = window.parent.document;
            const inputs = doc.querySelectorAll('input[type="password"]');
            inputs.forEach((inp) => {
              if (inp.dataset.dcEnterBound === '1') return;
              inp.dataset.dcEnterBound = '1';
              inp.addEventListener('keydown', (ev) => {
                if (ev.key !== 'Enter' || ev.isComposing) return;
                ev.preventDefault();
                const form = inp.closest('[data-testid="stForm"]');
                const btn = (form || doc).querySelector(
                  'button[kind="primaryFormSubmit"], button[kind="secondaryFormSubmit"]'
                );
                if (btn) btn.click();
              });
            });
          };
          attach();
          // Streamlit re-renders inputs; re-attach on DOM mutations.
          const obs = new MutationObserver(attach);
          obs.observe(window.parent.document.body, {childList: true, subtree: true});
          setTimeout(() => obs.disconnect(), 5000);
        </script>
        """,
        height=0,
    )
