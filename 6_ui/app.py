"""
MediAssist — Streamlit UI
==========================
Production chat interface. Clean, professional, no decorative emojis.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
import streamlit as st

st.set_page_config(
    page_title="MediAssist",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚕</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def _import_from_path(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_db_mod     = _import_from_path("db",           PROJECT_ROOT / "5_database" / "db.py")
_pipe_mod   = _import_from_path("pipeline",     PROJECT_ROOT / "3_rag"      / "pipeline.py")
_memory_mod = _import_from_path("memory_store", PROJECT_ROOT / "3_rag"      / "memory_store.py")

Database           = _db_mod.Database
MediAssistPipeline = _pipe_mod.MediAssistPipeline
MemoryStore        = _memory_mod.MemoryStore

DISCLAIMER = (
    "General health information only — not a substitute for professional "
    "medical advice. Always consult a qualified healthcare provider."
)

# ═══════════════════════════════════════════════════════════════════════════════
#  CSS
# ═══════════════════════════════════════════════════════════════════════════════

def _inject_css():
    st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg:          #06090f;
  --bg2:         #0b1018;
  --bg3:         #101720;
  --bg4:         #162035;
  --accent:      #00c9ad;
  --accent-dim:  rgba(0,201,173,0.09);
  --accent-glow: rgba(0,201,173,0.25);
  --text:        #d8e8f5;
  --text2:       #7d9ab8;
  --text3:       #3d5570;
  --border:      rgba(255,255,255,0.06);
  --border-a:    rgba(0,201,173,0.22);
  --low:         #10b981;
  --medium:      #f59e0b;
  --high:        #f43f5e;
  --r:           12px;
}

/* ── Hide all Streamlit chrome ────────────────────────────────────────────── */
header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
#MainMenu, footer { display: none !important; }

/* ── st.page_link styling ─────────────────────────────────────────────────── */
[data-testid="stPageLink"] a,
[data-testid="stPageLink-NavLink"] {
    background: transparent !important;
    border: none !important;
    color: var(--text2) !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 8px 10px !important;
    border-radius: 8px !important;
    transition: background 0.15s, color 0.15s !important;
    text-decoration: none !important;
    width: 100% !important;
    display: flex !important;
    align-items: center !important;
}
[data-testid="stPageLink"] a:hover,
[data-testid="stPageLink-NavLink"]:hover {
    background: rgba(255,255,255,0.05) !important;
    color: var(--text) !important;
}

/* ── Hide chat message avatars entirely ──────────────────────────────────── */
[data-testid="chatAvatarIcon-user"],
[data-testid="chatAvatarIcon-assistant"],
[data-testid="stChatMessageAvatarUser"],
[data-testid="stChatMessageAvatarAssistant"],
.stChatMessage [data-testid*="Avatar"] { display: none !important; }

/* ── App root ─────────────────────────────────────────────────────────────── */
.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 55% 40% at 8% 4%,  rgba(0,201,173,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 40% 35% at 92% 92%, rgba(99,102,241,0.04) 0%, transparent 55%);
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}
.main .block-container {
    max-width: 860px !important;
    padding: 1.5rem 2rem 6rem !important;
}

/* ── Typography ───────────────────────────────────────────────────────────── */
h1 { font-family: 'Crimson Pro', Georgia, serif !important; color: var(--text) !important; letter-spacing: -0.025em !important; }
h2, h3, h4, h5 { font-family: 'DM Sans', sans-serif !important; font-weight: 600 !important; color: var(--text) !important; }
p, span, li, label, div { font-family: 'DM Sans', sans-serif !important; color: var(--text) !important; }

/* ── Sidebar ──────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* ── Chat messages ────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: transparent !important;
    padding: 0.3rem 0 !important;
    animation: msgIn 0.28s cubic-bezier(0.16, 1, 0.3, 1) both;
}
[data-testid="stChatMessageContent"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: 1rem 1.25rem !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.25) !important;
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"])
[data-testid="stChatMessageContent"] {
    background: var(--accent-dim) !important;
    border-color: var(--border-a) !important;
}

/* ── Chat input — fix white bar + text color ──────────────────────────────── */
/* Kill the white bottom bar entirely */
[data-testid="stBottom"],
[data-testid="stBottom"] > div,
[data-testid="stBottom"] > div > div,
[data-testid="stBottom"] > div > div > div {
    background: var(--bg) !important;
    border-top: none !important;
    padding-top: 0.75rem !important;
}

[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 1px solid #d0d8e4 !important;
    border-radius: 24px !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
    outline: none !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-glow) !important;
    outline: none !important;
}
/* Text: user wants plain black */
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"],
[data-testid="stChatInputTextArea"] textarea,
.stChatInput textarea,
div[data-testid="stChatInput"] textarea {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
    background: transparent !important;
    caret-color: #111111 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    outline: none !important;
    box-shadow: none !important;
    border: none !important;
}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] textarea {
    color: #111111 !important;
    -webkit-text-fill-color: #111111 !important;
}
[data-testid="stChatInput"] textarea::placeholder,
[data-testid="stChatInputTextArea"] textarea::placeholder {
    color: #9aafc4 !important;
    -webkit-text-fill-color: #9aafc4 !important;
}
[data-testid="stChatInputSubmitButton"] > button {
    background: var(--accent) !important;
    border: none !important; border-radius: 50% !important;
    color: #ffffff !important;
    transition: all 0.18s ease !important;
}
[data-testid="stChatInputSubmitButton"] > button:hover {
    background: #009e8c !important;
    box-shadow: 0 4px 16px var(--accent-glow) !important;
    transform: scale(1.06) !important;
}

/* ── Buttons ──────────────────────────────────────────────────────────────── */
.stButton > button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--text2) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    border-radius: 8px !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    background: var(--accent-dim) !important;
    border-color: var(--border-a) !important;
    color: var(--accent) !important;
}

/* ── Expander ─────────────────────────────────────────────────────────────── */
[data-testid="stExpander"],
[data-testid="stExpander"] details {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    overflow: hidden !important;
    margin-top: 0.6rem !important;
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] details > summary {
    background: var(--bg3) !important;
    color: var(--text2) !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 0.65rem 1rem 0.65rem 1.25rem !important;
}
[data-testid="stExpander"] summary:hover,
[data-testid="stExpander"] details > summary:hover { color: var(--accent) !important; background: var(--bg4) !important; }

/* ── Misc ─────────────────────────────────────────────────────────────────── */
hr { border: none !important; border-top: 1px solid var(--border) !important; margin: 0.75rem 0 !important; }
.stSpinner > div { border-top-color: var(--accent) !important; }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text3) !important; font-size: 0.78rem !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

/* ── Animations ───────────────────────────────────────────────────────────── */
@keyframes msgIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes highPulse { 0%,100% { box-shadow: 0 0 0 0 rgba(244,63,94,.5); } 60% { box-shadow: 0 0 0 10px rgba(244,63,94,0); } }
@keyframes emergencyFlash { 0%,100% { opacity: 1; } 50% { opacity: 0.82; } }

/* ── Top-right hover panels ───────────────────────────────────────────────── */
.ma-top-actions {
    position: fixed; top: 12px; right: 16px;
    display: flex; gap: 8px; z-index: 9999;
}
.ma-action-wrap { position: relative; }
.ma-icon-btn {
    width: 36px; height: 36px;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    cursor: pointer;
    color: var(--text2);
    font-size: 14px;
    transition: all 0.18s;
    user-select: none;
}
.ma-icon-btn:hover { background: var(--bg4); border-color: var(--border-a); color: var(--accent); }
.ma-panel {
    display: none;
    position: absolute; top: 44px; right: 0;
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: var(--r);
    padding: 1rem;
    min-width: 220px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
    z-index: 9999;
}
.ma-action-wrap:hover .ma-panel { display: block; }
.ma-panel-title {
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--text3);
    font-family: 'DM Sans', sans-serif;
    margin-bottom: 0.65rem;
}
.ma-triage-row {
    display: flex; align-items: center; gap: 8px;
    padding: 5px 0;
    font-size: 0.8rem; font-family: 'DM Sans', sans-serif;
}
.ma-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.ma-stat-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 5px 0; font-size: 0.8rem;
    font-family: 'DM Sans', sans-serif;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.ma-stat-row:last-child { border-bottom: none; }
.ma-stat-count {
    font-weight: 700; font-size: 0.9rem;
    font-family: 'DM Sans', sans-serif;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════

def _top_actions_html(stats: dict, mem_count: int) -> str:
    """Fixed top-right icon buttons with hover panels."""
    # Triage legend panel
    triage_rows = ""
    for color, label, desc in [
        ("#10b981", "LOW",    "Self-care appropriate"),
        ("#f59e0b", "MEDIUM", "See a doctor within 24 h"),
        ("#f43f5e", "HIGH",   "Emergency care needed"),
    ]:
        triage_rows += f"""
<div class="ma-triage-row">
  <div class="ma-dot" style="background:{color};"></div>
  <span style="font-weight:600;color:{color};min-width:52px;">{label}</span>
  <span style="color:var(--text2);">{desc}</span>
</div>"""

    # Stats panel
    stat_rows = ""
    if stats:
        for level, color in [("HIGH","#f43f5e"),("MEDIUM","#f59e0b"),("LOW","#10b981")]:
            count = stats.get(level, 0)
            stat_rows += f"""
<div class="ma-stat-row">
  <span style="display:flex;align-items:center;gap:7px;">
    <div class="ma-dot" style="background:{color};"></div>
    <span style="color:{color};font-weight:600;">{level}</span>
  </span>
  <span class="ma-stat-count" style="color:var(--text);">{count}</span>
</div>"""
    else:
        stat_rows = '<div style="color:var(--text3);font-size:0.8rem;font-family:\'DM Sans\',sans-serif;">No queries yet</div>'

    mem_line = ""
    if mem_count > 0:
        mem_line = f'<div style="margin-top:0.6rem;padding-top:0.6rem;border-top:1px solid var(--border);font-size:0.78rem;color:var(--accent);font-family:\'DM Sans\',sans-serif;">{mem_count} patient fact{"s" if mem_count!=1 else ""} remembered</div>'

    return f"""
<div class="ma-top-actions">
  <div class="ma-action-wrap">
    <div class="ma-icon-btn" title="Triage Legend">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
    </div>
    <div class="ma-panel" style="min-width:260px;">
      <div class="ma-panel-title">Triage Legend</div>
      {triage_rows}
    </div>
  </div>
  <div class="ma-action-wrap">
    <div class="ma-icon-btn" title="Session Stats">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
    </div>
    <div class="ma-panel">
      <div class="ma-panel-title">Session Stats</div>
      {stat_rows}
      {mem_line}
    </div>
  </div>
</div>
"""


def _sidebar_html() -> str:
    """ChatGPT-style top section of sidebar: logo + navigation links."""
    return """
<div style="padding: 1rem 1rem 0;">
  <!-- Logo -->
  <div style="
    display:flex; align-items:center; gap:10px;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1rem;
  ">
    <div style="
      width:32px; height:32px;
      background: linear-gradient(135deg,#00c9ad,#006a8e);
      border-radius:9px;
      display:flex; align-items:center; justify-content:center;
      font-size:16px; flex-shrink:0;
      box-shadow: 0 2px 10px rgba(0,201,173,0.3);
      font-style:normal;
    ">⚕</div>
    <div>
      <div style="font-family:'Crimson Pro',Georgia,serif;font-size:1.1rem;font-weight:600;color:#d8e8f5;letter-spacing:-0.02em;">MediAssist</div>
    </div>
  </div>

  <!-- Navigation links -->
  <style>
  .ma-sb-link {
    display:flex; align-items:center; gap:10px;
    padding: 8px 10px; border-radius: 8px;
    color: #7d9ab8; font-size: 0.875rem; font-weight: 500;
    text-decoration: none; margin-bottom: 2px;
    transition: background 0.15s, color 0.15s;
  }
  .ma-sb-link:hover { background: rgba(255,255,255,0.05); color: #d8e8f5; }
  </style>

  <a href="http://localhost:8000/developer-docs" target="_blank" class="ma-sb-link">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg>
    API Documentation
  </a>

  <a href="https://github.com/Kisht2t/MediAssist-AI" target="_blank" class="ma-sb-link">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0 1 12 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"/></svg>
    GitHub Repository
  </a>

  <div style="height:1px;background:rgba(255,255,255,0.06);margin:0.75rem 0;"></div>
</div>
"""


def _header_html() -> str:
    return """
<div style="
    display:flex; align-items:center; gap:12px;
    padding: 0.25rem 0 1.25rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.25rem;
">
    <div style="
        width:38px; height:38px;
        background: linear-gradient(135deg,#00c9ad,#006a8e);
        border-radius:11px;
        display:flex; align-items:center; justify-content:center;
        font-size:20px; flex-shrink:0;
        box-shadow: 0 4px 16px rgba(0,201,173,0.28);
    ">⚕</div>
    <div>
        <div style="
            font-family:'Crimson Pro',Georgia,serif;
            font-size:1.55rem; font-weight:600;
            color:#d8e8f5; letter-spacing:-0.025em; line-height:1.1;
        ">MediAssist</div>
        <div style="
            font-size:0.68rem; font-weight:700; letter-spacing:0.12em;
            text-transform:uppercase; color:#3d5570; margin-top:1px;
        ">AI Medical Assistant</div>
    </div>
</div>
"""


def _triage_badge_html(triage: str) -> str:
    cfg = {
        "LOW":    ("#10b981", "rgba(16,185,129,0.12)", "rgba(16,185,129,0.28)", "LOW — Self-care may be appropriate",    ""),
        "MEDIUM": ("#f59e0b", "rgba(245,158,11,0.12)",  "rgba(245,158,11,0.28)",  "MEDIUM — See a doctor within 24 h",     ""),
        "HIGH":   ("#f43f5e", "rgba(244,63,94,0.14)",   "rgba(244,63,94,0.32)",   "HIGH — Seek emergency care now",         "animation:highPulse 1.6s ease-in-out infinite;"),
    }
    color, bg, glow, label, anim = cfg.get(triage, ("#7d9ab8","rgba(125,154,184,0.1)","transparent",triage,""))
    return f"""
<div style="
    display:inline-flex; align-items:center; gap:8px;
    background:{bg}; border:1px solid {glow};
    border-radius:999px; padding:5px 14px 5px 10px;
    margin-bottom:0.7rem; {anim}
">
    <div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;"></div>
    <span style="
        font-size:0.78rem; font-weight:600;
        letter-spacing:0.04em; color:{color};
        font-family:'DM Sans',sans-serif;
    ">{label}</span>
</div>
"""


def _emergency_banner_html() -> str:
    return """
<div style="
    background:rgba(244,63,94,0.09);
    border:1px solid rgba(244,63,94,0.32);
    border-radius:12px; padding:1rem 1.25rem;
    margin-bottom:1rem;
    animation:emergencyFlash 1.4s ease-in-out infinite;
    display:flex; gap:12px; align-items:flex-start;
">
    <div style="
        width:20px;height:20px;border-radius:50%;
        background:rgba(244,63,94,0.2);
        display:flex;align-items:center;justify-content:center;
        flex-shrink:0;margin-top:1px;
        font-size:11px;font-weight:700;color:#f43f5e;
    ">!</div>
    <div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;font-weight:700;color:#f43f5e;margin-bottom:4px;">
            EMERGENCY — Immediate attention required
        </div>
        <div style="font-family:'DM Sans',sans-serif;font-size:0.82rem;color:rgba(244,63,94,0.85);line-height:1.45;">
            Please <strong>call 911</strong> or go to the nearest Emergency Room immediately.
            Do not rely on online advice in an emergency.
        </div>
    </div>
</div>
"""


def _differentials_html(differentials: list) -> str:
    if not differentials:
        return ""
    rows = ""
    for i, d in enumerate(differentials):
        rank_color = ["#f59e0b", "#7d9ab8", "#3d5570"][min(i, 2)]
        rows += f"""
<div style="
    display:flex;gap:10px;align-items:flex-start;
    padding:8px 0;
    border-bottom:1px solid rgba(255,255,255,0.04);
">
    <div style="
        min-width:20px;height:20px;border-radius:50%;
        background:rgba(255,255,255,0.05);
        display:flex;align-items:center;justify-content:center;
        font-size:0.68rem;font-weight:700;color:{rank_color};
        flex-shrink:0;margin-top:1px;font-family:'DM Sans',sans-serif;
    ">{i+1}</div>
    <div style="flex:1;">
        <div style="font-size:0.83rem;font-weight:600;color:#d8e8f5;font-family:'DM Sans',sans-serif;">{d['name']}</div>
        <div style="font-size:0.75rem;color:#7d9ab8;font-family:'DM Sans',sans-serif;margin-top:2px;line-height:1.4;">{d['note']}</div>
    </div>
</div>"""

    return f"""
<div style="
    background:rgba(0,201,173,0.04);
    border:1px solid rgba(0,201,173,0.14);
    border-radius:10px;padding:0.85rem 1rem;
    margin-top:0.6rem;
">
    <div style="
        font-size:0.64rem;font-weight:700;letter-spacing:0.12em;
        text-transform:uppercase;color:#3d5570;
        font-family:'DM Sans',sans-serif;margin-bottom:6px;
    ">Possible Conditions</div>
    {rows}
    <div style="font-size:0.7rem;color:#3d5570;font-family:'DM Sans',sans-serif;margin-top:8px;font-style:italic;">
        For differential purposes only — consult a clinician for diagnosis.
    </div>
</div>"""


def _soap_note_html(note: str) -> str:
    """Renders a SOAP note in a styled clinical panel."""
    sections = {"SUBJECTIVE": "", "OBJECTIVE": "", "ASSESSMENT": "", "PLAN": ""}
    icons = {
        "SUBJECTIVE":  ("S", "#00c9ad"),
        "OBJECTIVE":   ("O", "#6366f1"),
        "ASSESSMENT":  ("A", "#f59e0b"),
        "PLAN":        ("P", "#10b981"),
    }
    current = None
    for line in note.splitlines():
        stripped = line.strip()
        if stripped.startswith("SUBJECTIVE"):
            current = "SUBJECTIVE"
        elif stripped.startswith("OBJECTIVE"):
            current = "OBJECTIVE"
        elif stripped.startswith("ASSESSMENT"):
            current = "ASSESSMENT"
        elif stripped.startswith("PLAN"):
            current = "PLAN"
        elif current and stripped:
            sections[current] += stripped + " "

    cards = ""
    for key, text in sections.items():
        letter, color = icons[key]
        cards += f"""
<div style="
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.06);
    border-left:3px solid {color};
    border-radius:8px;padding:0.8rem 1rem;
    margin-bottom:8px;
">
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <div style="
            width:22px;height:22px;border-radius:50%;
            background:{color};
            display:flex;align-items:center;justify-content:center;
            font-size:0.7rem;font-weight:700;color:#06090f;
            font-family:'DM Sans',sans-serif;flex-shrink:0;
        ">{letter}</div>
        <span style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;
            text-transform:uppercase;color:{color};font-family:'DM Sans',sans-serif;">
            {key.title()}
        </span>
    </div>
    <div style="font-size:0.82rem;color:#d8e8f5;font-family:'DM Sans',sans-serif;line-height:1.6;">
        {text.strip() or "—"}
    </div>
</div>"""

    return f"""
<div style="
    background:rgba(0,0,0,0.3);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:12px;padding:1.25rem;
    margin-top:0.5rem;
">
    <div style="
        font-size:0.7rem;font-weight:700;letter-spacing:0.14em;
        text-transform:uppercase;color:#3d5570;
        font-family:'DM Sans',sans-serif;margin-bottom:12px;
    ">Clinical Note — SOAP Format</div>
    {cards}
</div>"""


def _source_item_html(index: int, text: str) -> str:
    preview = text[:260] + ("..." if len(text) > 260 else "")
    return f"""
<div style="
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.055);
    border-radius:8px; padding:0.65rem 0.9rem; margin-bottom:0.5rem;
">
    <div style="
        font-size:0.66rem;font-weight:700;letter-spacing:0.1em;
        text-transform:uppercase;color:#3d5570;
        font-family:'DM Sans',sans-serif;margin-bottom:5px;
    ">Source {index}</div>
    <div style="
        font-size:0.81rem;color:#7d9ab8;
        font-family:'DM Sans',sans-serif;line-height:1.5;
    ">{preview}</div>
</div>
"""


def _entity_pill_html(value: str, color: str) -> str:
    return f"""
<span style="
    display:inline-flex;align-items:center;gap:4px;
    background:{color.replace('rgb(','rgba(').replace(')',',0.12)')};
    border:1px solid {color.replace('rgb(','rgba(').replace(')',',0.28)')};
    border-radius:999px; padding:3px 10px;
    font-size:0.74rem;font-weight:500;color:{color};
    font-family:'DM Sans',sans-serif;
    margin:2px 3px 2px 0;
">{value}</span>
"""


def _welcome_html() -> str:
    features = [
        ("Named Entity Recognition",   "Extracts symptoms, durations, and clinical measurements from your description"),
        ("RAG Medical Knowledge",       "Grounds responses in 2,000+ curated medical reference document chunks"),
        ("Triage Classification",       "Rates urgency as LOW, MEDIUM, or HIGH based on symptom analysis"),
        ("Fine-tuned Llama 3.2",        "Medical reasoning powered by a LoRA-adapted 3B parameter model"),
    ]
    cards = ""
    for title, desc in features:
        cards += f"""
<div style="
    background:rgba(255,255,255,0.025);
    border:1px solid rgba(255,255,255,0.055);
    border-radius:10px; padding:0.85rem 1rem;
    display:flex;gap:10px;align-items:flex-start;
">
    <div style="
        width:6px;height:6px;border-radius:50%;
        background:#00c9ad;flex-shrink:0;margin-top:6px;
    "></div>
    <div>
        <div style="font-size:0.83rem;font-weight:600;color:#d8e8f5;font-family:'DM Sans',sans-serif;">{title}</div>
        <div style="font-size:0.77rem;color:#7d9ab8;font-family:'DM Sans',sans-serif;line-height:1.45;margin-top:2px;">{desc}</div>
    </div>
</div>"""

    examples = [
        "I've had a headache and fever for 3 days",
        "My child has been vomiting since last night",
        "I have chest pain and shortness of breath",
    ]
    ex_html = "".join([
        f"""<div style="
            background:rgba(0,201,173,0.05);border:1px solid rgba(0,201,173,0.14);
            border-radius:8px;padding:7px 12px;margin-bottom:6px;
            font-size:0.82rem;color:#7d9ab8;font-family:'DM Sans',sans-serif;font-style:italic;
        ">&ldquo;{e}&rdquo;</div>"""
        for e in examples
    ])

    return f"""
<div style="
    background:rgba(0,201,173,0.04);
    border:1px solid rgba(0,201,173,0.14);
    border-radius:14px;padding:1.5rem 1.5rem 1.25rem;
    margin-bottom:0.5rem;
">
    <div style="font-family:'Crimson Pro',Georgia,serif;font-size:1.4rem;font-weight:400;font-style:italic;color:#d8e8f5;margin-bottom:6px;">Hello, I&rsquo;m MediAssist.</div>
    <div style="font-size:0.87rem;color:#7d9ab8;font-family:'DM Sans',sans-serif;line-height:1.55;margin-bottom:1.25rem;">
        I combine medical AI with retrieval-augmented knowledge to give you grounded, triage-aware health information.
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:1.25rem;">{cards}</div>
    <div style="font-size:0.68rem;font-weight:700;letter-spacing:0.12em;text-transform:uppercase;color:#3d5570;font-family:'DM Sans',sans-serif;margin-bottom:8px;">Try asking</div>
    {ex_html}
    <div style="margin-top:1rem;padding-top:0.75rem;border-top:1px solid rgba(255,255,255,0.055);font-size:0.74rem;color:#3d5570;font-family:'DM Sans',sans-serif;">
        {DISCLAIMER}
    </div>
</div>
"""


def _sidebar_section(title: str):
    st.sidebar.markdown(f"""
<div style="
    font-size:0.64rem;font-weight:700;letter-spacing:0.13em;
    text-transform:uppercase;color:#3d5570;
    margin:1rem 1rem 0.4rem;font-family:'DM Sans',sans-serif;
">{title}</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  CACHED RESOURCES
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initialising MediAssist pipeline...")
def load_pipeline() -> MediAssistPipeline:
    return MediAssistPipeline()


@st.cache_resource(show_spinner=False)
def load_db() -> Database:
    return Database()


@st.cache_resource(show_spinner=False)
def load_memory() -> MemoryStore:
    return MemoryStore()


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════

def init_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = load_db().create_session()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "soap_note" not in st.session_state:
        st.session_state.soap_note = None
    if "show_soap" not in st.session_state:
        st.session_state.show_soap = False


# ═══════════════════════════════════════════════════════════════════════════════
#  RENDER HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_sources(sources: list[str]):
    if not sources:
        return
    items_html = "".join(_source_item_html(i + 1, s) for i, s in enumerate(sources))
    st.markdown(f"""
<details style="
    background:var(--bg3);
    border:1px solid var(--border);
    border-radius:var(--r);
    overflow:hidden;
    margin-top:0.6rem;
">
  <summary style="
    list-style:none;
    background:var(--bg3);
    color:var(--text2);
    font-size:0.82rem;font-weight:500;
    padding:0.65rem 1rem 0.65rem 1.25rem;
    cursor:pointer;
    display:flex;align-items:center;gap:8px;
  ">
    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor"
         stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"
         class="ma-exp-arrow" style="transition:transform 0.2s;flex-shrink:0;">
      <polyline points="9 18 15 12 9 6"/>
    </svg>
    Retrieved Sources ({len(sources)})
  </summary>
  <div style="padding:0.5rem 0.75rem 0.75rem;">
    {items_html}
  </div>
</details>
<style>
details[open] .ma-exp-arrow {{ transform: rotate(90deg); }}
details > summary::-webkit-details-marker {{ display: none; }}
details > summary::marker {{ display: none; }}
</style>
""", unsafe_allow_html=True)


def render_entities(entities: dict):
    has = any(v for v in entities.values())
    if not has:
        return
    color_map = {
        "symptoms":     "rgb(244,63,94)",
        "body_parts":   "rgb(0,201,173)",
        "durations":    "rgb(245,158,11)",
        "measurements": "rgb(99,102,241)",
        "other":        "rgb(125,154,184)",
    }
    label_map = {
        "symptoms":     "Symptoms",
        "body_parts":   "Body Parts",
        "durations":    "Duration",
        "measurements": "Measurements",
        "other":        "Other",
    }
    _sidebar_section("Entities Detected")
    for etype, values in entities.items():
        if values:
            color = color_map.get(etype, "rgb(125,154,184)")
            lbl   = label_map.get(etype, etype.replace("_"," ").title())
            st.sidebar.markdown(f"""
<div style="font-size:0.67rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
    color:#3d5570;font-family:'DM Sans',sans-serif;margin:0.55rem 1rem 0.3rem;">{lbl}</div>
""", unsafe_allow_html=True)
            pills = "".join(_entity_pill_html(v, color) for v in values)
            st.sidebar.markdown(f'<div style="padding:0 1rem;line-height:1.9;">{pills}</div>', unsafe_allow_html=True)


def render_chat_message(msg: dict):
    role    = msg["role"]
    content = msg["content"]
    meta    = msg.get("meta", {})

    with st.chat_message(role, avatar=None):
        if role == "assistant" and meta:
            if meta.get("is_emergency"):
                st.markdown(_emergency_banner_html(), unsafe_allow_html=True)
            st.markdown(_triage_badge_html(meta.get("triage_level","LOW")), unsafe_allow_html=True)
        st.markdown(content)
        if role == "assistant" and meta:
            render_sources(meta.get("sources", []))
            diffs = meta.get("differentials", [])
            if diffs:
                st.markdown(_differentials_html(diffs), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    _inject_css()
    init_state()
    pipeline = load_pipeline()
    db       = load_db()
    memory   = load_memory()

    # Live data for hover panels
    stats     = db.get_triage_stats()
    mem_count = memory.count()

    # Inject top-right hover icons
    st.markdown(_top_actions_html(stats or {}, mem_count), unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # ChatGPT-style top nav (logo + links)
        st.markdown(_sidebar_html(), unsafe_allow_html=True)

        # Analytics page link — uses Streamlit's internal router (no hard reload)
        st.page_link("pages/1_Analytics.py", label="Analytics Dashboard", icon=None)

        # New conversation button
        if st.button("+ New Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = db.create_session()
            st.session_state.soap_note = None
            st.session_state.show_soap = False
            st.rerun()

        # SOAP Note button — only visible when there is a conversation
        if st.session_state.messages:
            if st.button("Generate Clinical Note", use_container_width=True):
                last_asst = next(
                    (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
                    None,
                )
                last_ents   = last_asst["meta"].get("entities", {}) if last_asst else {}
                last_triage = last_asst["meta"].get("triage_level", "LOW") if last_asst else "LOW"
                with st.spinner("Generating clinical note..."):
                    soap = pipeline.generate_soap_note(
                        st.session_state.messages,
                        last_ents,
                        last_triage,
                    )
                st.session_state.soap_note = soap
                st.session_state.show_soap = True
                st.rerun()

        st.divider()

        # About
        _sidebar_section("About")
        st.markdown("""
<div style="font-size:0.82rem;color:#7d9ab8;font-family:'DM Sans',sans-serif;line-height:1.6;padding:0 1rem;">
MediAssist combines a <strong style="color:#d8e8f5;">fine-tuned Llama 3.2 3B</strong>
medical model with <strong style="color:#d8e8f5;">RAG</strong> and
<strong style="color:#d8e8f5;">medical NER</strong> for grounded, triage-aware responses.
</div>
""", unsafe_allow_html=True)

        # Entities from last assistant message
        if st.session_state.messages:
            last_asst = next(
                (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
                None,
            )
            if last_asst and last_asst.get("meta"):
                ents = last_asst["meta"].get("entities", {})
                if any(v for v in ents.values()):
                    st.divider()
                    render_entities(ents)

        # Disclaimer
        st.divider()
        st.markdown(f"""
<div style="font-size:0.71rem;color:#3d5570;font-family:'DM Sans',sans-serif;line-height:1.5;padding:0 1rem;">
{DISCLAIMER}
</div>
""", unsafe_allow_html=True)

    # ── Main chat area ────────────────────────────────────────────────────────
    st.markdown(_header_html(), unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(_welcome_html(), unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            render_chat_message(msg)

    # ── SOAP Note Panel ───────────────────────────────────────────────────────
    if st.session_state.show_soap and st.session_state.soap_note:
        st.markdown(
            _soap_note_html(st.session_state.soap_note),
            unsafe_allow_html=True,
        )
        if st.button("Dismiss Clinical Note"):
            st.session_state.show_soap = False
            st.rerun()

    # ── Chat input ────────────────────────────────────────────────────────────
    if patient_text := st.chat_input("Describe your symptoms or ask a health question..."):

        st.session_state.messages.append({"role": "user", "content": patient_text, "meta": {}})
        with st.chat_message("user", avatar=None):
            st.markdown(patient_text)

        memory.save_facts(patient_text, st.session_state.session_id)
        memories = memory.retrieve(patient_text, n=5)

        raw_history = st.session_state.messages[:-1]
        history = [
            {"role": m["role"], "content": m["content"]}
            for m in raw_history[-6:]
            if m["role"] in ("user", "assistant")
        ]

        with st.chat_message("assistant", avatar=None):
            with st.spinner("Analysing..."):
                response = pipeline.run(patient_text, history=history, memories=memories)

            triage = response.triage_level

            if response.is_emergency:
                st.markdown(_emergency_banner_html(), unsafe_allow_html=True)

            st.markdown(_triage_badge_html(triage), unsafe_allow_html=True)
            st.markdown(response.answer)
            render_sources(response.sources)
            diffs = getattr(response, "differentials", [])
            if diffs:
                st.markdown(_differentials_html(diffs), unsafe_allow_html=True)

        query_id = db.save_query(
            session_id   = st.session_state.session_id,
            patient_text = patient_text,
            triage_level = triage,
            is_emergency = response.is_emergency,
        )
        db.save_entities(query_id, response.entities)
        db.save_retrieved_docs(
            query_id,
            [{"source": f"doc_{i+1}", "content": s} for i, s in enumerate(response.sources)],
        )
        db.save_response(query_id, response.answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "meta": {
                "triage_level": triage,
                "is_emergency": response.is_emergency,
                "sources":      response.sources,
                "entities":     response.entities,
                "differentials": getattr(response, "differentials", []),
            },
        })
        st.rerun()


if __name__ == "__main__":
    main()
