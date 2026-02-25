"""
MediAssist — Analytics Dashboard
==================================
Separate Streamlit page with Plotly charts pulled from the SQLite database.
"""

import sys
from pathlib import Path
from collections import Counter
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
import streamlit as st

st.set_page_config(
    page_title="MediAssist — Analytics",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚕</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _import_from_path(module_name: str, file_path: Path):
    spec   = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_db_mod  = _import_from_path("db", PROJECT_ROOT / "5_database" / "db.py")
Database = _db_mod.Database


# ═══════════════════════════════════════════════════════════════════════════════
#  CSS (same dark clinical theme as main app)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Crimson+Pro:ital,wght@0,400;0,600;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

:root {
  --bg:     #06090f;
  --bg2:    #0b1018;
  --bg3:    #101720;
  --bg4:    #162035;
  --accent: #00c9ad;
  --text:   #d8e8f5;
  --text2:  #7d9ab8;
  --text3:  #3d5570;
  --border: rgba(255,255,255,0.06);
  --low:    #10b981;
  --medium: #f59e0b;
  --high:   #f43f5e;
}

header[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"],
#MainMenu, footer { display: none !important; }

.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 55% 40% at 8% 4%, rgba(0,201,173,0.05) 0%, transparent 55%),
        radial-gradient(ellipse 40% 35% at 92% 92%, rgba(99,102,241,0.04) 0%, transparent 55%);
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text) !important;
}
.main .block-container { padding: 1.5rem 2rem 4rem !important; max-width: 1200px !important; }

h1, h2, h3 { color: var(--text) !important; }
p, span, li, label, div { font-family: 'DM Sans', sans-serif !important; color: var(--text) !important; }

[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

/* KPI metric cards */
[data-testid="metric-container"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.1rem 1.25rem !important;
}
[data-testid="stMetricLabel"]  { color: var(--text2) !important; font-size: 0.78rem !important; }
[data-testid="stMetricValue"]  { color: var(--text)  !important; font-size: 2rem  !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"]  { color: var(--accent) !important; }

/* Plotly chart containers */
.js-plotly-plot { border-radius: 12px !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--bg4); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Cached DB
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_db() -> Database:
    return Database()


# ═══════════════════════════════════════════════════════════════════════════════
#  Plotly theme helper
# ═══════════════════════════════════════════════════════════════════════════════

_PLOTLY_LAYOUT = dict(
    paper_bgcolor="#101720",
    plot_bgcolor="#101720",
    font=dict(family="DM Sans, sans-serif", color="#d8e8f5", size=12),
)

# Reusable axis style — reference directly in per-chart update_layout calls
_AXIS = dict(gridcolor="rgba(255,255,255,0.05)", linecolor="rgba(255,255,255,0.08)")

TRIAGE_COLORS = {"LOW": "#10b981", "MEDIUM": "#f59e0b", "HIGH": "#f43f5e"}


# ═══════════════════════════════════════════════════════════════════════════════
#  Page header
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="
    display:flex; align-items:center; gap:12px;
    padding: 0.25rem 0 1.25rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 1.5rem;
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
        ">Analytics Dashboard</div>
        <div style="
            font-size:0.68rem; font-weight:700; letter-spacing:0.12em;
            text-transform:uppercase; color:#3d5570; margin-top:1px;
        ">MediAssist — Usage Insights</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar nav ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<style>
.ma-nav-link {
    display:flex; align-items:center; gap:10px;
    padding: 8px 10px; border-radius: 8px;
    color: #7d9ab8; font-size: 0.875rem; font-weight: 500;
    text-decoration: none; margin-bottom: 2px;
    transition: background 0.15s, color 0.15s;
}
.ma-nav-link:hover { background: rgba(255,255,255,0.05); color: #d8e8f5; }
</style>
<div style="padding: 1rem 1rem 0;">
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
    ">⚕</div>
    <div style="font-family:'Crimson Pro',Georgia,serif;font-size:1.1rem;font-weight:600;color:#d8e8f5;">MediAssist</div>
  </div>

  <a href="http://localhost:8501" target="_self" class="ma-nav-link">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
    Chat Interface
  </a>

  <a href="http://localhost:8000/developer-docs" target="_blank" class="ma-nav-link">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
    API Documentation
  </a>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Load data
# ═══════════════════════════════════════════════════════════════════════════════

db = load_db()

total_queries   = db.get_total_query_count()
total_sessions  = db.get_session_count()
total_emergency = db.get_emergency_count()
triage_stats    = db.get_triage_stats()
all_queries     = db.get_all_queries()
top_symptoms    = db.get_entity_frequency("symptoms", top_n=12)


# ═══════════════════════════════════════════════════════════════════════════════
#  KPI Cards
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.13em;
    text-transform:uppercase;color:#3d5570;font-family:'DM Sans',sans-serif;
    margin-bottom:0.75rem;">Overview</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.metric("Total Queries", f"{total_queries:,}")
with k2:
    st.metric("Sessions", f"{total_sessions:,}")
with k3:
    high_count = triage_stats.get("HIGH", 0)
    st.metric("High-Urgency", f"{high_count:,}")
with k4:
    st.metric("Emergency Flags", f"{total_emergency:,}")


st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Check for data
# ═══════════════════════════════════════════════════════════════════════════════

if total_queries == 0:
    st.markdown("""
<div style="
    background:rgba(0,201,173,0.04);
    border:1px solid rgba(0,201,173,0.14);
    border-radius:14px;padding:2.5rem 2rem;text-align:center;
">
    <div style="font-family:'Crimson Pro',Georgia,serif;font-size:1.4rem;color:#d8e8f5;margin-bottom:8px;">
        No data yet
    </div>
    <div style="font-size:0.87rem;color:#7d9ab8;font-family:'DM Sans',sans-serif;">
        Start a conversation in the Chat Interface to generate analytics data.
    </div>
</div>
""", unsafe_allow_html=True)
    st.stop()


# ═══════════════════════════════════════════════════════════════════════════════
#  Row 1: Triage Donut + Queries per Day
# ═══════════════════════════════════════════════════════════════════════════════

import plotly.graph_objects as go
import plotly.express as px

col_left, col_right = st.columns([1, 2])

# ── Triage Donut ─────────────────────────────────────────────────────────────
with col_left:
    st.markdown("""
<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.13em;
    text-transform:uppercase;color:#3d5570;font-family:'DM Sans',sans-serif;
    margin-bottom:0.5rem;">Triage Distribution</div>
""", unsafe_allow_html=True)

    labels = [k for k in ("LOW", "MEDIUM", "HIGH") if k in triage_stats]
    values = [triage_stats[k] for k in labels]
    colors = [TRIAGE_COLORS[k] for k in labels]

    fig_donut = go.Figure(go.Pie(
        labels=labels,
        values=values,
        marker=dict(colors=colors, line=dict(color="#06090f", width=3)),
        hole=0.62,
        textinfo="percent",
        textfont=dict(size=12, color="#d8e8f5"),
        hovertemplate="<b>%{label}</b><br>%{value} queries (%{percent})<extra></extra>",
    ))
    fig_donut.add_annotation(
        text=f"<b>{total_queries}</b><br><span style='font-size:10px'>queries</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=16, color="#d8e8f5", family="DM Sans"),
        align="center",
    )
    fig_donut.update_layout(
        **_PLOTLY_LAYOUT,
        showlegend=True,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.05,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7d9ab8", size=11),
        ),
        height=320,
        margin=dict(t=20, b=40, l=20, r=20),
    )
    st.plotly_chart(fig_donut, use_container_width=True)


# ── Queries per Day trend ─────────────────────────────────────────────────────
with col_right:
    st.markdown("""
<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.13em;
    text-transform:uppercase;color:#3d5570;font-family:'DM Sans',sans-serif;
    margin-bottom:0.5rem;">Queries Over Time</div>
""", unsafe_allow_html=True)

    if all_queries:
        # Group by date × triage level
        daily: dict[str, dict[str, int]] = {}
        for q in all_queries:
            d = q["date"]
            t = q["triage_level"]
            daily.setdefault(d, {"LOW": 0, "MEDIUM": 0, "HIGH": 0})
            daily[d][t] = daily[d].get(t, 0) + 1

        dates = sorted(daily.keys())

        fig_trend = go.Figure()
        for level in ("LOW", "MEDIUM", "HIGH"):
            fig_trend.add_trace(go.Bar(
                name=level,
                x=dates,
                y=[daily[d].get(level, 0) for d in dates],
                marker_color=TRIAGE_COLORS[level],
                opacity=0.88,
                hovertemplate=f"<b>{level}</b><br>%{{x}}<br>%{{y}} queries<extra></extra>",
            ))

        fig_trend.update_layout(
            **_PLOTLY_LAYOUT,
            barmode="stack",
            height=320,
            xaxis=dict(
                **_AXIS,
                tickformat="%b %d",
                tickangle=-30,
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                **_AXIS,
                title=dict(text="Queries", font=dict(size=11, color="#7d9ab8")),
                tickfont=dict(size=10),
            ),
            legend=dict(
                orientation="h", x=1, xanchor="right", y=1.08,
                bgcolor="rgba(0,0,0,0)",
                font=dict(color="#7d9ab8", size=11),
            ),
            margin=dict(t=36, b=60, l=48, r=24),
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    else:
        st.caption("No query history to plot yet.")


st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Row 2: Top Symptoms + Emergency Ratio
# ═══════════════════════════════════════════════════════════════════════════════

col_sym, col_em = st.columns([2, 1])

# ── Top Symptoms horizontal bar ───────────────────────────────────────────────
with col_sym:
    st.markdown("""
<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.13em;
    text-transform:uppercase;color:#3d5570;font-family:'DM Sans',sans-serif;
    margin-bottom:0.5rem;">Most Frequently Reported Symptoms</div>
""", unsafe_allow_html=True)

    if top_symptoms:
        sym_names  = [s["entity"] for s in reversed(top_symptoms)]
        sym_counts = [s["count"]  for s in reversed(top_symptoms)]

        fig_sym = go.Figure(go.Bar(
            x=sym_counts,
            y=sym_names,
            orientation="h",
            marker=dict(
                color=sym_counts,
                colorscale=[[0, "rgba(0,201,173,0.35)"], [1, "rgba(0,201,173,0.9)"]],
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            hovertemplate="<b>%{y}</b><br>%{x} occurrences<extra></extra>",
            text=sym_counts,
            textposition="outside",
            textfont=dict(color="#7d9ab8", size=10),
        ))
        fig_sym.update_layout(
            **_PLOTLY_LAYOUT,
            height=380,
            xaxis=dict(
                **_AXIS,
                title=dict(text="Occurrences", font=dict(size=11, color="#7d9ab8")),
                tickfont=dict(size=10),
            ),
            yaxis=dict(
                gridcolor="rgba(255,255,255,0)", linecolor="rgba(255,255,255,0.08)",
                tickfont=dict(size=10, color="#7d9ab8"),
            ),
            showlegend=False,
            margin=dict(t=20, b=48, l=160, r=60),
        )
        st.plotly_chart(fig_sym, use_container_width=True)
    else:
        st.caption("No symptom data yet — NER entities will appear here after queries.")


# ── Emergency gauge / pie ─────────────────────────────────────────────────────
with col_em:
    st.markdown("""
<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.13em;
    text-transform:uppercase;color:#3d5570;font-family:'DM Sans',sans-serif;
    margin-bottom:0.5rem;">Emergency vs Routine</div>
""", unsafe_allow_html=True)

    non_emergency = max(total_queries - total_emergency, 0)
    fig_em = go.Figure(go.Pie(
        labels=["Routine", "Emergency"],
        values=[non_emergency, total_emergency],
        marker=dict(
            colors=["#10b981", "#f43f5e"],
            line=dict(color="#06090f", width=3),
        ),
        hole=0.55,
        textinfo="percent",
        textfont=dict(size=12, color="#d8e8f5"),
        hovertemplate="<b>%{label}</b><br>%{value} queries (%{percent})<extra></extra>",
        sort=False,
    ))
    pct_em = round(total_emergency / total_queries * 100, 1) if total_queries else 0
    fig_em.add_annotation(
        text=f"<b>{pct_em}%</b><br><span style='font-size:9px'>emergency</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=15, color="#f43f5e", family="DM Sans"),
        align="center",
    )
    fig_em.update_layout(
        **_PLOTLY_LAYOUT,
        showlegend=True,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.05,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color="#7d9ab8", size=11),
        ),
        height=380,
        margin=dict(t=20, b=48, l=20, r=20),
    )
    st.plotly_chart(fig_em, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  Recent Queries Table
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
st.markdown("""
<div style="font-size:0.68rem;font-weight:700;letter-spacing:0.13em;
    text-transform:uppercase;color:#3d5570;font-family:'DM Sans',sans-serif;
    margin-bottom:0.75rem;">Recent Queries</div>
""", unsafe_allow_html=True)

recent = db.get_recent_queries(limit=10)
if recent:
    rows_html = ""
    for q in recent:
        color = TRIAGE_COLORS.get(q["triage_level"], "#7d9ab8")
        em_badge = (
            '<span style="'
            'background:rgba(244,63,94,0.12);border:1px solid rgba(244,63,94,0.3);'
            'border-radius:999px;padding:1px 7px;font-size:0.68rem;color:#f43f5e;'
            'font-family:\'DM Sans\',sans-serif;margin-left:6px;">'
            'EMERGENCY</span>'
        ) if q["is_emergency"] else ""
        rows_html += f"""
<div style="
    display:flex;align-items:center;gap:12px;
    padding:9px 1rem;
    border-bottom:1px solid rgba(255,255,255,0.04);
">
    <div style="width:8px;height:8px;border-radius:50%;background:{color};flex-shrink:0;"></div>
    <div style="flex:1;font-size:0.82rem;color:#d8e8f5;font-family:'DM Sans',sans-serif;">
        {q['patient_text'][:90]}{'...' if len(q['patient_text']) >= 90 else ''}
        {em_badge}
    </div>
    <span style="
        font-size:0.72rem;font-weight:600;color:{color};
        min-width:52px;text-align:right;font-family:'DM Sans',sans-serif;
    ">{q['triage_level']}</span>
    <span style="font-size:0.7rem;color:#3d5570;min-width:80px;text-align:right;font-family:'DM Sans',sans-serif;">
        {q['timestamp'][:10]}
    </span>
</div>"""

    st.markdown(f"""
<div style="
    background:#101720;
    border:1px solid rgba(255,255,255,0.06);
    border-radius:12px;overflow:hidden;
">
    <div style="
        display:flex;gap:12px;align-items:center;
        padding:8px 1rem;
        border-bottom:1px solid rgba(255,255,255,0.06);
        background:rgba(255,255,255,0.02);
    ">
        <div style="width:8px;flex-shrink:0;"></div>
        <div style="flex:1;font-size:0.66rem;font-weight:700;letter-spacing:0.1em;
            text-transform:uppercase;color:#3d5570;font-family:'DM Sans',sans-serif;">Query</div>
        <span style="font-size:0.66rem;font-weight:700;letter-spacing:0.1em;
            text-transform:uppercase;color:#3d5570;min-width:52px;text-align:right;
            font-family:'DM Sans',sans-serif;">Triage</span>
        <span style="font-size:0.66rem;font-weight:700;letter-spacing:0.1em;
            text-transform:uppercase;color:#3d5570;min-width:80px;text-align:right;
            font-family:'DM Sans',sans-serif;">Date</span>
    </div>
    {rows_html}
</div>
""", unsafe_allow_html=True)
else:
    st.caption("No queries yet.")


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="
    margin-top:2rem;padding-top:1rem;
    border-top:1px solid rgba(255,255,255,0.06);
    font-size:0.72rem;color:#3d5570;
    font-family:'DM Sans',sans-serif;text-align:center;
">
    MediAssist Analytics &mdash; Data from local SQLite instance &mdash;
    General health information only, not medical advice.
</div>
""", unsafe_allow_html=True)
