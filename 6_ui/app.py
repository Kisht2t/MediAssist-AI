"""
Step 7 — MediAssist Streamlit UI
==================================
Full chat interface for the MediAssist medical assistant.

Features:
  - Chat interface with session history
  - Triage badge (LOW / MEDIUM / HIGH) per response
  - Emergency banner with 911 redirect
  - Expandable "Sources Used" panel showing RAG retrieved docs
  - "Entities Detected" sidebar showing NER output
  - Session persistence via SQLite

Run:
  streamlit run 6_ui/app.py
"""

import sys
from pathlib import Path

# Make project root importable
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
import streamlit as st

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="MediAssist — AI Medical Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Import modules from numbered directories using importlib ──────────────────
# Python identifiers can't start with digits, so we use importlib.util directly.
from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / ".env")


def _import_from_path(module_name: str, file_path: Path):
    """Import a module from an absolute file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_db_module       = _import_from_path("db",       PROJECT_ROOT / "5_database" / "db.py")
_pipeline_module = _import_from_path("pipeline", PROJECT_ROOT / "3_rag"      / "pipeline.py")

Database           = _db_module.Database
MediAssistPipeline = _pipeline_module.MediAssistPipeline


# ── Constants ─────────────────────────────────────────────────────────────────
TRIAGE_COLORS = {
    "LOW":    "#28a745",   # green
    "MEDIUM": "#fd7e14",   # orange
    "HIGH":   "#dc3545",   # red
}

TRIAGE_LABELS = {
    "LOW":    "🟢 LOW — Self-care may be appropriate",
    "MEDIUM": "🟠 MEDIUM — See a doctor within 24 hours",
    "HIGH":   "🔴 HIGH — Seek emergency care now",
}

DISCLAIMER = (
    "⚕️ **Disclaimer:** MediAssist provides general health information only. "
    "It is not a substitute for professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare provider."
)


# ── Cached resource initialisation ────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading MediAssist pipeline...")
def load_pipeline() -> MediAssistPipeline:
    return MediAssistPipeline()


@st.cache_resource(show_spinner=False)
def load_db() -> Database:
    return Database()


# ── Session State Initialisation ──────────────────────────────────────────────

def init_session_state():
    if "session_id" not in st.session_state:
        db = load_db()
        st.session_state.session_id = db.create_session()

    if "messages" not in st.session_state:
        st.session_state.messages = []   # list of {"role", "content", "meta"}

    if "pipeline_ready" not in st.session_state:
        st.session_state.pipeline_ready = False


# ── UI Components ─────────────────────────────────────────────────────────────

def render_emergency_banner():
    st.error(
        "🚨 **EMERGENCY DETECTED** — The symptoms described may require immediate attention. "
        "**Please call 911 or go to the nearest Emergency Room immediately.** "
        "Do not wait for online advice.",
        icon="🚨",
    )


def render_triage_badge(triage: str):
    color = TRIAGE_COLORS.get(triage, "#6c757d")
    label = TRIAGE_LABELS.get(triage, triage)
    st.markdown(
        f"""
        <div style="
            background-color: {color};
            color: white;
            padding: 8px 16px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 0.95em;
            display: inline-block;
            margin-bottom: 10px;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[str]):
    if not sources:
        return
    with st.expander("📚 Sources Used (RAG Retrieved Documents)", expanded=False):
        for i, src in enumerate(sources, 1):
            st.markdown(f"**Source {i}:**")
            st.caption(src)
            if i < len(sources):
                st.divider()


def render_entities(entities: dict):
    """Renders NER entities in the sidebar."""
    has_entities = any(v for v in entities.values())
    if not has_entities:
        return

    st.sidebar.markdown("### 🔬 Entities Detected")
    emoji_map = {
        "symptoms":     "🤒",
        "body_parts":   "🫀",
        "durations":    "⏱️",
        "measurements": "📊",
        "other":        "📌",
    }
    for etype, values in entities.items():
        if values:
            emoji = emoji_map.get(etype, "•")
            st.sidebar.markdown(f"**{emoji} {etype.replace('_', ' ').title()}**")
            for v in values:
                st.sidebar.markdown(f"  - {v}")


def render_chat_message(msg: dict):
    """Renders a single chat message bubble with metadata."""
    role = msg["role"]
    content = msg["content"]
    meta = msg.get("meta", {})

    with st.chat_message(role, avatar="🧑‍💻" if role == "user" else "🩺"):
        if role == "assistant" and meta:
            triage = meta.get("triage_level", "LOW")
            is_emergency = meta.get("is_emergency", False)

            if is_emergency:
                render_emergency_banner()

            render_triage_badge(triage)

        st.markdown(content)

        if role == "assistant" and meta:
            render_sources(meta.get("sources", []))


# ── Main App ──────────────────────────────────────────────────────────────────

def main():
    init_session_state()
    pipeline = load_pipeline()
    db = load_db()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        st.image(
            "https://img.icons8.com/color/96/caduceus.png",
            width=64,
        )
        st.title("MediAssist")
        st.caption("AI-Powered Medical Assistant")
        st.divider()

        st.markdown("#### About")
        st.markdown(
            "MediAssist combines a **fine-tuned Llama 3.2 3B** medical model with "
            "**Retrieval-Augmented Generation (RAG)** and **Medical Named Entity Recognition** "
            "to provide grounded, triage-aware health information."
        )
        st.divider()

        # Triage legend
        st.markdown("#### Triage Legend")
        st.markdown("🟢 **LOW** — Self-care appropriate")
        st.markdown("🟠 **MEDIUM** — See doctor within 24h")
        st.markdown("🔴 **HIGH** — Emergency care needed")
        st.divider()

        # Clear chat button
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = db.create_session()
            st.rerun()

        st.divider()

        # Triage stats
        stats = db.get_triage_stats()
        if stats:
            st.markdown("#### Session Stats")
            for level in ["HIGH", "MEDIUM", "LOW"]:
                count = stats.get(level, 0)
                if count:
                    emoji = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[level]
                    st.markdown(f"{emoji} {level}: **{count}** queries")

        # NER entities panel (populated after each query)
        if st.session_state.messages:
            last_assistant = next(
                (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"),
                None,
            )
            if last_assistant and last_assistant.get("meta"):
                render_entities(last_assistant["meta"].get("entities", {}))

    # ── Main Chat Area ────────────────────────────────────────────────────────
    st.title("🩺 MediAssist — AI Medical Assistant")
    st.caption(DISCLAIMER)
    st.divider()

    # Render existing chat history
    for msg in st.session_state.messages:
        render_chat_message(msg)

    # Welcome message if no history
    if not st.session_state.messages:
        with st.chat_message("assistant", avatar="🩺"):
            st.markdown(
                "Hello! I'm **MediAssist**, your AI-powered medical information assistant. "
                "I can help you understand symptoms, assess urgency, and provide general health guidance.\n\n"
                "**Tell me what's bothering you today.** For example:\n"
                "- *I have a headache and fever for 3 days*\n"
                "- *My child has been vomiting since last night*\n"
                "- *I have chest pain and it's hard to breathe*\n\n"
                "> ⚕️ I'm not a replacement for a real doctor — always seek professional care for serious concerns."
            )

    # ── Chat Input ────────────────────────────────────────────────────────────
    if patient_text := st.chat_input("Describe your symptoms..."):

        # Show user message immediately
        user_msg = {"role": "user", "content": patient_text, "meta": {}}
        st.session_state.messages.append(user_msg)

        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(patient_text)

        # Run pipeline
        with st.chat_message("assistant", avatar="🩺"):
            with st.spinner("Analyzing your symptoms..."):
                response = pipeline.run(patient_text)

            triage = response.triage_level

            # Emergency banner
            if response.is_emergency:
                render_emergency_banner()

            # Triage badge
            render_triage_badge(triage)

            # Answer
            st.markdown(response.answer)

            # Sources
            render_sources(response.sources)

        # Persist to database
        query_id = db.save_query(
            session_id=st.session_state.session_id,
            patient_text=patient_text,
            triage_level=triage,
            is_emergency=response.is_emergency,
        )
        db.save_entities(query_id, response.entities)
        db.save_retrieved_docs(
            query_id,
            [
                {"source": f"doc_{i+1}", "content": src}
                for i, src in enumerate(response.sources)
            ],
        )
        db.save_response(query_id, response.answer)

        # Append to session state with metadata for re-render
        assistant_msg = {
            "role": "assistant",
            "content": response.answer,
            "meta": {
                "triage_level": triage,
                "is_emergency": response.is_emergency,
                "sources": response.sources,
                "entities": response.entities,
            },
        }
        st.session_state.messages.append(assistant_msg)
        st.rerun()


if __name__ == "__main__":
    main()
