"""
4_app/api.py
============
FastAPI REST API for MediAssist.

Exposes the full MediAssist pipeline as a production-grade HTTP API.
Any client — mobile app, website, another developer's script — can
call these endpoints and get structured medical responses back.

This is exactly how OpenAI's API works:
  - ChatGPT is their UI  (like our Streamlit app)
  - The API is what developers call  (like this file)

Endpoints:
  POST /query              — main medical query endpoint
  POST /session            — create a new chat session
  GET  /session/{id}/stats — triage stats for a session
  GET  /memory/count       — number of long-term patient facts stored
  GET  /health             — full system health check

Run:
  python 4_app/api.py

Docs auto-generated at:
  http://localhost:8000/docs      ← interactive Swagger UI
  http://localhost:8000/redoc     ← ReDoc UI
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import importlib.util
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ── Dynamic imports from numbered directories ──────────────────────────────────
def _import_from_path(module_name: str, file_path: Path):
    spec   = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_db_mod     = _import_from_path("db",           PROJECT_ROOT / "5_database" / "db.py")
_pipe_mod   = _import_from_path("pipeline",     PROJECT_ROOT / "3_rag"      / "pipeline.py")
_memory_mod = _import_from_path("memory_store", PROJECT_ROOT / "3_rag"      / "memory_store.py")

Database           = _db_mod.Database
MediAssistPipeline = _pipe_mod.MediAssistPipeline
MemoryStore        = _memory_mod.MemoryStore


# ── App state (loaded once, shared across all requests) ───────────────────────
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all heavy resources once at startup, clean up on shutdown."""
    print("Starting MediAssist API — loading pipeline...")
    _state["pipeline"] = MediAssistPipeline()
    _state["db"]       = Database()
    _state["memory"]   = MemoryStore()
    print("MediAssist API is ready.\n")
    yield
    print("Shutting down MediAssist API.")


# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "MediAssist API",
    description = (
        "AI medical assistant API — symptom analysis, triage classification, "
        "and RAG-grounded responses powered by a fine-tuned Llama 3.2 3B model.\n\n"
        "**Model:** `kisht2t/mediassist-llama-3.2-3b` (LoRA fine-tuned on ChatDoctor)\n"
        "**Knowledge base:** 2,000+ medical document chunks (Wikipedia + curated docs)\n"
        "**NER:** scispaCy `en_core_sci_sm`\n"
    ),
    version     = "1.0.0",
    lifespan    = lifespan,
)

# CORS — allows the Streamlit UI (or any other client) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins  = ["*"],
    allow_methods  = ["*"],
    allow_headers  = ["*"],
)

# Serve static files (docs HTML, assets)
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/developer-docs", include_in_schema=False)
def developer_docs():
    """Custom documentation page."""
    return FileResponse(str(_STATIC_DIR / "docs.html"))


# ═══════════════════════════════════════════════════════════════════════════════
#  PYDANTIC SCHEMAS  (defines the shape of requests and responses)
# ═══════════════════════════════════════════════════════════════════════════════

class Message(BaseModel):
    """A single turn in a conversation."""
    role:    str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message content")

    model_config = {"json_schema_extra": {"example": {"role": "user", "content": "I have a fever"}}}


class QueryRequest(BaseModel):
    """Request body for POST /query"""
    text:       str           = Field(...,    description="Patient's message or symptom description")
    session_id: str | None    = Field(None,   description="Session ID from POST /session. One is auto-created if omitted.")
    history:    list[Message] = Field(default=[], description="Previous conversation turns (for context window memory)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "text":       "I've had a headache and fever for 3 days",
                "session_id": "a1b2c3",
                "history":    [],
            }
        }
    }


class QueryResponse(BaseModel):
    """Response from POST /query"""
    answer:       str       = Field(..., description="MediAssist's response to the patient")
    triage_level: str       = Field(..., description="Urgency level: LOW | MEDIUM | HIGH")
    is_emergency: bool      = Field(..., description="True if 911 / ER is recommended")
    entities:     dict      = Field(..., description="Medical entities extracted by NER")
    sources:      list[str] = Field(..., description="RAG source document snippets used")
    session_id:   str       = Field(..., description="Session ID (use this for follow-up messages)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer":       "That sounds like it could be a viral illness...",
                "triage_level": "MEDIUM",
                "is_emergency": False,
                "entities":     {"symptoms": ["headache", "fever"], "durations": ["3 days"]},
                "sources":      ["Fever is defined as a body temperature above 100.4°F..."],
                "session_id":   "a1b2c3",
            }
        }
    }


class SessionResponse(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")


class HealthResponse(BaseModel):
    status:          str  = Field(..., description="'ok' or 'degraded'")
    pipeline_loaded: bool = Field(..., description="True if the LLM model is loaded")
    chromadb_chunks: int  = Field(..., description="Number of indexed knowledge chunks")
    memory_facts:    int  = Field(..., description="Number of stored long-term patient facts")
    model:           str  = Field(..., description="Model identifier")


class StatsResponse(BaseModel):
    session_id:    str  = Field(..., description="Session ID")
    triage_counts: dict = Field(..., description="Count of queries per triage level")


class MemoryCountResponse(BaseModel):
    count: int = Field(..., description="Number of long-term patient facts stored")


# ═══════════════════════════════════════════════════════════════════════════════
#  ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    response_model = HealthResponse,
    tags           = ["System"],
    summary        = "Health check — are all components ready?",
)
def health():
    """
    Returns the status of all MediAssist components:
    - LLM model loaded
    - ChromaDB chunk count
    - Long-term memory fact count
    """
    pipeline: MediAssistPipeline = _state.get("pipeline")
    memory:   MemoryStore        = _state.get("memory")

    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialised yet — try again in a moment.")

    try:
        chunks = pipeline.retriever._store._collection.count()
    except Exception:
        chunks = -1

    return HealthResponse(
        status          = "ok",
        pipeline_loaded = pipeline._mlx_model is not None,
        chromadb_chunks = chunks,
        memory_facts    = memory.count() if memory else 0,
        model           = "kisht2t/mediassist-llama-3.2-3b (Llama 3.2 3B + LoRA)",
    )


@app.post(
    "/session",
    response_model = SessionResponse,
    tags           = ["Session"],
    summary        = "Create a new chat session",
)
def create_session():
    """
    Creates a new chat session and returns a `session_id`.

    Pass this `session_id` in subsequent `/query` calls to link them
    to the same conversation for analytics and history.
    """
    db: Database = _state["db"]
    return SessionResponse(session_id=str(db.create_session()))


@app.post(
    "/query",
    response_model = QueryResponse,
    tags           = ["Query"],
    summary        = "Send a medical query — the main endpoint",
)
def query(req: QueryRequest):
    """
    The core MediAssist endpoint. Send a patient message and receive:

    - **answer** — a natural, empathetic medical response
    - **triage_level** — LOW / MEDIUM / HIGH urgency classification
    - **is_emergency** — whether 911 or ER is recommended
    - **entities** — extracted symptoms, durations, measurements
    - **sources** — the RAG documents that informed the response

    ### Memory
    - Pass `history` (previous turns) for **short-term context** within a session
    - Long-term facts (age, allergies, chronic conditions) are **automatically extracted
      and remembered** across sessions

    ### Example
    ```
    POST /query
    {
        "text": "I have chest pain and shortness of breath",
        "session_id": "abc123"
    }
    ```
    """
    pipeline: MediAssistPipeline = _state["pipeline"]
    db:       Database           = _state["db"]
    memory:   MemoryStore        = _state["memory"]

    # Session
    session_id = str(req.session_id or db.create_session())

    # Long-term memory: save any stable facts from this message
    memory.save_facts(req.text, session_id)
    memories = memory.retrieve(req.text, n=5)

    # Short-term memory: last 6 turns of history
    history = [{"role": m.role, "content": m.content} for m in req.history[-6:]]

    # Run the full pipeline
    try:
        response = pipeline.run(req.text, history=history, memories=memories)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

    # Persist to database
    query_id = db.save_query(
        session_id   = session_id,
        patient_text = req.text,
        triage_level = response.triage_level,
        is_emergency = response.is_emergency,
    )
    db.save_entities(query_id, response.entities)
    db.save_retrieved_docs(
        query_id,
        [{"source": f"doc_{i+1}", "content": s} for i, s in enumerate(response.sources)],
    )
    db.save_response(query_id, response.answer)

    return QueryResponse(
        answer       = response.answer,
        triage_level = response.triage_level,
        is_emergency = response.is_emergency,
        entities     = response.entities,
        sources      = response.sources,
        session_id   = session_id,
    )


@app.get(
    "/session/{session_id}/stats",
    response_model = StatsResponse,
    tags           = ["Session"],
    summary        = "Triage stats for a session",
)
def session_stats(session_id: str):
    """Returns a count of LOW / MEDIUM / HIGH queries for the given session."""
    db: Database = _state["db"]
    stats = db.get_triage_stats()
    return StatsResponse(session_id=session_id, triage_counts=stats or {})


@app.get(
    "/memory/count",
    response_model = MemoryCountResponse,
    tags           = ["Memory"],
    summary        = "Number of long-term patient facts stored",
)
def memory_count():
    """Returns how many persistent patient facts are stored in the vector memory."""
    memory: MemoryStore = _state["memory"]
    return MemoryCountResponse(count=memory.count())


# ── Run directly ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False,
        workers = 1,   # 1 worker — MLX model can't be forked across processes
    )
