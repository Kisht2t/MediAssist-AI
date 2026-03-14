# MediAssist — AI Medical Assistant

> A production-grade AI medical assistant combining LoRA fine-tuning, Retrieval-Augmented Generation, Medical NER, triage classification, and a clinical Streamlit UI — built end-to-end on Apple Silicon.

---

## Links

| | |
|---|---|
| **HuggingFace Model** | [kisht2t/mediassist-llama-3.2-3b](https://huggingface.co/kisht2t/mediassist-llama-3.2-3b) |
| **GitHub Repository** | [Kisht2t/MediAssist-AI](https://github.com/Kisht2t/MediAssist-AI) |
| **API Docs** | `http://localhost:8000/developer-docs` (run locally) |
| **Analytics Dashboard** | `http://localhost:8501/1_Analytics` (run locally) |

---

## What It Does

A patient types their symptoms in plain English. MediAssist:

1. **Extracts medical entities** — symptoms, durations, body parts, measurements (scispaCy NER)
2. **Retrieves grounding knowledge** — semantic search over 2,000+ medical document chunks (ChromaDB)
3. **Reranks results** — cross-encoder ensures the most relevant 3 docs are kept
4. **Classifies urgency** — LOW / MEDIUM / HIGH triage with emergency flag
5. **Generates a response** — fine-tuned Llama 3.2 3B model responds like a medical professional
6. **Shows differential diagnoses** — top 3 possible conditions ranked by symptom match
7. **Remembers the patient** — long-term memory stores facts (age, allergies, conditions) across sessions
8. **Generates SOAP notes** — structured clinical documentation on demand
9. **Logs everything** — SQLite persists all sessions, entities, retrieved docs, and responses

---

## Architecture

```
Patient Input
      │
      ▼
┌─────────────────┐
│  Medical NER    │  scispaCy en_core_sci_sm — extracts symptoms, durations, body parts
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Long-term      │  ChromaDB patient_memory collection — recalls age, allergies,
│  Memory         │  chronic conditions across sessions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ChromaDB       │  BAAI/bge-small-en-v1.5 embeddings — 2,062 chunks from
│  Retriever      │  Wikipedia medical articles + curated seed docs
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Cross-Encoder  │  ms-marco-MiniLM-L-6-v2 — re-scores top-10 by true relevance
│  Reranker       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Triage Engine  │  Rule-based — LOW / MEDIUM / HIGH + emergency flag
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Differential   │  Rule-based — top 3 possible conditions from 12 symptom clusters
│  Diagnosis      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fine-tuned LLM │  Llama 3.2 3B + LoRA adapter trained on ChatDoctor
│  (MLX on-device)│  runs entirely on Apple Silicon — no API calls
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SOAP Note      │  On-demand clinical note generator (S/O/A/P format)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SQLite DB      │  Sessions, queries, NER entities, retrieved docs, responses
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI        │  REST API — POST /query, POST /session, GET /health, etc.
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit UI   │  Chat interface + Analytics Dashboard
└─────────────────┘
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Fine-tuning | MLX-LM + LoRA (Apple Silicon M-series) |
| Base model | `mlx-community/Llama-3.2-3B-Instruct` |
| Fine-tuned model | `kisht2t/mediassist-llama-3.2-3b` |
| Training data | `avaliev/chat_doctor` (4,498 patient-doctor pairs) |
| Embeddings | `BAAI/bge-small-en-v1.5` |
| Vector store | ChromaDB — 2,062 chunks (cosine similarity) |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Medical NER | scispaCy `en_core_sci_sm` |
| Memory | ChromaDB `patient_memory` collection (long-term) + context window (short-term) |
| Database | SQLite + SQLAlchemy |
| REST API | FastAPI + Swagger UI |
| UI | Streamlit (dark clinical theme) |
| Analytics | Plotly — triage donut, trend chart, symptom frequency, KPI cards |

---

## Features

### Chat Interface
- Dark clinical UI with triage badges (LOW / MEDIUM / HIGH)
- Emergency banner with pulsing animation for critical cases
- Retrieved sources panel (collapsible)
- Entity pills in sidebar (symptoms, body parts, durations)
- Two-layer memory: context window (last 6 turns) + long-term ChromaDB facts

### Differential Diagnosis
Automatically displayed after every response — top 3 possible conditions ranked by symptom keyword matching across 12 medical clusters (chest, headache, meningitis, fever, GI, rash, etc.).

### SOAP Note Generator
Click "Generate Clinical Note" in the sidebar to produce a structured clinical note from the conversation:
- **S** — Subjective (patient's chief complaint)
- **O** — Objective (extracted entities, triage level)
- **A** — Assessment (clinical impression)
- **P** — Plan (recommended next steps)

### Analytics Dashboard
Live usage insights at `/1_Analytics`:
- KPI cards: total queries, sessions, high-urgency count, emergency flags
- Triage distribution donut chart
- Queries over time (stacked bar by triage level)
- Most frequently reported symptoms (horizontal bar)
- Emergency vs routine breakdown
- Recent queries table

### REST API
Full FastAPI backend with interactive Swagger docs:

```
POST /query              — main medical query endpoint
POST /session            — create a new chat session
GET  /session/{id}/stats — triage stats for a session
GET  /memory/count       — number of long-term patient facts
GET  /health             — full system health check
GET  /developer-docs     — custom HTML documentation page
```

---

## Project Structure

```
MediAssist-AI/
├── 1_data/
│   ├── prepare_data.py        download & format ChatDoctor dataset
│   └── processed/             train.jsonl, valid.jsonl (gitignored)
│
├── 2_finetune/
│   ├── config.yaml            MLX-LM LoRA hyperparameters
│   ├── train.py               LoRA fine-tuning script
│   ├── fuse_and_push.py       fuse adapter + push to HF Hub
│   └── adapters/              saved LoRA checkpoints (gitignored)
│
├── 3_rag/
│   ├── index.py               build ChromaDB vector store
│   ├── add_wiki_docs.py       add 80 Wikipedia medical articles to ChromaDB
│   ├── pipeline.py            full RAG pipeline (NER → retrieve → rerank → triage → generate → differentials)
│   ├── memory_store.py        long-term patient memory (ChromaDB patient_memory collection)
│   └── chroma_db/             persistent vector store (gitignored)
│
├── 4_app/
│   ├── api.py                 FastAPI REST API
│   └── static/
│       └── docs.html          custom developer documentation page
│
├── 5_database/
│   └── db.py                  SQLite schema + Database helper (sessions, queries, entities, docs, responses)
│
├── 6_ui/
│   ├── app.py                 Streamlit chat UI
│   └── pages/
│       └── 1_Analytics.py     Analytics Dashboard (Plotly)
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup & Run

### 1. Clone and install

```bash
git clone https://github.com/Kisht2t/MediAssist-AI.git
cd MediAssist-AI

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Medical NER models
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in HF_TOKEN (HuggingFace token for model access)
```

### 3. Build the vector store

```bash
python 3_rag/index.py          # indexes the 4 seed medical documents
python 3_rag/add_wiki_docs.py  # adds 80 Wikipedia medical articles (optional, ~10 min)
```

### 4. Run fine-tuning (Apple Silicon only)

```bash
python 2_finetune/train.py
# ~2-3 hours on M2/M3, saves adapter to 2_finetune/adapters/
```

### 5. Start the app

```bash
# Streamlit UI
streamlit run 6_ui/app.py
# → http://localhost:8501

# FastAPI (optional, separate terminal)
python 4_app/api.py
# → http://localhost:8000
# → http://localhost:8000/developer-docs (interactive API docs)
```

---

## Key Design Decisions

**Why MLX-LM for fine-tuning?**
Apple's M-series chips run MLX 3-4x faster than PyTorch+MPS for LoRA training. On M2 16GB, a 3B model fine-tunes in ~2-3 hours vs 8-10 hours on MPS.

**Why LoRA (not full fine-tune)?**
LoRA trains only ~0.5% of parameters by injecting small rank-decomposition matrices. Memory drops from ~24GB (full FP16) to ~6GB — feasible on consumer Apple Silicon.

**Why cross-encoder reranking?**
Vector similarity (bi-encoder) encodes query and document independently — it misses nuanced relevance. A cross-encoder sees query + document together, giving much more accurate scores at the cost of speed. We run it only on the top-10 pre-filtered candidates.

**Why scispaCy over general spaCy?**
General models miss medical terminology entirely. `en_core_sci_sm` is trained on PubMed and MIMIC clinical notes — it recognises "dyspnea", "bilateral crackles", "myocardial infarction" as clinical entities.

**Why two memory layers?**
Short-term (context window, last 6 turns) handles within-session continuity. Long-term (ChromaDB `patient_memory`) persists stable facts — age, allergies, chronic conditions — across sessions, so the assistant never forgets a returning patient.

---

## Disclaimer

MediAssist is for **educational and informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider. In emergencies, call 911.


TO DOS:
 Plan: RAG Expansion with Full Collection Isolation   

 Context

 MediAssist currently has a single ChromaDB collection (medical_knowledge) with
 2,062 chunks (seed docs + Wikipedia articles). The user wants to add a drug/medication
 database and clinical guidelines without disturbing the existing collection, with
 guardrails so future additions also stay isolated. Image analysis (Claude API) is
 deferred until after AWS setup.

 ---
 Isolation Architecture

 ChromaDB supports multiple named collections inside the same persist directory.
 The existing chroma_db/ folder gains new collections — the original is never touched.

 3_rag/chroma_db/
   ├── medical_knowledge     ← EXISTING — never modified
   ├── drug_database         ← NEW: drug info, interactions, side effects
   └── clinical_guidelines   ← NEW: WHO/CDC/NIH protocols

 Each collection:
 - Has its own dedicated indexing script
 - Can be rebuilt independently (python 3_rag/index_drugs.py --rebuild)
 - Has source metadata on every chunk for traceability
 - If missing at runtime, pipeline skips it gracefully (no crash)

 ---
 Feature 2: Document & Scan Upload

 Users can upload two types of files from the Streamlit sidebar:

 Type A — Text Documents (PDF, TXT lab reports, discharge summaries)

 User uploads PDF/TXT
         ↓
 Extract text (pdfplumber for PDF, raw read for TXT)
         ↓
 Chunk (512 tokens, 64 overlap) + metadata {source: "patient_upload", session_id}
         ↓
 Store in isolated `patient_documents` ChromaDB collection
         ↓
 Auto-retrieved in future queries for this session (same MedicalRetriever flow)
         ↓
 "Your document has been indexed. Ask me anything about it."

 Type B — Scan Images (PNG, JPG — X-rays, MRIs, blood test photos)

 User uploads image
         ↓
 Send to Claude API (claude-sonnet-4-6) with medical imaging system prompt
         ↓
 Claude returns conversational analysis (what it sees, possible findings, caveats)
         ↓
 Displayed as assistant message in chat — same UI as normal responses
         ↓
 Analysis text optionally stored in patient memory for context

 No re-training or model changes needed. Claude handles image understanding.
 Requires ANTHROPIC_API_KEY in .env.

 ---
 Files to Create (additive only — nothing existing is deleted)

 1. 3_rag/index_drugs.py

 - Fetches drug data from OpenFDA public API (free, no API key)
 - Endpoints used:
   - /drug/label — indications, warnings, dosage, contraindications
   - /drug/event — common adverse events
 - Chunks to 512 tokens (same as existing), overlap 64
 - Stores into drug_database collection
 - Metadata per chunk: {"source": "openFDA", "drug_name": str, "category": str}
 - ~500–1000 drugs covered (top common medications)

 2. 3_rag/index_guidelines.py

 - Indexes curated clinical guideline summaries
 - Sources: hardcoded high-quality text (WHO/CDC/NIH protocols) — same pattern
 as existing SEED_DOCS in index.py (proven approach, no scraping needed)
 - Topics: hypertension, diabetes, asthma, antibiotics, pain management,
 mental health crisis, pediatric fever, pregnancy warning signs
 - Stores into clinical_guidelines collection
 - Metadata per chunk: {"source": "WHO"|"CDC"|"NIH", "category": str, "topic": str}

 ---
 Files to Modify

 3. 3_rag/pipeline.py — MedicalRetriever class only (lines 341–374)

 Current (single collection):
 self._store = Chroma(collection_name=COLLECTION, ...)
 results = self._store.similarity_search_with_relevance_scores(query, k=k)

 New (multi-collection with isolation):
 # Connect to each collection independently
 self._stores = {}
 for col in [COLLECTION, "drug_database", "clinical_guidelines"]:
     try:
         store = Chroma(collection_name=col, ...)
         if store._collection.count() > 0:
             self._stores[col] = store
     except Exception:
         pass  # collection doesn't exist yet — skip silently

 Fetch merges results from all available collections, deduplicates by content hash,
 then feeds the combined list to the existing cross-encoder reranker unchanged.
 The reranker already handles mixed-source docs — no changes needed there.

 ---
 Guardrails

 1. Script isolation: Each index_*.py only touches its own collection.
 Running index_drugs.py cannot affect medical_knowledge — different
 collection name, independent ChromaDB client.
 2. Graceful degradation: If drug_database doesn't exist yet (not indexed),
 MedicalRetriever skips it silently. Existing behavior is unchanged.
 3. Metadata tagging: Every chunk carries {"collection": name, "source": ...}
 so retrieved sources shown in the UI identify where the info came from.
 4. Future additions: To add a new knowledge domain later, just create
 index_newdomain.py targeting a new collection name. Zero changes to
 pipeline.py — it auto-discovers non-empty collections.
 5. Rebuild safety: Each index script wipes and rebuilds only its own collection
 (same client.delete_collection() pattern as existing index.py line 236).

 ---
 3. 3_rag/scan_analyzer.py (new)

 - Thin wrapper around Claude API for image analysis
 - analyze_scan(image_bytes, mime_type) -> str
 - Medical imaging system prompt: "You are a medical imaging assistant..."
 - Includes disclaimer in every response
 - Requires ANTHROPIC_API_KEY in .env

 ---
 Files to Modify

 4. 3_rag/pipeline.py — MedicalRetriever class only (lines 341–374)

 (As described above — multi-collection with graceful fallback)

 5. 4_app/api.py — add one new endpoint

 POST /upload
   - Accepts: multipart/form-data (file + session_id)
   - PDF/TXT → extract text → chunk → index into patient_documents collection
   - Image (PNG/JPG) → call scan_analyzer.analyze_scan() → return analysis
   - Returns: {"type": "document"|"image", "message": str, "analysis": str|None}
 New imports only: UploadFile, File, pdfplumber, scan_analyzer

 6. 6_ui/app.py — add upload section to existing sidebar

 - After "Generate Clinical Note" button (line ~911), add _sidebar_section("Upload Files")
 - st.file_uploader(type=["pdf", "txt", "png", "jpg", "jpeg"])
 - On upload: POST to /upload endpoint → show result as assistant message in chat
 - Matches existing sidebar style (_sidebar_section + teal accent)

 ---
 Critical Files

 - 3_rag/pipeline.py — modify MedicalRetriever.__init__ and fetch only
 - 3_rag/index.py — reference only, NOT modified
 - 3_rag/index_drugs.py — create new
 - 3_rag/index_guidelines.py — create new
 - 3_rag/scan_analyzer.py — create new
 - 4_app/api.py — add /upload endpoint only
 - 6_ui/app.py — add upload section to sidebar only

 Dependencies to add to requirements.txt

 - pdfplumber — PDF text extraction
 - anthropic — Claude API for image analysis

 Verification

 1. python 3_rag/index_drugs.py → drug_database collection built
 2. python 3_rag/index_guidelines.py → clinical_guidelines collection built
 3. python 3_rag/index.py → medical_knowledge chunk count unchanged
 4. Query "dosage for ibuprofen" → sources show drug_database chunks
 5. Query "chest pain" → sources show medical_knowledge chunks (existing behavior intact)
 6. Upload a PDF → "indexed" confirmation, follow-up query uses its content
 7. Upload a scan image → Claude returns analysis as chat message
 8. Delete drug_database → app still works on medical_knowledge alone

TO DOS:

Authentication
User records collection for context specification for user
n8n workflow
server hosting
AI Agent for for symptom collection (User query is converted oru structure to LLM + conext from database) Context and promt engineering
RAG Design database for better retrieval (Additional data + patient docs)
UI

-----------

Expand MediAssist seriously — add eval frameworks, benchmark it against GPT-4o, write up what you learned about RAG failures in medical contexts. That's publishable insight.
Build an interpretability micro-project — take a small open source model, visualize attention heads, write about what you found. Anthropic literally does interpretability research — this speaks their language directly.
Contribute to a real open source AI repo — LangChain, LlamaIndex, DSPy, or even Anthropic's own SDK. Even one meaningful PR gets your name in the commit history of something people use.
Build an eval pipeline for LLMs — evals are the hottest thing in AI engineering right now. Build a small public benchmark for something specific (medical QA, trade document parsing, anything), put it on GitHub, write about it.
