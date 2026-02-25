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
