# MediAssist — AI Medical Assistant

A production-style AI medical chatbot that combines **LoRA fine-tuning**, **Retrieval-Augmented Generation (RAG)**, **Medical Named Entity Recognition**, **triage classification**, and a **Streamlit UI** — all running on an Apple Silicon Mac.

---

## Architecture

```
Patient Input (text)
       │
       ▼
┌─────────────────┐
│  Medical NER    │  ← scispaCy extracts symptoms, body parts, durations
│  (scispaCy)     │
└────────┬────────┘
         │ enriched query
         ▼
┌─────────────────┐
│  ChromaDB       │  ← semantic vector search over medical knowledge base
│  Retriever      │     (BAAI/bge-small-en-v1.5 embeddings)
└────────┬────────┘
         │ top-10 candidate docs
         ▼
┌─────────────────┐
│  Cross-Encoder  │  ← re-scores docs by true relevance (ms-marco-MiniLM)
│  Reranker       │
└────────┬────────┘
         │ top-3 grounding docs
         ▼
┌─────────────────┐
│ Triage Classify │  ← rule-based: LOW / MEDIUM / HIGH
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Fine-tuned LLM │  ← Llama 3.2 3B + LoRA (ChatDoctor) via HF Inference API
│  (MediAssist)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  SQLite DB      │  ← stores sessions, NER entities, retrieved docs, responses
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit UI   │  ← chat interface + triage badge + sources panel
└─────────────────┘
```

---

## Tech Stack

| Component        | Technology                                      |
|------------------|-------------------------------------------------|
| Fine-tuning      | MLX-LM + LoRA (Apple Silicon M-series)          |
| Base model       | `meta-llama/Llama-3.2-3B-Instruct`              |
| Training data    | `avaliev/chat_doctor` (5k patient-doctor pairs) |
| Model hosting    | HuggingFace Hub + Inference API                 |
| Embeddings       | `BAAI/bge-small-en-v1.5`                        |
| Vector store     | ChromaDB (persistent)                           |
| Reranker         | `cross-encoder/ms-marco-MiniLM-L-6-v2`          |
| Medical NER      | scispaCy (`en_core_sci_sm`)                     |
| Database         | SQLite + SQLAlchemy                             |
| UI               | Streamlit                                       |
| Deployment       | HuggingFace Spaces                              |

---

## Project Structure

```
DocPatient Proejct/
├── 1_data/
│   ├── prepare_data.py       ← Step 1: Download & format ChatDoctor dataset
│   ├── raw/                  ← (gitignored) raw HF dataset cache
│   ├── processed/            ← (gitignored) train.jsonl, valid.jsonl
│   └── medical_docs/         ← (gitignored) optional custom .txt docs for RAG
│
├── 2_finetune/
│   ├── config.yaml           ← MLX-LM LoRA hyperparameters
│   ├── train.py              ← Step 2: Run fine-tuning
│   ├── fuse_and_push.py      ← Step 3: Fuse adapter + push to HF Hub
│   └── adapters/             ← (gitignored) saved LoRA checkpoints
│
├── 3_rag/
│   ├── index.py              ← Step 4: Build ChromaDB vector store
│   ├── pipeline.py           ← Step 5: Full RAG pipeline (NER + retrieve + rerank + generate)
│   └── chroma_db/            ← (gitignored) persistent vector store
│
├── 5_database/
│   └── db.py                 ← Step 6: SQLite schema + Database helper class
│
├── 6_ui/
│   └── app.py                ← Step 7: Streamlit chat UI
│
├── 7_deploy/
│   └── README.md             ← Step 8: HuggingFace Spaces deployment config
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Setup & Run

### 1. Clone and Install

```bash
git clone https://github.com/Kisht2t/MediAssist-AI.git
cd "MediAssist-AI"

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install spaCy models
python -m spacy download en_core_web_sm
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz
```

### 2. Configure Secrets

```bash
cp .env.example .env
# Edit .env and fill in HF_TOKEN and HF_MODEL_REPO
```

### 3. Prepare Data

```bash
python 1_data/prepare_data.py
# → generates 1_data/processed/train.jsonl and valid.jsonl
```

### 4. Fine-tune (Apple Silicon only)

```bash
python 2_finetune/train.py
# → takes ~2-3 hours on M2/M3, saves adapter to 2_finetune/adapters/
```

### 5. Push Model to HuggingFace

```bash
python 2_finetune/fuse_and_push.py
# → fuses LoRA weights and uploads to HF Hub
```

### 6. Build Vector Store

```bash
python 3_rag/index.py
# → indexes medical knowledge into 3_rag/chroma_db/
```

### 7. Run the App

```bash
streamlit run 6_ui/app.py
# → opens at http://localhost:8501
```

---

## Key Design Decisions

**Why MLX-LM?**
Apple's M-series chips run MLX dramatically faster than PyTorch+MPS for LoRA fine-tuning. On an M2 16GB, MLX achieves ~3-4x throughput vs. HuggingFace + MPS backend.

**Why LoRA (not full fine-tune)?**
LoRA trains only 0.1-1% of parameters by injecting small rank-decomposition matrices. This reduces memory from ~24GB (full FP16) to ~6GB, making it feasible on consumer Apple Silicon.

**Why cross-encoder reranking?**
Vector similarity (bi-encoder) retrieval is fast but imprecise — it encodes query and documents independently. A cross-encoder sees the query + document together and gives much more accurate relevance scores. We use it as a second-pass filter over the top-10 candidates.

**Why scispaCy?**
General NLP models (spaCy `en_core_web_sm`) don't understand medical terminology. scispaCy is trained on PubMed/MIMIC clinical notes and recognizes entities like "dyspnea", "myocardial infarction", "bilateral crackles" that general models miss.

---

## Disclaimer

MediAssist is for **educational and informational purposes only**. It is not a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns. In emergencies, call 911.
