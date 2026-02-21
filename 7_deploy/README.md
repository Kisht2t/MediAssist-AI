---
title: MediAssist
emoji: 🩺
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: "1.32.0"
app_file: 6_ui/app.py
pinned: false
license: mit
---

# MediAssist — AI Medical Assistant

A fine-tuned Llama 3.2 3B medical chatbot with RAG, Medical NER, and triage classification.

## Architecture
- **LLM**: Llama 3.2 3B Instruct fine-tuned on ChatDoctor (LoRA via MLX-LM)
- **RAG**: ChromaDB + BAAI/bge-small-en-v1.5 embeddings + cross-encoder reranking
- **NER**: scispaCy (en_core_sci_sm) for medical entity extraction
- **Storage**: SQLite via SQLAlchemy
- **UI**: Streamlit

## Setup (HuggingFace Spaces)
Add these secrets in your Space settings:
- `HF_TOKEN` — your HuggingFace write token
- `HF_MODEL_REPO` — e.g., `Kisht2t/mediassist-llama-3.2-3b`
