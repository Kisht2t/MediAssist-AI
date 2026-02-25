"""
3_rag/add_wiki_docs.py
======================
Fetches ~60 Wikipedia medical articles and adds them to the existing
ChromaDB collection WITHOUT wiping what's already there.

Before running, install the wikipedia package:
    pip install wikipedia

Run:
    python 3_rag/add_wiki_docs.py
"""

import time
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
CHROMA_DIR    = Path(__file__).parent / "chroma_db"
COLLECTION    = "medical_knowledge"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64

# ── Medical topics to fetch ────────────────────────────────────────────────────
TOPICS = [
    # Common infections
    "Common cold", "Influenza", "Pneumonia", "Bronchitis", "Sinusitis",
    "Otitis media", "Conjunctivitis", "Gastroenteritis",
    "Urinary tract infection", "Appendicitis",

    # Cardiovascular
    "Hypertension", "Myocardial infarction", "Heart failure",
    "Atrial fibrillation", "Angina pectoris", "Stroke",
    "Deep vein thrombosis", "Aortic aneurysm",

    # Metabolic / endocrine
    "Diabetes mellitus type 2", "Diabetes mellitus type 1",
    "Hypothyroidism", "Hyperthyroidism", "Obesity",

    # Neurological
    "Migraine", "Tension headache", "Epilepsy",
    "Parkinson's disease", "Alzheimer's disease",
    "Multiple sclerosis", "Meningitis", "Encephalitis",

    # Respiratory
    "Asthma", "Chronic obstructive pulmonary disease",
    "Pulmonary embolism", "Tuberculosis", "COVID-19",

    # Gastrointestinal
    "Gastroesophageal reflux disease", "Irritable bowel syndrome",
    "Crohn's disease", "Ulcerative colitis", "Peptic ulcer disease",
    "Celiac disease", "Pancreatitis", "Cholecystitis",

    # Musculoskeletal
    "Osteoarthritis", "Rheumatoid arthritis", "Osteoporosis",
    "Gout", "Fibromyalgia", "Herniated disc",

    # Mental health
    "Major depressive disorder", "Generalized anxiety disorder",
    "Bipolar disorder", "Schizophrenia", "Post-traumatic stress disorder",

    # Rare / serious
    "Systemic lupus erythematosus", "Huntington's disease",
    "Amyotrophic lateral sclerosis", "Cystic fibrosis",
    "Marfan syndrome", "Sarcoidosis",

    # Medications / pharmacology
    "Analgesic", "Antibiotic", "Antihistamine", "Corticosteroid",
    "Beta blocker", "ACE inhibitor", "Statin",

    # Paediatric
    "Kawasaki disease", "Chickenpox", "Measles", "Mumps",

    # Women's health
    "Endometriosis", "Polycystic ovary syndrome", "Preeclampsia",
    "Ectopic pregnancy",

    # Skin
    "Psoriasis", "Eczema", "Cellulitis", "Melanoma",
]
# ─────────────────────────────────────────────────────────────────────────────


def fetch_article(topic: str) -> str | None:
    """Fetch Wikipedia article text (first 8,000 characters)."""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        page = wikipedia.page(topic, auto_suggest=False)
        return page.content[:8_000]
    except Exception as exc:
        # Try with auto_suggest on as fallback
        try:
            import wikipedia
            page = wikipedia.page(topic, auto_suggest=True)
            return page.content[:8_000]
        except Exception:
            print(f"✗  ({exc})")
            return None


def main():
    # ── Verify wikipedia package ──────────────────────────────────────────────
    try:
        import wikipedia  # noqa: F401
    except ImportError:
        print("ERROR: 'wikipedia' package not installed.")
        print("Run:  pip install wikipedia")
        return

    print("── Expanding ChromaDB with Wikipedia medical articles ────────────────")
    print(f"   Topics planned   : {len(TOPICS)}")
    print(f"   Embedding model  : {EMBED_MODEL}")
    print(f"   Persisting to    : {CHROMA_DIR}")
    print()

    # ── Load embedding model ──────────────────────────────────────────────────
    print("   Loading embedding model (BAAI/bge-small-en-v1.5)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Connect to existing ChromaDB (no wipe) ────────────────────────────────
    vectorstore = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    existing_count = vectorstore._collection.count()
    print(f"   Existing chunks  : {existing_count}")
    print()

    # ── Fetch articles + chunk ────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_new_chunks = []
    success, failed = 0, 0

    for i, topic in enumerate(TOPICS, 1):
        print(f"   [{i:2d}/{len(TOPICS)}] {topic:<45}", end=" ", flush=True)
        content = fetch_article(topic)

        if content:
            doc = Document(
                page_content=content,
                metadata={
                    "source":   f"wikipedia:{topic.lower().replace(' ', '_')}",
                    "category": "wikipedia",
                    "topic":    topic,
                },
            )
            chunks = splitter.split_documents([doc])
            all_new_chunks.extend(chunks)
            print(f"✓  ({len(chunks)} chunks)")
            success += 1
        else:
            failed += 1

        time.sleep(0.4)   # be polite to Wikipedia servers

    # ── Add to existing collection ────────────────────────────────────────────
    print()
    print(f"   Fetched {success}/{len(TOPICS)} articles → {len(all_new_chunks)} new chunks")

    if all_new_chunks:
        print("   Embedding and indexing (this may take a few minutes)...")
        vectorstore.add_documents(all_new_chunks)
        new_total = vectorstore._collection.count()
        print()
        print(f"✅  Done!  Collection grew from {existing_count} → {new_total} chunks")
        if failed:
            print(f"⚠   {failed} articles could not be fetched (Wikipedia rate-limit or disambiguation).")
    else:
        print("   No new chunks to add — check your internet connection.")

    print()
    print("── Next Step ─────────────────────────────────────────────────────────")
    print("   Restart the Streamlit app — the expanded knowledge base is live.")
    print()


if __name__ == "__main__":
    main()
