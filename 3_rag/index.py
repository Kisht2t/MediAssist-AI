"""
Step 4 — Build ChromaDB Vector Store
======================================
Loads medical reference documents from 1_data/medical_docs/,
chunks them, embeds them with BAAI/bge-small-en-v1.5, and
stores them in a persistent ChromaDB collection.

Also seeds the DB with a curated set of medical FAQ documents
covering common symptoms, triage levels, and emergency signs.

Run:
  python 3_rag/index.py

Output:
  3_rag/chroma_db/   ← persistent vector store (gitignored)
"""

from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT  = Path(__file__).parent.parent
DOCS_DIR      = PROJECT_ROOT / "1_data" / "medical_docs"
CHROMA_DIR    = Path(__file__).parent / "chroma_db"
COLLECTION    = "medical_knowledge"
EMBED_MODEL   = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 64
# ─────────────────────────────────────────────────────────────────────────────

# ── Seed documents (inline) ──────────────────────────────────────────────────
# These core triage & safety documents are always indexed,
# regardless of what files are in medical_docs/.
SEED_DOCS = [
    {
        "content": """
EMERGENCY SIGNS — Call 911 Immediately
=======================================
The following symptoms require immediate emergency care (911 or nearest ER):

- Chest pain, pressure, tightness, or squeezing (possible heart attack)
- Sudden severe headache ("worst headache of my life") — possible stroke or aneurysm
- Sudden numbness or weakness on one side of the face, arm, or leg
- Sudden confusion, difficulty speaking, or trouble understanding speech
- Sudden trouble seeing in one or both eyes
- Trouble breathing or shortness of breath at rest
- Coughing or vomiting blood
- Severe abdominal pain, especially if the abdomen is rigid or board-like
- High fever (>103°F / 39.4°C) with stiff neck, confusion, or rash
- Seizures
- Loss of consciousness or unresponsiveness
- Severe allergic reaction (throat swelling, can't breathe, hives after eating or a sting)
- Signs of stroke: use FAST — Face drooping, Arm weakness, Speech difficulty, Time to call 911

ALWAYS err on the side of caution. If in doubt, call 911.
        """.strip(),
        "metadata": {"source": "emergency_signs", "triage": "HIGH", "category": "safety"},
    },
    {
        "content": """
TRIAGE LEVELS — How Urgency Is Assessed
=========================================
MediAssist classifies symptoms into three triage levels:

HIGH (Emergency — Go to ER or Call 911):
  - Chest pain, difficulty breathing, signs of stroke, unresponsiveness
  - Severe bleeding that won't stop, major trauma
  - Suspected overdose or poisoning
  - High fever with confusion or stiff neck

MEDIUM (Urgent — See a Doctor Within 24 Hours):
  - Fever above 101°F lasting more than 2 days
  - Moderate pain that does not respond to OTC medication
  - Ear infection symptoms (ear pain, discharge, hearing loss)
  - Urinary tract infection symptoms (burning, frequent urination, blood in urine)
  - Rash with no known cause
  - Persistent vomiting or diarrhea (>24 hours)
  - Minor injuries with significant swelling or bruising

LOW (Self-Care / Schedule an Appointment):
  - Common cold symptoms (runny nose, mild sore throat, mild cough)
  - Minor headache responsive to OTC pain relief
  - Mild allergies
  - Low-grade fever (<100.4°F) without other concerning symptoms
  - Minor cuts or scrapes (properly cleaned)
  - Mild muscle aches after exercise

Note: These are general guidelines. Always consult a qualified physician
for an accurate diagnosis. MediAssist is for informational purposes only.
        """.strip(),
        "metadata": {"source": "triage_levels", "category": "triage"},
    },
    {
        "content": """
COMMON SYMPTOMS — What They May Indicate
==========================================

HEADACHE
  - Tension headache: dull, pressure-like pain on both sides of head; often from stress
  - Migraine: throbbing, often one-sided; may include nausea, light/sound sensitivity
  - Cluster headache: severe pain around one eye, recurring in clusters
  - WARNING: sudden severe headache, or headache with stiff neck/fever → EMERGENCY

FEVER
  - Low-grade (99–100.4°F): usually viral; rest and fluids
  - Moderate (100.4–103°F): consult doctor if lasting > 2 days or with other symptoms
  - High (>103°F): seek medical care; >104°F in adults is a medical emergency
  - Fever with rash, stiff neck, or confusion → EMERGENCY

CHEST PAIN
  - Sharp pain that worsens with deep breath → may be pleuritis or costochondritis
  - Burning sensation after eating → may be acid reflux/GERD
  - Pressure, squeezing, or tightness → possible cardiac; treat as EMERGENCY
  - Accompanied by shortness of breath, sweating, arm pain → EMERGENCY

SHORTNESS OF BREATH
  - With exertion only, mild → monitor; may be deconditioning
  - At rest, sudden onset → EMERGENCY (possible pulmonary embolism, heart failure)
  - With wheezing → may be asthma; use rescue inhaler if prescribed

ABDOMINAL PAIN
  - Upper right quadrant: may indicate gallbladder or liver
  - Lower right quadrant: may indicate appendicitis (with fever/nausea → urgent)
  - Generalized with rigidity or "board-like" abdomen → EMERGENCY
  - Accompanied by vomiting blood or black stools → EMERGENCY
        """.strip(),
        "metadata": {"source": "common_symptoms", "category": "symptom_guide"},
    },
    {
        "content": """
WHEN TO SEEK CARE — Decision Guide
=====================================

Go to the Emergency Room (ER) immediately if you have:
  - Any symptom listed under EMERGENCY SIGNS
  - Severe pain (8-10/10) with no relief
  - Symptoms of stroke (FAST: Face, Arm, Speech, Time)
  - Suspected heart attack

Visit an Urgent Care clinic (within a few hours) if you have:
  - Moderate pain not relieved by OTC medication
  - Fever in an adult (> 101°F lasting > 48 hours)
  - Minor broken bone or sprain
  - Pink eye or ear infection
  - UTI symptoms

Schedule a doctor appointment (within a few days) if you have:
  - Symptoms that are mild but persistent (> 1 week)
  - Medication refill needs
  - Routine follow-up questions
  - Mild skin conditions

Manage at home with self-care if:
  - Common cold (rest, fluids, OTC decongestants/analgesics)
  - Mild sore throat without fever or difficulty swallowing
  - Minor cut (clean with soap and water, apply bandage)
  - Muscle soreness from exercise (rest, ice, gentle stretching)

DISCLAIMER: This guide is for general informational purposes only and does not
constitute medical advice. Always consult a licensed healthcare provider for
diagnosis and treatment.
        """.strip(),
        "metadata": {"source": "care_guide", "category": "decision_support"},
    },
]
# ─────────────────────────────────────────────────────────────────────────────


def load_local_docs():
    """Load any .txt files the user has placed in 1_data/medical_docs/."""
    if not DOCS_DIR.exists() or not any(DOCS_DIR.glob("*.txt")):
        print(f"   No local docs found in {DOCS_DIR} — using seed documents only.")
        return []

    loader = DirectoryLoader(
        str(DOCS_DIR),
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    docs = loader.load()
    print(f"   Loaded {len(docs)} local document(s) from {DOCS_DIR}")
    return docs


def main():
    print("── Step 4: Building ChromaDB vector store ────────────────────────────")
    print(f"   Embedding model : {EMBED_MODEL}")
    print(f"   Collection      : {COLLECTION}")
    print(f"   Output dir      : {CHROMA_DIR}")
    print()

    # ── 1. Load local docs (optional) ────────────────────────────────────────
    local_docs = load_local_docs()

    # ── 2. Split local docs into chunks ──────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(local_docs) if local_docs else []
    print(f"   Chunked local docs: {len(chunks)} chunks")

    # ── 3. Convert seed docs into LangChain Document objects ─────────────────
    from langchain_core.documents import Document

    seed_documents = [
        Document(page_content=d["content"], metadata=d["metadata"])
        for d in SEED_DOCS
    ]

    # Split seed docs too (they're already reasonably sized but be consistent)
    seed_chunks = splitter.split_documents(seed_documents)
    print(f"   Seed documents   : {len(seed_chunks)} chunks")

    all_chunks = chunks + seed_chunks
    print(f"   Total chunks     : {len(all_chunks)}")
    print()

    # ── 4. Embed and store in ChromaDB ───────────────────────────────────────
    print("   Embedding and indexing (this may take a minute)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Wipe old collection if it exists, then rebuild
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION)
        print("   Deleted old collection — rebuilding fresh.")
    except Exception:
        pass

    vectorstore = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=str(CHROMA_DIR),
    )

    print()
    print(f"✅  Vector store built successfully!")
    print(f"   {len(all_chunks)} chunks indexed in '{COLLECTION}'")
    print(f"   Stored at: {CHROMA_DIR}")
    print()
    print("── Quick Test ────────────────────────────────────────────────────────")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    test_query = "I have severe chest pain and difficulty breathing"
    results = retriever.invoke(test_query)
    print(f"   Query  : \"{test_query}\"")
    print(f"   Top hit: {results[0].page_content[:200]}...")
    print()
    print("── Next Step ─────────────────────────────────────────────────────────")
    print("   python 3_rag/pipeline.py")
    print()


if __name__ == "__main__":
    main()
