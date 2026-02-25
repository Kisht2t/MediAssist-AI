"""
3_rag/memory_store.py
======================
Long-term patient memory backed by a dedicated ChromaDB collection.

How it works:
  - A separate "patient_memory" collection stores facts extracted from
    patient messages — things that are STABLE over time:
      · age ("I'm 34 years old")
      · allergies ("allergic to penicillin")
      · chronic conditions ("I have diabetes")
      · current medications ("taking metformin daily")
  - Facts are extracted using regex patterns (no extra LLM call needed)
  - On every query, relevant memories are retrieved and injected into
    the LLM system prompt so the model can personalise responses

Contrast with context-window memory (short-term):
  - Context window: raw chat history sent per-request → forgotten on session end
  - MemoryStore:   extracted facts persist across ALL sessions → true long-term memory

Usage:
    store = MemoryStore()
    store.save_facts("I'm 34 and allergic to penicillin", session_id="abc")
    memories = store.retrieve("what antibiotics can I take?")
    # → ["Patient allergy: penicillin", "Patient age: 34"]
"""

import re
import uuid
from pathlib import Path

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT      = Path(__file__).parent.parent
CHROMA_DIR        = Path(__file__).parent / "chroma_db"
MEMORY_COLLECTION = "patient_memory"
EMBED_MODEL       = "BAAI/bge-small-en-v1.5"

# ── Regex patterns for stable patient facts ────────────────────────────────────
# These intentionally do NOT catch acute symptoms like "I have a headache today"
# They target facts that remain true across visits.
_PATTERNS: dict[str, list[str]] = {
    "age": [
        r"\bI[''`]?m\s+(\d{1,3})\s*(?:years?\s*old)?\b",
        r"\b(\d{1,3})\s*years?\s*old\b",
        r"\bage[d]?\s*[:\-]?\s*(\d{1,3})\b",
    ],
    "allergy": [
        r"(?:I[''`]?m\s+)?allergic\s+to\s+([\w\s\-,]+?)(?=\s*[.,]|\s+and\b|\s+but\b|$)",
        r"allerg(?:y|ies)\s+to\s+([\w\s\-,]+?)(?=\s*[.,]|\s+and\b|$)",
        r"can(?:not|[''`]?t)\s+take\s+([\w\s\-]+?)(?=\s+due|\s+because|\s*[.,]|$)",
    ],
    "chronic_condition": [
        (
            r"\bI\s+(?:have|had|was diagnosed with)\s+"
            r"(diabetes(?:\s+type\s*[12])?|hypertension|asthma|COPD|epilepsy|"
            r"arthritis|lupus|cancer|HIV|depression|anxiety|"
            r"hypothyroidism|hyperthyroidism|Crohn[''`]?s?\s*disease|"
            r"colitis|fibromyalgia|multiple sclerosis|"
            r"Parkinson[''`]?s?\s*disease|Alzheimer[''`]?s?\s*disease|"
            r"celiac disease|psoriasis|eczema|heart disease|kidney disease)\b"
        ),
        r"\b(diabetic|hypertensive|asthmatic|epileptic)\b",
        r"diagnosed\s+with\s+([\w\s\-]+?)(?=\s*[.,]|\s+and\b|$)",
    ],
    "medication": [
        r"(?:I[''`]?m?\s+)?(?:taking|on|prescribed)\s+([\w\s\-]+?)(?=\s+for|\s+daily|\s+mg|\s+twice|\s*[.,]|$)",
        r"my\s+(?:current\s+)?medications?\s+(?:is|are|include[s]?)\s+([\w\s,\-]+?)(?=\s*[.]|$)",
    ],
}
# ─────────────────────────────────────────────────────────────────────────────


class MemoryStore:
    """
    Persistent cross-session patient memory.

    Initialise once (cached via @st.cache_resource) and share across requests.
    """

    def __init__(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store = Chroma(
            collection_name=MEMORY_COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
        count = self._store._collection.count()
        print(f"   Memory store ready — {count} fact(s) stored")

    # ── Internal: extract facts from text ────────────────────────────────────

    def _extract_facts(self, text: str) -> list[dict]:
        """
        Run all regex patterns against patient text.
        Returns a list of fact dicts with keys: type, value, readable.
        """
        facts: list[dict] = []
        seen: set[str] = set()   # dedup

        for fact_type, patterns in _PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    value = (
                        match.group(1)
                        if match.lastindex and match.lastindex >= 1
                        else match.group(0)
                    ).strip()

                    # Clean up whitespace and trailing punctuation
                    value = re.sub(r"\s+", " ", value).strip("., ")

                    if not value or len(value) < 2 or len(value) > 80:
                        continue

                    key = f"{fact_type}:{value.lower()}"
                    if key in seen:
                        continue
                    seen.add(key)

                    readable = f"Patient {fact_type.replace('_', ' ')}: {value}"
                    facts.append({"type": fact_type, "value": value, "readable": readable})

        return facts

    # ── Public: save ─────────────────────────────────────────────────────────

    def save_facts(self, patient_text: str, session_id: str) -> int:
        """
        Extract persistent facts from patient_text and store in ChromaDB.

        Args:
            patient_text: The raw patient message.
            session_id:   Current session ID for provenance tracking.

        Returns:
            Number of new facts saved.
        """
        facts = self._extract_facts(patient_text)
        if not facts:
            return 0

        texts  = [f["readable"] for f in facts]
        metas  = [{"type": f["type"], "value": f["value"], "session_id": session_id} for f in facts]
        ids    = [str(uuid.uuid4()) for _ in facts]

        self._store.add_texts(texts=texts, metadatas=metas, ids=ids)
        print(f"   Memory: +{len(facts)} fact(s) → {[f['readable'] for f in facts]}")
        return len(facts)

    # ── Public: retrieve ─────────────────────────────────────────────────────

    def retrieve(self, query: str, n: int = 5) -> list[str]:
        """
        Retrieve the most relevant patient facts for the current query.

        Args:
            query: The current patient message (used for semantic search).
            n:     Max number of facts to return.

        Returns:
            List of human-readable fact strings, e.g.:
            ["Patient allergy: penicillin", "Patient age: 34"]
        """
        total = self._store._collection.count()
        if total == 0:
            return []

        results = self._store.similarity_search(query, k=min(n, total))
        return [doc.page_content for doc in results]

    # ── Public: count ─────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._store._collection.count()
