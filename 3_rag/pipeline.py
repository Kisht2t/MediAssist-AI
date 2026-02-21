"""
Step 5 — RAG Pipeline
======================
Full pipeline that takes raw patient text and returns a grounded response:

  Patient text
       │
       ▼
  [Medical NER]          ← scispaCy extracts symptoms, body parts, durations
       │
       ▼
  [ChromaDB Retrieval]   ← semantic search for relevant medical knowledge
       │
       ▼
  [Cross-Encoder Rerank] ← re-scores docs for true relevance to the query
       │
       ▼
  [Triage Classifier]    ← determines LOW / MEDIUM / HIGH urgency
       │
       ▼
  [LLM Generation]       ← fine-tuned MediAssist model via HF Inference API
       │
       ▼
  MediAssistResponse (text + triage + sources + entities)

Usage:
  from 3_rag.pipeline import MediAssistPipeline

  pipeline = MediAssistPipeline()
  response = pipeline.run("I have a severe headache and stiff neck for 2 days")
  print(response.answer)
  print(response.triage_level)  # HIGH / MEDIUM / LOW
"""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ── Config ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
CHROMA_DIR   = Path(__file__).parent / "chroma_db"
COLLECTION   = "medical_knowledge"
EMBED_MODEL  = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_FETCH  = 10    # docs fetched from vector store
TOP_K_RERANK = 3     # docs kept after reranking

load_dotenv(PROJECT_ROOT / ".env")
HF_TOKEN   = os.getenv("HF_TOKEN", "")
HF_REPO    = os.getenv("HF_MODEL_REPO", "Kisht2t/mediassist-llama-3.2-3b")
# ─────────────────────────────────────────────────────────────────────────────

EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "can't breathe", "cannot breathe",
    "difficulty breathing", "shortness of breath", "unresponsive", "unconscious",
    "seizure", "aneurysm", "worst headache", "vomiting blood", "coughing blood",
    "severe allergic", "anaphylaxis", "overdose", "poisoning", "call 911",
    "go to er", "emergency room",
]

SYSTEM_PROMPT = (
    "You are MediAssist, a helpful and empathetic AI medical assistant. "
    "Your role is to listen to the patient's symptoms, ask clarifying "
    "questions when needed, and provide clear, accurate health information. "
    "Always remind users that your responses are for informational purposes "
    "only and not a substitute for professional medical advice. "
    "For emergencies, always direct users to call 911 or visit the nearest ER."
)


@dataclass
class MediAssistResponse:
    answer: str
    triage_level: str                    # LOW | MEDIUM | HIGH
    entities: dict = field(default_factory=dict)   # NER results
    sources: list  = field(default_factory=list)   # retrieved doc snippets
    is_emergency: bool = False


# ── 1. Medical NER ───────────────────────────────────────────────────────────

class MedicalNER:
    """
    Extracts medical entities from patient text using spaCy + scispaCy.

    Entity types extracted:
      - symptoms / findings  (DISEASE, SYMPTOM)
      - body parts           (BODY_PART)
      - duration             (TIME, DATE)
      - measurements         (QUANTITY, CARDINAL)

    Falls back to simple regex if scispaCy is not installed.
    """

    def __init__(self):
        self._nlp = None
        self._load_model()

    def _load_model(self):
        try:
            import spacy
            # Try scientific NLP model first
            try:
                self._nlp = spacy.load("en_core_sci_sm")
                print("   NER: loaded en_core_sci_sm (scispaCy)")
            except OSError:
                # Fall back to general English model
                self._nlp = spacy.load("en_core_web_sm")
                print("   NER: loaded en_core_web_sm (general spaCy — install scispaCy for better medical NER)")
        except ImportError:
            print("   NER: spaCy not installed — using regex fallback")

    def extract(self, text: str) -> dict:
        if self._nlp is None:
            return self._regex_fallback(text)

        doc = self._nlp(text)
        entities = {
            "symptoms":     [],
            "body_parts":   [],
            "durations":    [],
            "measurements": [],
            "other":        [],
        }

        for ent in doc.ents:
            label = ent.label_.upper()
            text_val = ent.text.strip()

            if label in ("DISEASE", "SYMPTOM", "PROBLEM", "CONDITION"):
                entities["symptoms"].append(text_val)
            elif label in ("BODY_PART", "TISSUE", "ANATOMY"):
                entities["body_parts"].append(text_val)
            elif label in ("TIME", "DATE", "DURATION"):
                entities["durations"].append(text_val)
            elif label in ("QUANTITY", "CARDINAL", "PERCENT"):
                entities["measurements"].append(text_val)
            else:
                entities["other"].append(f"{text_val} ({label})")

        # Deduplicate
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))

        return entities

    @staticmethod
    def _regex_fallback(text: str) -> dict:
        """Minimal regex-based entity extraction as a fallback."""
        duration_pattern = re.compile(
            r"\b(\d+\s+(?:day|days|week|weeks|month|months|hour|hours|year|years))\b",
            re.IGNORECASE,
        )
        measurement_pattern = re.compile(
            r"\b(\d+(?:\.\d+)?)\s*(?:°F|°C|mg|ml|mmHg|bpm|kg|lbs?)\b",
            re.IGNORECASE,
        )
        return {
            "symptoms":     [],
            "body_parts":   [],
            "durations":    duration_pattern.findall(text),
            "measurements": measurement_pattern.findall(text),
            "other":        [],
        }


# ── 2. Retriever ─────────────────────────────────────────────────────────────

class MedicalRetriever:
    """Fetches relevant chunks from ChromaDB using semantic similarity."""

    def __init__(self):
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBED_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._store = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

    def fetch(self, query: str, entities: dict, k: int = TOP_K_FETCH) -> list:
        """
        Augments the raw query with extracted entities for better retrieval.
        Returns list of (document, score) tuples.
        """
        # Build an enriched query string using extracted entities
        enriched_parts = [query]
        if entities.get("symptoms"):
            enriched_parts.append("Symptoms: " + ", ".join(entities["symptoms"]))
        if entities.get("body_parts"):
            enriched_parts.append("Body parts: " + ", ".join(entities["body_parts"]))
        if entities.get("durations"):
            enriched_parts.append("Duration: " + ", ".join(entities["durations"]))

        enriched_query = " | ".join(enriched_parts)

        results = self._store.similarity_search_with_relevance_scores(
            enriched_query, k=k
        )
        return results   # list of (Document, float)


# ── 3. Cross-Encoder Reranker ─────────────────────────────────────────────────

class Reranker:
    """
    Re-ranks retrieved documents using a cross-encoder for higher precision.

    Unlike bi-encoders (which encode query and doc separately),
    a cross-encoder sees query + doc together → more accurate relevance score.
    """

    def __init__(self):
        try:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(RERANK_MODEL)
            print(f"   Reranker: loaded {RERANK_MODEL}")
        except ImportError:
            self._model = None
            print("   Reranker: sentence-transformers not installed — skipping rerank")

    def rerank(self, query: str, docs: list, top_k: int = TOP_K_RERANK) -> list:
        """
        docs: list of (Document, float) from ChromaDB
        Returns: top_k Document objects, ordered by cross-encoder score
        """
        if self._model is None or not docs:
            # Return top_k without reranking
            return [doc for doc, _ in docs[:top_k]]

        pairs = [(query, doc.page_content) for doc, _ in docs]
        scores = self._model.predict(pairs)

        # Zip docs with new cross-encoder scores, sort descending
        scored = sorted(zip(scores, [doc for doc, _ in docs]), reverse=True)
        return [doc for _, doc in scored[:top_k]]


# ── 4. Triage Classifier ─────────────────────────────────────────────────────

def classify_triage(query: str, entities: dict) -> tuple[str, bool]:
    """
    Rule-based triage classification.
    Returns (level, is_emergency) where level is LOW | MEDIUM | HIGH.
    """
    text_lower = query.lower()

    # Check emergency keywords
    if any(kw in text_lower for kw in EMERGENCY_KEYWORDS):
        return "HIGH", True

    # Check for concerning symptom combinations
    symptoms = [s.lower() for s in entities.get("symptoms", [])]
    high_symptoms = {"chest pain", "stroke", "seizure", "unconscious", "anaphylaxis"}
    medium_symptoms = {"fever", "infection", "vomiting", "diarrhea", "pain", "swelling"}

    if any(s in high_symptoms for s in symptoms):
        return "HIGH", False

    if any(s in medium_symptoms for s in symptoms):
        return "MEDIUM", False

    # Duration as a severity signal
    durations = entities.get("durations", [])
    if durations:
        # e.g., "3 days" → likely needs attention
        for dur in durations:
            match = re.search(r"(\d+)\s+(day|week|month)", dur, re.IGNORECASE)
            if match:
                num = int(match.group(1))
                unit = match.group(2).lower()
                if unit == "day" and num >= 3:
                    return "MEDIUM", False
                if unit in ("week", "month"):
                    return "MEDIUM", False

    return "LOW", False


# ── 5. LLM Generation ────────────────────────────────────────────────────────

def generate_response(query: str, context_docs: list, triage: str) -> str:
    """
    Calls the fine-tuned MediAssist model via HuggingFace Inference API.
    Falls back to a rule-based response if HF_TOKEN is not set.
    """
    if not HF_TOKEN:
        return _fallback_response(query, context_docs, triage)

    try:
        from huggingface_hub import InferenceClient

        context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)
        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}\n\n"
            f"Use the following medical reference information to inform your response:\n\n"
            f"{context}"
            f"<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{query}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        client = InferenceClient(model=HF_REPO, token=HF_TOKEN)
        result = client.text_generation(
            prompt,
            max_new_tokens=512,
            temperature=0.3,
            repetition_penalty=1.1,
            stop=["<|eot_id|>"],
        )
        return result.strip()

    except Exception as e:
        print(f"   Warning: HF Inference API error ({e}) — using fallback response")
        return _fallback_response(query, context_docs, triage)


def _fallback_response(query: str, context_docs: list, triage: str) -> str:
    """Rule-based fallback when the LLM is not available."""
    if triage == "HIGH":
        return (
            "Based on the symptoms you've described, this may be a medical emergency. "
            "Please call 911 or go to the nearest emergency room immediately. "
            "Do not wait. This response is for informational purposes only and is not "
            "a substitute for professional medical advice."
        )

    context_snippet = context_docs[0].page_content[:500] if context_docs else ""
    return (
        f"Thank you for describing your symptoms. Based on your description, "
        f"here is some relevant information:\n\n{context_snippet}\n\n"
        f"Please consult a healthcare professional for an accurate diagnosis. "
        f"This information is for educational purposes only."
    )


# ── Main Pipeline ─────────────────────────────────────────────────────────────

class MediAssistPipeline:
    """
    End-to-end medical assistant pipeline.

    Example:
        pipeline = MediAssistPipeline()
        response = pipeline.run("I have chest pain and shortness of breath")
        print(response.triage_level)   # HIGH
        print(response.answer)
    """

    def __init__(self):
        print("Loading MediAssist pipeline...")
        self.ner       = MedicalNER()
        self.retriever = MedicalRetriever()
        self.reranker  = Reranker()
        print("Pipeline ready.\n")

    def run(self, patient_text: str) -> MediAssistResponse:
        print(f"Query: {patient_text[:100]}...")

        # Step A: Extract medical entities
        entities = self.ner.extract(patient_text)
        print(f"   NER entities: {entities}")

        # Step B: Retrieve relevant docs
        raw_docs = self.retriever.fetch(patient_text, entities)
        print(f"   Retrieved: {len(raw_docs)} docs")

        # Step C: Rerank
        top_docs = self.reranker.rerank(patient_text, raw_docs)
        print(f"   After rerank: {len(top_docs)} docs")

        # Step D: Classify triage
        triage, is_emergency = classify_triage(patient_text, entities)
        print(f"   Triage: {triage} | Emergency: {is_emergency}")

        # Step E: Generate answer
        answer = generate_response(patient_text, top_docs, triage)

        return MediAssistResponse(
            answer=answer,
            triage_level=triage,
            entities=entities,
            sources=[doc.page_content[:300] for doc in top_docs],
            is_emergency=is_emergency,
        )


# ── Quick demo when run directly ─────────────────────────────────────────────
if __name__ == "__main__":
    pipeline = MediAssistPipeline()

    test_cases = [
        "I have had a headache for 3 days along with a high fever and stiff neck.",
        "My throat is a bit sore and I have a runny nose. Started yesterday.",
        "I'm having severe chest pain and I can't breathe properly. It started 10 minutes ago.",
    ]

    for query in test_cases:
        print("\n" + "=" * 70)
        response = pipeline.run(query)
        print(f"\nTriage Level : {response.triage_level}")
        print(f"Is Emergency : {response.is_emergency}")
        print(f"\nAnswer:\n{response.answer}")
        print(f"\nSources used :")
        for i, src in enumerate(response.sources, 1):
            print(f"  [{i}] {src[:120]}...")
