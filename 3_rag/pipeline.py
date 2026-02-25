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

SYSTEM_PROMPT = """\
You are MediAssist, an empathetic and knowledgeable AI medical assistant.

Your personality and response style:
- Warm, human, and conversational — never robotic or clinical
- Acknowledge the patient's situation before giving information ("That sounds uncomfortable", "I understand that must be worrying")
- Use natural flowing paragraphs for most responses, not bullet lists
- Only use bullet points when listing multiple distinct symptoms or steps where a list genuinely helps clarity
- Use **bold** to highlight key medical terms or urgent warnings
- Keep responses concise — 3 to 5 sentences for simple cases, slightly more for complex ones
- Always ask ONE natural follow-up question at the end to better understand the situation
- Never copy raw text from documents verbatim — synthesize and explain in your own words
- Match the tone to urgency: calm and practical for LOW, attentive for MEDIUM, direct and urgent for HIGH

Rules you must follow:
- Always end with a one-line reminder that this is informational only and not a substitute for professional medical advice
- For HIGH urgency or emergencies: lead with the urgent action (call 911, go to ER) immediately, do not bury it
- Never diagnose definitively — say "this could suggest", "this may indicate", "it's worth checking for"
- If the patient mentions duration (e.g. 2 weeks), treat it as a significant factor — longer duration = higher concern
"""


@dataclass
class MediAssistResponse:
    answer: str
    triage_level: str                    # LOW | MEDIUM | HIGH
    entities: dict = field(default_factory=dict)   # NER results
    sources: list  = field(default_factory=list)   # retrieved doc snippets
    is_emergency: bool = False
    differentials: list = field(default_factory=list)  # differential diagnoses


# ── Differential Diagnosis Engine ────────────────────────────────────────────

# Rule-based differential diagnosis map.
# Each rule has trigger keywords and a list of candidate conditions.
_DIFFERENTIAL_RULES = [
    {
        "triggers": ["chest pain", "chest tightness", "chest pressure", "heart", "myocardial"],
        "conditions": [
            {"name": "Myocardial Infarction (Heart Attack)", "note": "Emergency — call 911 immediately"},
            {"name": "Unstable Angina", "note": "Urgent cardiac evaluation required"},
            {"name": "Pericarditis", "note": "Pain often worsens lying flat, eases leaning forward"},
        ],
    },
    {
        "triggers": ["shortness of breath", "difficulty breathing", "can't breathe", "breathless"],
        "conditions": [
            {"name": "Pulmonary Embolism", "note": "Emergency if sudden onset with chest pain"},
            {"name": "Asthma Exacerbation", "note": "Wheezing and reversible airflow obstruction"},
            {"name": "Pneumonia", "note": "Often accompanied by fever, productive cough"},
        ],
    },
    {
        "triggers": ["headache", "head pain", "migraine", "head ache"],
        "conditions": [
            {"name": "Tension Headache", "note": "Most common — bilateral, pressure-like"},
            {"name": "Migraine", "note": "Throbbing, unilateral, with nausea or photophobia"},
            {"name": "Cluster Headache", "note": "Severe, unilateral around the eye, cyclical"},
        ],
    },
    {
        "triggers": ["stiff neck", "neck stiffness", "meningitis", "photophobia", "light sensitivity"],
        "conditions": [
            {"name": "Bacterial Meningitis", "note": "Emergency — requires immediate hospitalisation"},
            {"name": "Viral Meningitis", "note": "Less severe; lumbar puncture needed to confirm"},
            {"name": "Subarachnoid Haemorrhage", "note": "Worst headache of life — emergency imaging required"},
        ],
    },
    {
        "triggers": ["fever", "temperature", "high temp", "chills"],
        "conditions": [
            {"name": "Viral Upper Respiratory Infection", "note": "Most common cause of fever with cough/cold symptoms"},
            {"name": "Bacterial Infection (Sepsis risk)", "note": "High fever with rigors warrants blood cultures"},
            {"name": "Influenza", "note": "Sudden onset with myalgia and high fever"},
        ],
    },
    {
        "triggers": ["cough", "sore throat", "runny nose", "congestion", "cold"],
        "conditions": [
            {"name": "Viral Upper Respiratory Infection", "note": "Most likely — rest, fluids, supportive care"},
            {"name": "Streptococcal Pharyngitis", "note": "Throat culture needed; responds to antibiotics"},
            {"name": "Influenza", "note": "More severe — consider antiviral therapy if within 48 h"},
        ],
    },
    {
        "triggers": ["nausea", "vomiting", "diarrhoea", "diarrhea", "stomach", "abdominal pain", "stomach pain"],
        "conditions": [
            {"name": "Viral Gastroenteritis", "note": "Most common — self-limiting, maintain hydration"},
            {"name": "Food Poisoning", "note": "Rapid onset after eating; usually resolves in 24–48 h"},
            {"name": "Appendicitis", "note": "Right lower quadrant pain with vomiting warrants evaluation"},
        ],
    },
    {
        "triggers": ["rash", "hives", "itching", "skin", "allergy"],
        "conditions": [
            {"name": "Allergic Contact Dermatitis", "note": "Localised reaction to a contact allergen"},
            {"name": "Urticaria (Hives)", "note": "Systemic allergic reaction — antihistamines usually effective"},
            {"name": "Eczema (Atopic Dermatitis)", "note": "Chronic, recurrent; managed with emollients and steroids"},
        ],
    },
    {
        "triggers": ["back pain", "lower back", "spine", "lumbar"],
        "conditions": [
            {"name": "Musculoskeletal Strain", "note": "Most common — rest, NSAIDs, physiotherapy"},
            {"name": "Lumbar Disc Herniation", "note": "Pain radiating to leg (sciatica) suggests nerve involvement"},
            {"name": "Kidney Stone", "note": "Colicky flank pain radiating to groin warrants imaging"},
        ],
    },
    {
        "triggers": ["dizziness", "dizzy", "lightheaded", "vertigo", "spinning"],
        "conditions": [
            {"name": "Benign Paroxysmal Positional Vertigo (BPPV)", "note": "Most common — positional; treated with Epley manoeuvre"},
            {"name": "Vestibular Neuritis", "note": "Prolonged vertigo following viral illness"},
            {"name": "Orthostatic Hypotension", "note": "Dizziness on standing — check BP lying and standing"},
        ],
    },
    {
        "triggers": ["fatigue", "tired", "exhaustion", "weak", "weakness"],
        "conditions": [
            {"name": "Anaemia", "note": "Blood test (FBC) recommended to check haemoglobin"},
            {"name": "Hypothyroidism", "note": "TSH blood test; fatigue with weight gain and cold intolerance"},
            {"name": "Chronic Fatigue Syndrome", "note": "Diagnosis of exclusion after ruling out organic causes"},
        ],
    },
    {
        "triggers": ["anxiety", "panic", "heart racing", "palpitations", "palpitation"],
        "conditions": [
            {"name": "Panic Disorder", "note": "Recurrent panic attacks with intense fear — CBT effective"},
            {"name": "Cardiac Arrhythmia", "note": "ECG needed to rule out SVT or atrial fibrillation"},
            {"name": "Hyperthyroidism", "note": "TSH blood test; palpitations with weight loss and heat intolerance"},
        ],
    },
]


def get_differentials(patient_text: str, entities: dict, triage: str) -> list:
    """
    Rule-based differential diagnosis.

    Scores each rule by how many of its trigger keywords appear in the
    patient's text or extracted symptoms, then returns conditions from
    the top-scoring rules (up to 3 conditions total).

    Returns a list of dicts: [{"name": str, "note": str}, ...]
    """
    combined_text = patient_text.lower()
    # Also include extracted symptom strings
    for sym in entities.get("symptoms", []):
        combined_text += " " + sym.lower()

    scored: list[tuple[int, list]] = []
    for rule in _DIFFERENTIAL_RULES:
        score = sum(1 for t in rule["triggers"] if t in combined_text)
        if score > 0:
            scored.append((score, rule["conditions"]))

    if not scored:
        # Generic fallback based on triage level
        if triage == "HIGH":
            return [
                {"name": "Acute Condition — Emergency Evaluation Needed", "note": "Seek immediate care; differential requires in-person assessment"},
            ]
        return []

    # Sort by score descending; collect up to 3 conditions from top rules
    scored.sort(key=lambda x: x[0], reverse=True)
    result = []
    for _, conditions in scored:
        for c in conditions:
            if c not in result:
                result.append(c)
            if len(result) >= 3:
                return result
    return result


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

        # scispaCy en_core_sci_sm uses "ENTITY" for all medical concepts.
        # Larger models use specific labels — we handle both cases.
        for ent in doc.ents:
            label = ent.label_.upper()
            text_val = ent.text.strip()

            if label in ("DISEASE", "SYMPTOM", "PROBLEM", "CONDITION", "ENTITY"):
                entities["symptoms"].append(text_val)
            elif label in ("BODY_PART", "TISSUE", "ANATOMY"):
                entities["body_parts"].append(text_val)
            elif label in ("TIME", "DATE", "DURATION"):
                entities["durations"].append(text_val)
            elif label in ("QUANTITY", "CARDINAL", "PERCENT"):
                entities["measurements"].append(text_val)
            else:
                entities["other"].append(f"{text_val} ({label})")

        # Always run regex on top to catch durations & measurements
        # that scispaCy's small model may miss
        regex_entities = self._regex_fallback(text)
        for key in ("durations", "measurements"):
            entities[key].extend(regex_entities[key])

        # Deduplicate all buckets
        for key in entities:
            entities[key] = list(dict.fromkeys(v for v in entities[key] if v))

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
        # Use index as tiebreaker so Document objects are never compared directly
        scored = sorted(
            zip(scores, range(len(docs)), [doc for doc, _ in docs]),
            reverse=True,
        )
        return [doc for _, _, doc in scored[:top_k]]


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

def generate_response(
    query: str,
    context_docs: list,
    triage: str,
    mlx_model=None,
    mlx_tokenizer=None,
    mlx_generate=None,
    mlx_sampler=None,
    history: list | None = None,
    memories: list | None = None,
) -> str:
    """
    Runs inference using the preloaded fine-tuned MediAssist MLX model.

    Args:
        query:        Current patient message.
        context_docs: Top RAG documents (reranked).
        triage:       LOW | MEDIUM | HIGH.
        history:      List of {"role": "user"|"assistant", "content": str}
                      representing the conversation so far (short-term memory).
        memories:     List of retrieved long-term patient facts
                      e.g. ["Patient allergy: penicillin", "Patient age: 34"].
    """
    if mlx_model is None or mlx_generate is None:
        return _fallback_response(query, context_docs, triage)

    try:
        context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

        # Build system prompt — inject long-term memories if present
        system = (
            f"{SYSTEM_PROMPT}\n\n"
            f"Urgency level assessed: {triage}\n\n"
            f"Reference information to draw from (do NOT copy verbatim):\n{context}"
        )
        if memories:
            mem_block = "\n".join(f"  - {m}" for m in memories)
            system += (
                f"\n\nKnown facts about this patient (from past conversations):\n"
                f"{mem_block}\n"
                f"Use this background information to personalise your response where relevant."
            )

        # Build Llama 3.2 multi-turn chat template
        # Format: <system> then alternating <user>/<assistant> turns, ending with <user>
        prompt = (
            f"<|begin_of_text|>"
            f"<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}<|eot_id|>"
        )

        # Inject conversation history (short-term memory) — last N turns
        if history:
            for turn in history:
                role    = turn.get("role", "user")
                content = turn.get("content", "").strip()
                if content:
                    prompt += (
                        f"<|start_header_id|>{role}<|end_header_id|>\n\n"
                        f"{content}<|eot_id|>"
                    )

        # Current query
        prompt += (
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{query}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

        response = mlx_generate(
            mlx_model, mlx_tokenizer,
            prompt=prompt,
            max_tokens=400,
            verbose=False,
            sampler=mlx_sampler,
        )
        # Strip prompt echo if present
        if "<|start_header_id|>assistant<|end_header_id|>" in response:
            response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
        return response.split("<|eot_id|>")[0].strip()

    except Exception as e:
        print(f"   Warning: MLX generation failed ({e}) — using fallback")
        return _fallback_response(query, context_docs, triage)


def _unused_hf_api_generate(query: str, context_docs: list, triage: str) -> str:
    """
    Legacy HuggingFace Inference API approach — kept for reference.
    HF discontinued free serverless inference for custom models in 2025.
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


# ── Conversational Message Detector ──────────────────────────────────────────

_CONVERSATIONAL_PATTERNS = [
    (r"^(hi|hello|hey|good\s*(morning|evening|afternoon|night))[!.,\s]*$",
     "Hello! I'm MediAssist, your AI medical assistant. How are you feeling today? Please describe your symptoms and I'll do my best to help."),
    (r"^(thank(s| you)|ty|thx|thank you so much)[!.,\s]*$",
     "You're welcome! If you have any more symptoms or questions, feel free to ask. Please remember to consult a doctor for a proper diagnosis. Take care!"),
    (r"^(ok|okay|got it|i see|alright|sure|great|sounds good)[!.,\s]*$",
     "I'm here whenever you need help. Feel free to describe any symptoms you're experiencing and I'll assist you."),
    (r"^(bye|goodbye|see you|take care)[!.,\s]*$",
     "Take care! Remember, if your symptoms worsen or you feel uncertain, please consult a healthcare professional. Goodbye!"),
    (r"^(how are you|how r u)[?.,\s]*$",
     "I'm here and ready to help! I'm MediAssist — tell me how YOU are feeling. What symptoms are you experiencing?"),
]

def _detect_conversational(text: str):
    """
    Returns a friendly response string if the message is conversational
    (not a medical query), otherwise returns None.
    """
    import re
    cleaned = text.strip().lower()
    # Very short messages with no medical content are likely conversational
    for pattern, response in _CONVERSATIONAL_PATTERNS:
        if re.match(pattern, cleaned, re.IGNORECASE):
            return response
    # If the message is very short (< 8 chars) and has no medical keywords, treat as conversational
    medical_hints = ["pain", "ache", "fever", "cough", "bleed", "swel", "hurt",
                     "sick", "nause", "dizz", "breath", "chest", "head", "throat",
                     "stomach", "rash", "tired", "weak", "vomit", "infect"]
    if len(cleaned) < 15 and not any(h in cleaned for h in medical_hints):
        return "I'm here to help with any health concerns. Please describe your symptoms and I'll assess them for you."
    return None


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

        # Load MLX model ONCE at startup — reused for every query
        print("   Loading fine-tuned Llama model (MLX)...")
        try:
            from mlx_lm import load, generate as mlx_generate
            from mlx_lm.sample_utils import make_sampler
            self._mlx_generate = mlx_generate
            self._mlx_sampler   = make_sampler(temp=0.4, top_p=0.9)
            self._mlx_model, self._mlx_tokenizer = load(
                "mlx-community/Llama-3.2-3B-Instruct",
                adapter_path=str(PROJECT_ROOT / "2_finetune" / "adapters"),
            )
            print("   Model loaded and ready.")
        except Exception as e:
            print(f"   Warning: Could not load MLX model ({e}) — will use fallback")
            self._mlx_model     = None
            self._mlx_tokenizer = None
            self._mlx_generate  = None
            self._mlx_sampler   = None

        print("Pipeline ready.\n")

    def generate_soap_note(
        self,
        messages: list,
        last_entities: dict,
        last_triage: str,
    ) -> str:
        """
        Generate a structured SOAP clinical note from the conversation.

        Args:
            messages:      List of {"role", "content"} dicts (full chat history).
            last_entities: NER entities from the most recent query.
            last_triage:   Triage level from the most recent query.

        Returns:
            A formatted SOAP note string, or a fallback rule-based version.
        """
        # Build conversation transcript
        transcript = ""
        for m in messages:
            role_label = "Patient" if m["role"] == "user" else "MediAssist"
            transcript += f"{role_label}: {m['content']}\n\n"

        entity_summary = []
        if last_entities.get("symptoms"):
            entity_summary.append("Symptoms: " + ", ".join(last_entities["symptoms"]))
        if last_entities.get("durations"):
            entity_summary.append("Duration: " + ", ".join(last_entities["durations"]))
        if last_entities.get("measurements"):
            entity_summary.append("Measurements: " + ", ".join(last_entities["measurements"]))
        entity_str = "\n".join(entity_summary) if entity_summary else "None extracted"

        soap_system = (
            "You are a clinical documentation assistant. "
            "Generate a concise, structured SOAP note from the given patient conversation. "
            "Format EXACTLY as:\n\n"
            "SUBJECTIVE:\n[Chief complaint and patient's description of symptoms in their own words]\n\n"
            "OBJECTIVE:\n[Measurable/observable findings — extracted symptoms, duration, measurements, triage level]\n\n"
            "ASSESSMENT:\n[Clinical impression — most likely diagnoses to consider given the symptoms]\n\n"
            "PLAN:\n[Recommended next steps — follow-up, referral, red flags to watch]\n\n"
            "Keep each section to 2-3 concise sentences. Do not add extra headers or preamble."
        )

        soap_user = (
            f"Patient Conversation:\n{transcript.strip()}\n\n"
            f"Extracted Entities:\n{entity_str}\n\n"
            f"Triage Level: {last_triage}\n\n"
            "Generate the SOAP note now."
        )

        if self._mlx_model is None or self._mlx_generate is None:
            # Fallback: rule-based SOAP note
            return self._soap_fallback(messages, last_entities, last_triage)

        try:
            prompt = (
                f"<|begin_of_text|>"
                f"<|start_header_id|>system<|end_header_id|>\n\n"
                f"{soap_system}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n"
                f"{soap_user}<|eot_id|>"
                f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            )
            result = self._mlx_generate(
                self._mlx_model,
                self._mlx_tokenizer,
                prompt=prompt,
                max_tokens=500,
                verbose=False,
                sampler=self._mlx_sampler,
            )
            if "<|start_header_id|>assistant<|end_header_id|>" in result:
                result = result.split("<|start_header_id|>assistant<|end_header_id|>")[-1]
            note = result.split("<|eot_id|>")[0].strip()
            if note and len(note) > 50:
                return note
        except Exception as e:
            print(f"   SOAP generation failed ({e}) — using fallback")

        return self._soap_fallback(messages, last_entities, last_triage)

    @staticmethod
    def _soap_fallback(messages: list, entities: dict, triage: str) -> str:
        """Rule-based SOAP note when the LLM is unavailable."""
        symptoms   = ", ".join(entities.get("symptoms",     [])) or "Not specified"
        durations  = ", ".join(entities.get("durations",    [])) or "Not specified"
        measures   = ", ".join(entities.get("measurements", [])) or "Not specified"
        body_parts = ", ".join(entities.get("body_parts",   [])) or "Not specified"

        # Extract first patient message as chief complaint
        chief = next(
            (m["content"] for m in messages if m.get("role") == "user"),
            "Symptom description not recorded",
        )

        triage_plan = {
            "HIGH":   "Immediate emergency evaluation. Call 911 or proceed to ER without delay. Do not drive yourself.",
            "MEDIUM": "Schedule appointment with primary care physician within 24-48 hours. Monitor for worsening symptoms.",
            "LOW":    "Rest and supportive self-care. Follow up with GP if symptoms persist beyond 5-7 days.",
        }.get(triage, "Follow up with a healthcare provider as appropriate.")

        return (
            f"SUBJECTIVE:\n"
            f"Patient presents with: {chief[:300]}.\n\n"
            f"OBJECTIVE:\n"
            f"Symptoms: {symptoms}\n"
            f"Duration: {durations}\n"
            f"Body parts involved: {body_parts}\n"
            f"Measurements noted: {measures}\n"
            f"Triage level assessed: {triage}\n\n"
            f"ASSESSMENT:\n"
            f"Based on symptom presentation, {triage.lower()}-urgency evaluation is indicated. "
            f"Differential diagnosis should be guided by a qualified healthcare professional.\n\n"
            f"PLAN:\n"
            f"{triage_plan}"
        )

    def run(
        self,
        patient_text: str,
        history: list | None = None,
        memories: list | None = None,
    ) -> MediAssistResponse:
        """
        Run the full pipeline.

        Args:
            patient_text: Current patient message.
            history:      List of {"role", "content"} dicts — recent chat turns
                          (short-term context window memory).
            memories:     List of retrieved long-term patient fact strings
                          e.g. ["Patient allergy: penicillin"].
        """
        print(f"Query: {patient_text[:100]}...")
        if history:
            print(f"   History turns   : {len(history)}")
        if memories:
            print(f"   Long-term facts : {len(memories)}")

        # Short-circuit for non-medical conversational messages
        conversational = _detect_conversational(patient_text)
        if conversational:
            return MediAssistResponse(
                answer=conversational,
                triage_level="LOW",
                entities={},
                sources=[],
                is_emergency=False,
            )

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

        # Step E: Generate answer with history + long-term memories
        answer = generate_response(
            patient_text, top_docs, triage,
            mlx_model=self._mlx_model,
            mlx_tokenizer=self._mlx_tokenizer,
            mlx_generate=self._mlx_generate,
            mlx_sampler=self._mlx_sampler,
            history=history or [],
            memories=memories or [],
        )

        # Step F: Differential diagnosis (rule-based, no extra LLM call)
        differentials = get_differentials(patient_text, entities, triage)
        print(f"   Differentials: {[d['name'] for d in differentials]}")

        return MediAssistResponse(
            answer=answer,
            triage_level=triage,
            entities=entities,
            sources=[doc.page_content[:300] for doc in top_docs],
            is_emergency=is_emergency,
            differentials=differentials,
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
