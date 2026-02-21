"""
Step 1 — Data Preparation
=========================
Downloads the ChatDoctor dataset from HuggingFace and formats it
into the Llama 3.2 Instruct chat format required by MLX-LM for
LoRA fine-tuning.

Output files (in 1_data/processed/):
  train.jsonl   — 90% of data  (~4500 examples)
  valid.jsonl   — 10% of data  (~500 examples)

Run:
  python 1_data/prepare_data.py
"""

import json
import random
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_NAME   = "avaliev/chat_doctor"
NUM_SAMPLES    = 5000       # use 5k out of 100k — enough for LoRA
TRAIN_SPLIT    = 0.90       # 90% train, 10% validation
OUTPUT_DIR     = Path(__file__).parent / "processed"
RANDOM_SEED    = 42

SYSTEM_PROMPT = (
    "You are MediAssist, a helpful and empathetic AI medical assistant. "
    "Your role is to listen to the patient's symptoms, ask clarifying "
    "questions when needed, and provide clear, accurate health information. "
    "Always remind users that your responses are for informational purposes "
    "only and not a substitute for professional medical advice. "
    "For emergencies, always direct users to call 911 or visit the nearest ER."
)
# ─────────────────────────────────────────────────────────────────────────────


def format_as_llama3(patient: str, doctor: str) -> dict:
    """
    Format a patient-doctor pair into Llama 3.2 Instruct chat template.

    MLX-LM expects a single "text" field with the full formatted conversation.

    Format:
        <|begin_of_text|>
        <|start_header_id|>system<|end_header_id|>
        {system_prompt}<|eot_id|>
        <|start_header_id|>user<|end_header_id|>
        {patient_message}<|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        {doctor_response}<|eot_id|>
    """
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{patient.strip()}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{doctor.strip()}<|eot_id|>"
    )
    return {"text": text}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Download dataset ───────────────────────────────────────────────────
    print(f"Downloading '{DATASET_NAME}' from HuggingFace...")
    dataset = load_dataset(DATASET_NAME, split="train")
    print(f"  Total examples available: {len(dataset):,}")

    # ── 2. Sample & shuffle ───────────────────────────────────────────────────
    random.seed(RANDOM_SEED)
    indices  = random.sample(range(len(dataset)), min(NUM_SAMPLES, len(dataset)))
    samples  = dataset.select(indices)
    print(f"  Sampled: {len(samples):,} examples")

    # ── 3. Format into Llama 3.2 chat template ───────────────────────────────
    formatted = []
    skipped   = 0

    for row in tqdm(samples, desc="Formatting"):
        patient = row.get("input", "").strip()
        doctor  = row.get("output", "").strip()

        # Skip empty or very short examples
        if len(patient) < 10 or len(doctor) < 20:
            skipped += 1
            continue

        formatted.append(format_as_llama3(patient, doctor))

    print(f"  Formatted: {len(formatted):,} | Skipped (too short): {skipped}")

    # ── 4. Split into train / validation ─────────────────────────────────────
    random.shuffle(formatted)
    split_idx   = int(len(formatted) * TRAIN_SPLIT)
    train_data  = formatted[:split_idx]
    valid_data  = formatted[split_idx:]

    # ── 5. Save as JSONL ─────────────────────────────────────────────────────
    train_path = OUTPUT_DIR / "train.jsonl"
    valid_path = OUTPUT_DIR / "valid.jsonl"

    for path, data in [(train_path, train_data), (valid_path, valid_data)]:
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record) + "\n")

    print(f"\n✅ Done!")
    print(f"   Train: {len(train_data):,} examples → {train_path}")
    print(f"   Valid: {len(valid_data):,} examples → {valid_path}")

    # ── 6. Preview a sample ──────────────────────────────────────────────────
    print("\n── Sample record (first training example) ──────────────────────")
    print(train_data[0]["text"][:600] + "...")


if __name__ == "__main__":
    main()
