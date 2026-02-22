"""
Step 3 — Fuse LoRA Adapter & Push to HuggingFace Hub
======================================================
After fine-tuning completes, this script:
  1. Fuses the LoRA adapter weights into the base model
  2. Converts the fused model to safetensors format
  3. Pushes the final model to your HuggingFace Hub repo

Prerequisites:
  - Fine-tuning completed (2_finetune/adapters/ exists)
  - HF_TOKEN set in .env (needs write access)
  - HF_MODEL_REPO set in .env  (e.g. "Kisht2t/mediassist-llama-3.2-3b")

Run:
  python 2_finetune/fuse_and_push.py
"""

import os
import subprocess
import sys
from pathlib import Path
from dotenv import load_dotenv

# ── Paths & Config ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
ADAPTER_DIR  = PROJECT_ROOT / "2_finetune" / "adapters"
FUSED_DIR    = PROJECT_ROOT / "2_finetune" / "fused_model"
BASE_MODEL   = "mlx-community/Llama-3.2-3B-Instruct"

load_dotenv(PROJECT_ROOT / ".env")
HF_TOKEN     = os.getenv("HF_TOKEN")
HF_REPO      = os.getenv("HF_MODEL_REPO", "Kisht2t/mediassist-llama-3.2-3b")
# ──────────────────────────────────────────────────────────────────────────────


def check_prerequisites():
    if not ADAPTER_DIR.exists() or not any(ADAPTER_DIR.iterdir()):
        print("❌  Adapter not found. Run training first:")
        print("     python 2_finetune/train.py")
        sys.exit(1)

    if not HF_TOKEN:
        print("❌  HF_TOKEN not set. Add it to your .env file.")
        sys.exit(1)

    print("✅  Prerequisites OK")
    print(f"   Adapter : {ADAPTER_DIR}")
    print(f"   Target  : {HF_REPO}")
    print()


def fuse_adapter():
    """Merge LoRA weights into base model weights."""
    print("── Step 3a: Fusing adapter into base model ──────────────────────────")
    FUSED_DIR.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", BASE_MODEL,
        "--adapter-path", str(ADAPTER_DIR),
        "--save-path", str(FUSED_DIR),
        "--dequantize",           # convert back to float16 for HF compatibility
    ]

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        print("❌  Fusing failed.")
        sys.exit(result.returncode)

    print(f"✅  Fused model saved to: {FUSED_DIR}")
    print()


def push_to_hub():
    """Upload the fused model to HuggingFace Hub."""
    print("── Step 3b: Pushing to HuggingFace Hub ──────────────────────────────")
    print(f"   Repo: https://huggingface.co/{HF_REPO}")
    print()

    # Use huggingface_hub Python API to upload the folder
    from huggingface_hub import HfApi

    api = HfApi(token=HF_TOKEN)

    # Create repo if it doesn't exist yet
    api.create_repo(
        repo_id=HF_REPO,
        repo_type="model",
        exist_ok=True,
        private=False,
    )

    # Upload all files in the fused model directory
    api.upload_folder(
        folder_path=str(FUSED_DIR),
        repo_id=HF_REPO,
        repo_type="model",
        commit_message="Add MediAssist fine-tuned Llama 3.2 3B",
    )

    print(f"✅  Model pushed to: https://huggingface.co/{HF_REPO}")
    print()


def print_next_steps():
    print("── Next Steps ────────────────────────────────────────────────────────")
    print()
    print("  Your fine-tuned model is now public on HuggingFace!")
    print()
    print("  Test it with the Inference API:")
    print(f"    curl -X POST https://api-inference.huggingface.co/models/{HF_REPO} \\")
    print('      -H "Authorization: Bearer $HF_TOKEN" \\')
    print('      -d \'{"inputs": "I have chest pain. What should I do?"}\' ')
    print()
    print("  Now run Step 4 — build the RAG vector store:")
    print("    python 3_rag/index.py")
    print()


if __name__ == "__main__":
    check_prerequisites()
    fuse_adapter()
    push_to_hub()
    print_next_steps()
