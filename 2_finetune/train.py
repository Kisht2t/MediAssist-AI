"""
Step 2 — Fine-Tuning with MLX-LM LoRA
======================================
Thin wrapper around mlx_lm.lora that:
  1. Validates the training data exists
  2. Kicks off fine-tuning using config.yaml
  3. Prints a summary with next steps when done

Run:
  python 2_finetune/train.py

Or equivalently (runs the same config directly):
  python -m mlx_lm.lora --config 2_finetune/config.yaml
"""

import subprocess
import sys
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "1_data" / "processed"
CONFIG_FILE  = PROJECT_ROOT / "2_finetune" / "config.yaml"
ADAPTER_DIR  = PROJECT_ROOT / "2_finetune" / "adapters"
# ──────────────────────────────────────────────────────────────────────────────


def check_prerequisites():
    """Make sure training data is present before we start."""
    train_path = DATA_DIR / "train.jsonl"
    valid_path = DATA_DIR / "valid.jsonl"

    if not train_path.exists() or not valid_path.exists():
        print("❌  Training data not found!")
        print(f"   Expected: {train_path}")
        print(f"   Expected: {valid_path}")
        print()
        print("   Run this first:")
        print("     python 1_data/prepare_data.py")
        sys.exit(1)

    # Count lines as a sanity check
    with open(train_path) as f:
        n_train = sum(1 for _ in f)
    with open(valid_path) as f:
        n_valid = sum(1 for _ in f)

    print(f"✅  Training data found:")
    print(f"   Train: {n_train:,} examples  →  {train_path}")
    print(f"   Valid: {n_valid:,} examples  →  {valid_path}")
    print()


def run_training():
    """Launch mlx_lm.lora with our config file."""
    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--config", str(CONFIG_FILE),
    ]

    print("── Starting MLX-LM LoRA fine-tuning ──────────────────────────────────")
    print(f"   Config : {CONFIG_FILE}")
    print(f"   Output : {ADAPTER_DIR}")
    print()
    print("   This will take ~2-3 hours on an M2/M3 Mac.")
    print("   Watch the loss — it should drop from ~2.5 down to ~1.0-1.5.")
    print()

    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        print("\n❌  Training failed. Check output above for details.")
        sys.exit(result.returncode)


def print_next_steps():
    """Tell the user what to do after training."""
    fused_dir = PROJECT_ROOT / "2_finetune" / "fused_model"
    print()
    print("✅  Fine-tuning complete!")
    print(f"   LoRA adapter saved to: {ADAPTER_DIR}")
    print()
    print("── Next Steps ────────────────────────────────────────────────────────")
    print()
    print("  1. Test your adapter locally:")
    print("     python -m mlx_lm.generate \\")
    print(f"       --model meta-llama/Llama-3.2-3B-Instruct \\")
    print(f"       --adapter-path {ADAPTER_DIR} \\")
    print('       --prompt "I have had a headache for 3 days with fever. What could this be?"')
    print()
    print("  2. Fuse the adapter into a standalone model (run Step 3):")
    print("     python 2_finetune/fuse_and_push.py")
    print()


if __name__ == "__main__":
    check_prerequisites()
    run_training()
    print_next_steps()
