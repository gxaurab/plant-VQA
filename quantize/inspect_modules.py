"""
inspect_modules.py

Print the real top-level module names of the merged Qwen3-VL model so the
quantization `ignore` list is based on fact, not guesses.

We must confirm:
  - what the vision encoder is called   (must be IGNORED -> stays bf16)
  - what the multimodal projector/merger is called (must be IGNORED)
  - what the language backbone layers are called (these get quantized)

    python inspect_modules.py --model /scratch/.../merged
"""

import argparse
import torch
from transformers import Qwen3VLForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    args = ap.parse_args()

    print("Loading model (meta/cpu is fine, we only read names) ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu")

    # top-level children: this reveals the vision vs language split
    print("\n=== TOP-LEVEL MODULES ===")
    for name, _ in model.named_children():
        print("  ", name)

    # one level deeper on the main model
    print("\n=== SECOND-LEVEL (model.*) ===")
    if hasattr(model, "model"):
        for name, _ in model.model.named_children():
            print("   model.", name)

    # collect unique 'families' of Linear module paths
    print("\n=== Linear module path prefixes (first 3 path parts) ===")
    seen = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear):
            parts = name.split(".")
            prefix = ".".join(parts[:3])
            seen[prefix] = seen.get(prefix, 0) + 1
    for prefix, count in sorted(seen.items()):
        tag = ""
        low = prefix.lower()
        if any(k in low for k in ["visual", "vision", "vit"]):
            tag = "  <- VISION: ignore (keep bf16)"
        elif any(k in low for k in ["merger", "projector", "connector"]):
            tag = "  <- PROJECTOR: ignore (keep bf16)"
        elif "lm_head" in low:
            tag = "  <- lm_head: ignore"
        print(f"   {prefix:45s} {count:4d} Linear{tag}")

    print("\nUse the tagged names to build the recipe `ignore` list.")
    print("Anything NOT tagged is language backbone -> gets quantized.")


if __name__ == "__main__":
    main()