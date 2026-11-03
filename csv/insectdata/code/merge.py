"""
merge_only.py -- fold the LoRA adapter into base weights and save. No eval.

Merge does NO forward pass, so it cannot hit the vision-encoder / rot_pos_emb
CUDA error seen during evaluation. Produces a standalone (no-PEFT) checkpoint
ready for quantization and Jetson deployment.

    python merge_only.py \
        --model_id /scratch/.../qwen3VL2BInstruct \
        --adapter  /scratch/.../InsectAI/outputs/best_adapter \
        --out      /scratch/.../InsectAI/outputs/merged
"""

import argparse
import os

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True, help="local base model dir (offline)")
    ap.add_argument("--adapter", required=True, help="best_adapter dir")
    ap.add_argument("--out", required=True, help="output dir for merged model")
    ap.add_argument("--safetensors", action="store_true",
                    help="save as .safetensors (try this first; if it raises a "
                         "shared-tensor error, omit this flag to save as .bin)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading base model ...")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map=None,
        local_files_only=True,
    ).to(device)
    print("Base loaded.")

    print("Attaching adapter ...")
    model = PeftModel.from_pretrained(base, args.adapter, local_files_only=True)

    print("Merging LoRA into base weights (merge_and_unload) ...")
    merged = model.merge_and_unload()
    print("Merge complete.")

    os.makedirs(args.out, exist_ok=True)
    safe = bool(args.safetensors)
    print(f"Saving merged model to {args.out} (safe_serialization={safe}) ...")
    try:
        merged.save_pretrained(args.out, safe_serialization=safe)
    except RuntimeError as e:
        if "shared" in str(e).lower() and safe:
            print(f"[warn] safetensors rejected shared/tied tensors ({e});")
            print("[warn] retrying with safe_serialization=False (.bin) ...")
            merged.save_pretrained(args.out, safe_serialization=False)
        else:
            raise

    # save the processor alongside (from adapter dir if present, else base)
    proc_src = args.adapter
    if not os.path.exists(os.path.join(args.adapter, "preprocessor_config.json")):
        proc_src = args.model_id
    AutoProcessor.from_pretrained(proc_src, local_files_only=True).save_pretrained(args.out)

    print(f"\nMerged model ready at {args.out}")
    print("Contains: config.json + weight shards + processor/tokenizer files.")


if __name__ == "__main__":
    main()