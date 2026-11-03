"""
merge_llava.py -- merge the LoRA adapter into LLaVA-1.5-7B-HF base and
save a standalone FP16 HF model (input to eval and to quantization).

    python merge_llava.py \
        --base   /scratch/ptolemy/users/ba974/weedAI/llava-1.5-7b-hf \
        --adapter outputs/llava-1.5-7b-plant-12k/best_adapter \
        --out    outputs/llava-1.5-7b-plant-merged

Offline-safe: only reads local paths. Copies the processor so the merged
dir is self-contained (patch_size / select-strategy preserved at save).
"""

import argparse

import torch
from peft import PeftModel
from transformers import AutoProcessor, LlavaForConditionalGeneration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="local base model dir")
    ap.add_argument("--adapter", required=True, help="best_adapter dir")
    ap.add_argument("--out", required=True, help="merged output dir")
    args = ap.parse_args()

    print("Loading base ...")
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="auto",
    )

    print("Applying adapter ...")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging ...")
    model = model.merge_and_unload()

    print(f"Saving merged model -> {args.out}")
    model.save_pretrained(args.out, safe_serialization=True)

    # processor: load from the ADAPTER dir (train.py saved it there with the
    # correct patch_size / vision_feature_select_strategy), fall back to base.
    try:
        proc = AutoProcessor.from_pretrained(
            args.adapter, patch_size=14,
            vision_feature_select_strategy="default",
        )
    except Exception:
        proc = AutoProcessor.from_pretrained(
            args.base, patch_size=14,
            vision_feature_select_strategy="default",
        )
    proc.save_pretrained(args.out)

    print("Done. Merged FP16 model is self-contained at:", args.out)


if __name__ == "__main__":
    main()