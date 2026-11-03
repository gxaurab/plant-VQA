"""
quantize_awq_w4a16.py  -- FINAL

W4A16 AWQ quantization of merged Qwen3-VL-2B.
  - 4-bit grouped weights, 16-bit activations, TEXT DECODER only.
  - Vision encoder + projector + lm_head stay BF16.
  - Explicit Qwen3 AWQ mappings (Qwen3VL is NOT in the auto-registry, but its
    language backbone IS Qwen3ForCausalLM, so the Qwen3 decoder mappings apply).
  - Calibration on UNIQUE plant images from the train split.

Run in .venvWeedAI (llmcompressor 0.8.1 + AWQ + replace_modules_for_calibration).
"""

import argparse
import os
import random

import torch
import pandas as pd
from datasets import Dataset
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.awq import AWQModifier, AWQMapping

PIXELS_PER_TOKEN = 32 * 32
SYSTEM_PROMPT = (
    "You are a plant pathology expert. Answer the question about the plant "
    "image accurately and concisely."
)

# Text decoder only; vision + head stay BF16.
IGNORE = [
    "lm_head",
    "re:.*model\\.visual\\..*",
]

# Explicit AWQ mappings for the Qwen3 decoder layers (the language backbone of
# Qwen3-VL). smooth_layer feeds balance_layers; AWQ scales balance_layers using
# activations seen at smooth_layer. These names match model.language_model.layers.*
QWEN3_AWQ_MAPPINGS = [
    AWQMapping(
        "re:.*input_layernorm$",
        ["re:.*q_proj$", "re:.*k_proj$", "re:.*v_proj$"],
    ),
    AWQMapping("re:.*v_proj$", ["re:.*o_proj$"]),
    AWQMapping(
        "re:.*post_attention_layernorm$",
        ["re:.*gate_proj$", "re:.*up_proj$"],
    ),
    AWQMapping("re:.*up_proj$", ["re:.*down_proj$"]),
]


def build_calibration_dataset(csv, image_root, processor, n, seed=42):
    from qwen_vl_utils import process_vision_info

    df = pd.read_csv(csv, low_memory=False)
    df = df.dropna(subset=["question_text"])
    df["abs_path"] = df["image_path"].apply(
        lambda p: os.path.join(image_root, os.path.basename(str(p))))
    df = df[df["abs_path"].apply(os.path.exists)].reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(f"No calibration images found under {image_root}")

    # unique images first -> N samples = N distinct images
    per_image = df.drop_duplicates(subset="image_id").reset_index(drop=True)
    random.seed(seed)
    k = min(n, len(per_image))
    idx = random.sample(range(len(per_image)), k)
    print(f"[calib] {k} unique images (of {len(per_image)}); "
          f"{df['crop'].nunique()} crops present")

    records = []
    for i in idx:
        r = per_image.iloc[i]
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": r["abs_path"]},
                {"type": "text", "text": str(r["question_text"])},
            ]},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(messages, image_patch_size=16)
        enc = processor(text=[text], images=imgs, return_tensors="pt")
        records.append({kk: v[0] for kk, v in enc.items()})

    has_px = sum(1 for r in records if "pixel_values" in r)
    print(f"[calib] pixel_values: {has_px}/{len(records)}")
    assert has_px == len(records) > 0, "calibration records missing images!"
    return Dataset.from_list(records)


def data_collator(batch):
    assert len(batch) == 1
    return {k: (v if torch.is_tensor(v) else torch.tensor(v))
            for k, v in batch[0].items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calib", type=int, default=256)
    ap.add_argument("--group_size", type=int, default=128)
    ap.add_argument("--min_pixels_factor", type=int, default=128)
    ap.add_argument("--max_pixels_factor", type=int, default=256)
    ap.add_argument("--max_seq_length", type=int, default=1024)
    args = ap.parse_args()

    import llmcompressor
    print(f"llmcompressor {llmcompressor.__version__}")

    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=args.min_pixels_factor * PIXELS_PER_TOKEN,
        max_pixels=args.max_pixels_factor * PIXELS_PER_TOKEN,
    )

    print("Loading merged model ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=None, local_files_only=True)

    # NOTE: replace_modules_for_calibration is for MoE models (e.g. 30B-A3B).
    # Qwen3-VL-2B is DENSE - no expert blocks to replace - so it is NOT called.

    calib = build_calibration_dataset(
        args.train_csv, args.images, processor, args.calib)

    # AWQ W4A16 with explicit Qwen3 decoder mappings.
    recipe = AWQModifier(
        mappings=QWEN3_AWQ_MAPPINGS,
        ignore=IGNORE,
        config_groups={
            "group_0": {
                "targets": ["Linear"],
                "input_activations": None,     # A16
                "output_activations": None,
                "weights": {
                    "num_bits": 4,
                    "type": "int",
                    "symmetric": False,        # asymmetric: better 4-bit accuracy
                    "strategy": "group",
                    "group_size": args.group_size,
                    "observer": "mse",
                },
            },
        },
    )

    print(f"Running AWQ W4A16 (group_size={args.group_size}) ...")
    oneshot(
        model=model,
        dataset=calib,
        recipe=recipe,
        data_collator=data_collator,
        max_seq_length=args.max_seq_length,
        num_calibration_samples=len(calib),
        sequential_targets=["Qwen3VLTextDecoderLayer"],
    )

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out, save_compressed=True)
    processor.save_pretrained(args.out)
    print(f"\nAWQ W4A16 saved to {args.out}")


if __name__ == "__main__":
    main()