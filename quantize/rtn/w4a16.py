"""
quantize_w4a16_rtn.py

W4A16 weight-only INT4 quantization of merged Qwen3-VL-2B via RTN.

DIFFERENCE FROM W8A16 (only two real changes):
  1. scheme = "W4A16"   (4-bit weights instead of 8-bit)
  2. group_size = 128    <-- REQUIRED at 4-bit. Per-channel scales (the W8A16
     default) are too coarse for 4-bit and cost accuracy; group quantization
     uses one scale per 128 weights, the standard GPTQ/AWQ/int4 setting.

CAVEAT (be honest in the writeup): at 8-bit, RTN ~= GPTQ ~= AWQ (near-lossless).
That equivalence BREAKS at 4-bit -- calibration-based GPTQ/AWQ typically beat
data-free RTN, especially on a small 2B model. So expect a larger gap vs bf16
than W8A16 showed. This is still a valid, reproducible comparison point.

Weight-only RTN still needs NO calibration data, NO forward pass, NO images:
    - no vision forward pass  -> no grid_thw / rot_pos_emb CUDA error
    - no DataLoader, no collator, no calibration records
    - the 'datafree' pipeline is used (llmcompressor's own recommendation)

Module split (unchanged):
    quantize : model.language_model.layers.*   (196 Linear)
    ignore   : model.visual.*  + lm_head        (stay BF16)

Save: compressed-tensors pack-quantized so vLLM loads it as INT4.
"""

import argparse
import os
import torch

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

IGNORE = [
    "lm_head",
    r"re:.*model\.visual\..*",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group_size", type=int, default=128,
                    help="4-bit group size (128 is the GPTQ/AWQ/int4 standard)")
    args = ap.parse_args()

    print("Loading merged model (single device, no auto-shard) ...")
    # device_map={"":0} pins everything on GPU 0. Even though weight-only RTN
    # does no forward pass, single-device avoids any placement edge cases.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0})
    print("Model loaded.")

    # W4A16 weight-only: 4-bit grouped weights, activations untouched (16-bit).
    # group_size=128 is the key addition vs the W8A16 recipe.
    # No dataset/calibration -> oneshot runs the datafree pipeline.
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W4A16",
        group_size=args.group_size,
        ignore=IGNORE,
    )

    print(f"Applying W4A16 weight-only RTN (group_size={args.group_size}, "
          f"datafree, no calibration) ...")
    oneshot(model=model, recipe=recipe)      # NO dataset -> datafree path
    print("Quantization applied.")

    # ---- save as compressed-tensors so it's actually packed on disk ----
    fmt = _resolve_pack_format()
    print(f"Saving with quantization_format={fmt!r} ...")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(
        args.out,
        save_compressed=True,
        quantization_format=fmt,
    )

    processor = AutoProcessor.from_pretrained(args.model)
    processor.save_pretrained(args.out)

    print(f"\nSaved W4A16 model to {args.out}")
    print("Verify: config.json should contain a quantization_config block with")
    print("num_bits=4 and group_size=128, and model.safetensors should be")
    print("~1/4 the bf16 size of the quantized (language) layers.")


def _resolve_pack_format():
    """
    Return whatever pack-quantized format this compressed_tensors version wants.
    Tries the enum first, falls back to the string form.
    """
    try:
        import compressed_tensors, importlib, pkgutil
        for m in pkgutil.walk_packages(compressed_tensors.__path__,
                                       "compressed_tensors."):
            try:
                mod = importlib.import_module(m.name)
            except Exception:
                continue
            QF = getattr(mod, "QuantizationFormat", None)
            if QF is not None:
                for name in ("pack_quantized", "PACK_QUANTIZED", "pack-quantized"):
                    if hasattr(QF, name):
                        return getattr(QF, name)
                members = [x for x in dir(QF) if not x.startswith("_")]
                print(f"[warn] QuantizationFormat members: {members}")
                for x in members:
                    if "pack" in x.lower():
                        return getattr(QF, x)
    except Exception as e:
        print(f"[warn] could not resolve QuantizationFormat enum: {e}")
    return "pack-quantized"


if __name__ == "__main__":
    main()