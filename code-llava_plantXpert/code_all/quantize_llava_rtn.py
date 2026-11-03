"""
quantize_llava_rtn.py -- RTN (data-free) weight quantization of the MERGED
LLaVA-1.5-7B model, saved in compressed-tensors pack-quantized format for vLLM
(incl. Jetson aarch64 vLLM). Mirrors the proven Qwen3-VL W4A16 pipeline.

    # W8A16 (per-channel, no group_size needed)
    python quantize_llava_rtn.py --model <merged> --scheme W8A16 --out <dir>

    # W4A16 (group_size=128 REQUIRED)
    python quantize_llava_rtn.py --model <merged> --scheme W4A16 --group_size 128 --out <dir>

RTN = Round-To-Nearest, data-free: no calibration dataset, no forward pass,
no images. This is the reliable path used for the Qwen and Ministral
checkpoints; it avoids the calibration failures (FX tracing, num_samples=0,
token desync) seen on calibration-based methods for multimodal models.

Scheme notes:
  W8A16 -> int8 weights, 16-bit activations. Per-channel scales fine.
           RTN ~= GPTQ ~= AWQ at 8-bit (near-lossless).
  W4A16 -> int4 weights, 16-bit activations. group_size=128 REQUIRED:
           per-channel is too coarse at 4-bit. Expect a larger gap vs FP16
           than W8A16 -- data-free RTN trails calibration-based GPTQ/AWQ at
           4-bit. Still a valid, reproducible comparison point.

LLaVA-specific module split:
  quantize : language_model.* Linear layers
  ignore   : vision_tower.*, multi_modal_projector.*, lm_head  (stay FP16)

Save: compressed-tensors pack-quantized (quantization_format resolved the same
way as the Qwen pipeline) so vLLM -- including the Jetson aarch64 image --
loads it correctly.
"""

import argparse
import os
import shutil

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier


# language-backbone-only: leave vision tower + projector + lm_head at FP16
IGNORE = [
    "lm_head",
    r"re:.*vision_tower.*",
    r"re:.*multi_modal_projector.*",
]


def _resolve_pack_format():
    """
    Return whatever pack-quantized format this compressed_tensors version wants.
    Tries the enum first, falls back to the string form. (Same resolver as the
    Qwen W4A16 pipeline -- keeps the on-disk format identical to what already
    runs on Jetson vLLM.)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="merged FP16 LLaVA dir (input)")
    ap.add_argument("--scheme", default="W4A16", choices=["W8A16", "W4A16"])
    ap.add_argument("--out", required=True, help="output dir for quantized model")
    ap.add_argument("--group_size", type=int, default=0,
                    help="4-bit group size; use 128 for W4A16, ignored for W8A16")
    args = ap.parse_args()

    # W4A16 REQUIRES a group size; default it to 128 if the user forgot.
    group_size = args.group_size
    if args.scheme == "W4A16" and group_size <= 0:
        group_size = 128
        print("W4A16 with no group_size -> defaulting to 128 (int4 standard).")

    print(f"Loading merged model (single device, no auto-shard): {args.model}")
    # device_map={"":0} pins everything on GPU 0. Even though weight-only RTN
    # does no forward pass, single-device avoids placement edge cases.
    # dtype='auto' lets llmcompressor place weights as the recipe expects.
    model = LlavaForConditionalGeneration.from_pretrained(
        args.model, torch_dtype="auto", device_map={"": 0})
    print("Model loaded.")

    recipe_kwargs = dict(targets="Linear", scheme=args.scheme, ignore=IGNORE)
    if args.scheme == "W4A16":
        recipe_kwargs["group_size"] = group_size

    recipe = QuantizationModifier(**recipe_kwargs)

    gs_note = f", group_size={group_size}" if args.scheme == "W4A16" else ""
    print(f"Applying {args.scheme} weight-only RTN (datafree, no calibration{gs_note}) ...")
    oneshot(model=model, recipe=recipe)      # NO dataset -> datafree path
    print("Quantization applied.")

    # ---- save as compressed-tensors PACK-QUANTIZED so vLLM loads it right ----
    fmt = _resolve_pack_format()
    print(f"Saving with quantization_format={fmt!r} -> {args.out}")
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(
        args.out,
        save_compressed=True,
        quantization_format=fmt,
    )

    # processor: copy verbatim from merged dir so patch_size /
    # vision_feature_select_strategy and tokenizer are preserved exactly.
    try:
        proc = AutoProcessor.from_pretrained(
            args.model, patch_size=14,
            vision_feature_select_strategy="default",
        )
        proc.save_pretrained(args.out)
    except Exception as e:
        print(f"[warn] processor re-save failed ({e}); copying files verbatim")
        for fn in os.listdir(args.model):
            if ("processor" in fn or "tokenizer" in fn) and os.path.isfile(
                    os.path.join(args.model, fn)):
                shutil.copy2(os.path.join(args.model, fn),
                             os.path.join(args.out, fn))

    def dir_gb(p):
        tot = 0
        for root, _, files in os.walk(p):
            for f in files:
                tot += os.path.getsize(os.path.join(root, f))
        return tot / 1e9

    print("\n========== QUANTIZATION COMPLETE ==========")
    print(f"scheme : {args.scheme}{gs_note}")
    print(f"input  : {args.model}  ({dir_gb(args.model):.2f} GB)")
    print(f"output : {args.out}  ({dir_gb(args.out):.2f} GB)")
    print("Verify: config.json has a quantization_config block "
          "(num_bits, pack-quantized format).")
    print("Next: eval with eval_llava.py --model <out> (set dtype='auto' if it "
          "errors on load).")
    print("===========================================")


if __name__ == "__main__":
    main()