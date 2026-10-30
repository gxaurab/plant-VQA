"""
quantize_w8a16_rtn.py

W8A16 weight-only INT8 quantization of merged Qwen3-VL-2B via RTN.

WHY THIS IS SIMPLE AND CRASH-FREE:
  W8A16 = 8-bit WEIGHTS, 16-bit ACTIVATIONS (activations left alone).
  Weight-only RTN just rounds each weight to the nearest int8 level using
  per-channel scales computed from the weights themselves. It needs NO
  calibration data, NO forward pass, NO images. Therefore:
    - no vision forward pass  -> no grid_thw / rot_pos_emb CUDA error
    - no DataLoader, no collator, no calibration records
    - the 'datafree' pipeline is used (llmcompressor's own recommendation)

WHY W8A16 (not W8A8), per Zheng et al. 2025 (arXiv:2505.02214):
  At 8-bit WEIGHTS, RTN/GPTQ/AWQ are all near-lossless and nearly identical.
  The accuracy damage in W8A8 comes from the ACTIVATION quantization
  (activation outliers), which small models are especially sensitive to.
  W8A16 keeps the near-lossless 8-bit weights and avoids the activation hit.

Module split (confirmed by inspection):
    quantize : model.language_model.layers.*   (196 Linear)
    ignore   : model.visual.*  + lm_head        (stay BF16)

Save: compressed-tensors pack-quantized so vLLM loads it as INT8.
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
    args = ap.parse_args()

    print("Loading merged model (single device, no auto-shard) ...")
    # device_map={"":0} pins everything on GPU 0. Even though weight-only RTN
    # does no forward pass, single-device avoids any placement edge cases.
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0})
    print("Model loaded.")

    # W8A16 weight-only: 8-bit weights, activations untouched (16-bit).
    # No dataset/calibration -> oneshot runs the datafree pipeline.
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="W8A16",
        ignore=IGNORE,
    )

    print("Applying W8A16 weight-only RTN (datafree, no calibration) ...")
    oneshot(model=model, recipe=recipe)      # NO dataset -> datafree path
    print("Quantization applied.")

    # ---- save as compressed-tensors so it's actually packed on disk ----
    # Resolve the format enum/string robustly across compressed_tensors versions.
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

    print(f"\nSaved W8A16 model to {args.out}")
    print("Verify: config.json should contain a quantization_config block,")
    print("and model.safetensors should be ~half the bf16 size.")


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
                # enum exists but member name differs; print options
                members = [x for x in dir(QF) if not x.startswith("_")]
                print(f"[warn] QuantizationFormat members: {members}")
                # best-effort: first member containing 'pack'
                for x in members:
                    if "pack" in x.lower():
                        return getattr(QF, x)
    except Exception as e:
        print(f"[warn] could not resolve QuantizationFormat enum: {e}")
    # string fallback - llmcompressor accepts this in most versions
    return "pack-quantized"


if __name__ == "__main__":
    main()