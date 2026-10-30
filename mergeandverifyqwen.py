"""
merge_and_verify.py

Step 1 of deployment: fold the LoRA adapter into the base weights, save a
standalone checkpoint, and PROVE the merge did not change model behaviour.

    python merge_and_verify.py --config configs/train.yaml

What it does:
  1. Load base + adapter, compute eval loss on a fixed val slice.  (A)
  2. Merge the adapter into the base weights.
  3. Save the merged model as a plain HF checkpoint.
  4. RELOAD that checkpoint from disk in vanilla transformers (no PEFT).
  5. Compute eval loss again on the same slice.                     (B)
  6. Assert A == B within tolerance. If they differ, the merge or the
     reload is broken and every downstream step would inherit it.

The reload-from-disk step matters: it confirms the saved checkpoint is
self-contained and loads WITHOUT peft, which is exactly how the quantizer
and the Jetson will load it.
"""

import argparse
import os

import torch
import yaml
from peft import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.dataset import load_qa_table, PlantVQADataset, QwenVLCollator

PIXELS_PER_TOKEN = 32 * 32


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def eval_loss(model, loader, device, max_batches=None):
    """Mean loss over the loader. Deterministic (no dropout, no sampling)."""
    model.eval()
    total, n = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total += out.loss.item()
        n += 1
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--model_id", default=None,
                    help="base model path/id (default: cfg['model_id']). "
                         "On the cluster point this at the LOCAL offline dir.")
    ap.add_argument("--adapter", default=None,
                    help="path to best_adapter (default: <output_dir>/best_adapter)")
    ap.add_argument("--out", default=None,
                    help="where to save merged model (default: <output_dir>/merged)")
    ap.add_argument("--val_csv", default=None,
                    help="verification split CSV (default: cfg['data']['val_csv'])")
    ap.add_argument("--val_images", default=None,
                    help="image root for the verification split "
                         "(default: cfg['data']['val_images'])")
    ap.add_argument("--n-eval", type=int, default=300,
                    help="number of val QA pairs to verify on")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max allowed |loss_adapter - loss_merged|")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    dcfg, tcfg = cfg["data"], cfg["train"]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model_id = args.model_id or cfg["model_id"]
    adapter_path = args.adapter or os.path.join(tcfg["output_dir"], "best_adapter")
    merged_path = args.out or os.path.join(tcfg["output_dir"], "merged")
    val_csv = args.val_csv or dcfg["val_csv"]
    val_images = args.val_images or dcfg["val_images"]

    # processor: identical vision budget as training.
    # train.py saves the processor into the adapter dir; if it isn't there
    # (e.g. adapter copied without it), fall back to the base model dir.
    proc_src = adapter_path
    if not os.path.exists(os.path.join(adapter_path, "preprocessor_config.json")):
        print(f"  (no processor in adapter dir; loading from base {base_model_id})")
        proc_src = base_model_id
    processor = AutoProcessor.from_pretrained(
        proc_src,
        min_pixels=dcfg["min_pixels_factor"] * PIXELS_PER_TOKEN,
        max_pixels=dcfg["max_pixels_factor"] * PIXELS_PER_TOKEN,
    )

    # fixed verification slice
    val_df = load_qa_table(val_csv, val_images)
    val_df = val_df.sample(n=min(args.n_eval, len(val_df)),
                           random_state=tcfg["seed"]).reset_index(drop=True)
    ds = PlantVQADataset(val_df)
    collator = QwenVLCollator(processor, max_length=dcfg["max_length"])
    loader = DataLoader(ds, batch_size=tcfg.get("per_device_eval_batch_size", 4),
                        collate_fn=collator, shuffle=False)

    # ---------- (A) base + adapter ----------
    print("Loading base + adapter ...")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        base_model_id, dtype=torch.bfloat16, device_map=None).to(device)
    model = PeftModel.from_pretrained(base, adapter_path).to(device)
    loss_adapter = eval_loss(model, loader, device)
    print(f"  eval loss (adapter)  : {loss_adapter:.5f}")

    # ---------- merge + save ----------
    print("Merging adapter into base weights ...")
    merged = model.merge_and_unload()      # returns the base with LoRA folded in
    os.makedirs(merged_path, exist_ok=True)
    merged.save_pretrained(merged_path, safe_serialization=True)
    processor.save_pretrained(merged_path)
    print(f"  saved merged model -> {merged_path}")

    # free memory before reloading
    del base, model, merged
    torch.cuda.empty_cache()

    # ---------- (B) reload merged FROM DISK, no peft ----------
    print("Reloading merged checkpoint in vanilla transformers ...")
    merged2 = Qwen3VLForConditionalGeneration.from_pretrained(
        merged_path, dtype=torch.bfloat16, device_map=None).to(device)
    loss_merged = eval_loss(merged2, loader, device)
    print(f"  eval loss (merged)   : {loss_merged:.5f}")

    # ---------- verdict ----------
    diff = abs(loss_adapter - loss_merged)
    print(f"\n  |adapter - merged| = {diff:.6f}  (tolerance {args.tol})")
    if diff <= args.tol:
        print("  PASS: merge preserved model behaviour.")
        print(f"\n  Deployment-ready checkpoint: {merged_path}")
        print("  This loads without PEFT and is the input to quantization.")
    else:
        print("  FAIL: merged model behaves differently from the adapter.")
        print("  Do NOT proceed to quantization. Likely causes:")
        print("    - dtype mismatch between save and reload")
        print("    - processor min/max_pixels differ from training")
        print("    - wrong adapter path")
        raise SystemExit(1)


if __name__ == "__main__":
    main()