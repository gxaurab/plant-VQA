"""
smoke_test.py  -- run this BEFORE writing train.py

Verifies the one thing that fails silently: that loss is computed on the
answer tokens and nothing else. Decodes the unmasked positions back to text
so you can read exactly what the model will be trained to produce.

    python smoke_test.py
"""
import os
from transformers import AutoProcessor
from torch.utils.data import DataLoader

from src.dataset import load_qa_table, PlantVQADataset, QwenVLCollator

MODEL_ID   = "Qwen/Qwen3-VL-2B-Instruct"
TRAIN_CSV  = "data/train.csv"
IMAGE_ROOT = "data/images/train_images"
MAX_PIXELS = 1024 * 28 * 28           # pin this; reuse the SAME value at eval
MAX_LENGTH = 1024

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=256 * 28 * 28,
    max_pixels=MAX_PIXELS,
)
if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

df = load_qa_table(TRAIN_CSV, IMAGE_ROOT)
ds = PlantVQADataset(df.sample(n=min(8, len(df)), random_state=0))
collator = QwenVLCollator(processor, max_length=MAX_LENGTH)

loader = DataLoader(ds, batch_size=4, collate_fn=collator)
batch = next(iter(loader))

print("\n--- tensor shapes ---")
for k, v in batch.items():
    if hasattr(v, "shape"):
        print(f"{k:22s} {tuple(v.shape)}")

print("\n--- what the loss actually sees ---")
ids, labels = batch["input_ids"], batch["labels"]
tok = processor.tokenizer

for i in range(ids.size(0)):
    keep = labels[i] != -100
    n_total, n_sup = int(keep.numel()), int(keep.sum())
    target = tok.decode(ids[i][keep], skip_special_tokens=False)
    print(f"\n[{i}] supervised {n_sup}/{n_total} tokens")
    print(f"    TARGET: {target!r}")

print("\nPASS CRITERIA")
print("  * TARGET is the ANSWER only (plus <|im_end|>)")
print("  * no question text, no system prompt, no <|vision|> tokens in TARGET")
print("  * supervised token count is small relative to total")
print("  * no row shows 0 supervised tokens (means marker was truncated away)")