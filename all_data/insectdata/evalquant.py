"""
eval_quant_merged.py -- evaluate a QUANTIZED (W8A16 or W4A16) Qwen3-VL model.

Based on the proven eval_merged.py (same metrics, same loader, same env),
with ONE addition: the compress-on-load patch so compressed-tensors loads the
packed INT8/INT4 checkpoint without the "weight_scale on meta device" error.

Scheme-agnostic: the SAME script evaluates both W8A16 and W4A16 -- just point
--model at the respective merged-w8a16 / merged-w4a16 dir.

    python eval_quant_merged.py \
        --model    outputs/qwen3vl/merged-w8a16 \
        --test_csv data/test/test.csv \
        --images   data/test/images \
        --out      outputs/eval_int8_withbert.csv \
        --bertscore_model /scratch/.../roberta-large \
        --min_pixels_factor 128 --max_pixels_factor 256
"""

import argparse
import os
import re
import string
from collections import Counter

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from src.dataset import load_qa_table, build_messages

PIXELS_PER_TOKEN = 32 * 32


# ---------------------------------------------------------------------------
# Compress-on-load patch: stops CompressedTensors from trying to re-compress an
# already-compressed checkpoint (which raises "weight_scale is on the meta
# device"). Required to load the packed W8A16 / W4A16 model.
# ---------------------------------------------------------------------------
def apply_compress_load_patch():
    import transformers.quantizers.quantizer_compressed_tensors as qct
    orig = qct.CompressedTensorsHfQuantizer._process_model_before_weight_loading

    def patched(self, model, **kwargs):
        save = self.compressor.compress_model
        self.compressor.compress_model = lambda *a, **k: None
        try:
            return orig(self, model, **kwargs)
        finally:
            self.compressor.compress_model = save

    qct.CompressedTensorsHfQuantizer._process_model_before_weight_loading = patched
    print("[patch] compress-on-load disabled (quantized model loads directly)")


def norm(s):
    s = str(s).lower()
    s = "".join(c for c in s if c not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def token_f1(h, r):
    ht, rt = h.split(), r.split()
    if not ht or not rt:
        return float(ht == rt)
    overlap = sum((Counter(ht) & Counter(rt)).values())
    if overlap == 0:
        return 0.0
    p, rec = overlap / len(ht), overlap / len(rt)
    return 2 * p * rec / (p + rec)


def _get_transition_scorer(model):
    return (getattr(model, "compute_transition_scores", None)
            or getattr(getattr(model, "base_model", None),
                       "compute_transition_scores", None))


@torch.no_grad()
def generate_batch(model, processor, rows, max_new_tokens):
    texts, images = [], []
    for r in rows:
        msgs = build_messages(r["abs_path"], str(r["question_text"]), answer=None)
        texts.append(processor.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))
        img, _ = process_vision_info(msgs)          # NO image_patch_size (Qwen3-VL)
        images.extend(img)
    inputs = processor(text=texts, images=images, padding=True,
                       return_tensors="pt").to(model.device)

    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False, num_beams=1,
                         output_scores=True, return_dict_in_generate=True)

    gen = out.sequences[:, inputs["input_ids"].shape[1]:]
    texts_out = [t.strip() for t in
                 processor.tokenizer.batch_decode(gen, skip_special_tokens=True)]

    eos_id = processor.tokenizer.eos_token_id
    conf = [float("nan")] * gen.size(0)
    scorer = _get_transition_scorer(model)
    if scorer is not None:
        try:
            trans = scorer(out.sequences, out.scores, normalize_logits=True)
            for b in range(gen.size(0)):
                row = []
                for t in range(trans.size(1)):
                    if gen[b, t].item() == eos_id:
                        break
                    lp = trans[b, t].item()
                    if lp != float("-inf"):
                        row.append(lp)
                conf[b] = sum(row) / len(row) if row else float("-inf")
        except Exception as e:
            print(f"[warn] confidence skipped for this batch: {e}")

    return texts_out, conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="quantized merged model dir (w8a16 or w4a16)")
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="outputs/eval_quant.csv")
    ap.add_argument("--min_pixels_factor", type=int, default=128)  # MUST match training
    ap.add_argument("--max_pixels_factor", type=int, default=256)  # MUST match training
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_bertscore", action="store_true")
    ap.add_argument("--bertscore_model", default=None)
    args = ap.parse_args()

    apply_compress_load_patch()          # MUST run before loading the quantized model

    min_px = args.min_pixels_factor * PIXELS_PER_TOKEN
    max_px = args.max_pixels_factor * PIXELS_PER_TOKEN
    print(f"vision budget: min={min_px:,} max={max_px:,} px")

    processor = AutoProcessor.from_pretrained(
        args.model, min_pixels=min_px, max_pixels=max_px, local_files_only=True)
    processor.tokenizer.padding_side = "left"

    print("loading quantized merged model ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, local_files_only=True)
    model.eval()
    print("model loaded.")

    df = load_qa_table(args.test_csv, args.images)
    df = df[df["answer"].notna() & (df["answer"].astype(str).str.strip() != "")]
    df = df.reset_index(drop=True)
    if args.limit:
        df = df.sample(n=min(args.limit, len(df)), random_state=0).reset_index(drop=True)
        print(f"LIMIT {len(df)} rows (debug only -- do NOT report)")

    rows = df.to_dict("records")
    preds, confs = [], []
    for i in tqdm(range(0, len(rows), args.batch_size), desc="generate"):
        batch = rows[i:i + args.batch_size]
        try:
            p, c = generate_batch(model, processor, batch, args.max_new_tokens)
            preds.extend(p); confs.extend(c)
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[warn] batch {i}: {e}")
            preds.extend([""] * len(batch)); confs.extend([float("nan")] * len(batch))

    df["prediction"] = preds
    df["confidence"] = confs
    df = score(df, with_bertscore=not args.no_bertscore,
               bertscore_model=args.bertscore_model)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"saved -> {args.out}")


def score(df, with_bertscore=True, bertscore_model=None):
    from rouge_score import rouge_scorer
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    def rl(preds, refs):
        if len(preds) == 0:
            return 0.0
        return 100.0 * sum(rs.score(norm(r), norm(p))["rougeL"].fmeasure
                           for p, r in zip(preds, refs)) / len(preds)

    hyps = [norm(p) for p in df["prediction"]]
    refs = [norm(a) for a in df["answer"]]
    em = 100.0 * sum(h == r for h, r in zip(hyps, refs)) / len(refs)
    f1 = 100.0 * sum(token_f1(h, r) for h, r in zip(hyps, refs)) / len(refs)

    print("\n============ OVERALL (QUANT) ============")
    print(f"n           : {len(df)}")
    print(f"ROUGE-L     : {rl(list(df['prediction']), list(df['answer'])):.2f}")
    print(f"Exact Match : {em:.2f}")
    print(f"Token-F1    : {f1:.2f}")

    if with_bertscore:
        from bert_score import score as bscore
        print("computing BERTScore ...")
        P = df["prediction"].astype(str).tolist()
        R = df["answer"].astype(str).tolist()
        kw = dict(batch_size=64, rescale_with_baseline=False, num_layers=17)
        kw["model_type"] = bertscore_model or "/scratch/ptolemy/users/ba974/weedAI/roberta-large"
        _, _, Fb = bscore(P, R, **kw)
        df["bertscore_f1"] = (100.0 * Fb).tolist()
        print(f"BERTScore-F1: {100.0 * Fb.mean().item():.2f}")

    print(f"\nunique preds {df['prediction'].nunique()} / unique refs {df['answer'].nunique()}")

    if "confidence" in df.columns:
        finite = df.loc[df["confidence"].apply(
            lambda x: x not in (float("inf"), float("-inf")) and x == x), "confidence"]
        if len(finite):
            print(f"\nconfidence (mean logprob): mean {finite.mean():.3f}  "
                  f"p10 {finite.quantile(0.10):.3f}  p50 {finite.quantile(0.50):.3f}  "
                  f"p90 {finite.quantile(0.90):.3f}")

    if "question_category" in df.columns:
        print("\n---- ROUGE-L by question_category ----")
        for cat, g in df.groupby("question_category"):
            line = f"  {cat:34s} {rl(list(g['prediction']), list(g['answer'])):6.2f}  (n={len(g)})"
            if with_bertscore and "bertscore_f1" in df.columns:
                line += f"   BERT {g['bertscore_f1'].mean():6.2f}"
            print(line)

    df["rougeL"] = [100.0 * rs.score(norm(a), norm(p))["rougeL"].fmeasure
                    for p, a in zip(df["prediction"], df["answer"])]
    df["token_f1"] = [100.0 * token_f1(norm(p), norm(a))
                      for p, a in zip(df["prediction"], df["answer"])]
    df["exact_match"] = [100.0 * float(norm(p) == norm(a))
                         for p, a in zip(df["prediction"], df["answer"])]
    df["ref_words"] = df["answer"].astype(str).str.split().str.len()
    return df


if __name__ == "__main__":
    main()