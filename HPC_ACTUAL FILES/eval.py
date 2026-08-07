"""
eval.py -- generate answers for a split and score them.

    python eval.py --adapter outputs/qwen3vl-2b-plant-12k/best_adapter \
                   --test_csv data/test_dev_2000.csv \
                   --images data/images/test \
                   --out outputs/eval_test_dev.csv

Reports: ROUGE-L (overall + per category), Exact Match, Token-F1, BERTScore-F1,
a memorisation check (unique preds vs refs), and a per-answer confidence signal
(mean token log-prob) captured in the same pass for later routing experiments.

Per-row metrics (rougeL, token_f1, exact_match, bertscore_f1, ref_words) are
written into the output CSV for downstream routing/calibration analysis.

Deps: pip install rouge_score bert_score tqdm
"""

import argparse
import os
import re
import string
from collections import Counter

import torch
import yaml
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from src.dataset import load_qa_table, build_messages

PIXELS_PER_TOKEN = 32 * 32


def norm(s):
    """SQuAD-style: lowercase, drop punctuation + articles, collapse spaces."""
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


def load_vision_budget(adapter_dir):
    cfg_path = os.path.join(adapter_dir, "train_config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(
            f"No train_config.yaml in {adapter_dir}. Eval must use the same "
            f"vision budget as training.")
    d = yaml.safe_load(open(cfg_path))["data"]
    return (d["min_pixels_factor"] * PIXELS_PER_TOKEN,
            d["max_pixels_factor"] * PIXELS_PER_TOKEN)


def _get_transition_scorer(model):
    """compute_transition_scores may live on the wrapper or the base model."""
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
        img, _ = process_vision_info(msgs)
        images.extend(img)
    inputs = processor(text=texts, images=images, padding=True,
                       return_tensors="pt").to(model.device)

    out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                         do_sample=False, num_beams=1, #sampling disabled = pick one with highest probability
                         output_scores=True, return_dict_in_generate=True)

    gen = out.sequences[:, inputs["input_ids"].shape[1]:]   # left-pad -> uniform slice
    texts_out = [t.strip() for t in
                 processor.tokenizer.batch_decode(gen, skip_special_tokens=True)]

    # --- confidence: mean log-prob of chosen answer tokens, up to EOS ---
    # trans is aligned 1:1 with `gen` (only generated steps). pad == eos here,
    # so a single EOS check ends the real answer cleanly.
    eos_id = processor.tokenizer.eos_token_id
    conf = [float("nan")] * gen.size(0)
    scorer = _get_transition_scorer(model)
    if scorer is not None:
        try:
            trans = scorer(out.sequences, out.scores, normalize_logits=True)  # [B, T_gen]
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
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--test_csv", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", default="outputs/eval_predictions.csv")
    ap.add_argument("--model_id", default="/scratch/ptolemy/users/ba974/weedAI/qwen3VL2BInstruct") #change path here
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--limit", type=int, default=0,
                    help="0=full; >0 random debug subset, do NOT report")
    ap.add_argument("--no_bertscore", action="store_true",
                    help="skip BERTScore for fast checkpoint comparison")
    ap.add_argument("--bertscore_model", default=None,
                    help="override BERTScore model (default: roberta-large via lang=en)")
    args = ap.parse_args()

    min_px, max_px = load_vision_budget(args.adapter)
    print(f"vision budget: min={min_px:,} max={max_px:,} px")

    processor = AutoProcessor.from_pretrained(
        args.model_id, min_pixels=min_px, max_pixels=max_px)
    processor.tokenizer.padding_side = "left"

    print("loading base + adapter (unmerged; safe under device_map=auto) ...")
    base = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_id, dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, args.adapter)   # NOT merged: reliable eval
    model.eval()

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
            preds.extend(p)
            confs.extend(c)
        except Exception as e:
            print(f"[warn] batch {i}: {e}")
            preds.extend([""] * len(batch))
            confs.extend([float("nan")] * len(batch))

    df["prediction"] = preds
    df["confidence"] = confs           # mean token log-prob, for routing later

    # score FIRST so per-row metric columns exist, THEN save
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

    print("\n============ OVERALL ============")
    print(f"n           : {len(df)}")
    print(f"ROUGE-L     : {rl(list(df['prediction']), list(df['answer'])):.2f}")
    print(f"Exact Match : {em:.2f}")
    print(f"Token-F1    : {f1:.2f}")

    if with_bertscore:
        from bert_score import score as bscore
        print("computing BERTScore ...")

        # do not normalize for bERT hai

        P = df["prediction"].astype(str).tolist()
        R = df["answer"].astype(str).tolist()
        kw = dict(batch_size=64, rescale_with_baseline=False)
        if bertscore_model:
            kw["model_type"] = bertscore_model
            kw["num_layers"] = 17
        else:
            kw["model_type"] = "/scratch/ptolemy/users/ba974/weedAI/roberta-large"
            kw["num_layers"] = 17
        _, _, Fb = bscore(P, R, **kw)
        bert_f1 = 100.0 * Fb.mean().item()
        df["bertscore_f1"] = (100.0 * Fb).tolist()
        print(f"BERTScore-F1: {bert_f1:.2f}")

    print(f"\nunique preds {df['prediction'].nunique()} / unique refs {df['answer'].nunique()}")
    print("(preds << refs => model memorised templates rather than grounding)")

    if "confidence" in df.columns:
        finite = df.loc[df["confidence"].apply(
            lambda x: x not in (float("inf"), float("-inf")) and x == x), "confidence"]
        if len(finite):
            print(f"\nconfidence (mean logprob): "
                  f"mean {finite.mean():.3f}  p10 {finite.quantile(0.10):.3f}  "
                  f"p50 {finite.quantile(0.50):.3f}  p90 {finite.quantile(0.90):.3f}")

    if "question_category" in df.columns:
        print("\n---- ROUGE-L by question_category ----")
        for cat, g in df.groupby("question_category"):
            line = f"  {cat:32s} {rl(list(g['prediction']), list(g['answer'])):6.2f}  (n={len(g)})"
            if with_bertscore and "bertscore_f1" in df.columns:
                line += f"   BERT {g['bertscore_f1'].mean():6.2f}"
            print(line)

    # --- persist PER-ROW metrics into the dataframe so they land in the CSV ---
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