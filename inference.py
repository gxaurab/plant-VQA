
# NOT READY HAIIIII

"""
inference.py -- single-query serving with edge-cloud routing.

    # local-only (no routing yet -- use this until you've calibrated T)
    python inference.py --adapter outputs/qwen3vl-2b-plant-12k/best_adapter \
        --image path/to/leaf.jpg --question "What disease is shown here?"

    # with routing: escalate to Gemini when local confidence < threshold
    export GEMINI_API_KEY=...
    python inference.py --adapter ... --image ... --question ... \
        --threshold -0.55 --cloud

Routing logic (the whole point):
    answer, confidence = local_generate(...)   # ONE pass, both values
    if confidence >= T:  return local answer     # most queries, no cloud
    else:                return gemini(...)       # low-confidence minority

Confidence = mean log-prob of the generated answer tokens. Same signal you
captured in eval.py; you pick T from that eval data before deploying.
"""

import argparse
import os

import torch
import yaml
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel
from qwen_vl_utils import process_vision_info

from src.dataset import build_messages

PIXELS_PER_TOKEN = 32 * 32


def load_vision_budget(adapter_dir):
    cfg = os.path.join(adapter_dir, "train_config.yaml")
    if not os.path.exists(cfg):
        raise FileNotFoundError(f"No train_config.yaml in {adapter_dir}")
    d = yaml.safe_load(open(cfg))["data"]
    return (d["min_pixels_factor"] * PIXELS_PER_TOKEN,
            d["max_pixels_factor"] * PIXELS_PER_TOKEN)


def _scorer(model):
    return (getattr(model, "compute_transition_scores", None)
            or getattr(getattr(model, "base_model", None),
                       "compute_transition_scores", None))


class LocalExpert:
    """The fine-tuned on-device model. Returns (answer, confidence)."""

    def __init__(self, adapter, model_id, device_map="auto"):
        min_px, max_px = load_vision_budget(adapter)
        self.proc = AutoProcessor.from_pretrained(
            model_id, min_pixels=min_px, max_pixels=max_px)
        self.proc.tokenizer.padding_side = "left"
        base = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=device_map)
        self.model = PeftModel.from_pretrained(base, adapter)
        self.model.eval()
        self.eos = self.proc.tokenizer.eos_token_id

    @torch.no_grad()
    def answer(self, image, question, max_new_tokens=160):
        msgs = build_messages(image, question, answer=None)
        text = self.proc.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)
        imgs, _ = process_vision_info(msgs)
        inputs = self.proc(text=[text], images=imgs,
                           return_tensors="pt").to(self.model.device)

        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens,
                                  do_sample=False, num_beams=1,
                                  output_scores=True, return_dict_in_generate=True)
        gen = out.sequences[:, inputs["input_ids"].shape[1]:]
        ans = self.proc.tokenizer.decode(gen[0], skip_special_tokens=True).strip()

        conf = float("nan")
        sc = _scorer(self.model)
        if sc is not None:
            try:
                trans = sc(out.sequences, out.scores, normalize_logits=True)
                lps = []
                for t in range(trans.size(1)):
                    if gen[0, t].item() == self.eos:
                        break
                    lp = trans[0, t].item()
                    if lp != float("-inf"):
                        lps.append(lp)
                conf = sum(lps) / len(lps) if lps else float("-inf")
            except Exception as e:
                print(f"[warn] confidence unavailable: {e}")
        return ans, conf


def gemini_answer(image, question, model="gemini-2.5-flash"):
    """Cloud fallback. Requires GEMINI_API_KEY and `pip install google-genai`."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    with open(image, "rb") as f:
        img_bytes = f.read()
    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            question,
        ],
    )
    return resp.text.strip()


def route(local, image, question, threshold=None, use_cloud=False):
    """One local pass, then escalate only if below threshold."""
    ans, conf = local.answer(image, question)

    if threshold is None or not use_cloud:
        return {"answer": ans, "confidence": conf, "source": "local",
                "note": "routing disabled (no threshold/cloud)"}

    if conf >= threshold:
        return {"answer": ans, "confidence": conf, "source": "local"}

    # low confidence -> escalate
    try:
        cloud = gemini_answer(image, question)
        return {"answer": cloud, "confidence": conf, "source": "cloud",
                "local_answer": ans}
    except Exception as e:
        # cloud failed (no key, offline, quota) -> graceful local fallback
        return {"answer": ans, "confidence": conf, "source": "local",
                "note": f"cloud escalation failed ({e}); returned local"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--question", required=True)
    ap.add_argument("--model_id", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--threshold", type=float, default=None,
                    help="escalate to cloud when confidence < this (calibrate in eval)")
    ap.add_argument("--cloud", action="store_true",
                    help="enable cloud escalation (needs GEMINI_API_KEY)")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    args = ap.parse_args()

    print("loading local expert ...")
    local = LocalExpert(args.adapter, args.model_id)

    result = route(local, args.image, args.question,
                   threshold=args.threshold, use_cloud=args.cloud)

    print("\n" + "=" * 50)
    print(f"Q          : {args.question}")
    print(f"ANSWER     : {result['answer']}")
    print(f"confidence : {result['confidence']:.3f}")
    print(f"source     : {result['source']}")
    if "local_answer" in result:
        print(f"(local was: {result['local_answer']})")
    if "note" in result:
        print(f"note       : {result['note']}")
    print("=" * 50)


if __name__ == "__main__":
    main()