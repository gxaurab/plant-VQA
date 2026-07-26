"""
train.py -- LoRA fine-tune Qwen3-VL-2B on PlantExpertVQA.

    python train.py --config configs/train.yaml --inspect    # verify, then exit
    python train.py --config configs/train.yaml              # train
    python train.py --config configs/train.yaml --resume     # resume from checkpoint

Vision encoder stays frozen: LoRA targets only language-backbone projections
via a regex. Always run --inspect first and confirm zero vision-tower hits.
"""

import argparse
import os
import re

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    EarlyStoppingCallback,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from src.dataset import load_qa_table, PlantVQADataset, QwenVLCollator

# Qwen3-VL: patch_size 16 x spatial_merge 2 -> one vision token per 32x32 px.
# So `factor` in the config IS the vision-token count.
PIXELS_PER_TOKEN = 32 * 32


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def inspect_modules(model, target):
    """
    Print which Linear modules LoRA would wrap.
    `target` may be a regex string (preferred) or a list of suffixes.
    """
    if isinstance(target, str):
        pat = re.compile(target)
        def hit(name):
            return bool(pat.fullmatch(name) or pat.match(name))
    else:
        def hit(name):
            return any(name.endswith(t) for t in target)
        print("NOTE: target_modules is a LIST -> suffix matching across the WHOLE "
              "model, including the vision tower. A regex string is safer.")

    matched, vision_hits = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and hit(name):
            matched.append(name)
            if "visual" in name or "vision" in name:
                vision_hits.append(name)

    print(f"\nmatched {len(matched)} Linear modules")
    for n in matched[:10]:
        print("   ", n)
    if len(matched) > 10:
        print(f"    ... and {len(matched) - 10} more")

    if not matched:
        print("\nNO MATCHES. Sample of actual Linear module names:")
        names = [n for n, m in model.named_modules()
                 if isinstance(m, torch.nn.Linear)]
        for n in names[:40]:
            print("   ", n)
        print("\nFix `lora.target_modules` to match these.")

    if vision_hits:
        print(f"\n*** WARNING: {len(vision_hits)} matches are in the VISION tower ***")
        for n in vision_hits[:5]:
            print("   ", n)
        print("The vision encoder must stay frozen. Tighten the regex.")
    else:
        print("\nOK: no vision-tower modules matched. Vision encoder stays frozen.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train.yaml")
    ap.add_argument("--inspect", action="store_true",
                    help="print LoRA target modules and exit")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the last checkpoint in output_dir")
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    dcfg, lcfg, tcfg = cfg["data"], cfg["lora"], cfg["train"]

    use_bf16 = bool(tcfg.get("bf16", False))
    use_fp16 = bool(tcfg.get("fp16", False))
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    print("Loading model ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        cfg["model_id"],
        dtype=dtype,
        device_map=None,          # Trainer handles placement
    )

    if args.inspect:
        inspect_modules(model, lcfg["target_modules"])
        return

    # ---- processor: vision budget is pinned here and MUST match at eval ----
    min_px = dcfg["min_pixels_factor"] * PIXELS_PER_TOKEN
    max_px = dcfg["max_pixels_factor"] * PIXELS_PER_TOKEN
    est_tokens = dcfg["max_pixels_factor"] + 95      # + system/question/answer
    print(f"vision budget: <= {dcfg['max_pixels_factor']} tokens "
          f"({max_px:,} px); est. total seq ~{est_tokens} vs "
          f"max_length {dcfg['max_length']}")
    if est_tokens > dcfg["max_length"]:
        raise ValueError(
            f"max_pixels_factor ({dcfg['max_pixels_factor']}) leaves no room for "
            f"the answer inside max_length ({dcfg['max_length']}). "
            f"Lower the factor or raise max_length."
        )

    processor = AutoProcessor.from_pretrained(
        cfg["model_id"], min_pixels=min_px, max_pixels=max_px
    )

    # ---- gradient checkpointing needs grads to flow into embeddings ----
    if tcfg.get("gradient_checkpointing"):
        model.enable_input_require_grads()
    model.config.use_cache = False

    # ---- LoRA: language backbone only ----
    peft_cfg = LoraConfig(
        r=lcfg["r"],
        lora_alpha=lcfg["alpha"],
        lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)

    print("\n========== MODEL SUMMARY ==========")
    model.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Frozen parameters    : {total_params - trainable_params:,}")

    vision_trainable = sum(
        p.numel()
        for n, p in model.named_parameters()
        if ("visual" in n or "vision" in n) and p.requires_grad
    )

    print(f"Vision trainable params : {vision_trainable:,}")

    if vision_trainable == 0:
        print("✅ Vision encoder is frozen.")
    else:
        print("❌ WARNING: Vision encoder is NOT frozen!")

    print("===================================\n")

    # ---- data ----
    train_df = load_qa_table(dcfg["train_csv"], dcfg["train_images"])
    val_df = load_qa_table(dcfg["val_csv"], dcfg["val_images"])

    # eval on a fixed subsample during training - the full val set every
    # eval_steps would add ~25% to runtime for no benefit in checkpoint choice
    val_limit = dcfg.get("val_limit")
    if val_limit and len(val_df) > val_limit:
        val_df = val_df.sample(n=val_limit, random_state=tcfg["seed"]).reset_index(drop=True)
        print(f"eval subsample: {len(val_df)} QA pairs")

    train_ds = PlantVQADataset(train_df)
    val_ds = PlantVQADataset(val_df)
    collator = QwenVLCollator(processor, max_length=dcfg["max_length"])

    eval_steps = tcfg.get("eval_steps", 500)

    targs = TrainingArguments(
        output_dir=tcfg["output_dir"],
        num_train_epochs=tcfg["epochs"],
        learning_rate=float(tcfg["lr"]),
        lr_scheduler_type=tcfg["scheduler"],
        warmup_ratio=tcfg["warmup_ratio"],
        weight_decay=tcfg["weight_decay"],
        per_device_train_batch_size=tcfg["per_device_batch_size"],
        per_device_eval_batch_size=tcfg.get("per_device_eval_batch_size",
                                            tcfg["per_device_batch_size"]),
        gradient_accumulation_steps=tcfg["grad_accum"],
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=tcfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=tcfg.get("logging_steps", 25),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,               # must match for load_best_model_at_end
        save_total_limit=tcfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,         # our collator takes raw dicts
        label_names=["labels"],              # PeftModel cannot infer this
        dataloader_num_workers=tcfg.get("num_workers", 4),
        dataloader_pin_memory=True,
        report_to=tcfg.get("report_to", "none"),
        seed=tcfg["seed"],
        logging_dir=os.path.join(tcfg["output_dir"], "logs"),

    )

    callbacks = []
    patience = tcfg.get("early_stopping_patience", 3)
    if patience:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=patience))

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=callbacks,
    )

    steps_per_epoch = max(
        1,
        len(train_ds) // (tcfg["per_device_batch_size"] * tcfg["grad_accum"]),
    )
    print(f"\n~{steps_per_epoch} optimizer steps/epoch, "
          f"~{steps_per_epoch * tcfg['epochs']} total, "
          f"eval every {eval_steps}\n")

    trainer.train(resume_from_checkpoint=args.resume or None)

    trainer.save_state()

    out = os.path.join(tcfg["output_dir"], "best_adapter")
    trainer.save_model(out)
    processor.save_pretrained(out)
    with open(os.path.join(out, "train_config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Adapter saved : {out}")
    print(f"Config saved  : {os.path.join(out, 'train_config.yaml')}")
    print("=======================================\n")

    if getattr(collator, "dropped", 0):
        print(f"WARNING: {collator.dropped} samples had no assistant marker "
              f"(truncated) and contributed nothing to the loss.")


if __name__ == "__main__":
    main()