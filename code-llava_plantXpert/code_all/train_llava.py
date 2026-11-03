"""
train_llava.py -- LoRA fine-tune LLaVA-1.5-7B-HF on PlantExpertVQA.

    python train_llava.py --config configs/train_llava.yaml --inspect   # verify, exit
    python train_llava.py --config configs/train_llava.yaml             # train
    python train_llava.py --config configs/train_llava.yaml --resume    # resume

Vision encoder + multimodal projector stay FROZEN: LoRA targets only the
language_model projections via a regex. Always run --inspect first and
confirm zero vision_tower / multi_modal_projector hits, matching the
Qwen and Ministral runs.
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
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from src.dataset_llava import load_qa_table, PlantVQADataset, LlavaCollator


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def inspect_modules(model, target):
    """Print which Linear modules LoRA would wrap; flag vision-tower hits."""
    if isinstance(target, str):
        pat = re.compile(target)
        def hit(name):
            return bool(pat.fullmatch(name) or pat.match(name))
    else:
        def hit(name):
            return any(name.endswith(t) for t in target)
        print("NOTE: target_modules is a LIST -> suffix matching across the "
              "WHOLE model, including the vision tower. A regex string is safer.")

    matched, vision_hits = [], []
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and hit(name):
            matched.append(name)
            if ("vision_tower" in name or "visual" in name
                    or "multi_modal_projector" in name):
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
        print(f"\n*** WARNING: {len(vision_hits)} matches are in the VISION "
              f"tower / projector ***")
        for n in vision_hits[:5]:
            print("   ", n)
        print("Those must stay frozen. Tighten the regex.")
    else:
        print("\nOK: no vision-tower / projector modules matched. "
              "Vision encoder stays frozen.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/train_llava.yaml")
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
    model = LlavaForConditionalGeneration.from_pretrained(
        cfg["model_id"],
        dtype=dtype,
        device_map=None,          # Trainer handles placement
    )

    if args.inspect:
        inspect_modules(model, lcfg["target_modules"])
        return

    # ---- processor: patch_size + select strategy are what expand <image>
    #      into 576 vision tokens. Missing these -> token/feature mismatch.
    processor = AutoProcessor.from_pretrained(
        cfg["model_id"],
        patch_size=14,
        vision_feature_select_strategy="default",
    )

    # ---- freeze vision tower + projector explicitly (belt and suspenders;
    #      LoRA regex already excludes them, but this zeroes their grads) ----
    for name, p in model.named_parameters():
        if ("vision_tower" in name or "multi_modal_projector" in name):
            p.requires_grad = False

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
        p.numel() for n, p in model.named_parameters()
        if ("vision_tower" in n or "visual" in n
            or "multi_modal_projector" in n) and p.requires_grad
    )
    print(f"Vision/projector trainable params : {vision_trainable:,}")
    if vision_trainable == 0:
        print("Vision encoder + projector are frozen.")
    else:
        print("WARNING: vision/projector is NOT fully frozen!")
    print("===================================\n")

    # ---- data ----
    train_df = load_qa_table(dcfg["train_csv"], dcfg["train_images"])
    val_df = load_qa_table(dcfg["val_csv"], dcfg["val_images"])

    val_limit = dcfg.get("val_limit")
    if val_limit and len(val_df) > val_limit:
        val_df = val_df.sample(
            n=val_limit, random_state=tcfg["seed"]
        ).reset_index(drop=True)
        print(f"eval subsample: {len(val_df)} QA pairs")

    train_ds = PlantVQADataset(train_df)
    val_ds = PlantVQADataset(val_df)
    collator = LlavaCollator(processor, max_length=dcfg["max_length"])

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
        save_steps=eval_steps,
        save_total_limit=tcfg.get("save_total_limit", 2),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
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
        1, len(train_ds) // (tcfg["per_device_batch_size"] * tcfg["grad_accum"])
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
    print("=======================================\n")

    if getattr(collator, "dropped", 0):
        print(f"WARNING: {collator.dropped} samples had no ASSISTANT marker "
              f"(truncated) and contributed nothing to the loss.")


if __name__ == "__main__":
    main()
