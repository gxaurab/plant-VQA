"""
src/dataset_llava.py

Dataset pipeline for LLaVA-1.5-7B-HF fine-tuning.
Mirrors the Qwen dataset.py: load table -> Dataset -> collator with
answer-only supervision. Consumes the SAME CSV + image folder as the
Qwen/Ministral pipelines. No conversations-JSON conversion.

Key LLaVA-1.5 facts baked in here:
* Prompt template is "USER: <image>\n{question} ASSISTANT:{answer}".
* The processor MUST expand the single <image> token into 576 vision
  tokens. That only happens when patch_size=14 and
  vision_feature_select_strategy="default" are set on the processor
  (done in train.py, not here). If they are unset you get the
  "image tokens do not match features" error at forward/generate.
* Answer-only supervision: mask everything up to and including
  "ASSISTANT:" so loss is computed on the answer span only.
"""

import os

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


# LLaVA-1.5 has no separate system slot in its template the way Qwen does;
# we prepend the instruction into the user turn so the model still gets it.
SYSTEM_PREFIX = (
    "You are a plant pathology expert. "
    "Answer the question about the plant image accurately and concisely.\n"
)

# Everything up to and including this marker is masked out of the loss.
ASSISTANT_MARKER = "ASSISTANT:"


# ---------------------------------------------------------------------
# Table loading  (identical contract to the Qwen load_qa_table)
# ---------------------------------------------------------------------

def load_qa_table(csv_path, image_root):
    """Load one dataset split; build absolute image paths."""
    df = pd.read_csv(csv_path)

    df["abs_path"] = df["image_path"].apply(
        lambda p: os.path.join(image_root, os.path.basename(p))
    )

    print(
        f"Loaded {len(df):,} QA pairs "
        f"from {df['image_id'].nunique():,} images."
    )
    return df


def build_prompt(question, answer=None):
    """
    LLaVA-1.5 single-image prompt.
    Training form includes the answer; inference form stops at 'ASSISTANT:'.
    """
    prompt = f"USER: <image>\n{SYSTEM_PREFIX}{question} ASSISTANT:"
    if answer is not None:
        prompt = prompt + f" {answer}"
    return prompt


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class PlantVQADataset(Dataset):

    def __init__(self, dataframe):
        self.df = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "image_path": row["abs_path"],
            "question": row["question_text"],
            "answer": row["answer"],
        }


# ---------------------------------------------------------------------
# Collator
# ---------------------------------------------------------------------

class LlavaCollator:
    """
    Builds text+image tensors with answer-only labels.

    We tokenize the full prompt (with answer) via the processor so the
    <image> token gets expanded to 576 vision tokens, then locate the
    'ASSISTANT:' marker in each row and mask everything before the answer.
    """

    def __init__(self, processor, max_length=1024):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length

        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # token ids for the marker, used to find the answer boundary
        self.marker_ids = self.tokenizer(
            ASSISTANT_MARKER, add_special_tokens=False
        )["input_ids"]

        self.dropped = 0

    def _find_answer_start(self, input_ids):
        marker = self.marker_ids
        ids = input_ids.tolist()
        for i in range(len(ids) - len(marker), -1, -1):
            if ids[i:i + len(marker)] == marker:
                return i + len(marker)
        return None

    def __call__(self, batch):
        texts, images = [], []

        for sample in batch:
            texts.append(build_prompt(sample["question"], sample["answer"]))
            images.append(Image.open(sample["image_path"]).convert("RGB"))

        encoding = self.processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        labels = encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        # also mask the expanded image tokens so loss is text-answer only
        image_token_id = getattr(
            self.processor, "image_token_id",
            self.tokenizer.convert_tokens_to_ids("<image>"),
        )
        labels[labels == image_token_id] = -100

        for i in range(labels.size(0)):
            start = self._find_answer_start(encoding["input_ids"][i])
            if start is None:
                labels[i] = -100
                self.dropped += 1
            else:
                labels[i, :start] = -100

        encoding["labels"] = labels
        return encoding
