"""
src/dataset.py

Minimal dataset pipeline for Qwen3-VL fine-tuning.

Responsibilities:
1. Read the CSV split.
2. Build absolute image paths.
3. Create Qwen conversation messages.
4. Convert a batch into model tensors with answer-only supervision.
"""

import os

import pandas as pd
import torch
from torch.utils.data import Dataset

from qwen_vl_utils import process_vision_info


SYSTEM_PROMPT = (
    "You are a plant pathology expert. "
    "Answer the question about the plant image accurately and concisely."
)

ASSISTANT_MARKER = "<|im_start|>assistant\n"


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def load_qa_table(csv_path, image_root):
    """Load one dataset split."""

    df = pd.read_csv(csv_path)

    # image_path in CSV is like: images/image_0001.JPG
    df["abs_path"] = df["image_path"].apply(
        lambda p: os.path.join(image_root, os.path.basename(p))
    )

    print(
        f"Loaded {len(df):,} QA pairs "
        f"from {df['image_id'].nunique():,} images."
    )

    return df


def build_messages(image_path, question, answer=None):
    """
    Create one Qwen conversation.
    """

    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        },
    ]

    if answer is not None:
        messages.append(
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": answer,
                    }
                ],
            }
        )

    return messages


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

class QwenVLCollator:

    def __init__(self, processor, max_length=1024):

        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.dropped = 0

        self.tokenizer.padding_side = "right"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.marker_ids = self.tokenizer(
            ASSISTANT_MARKER,
            add_special_tokens=False,
        )["input_ids"]

    def _find_answer_start(self, input_ids):

        marker = self.marker_ids
        ids = input_ids.tolist()

        for i in range(len(ids) - len(marker), -1, -1):
            if ids[i:i + len(marker)] == marker:
                return i + len(marker)

        return None

    def __call__(self, batch):

        texts = []
        images = []

        for sample in batch:

            messages = build_messages(
                sample["image_path"],
                sample["question"],
                sample["answer"],
            )

            texts.append(
                self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

            img, _ = process_vision_info(messages)
            images.extend(img)

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

        for i in range(labels.size(0)):

            start = self._find_answer_start(
                encoding["input_ids"][i]
            )

            if start is None:
                labels[i] = -100
                self.dropped += 1
            else:
                labels[i, :start] = -100

        encoding["labels"] = labels

        return encoding