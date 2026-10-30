"""
quantize_int8.py

INT8 W8A8 quantization of merged Qwen3-VL-2B using llmcompressor 0.6.0.

Method:
    RTN (Round-To-Nearest) via QuantizationModifier.

Why RTN:
    GPTQ uses llmcompressor's sequential pipeline, which triggers
    torch.fx symbolic tracing. Qwen3-VL's causal-mask implementation
    contains data-dependent control flow that FX cannot trace.

    SmoothQuant also caused llmcompressor 0.6.0 to infer a SequentialPipeline,
    which again enters FX tracing.

    Therefore this version deliberately uses ONLY QuantizationModifier
    with W8A8 and the basic calibration pipeline.

Goal:
    - model.language_model.layers.* -> W8A8 INT8
    - model.visual.*             -> BF16
    - lm_head                    -> BF16

Calibration:
    - Real PlantExpertVQA images/questions
    - 256 multimodal calibration samples
    - Vision budget matches training:
        min_pixels_factor = 128
        max_pixels_factor = 256
"""

import argparse
import os
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader

from transformers import (
    AutoProcessor,
    Qwen3VLForConditionalGeneration,
)

from llmcompressor import active_session
from llmcompressor.core.session_functions import reset_session
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.pipelines import CalibrationPipeline
from llmcompressor.args import DatasetArguments


# ---------------------------------------------------------
# Same vision budget used during training
# ---------------------------------------------------------

PIXELS_PER_TOKEN = 32 * 32

SYSTEM_PROMPT = (
    "You are a plant pathology expert. Answer the question about the plant "
    "image accurately and concisely."
)


# ---------------------------------------------------------
# Modules that must remain BF16
# ---------------------------------------------------------

IGNORE = [
    "lm_head",
    r"re:.*model\.visual\..*",
]


def build_calibration_records(
    val_csv,
    image_root,
    processor,
    n,
    seed=42,
):
    """
    Build real multimodal calibration examples.

    Each record contains the processor outputs required by Qwen3-VL:

        input_ids
        attention_mask
        pixel_values
        image_grid_thw
        ...

    The processor creates a batch dimension of 1.
    We remove that dimension here and restore it in the collator.
    """

    from qwen_vl_utils import process_vision_info

    df = pd.read_csv(
        val_csv,
        low_memory=False,
    )

    # Remove unusable questions.
    df = df.dropna(
        subset=["question_text"]
    )

    # -----------------------------------------------------
    # Build image paths
    # -----------------------------------------------------

    df["abs_path"] = df["image_path"].apply(
        lambda p: os.path.join(
            image_root,
            os.path.basename(str(p)),
        )
    )

    # Keep only examples whose images exist.
    df = df[
        df["abs_path"].apply(os.path.exists)
    ].reset_index(drop=True)

    if len(df) == 0:
        raise RuntimeError(
            f"No valid calibration examples found.\n"
            f"CSV: {val_csv}\n"
            f"Image root: {image_root}"
        )

    # -----------------------------------------------------
    # Reproducible calibration subset
    # -----------------------------------------------------

    random.seed(seed)

    idx = random.sample(
        range(len(df)),
        min(n, len(df)),
    )

    records = []

    # -----------------------------------------------------
    # Build multimodal records
    # -----------------------------------------------------

    for i in idx:

        row = df.iloc[i]

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
                        "image": row["abs_path"],
                    },
                    {
                        "type": "text",
                        "text": str(row["question_text"]),
                    },
                ],
            },
        ]

        # Qwen3-VL chat template
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process image
        imgs, _ = process_vision_info(
            messages,
            image_patch_size=16,
        )

        # Processor creates batch dimension [1, ...]
        enc = processor(
            text=[text],
            images=imgs,
            return_tensors="pt",
        )

        # Remove processor batch dimension.
        records.append(
            {
                key: value[0]
                for key, value in enc.items()
            }
        )

    # -----------------------------------------------------
    # Validate calibration records
    # -----------------------------------------------------

    has_pixel_values = sum(
        1
        for record in records
        if "pixel_values" in record
    )

    print(
        f"[calib] {len(records)} records | "
        f"with pixel_values: "
        f"{has_pixel_values}/{len(records)}"
    )

    assert len(records) > 0, (
        "No calibration records were created!"
    )

    assert has_pixel_values == len(records), (
        "Some calibration records are missing pixel_values!"
    )

    return records


def multimodal_collate(batch):
    """
    Calibration batch size = 1.

    build_calibration_records() removes the processor's batch
    dimension. Restore it before the model forward pass.
    """

    assert len(batch) == 1, (
        "Calibration must use batch_size=1 "
        "for this multimodal setup."
    )

    output = {}

    for key, value in batch[0].items():

        if not torch.is_tensor(value):
            value = torch.tensor(value)

        output[key] = value.unsqueeze(0)

    return output


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--val_csv",
        required=True,
    )

    parser.add_argument(
        "--images",
        required=True,
    )

    parser.add_argument(
        "--out",
        required=True,
    )

    parser.add_argument(
        "--calib",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--min_pixels_factor",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--max_pixels_factor",
        type=int,
        default=256,
    )

    args = parser.parse_args()

    # =====================================================
    # PROCESSOR
    # =====================================================

    processor = AutoProcessor.from_pretrained(
        args.model,
        min_pixels=(
            args.min_pixels_factor
            * PIXELS_PER_TOKEN
        ),
        max_pixels=(
            args.max_pixels_factor
            * PIXELS_PER_TOKEN
        ),
    )

    print(
        f"Vision budget: "
        f"min_factor={args.min_pixels_factor}, "
        f"max_factor={args.max_pixels_factor}"
    )

    # =====================================================
    # LOAD MERGED BF16 MODEL
    # =====================================================

    print("Loading merged model ...")

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
    )

    print("Merged model loaded.")

    # =====================================================
    # BUILD MULTIMODAL CALIBRATION DATA
    # =====================================================

    records = build_calibration_records(
        args.val_csv,
        args.images,
        processor,
        args.calib,
    )

    # =====================================================
    # OUR OWN DATALOADER
    #
    # This completely bypasses llmcompressor's broken
    # HF Dataset -> text preprocessing path.
    # =====================================================

    calib_loader = DataLoader(
        records,
        batch_size=1,
        shuffle=False,
        collate_fn=multimodal_collate,
    )

    print(
        f"[calib] DataLoader ready: "
        f"{len(calib_loader)} batches (bs=1)"
    )

    # =====================================================
    # RTN W8A8 QUANTIZATION
    #
    # IMPORTANT:
    #
    # No GPTQ.
    # No SmoothQuant.
    #
    # This prevents llmcompressor from selecting the
    # SequentialPipeline and therefore avoids torch.fx
    # tracing of Qwen3-VL.
    # =====================================================

    recipe = [
        QuantizationModifier(
            targets="Linear",
            scheme="W8A8",
            ignore=IGNORE,
        ),
    ]

    print()
    print("========================================")
    print("INT8 W8A8 QUANTIZATION (RTN)")
    print("========================================")
    print(f"Ignore modules: {IGNORE}")
    print("Vision: BF16")
    print("Language backbone: W8A8 INT8")
    print(
        f"Calibration samples: {len(records)}"
    )
    print("========================================")
    print()

    # =====================================================
    # MANUAL LLMCOMPRESSOR PIPELINE
    #
    # We do NOT call oneshot().
    #
    # oneshot() would send the Dataset through its internal
    # dataset-processing path.
    #
    # Instead we reproduce the useful part of oneshot()
    # and provide our already-built multimodal DataLoader.
    # =====================================================

    print(
        "Initializing compression session ..."
    )

    reset_session()

    session = active_session()

    session.initialize(
        model=model,
        start=-1,
        recipe=recipe,
        calib_data=calib_loader,
    )

    modifiers = session.get_modifiers()

    print(
        f"[session] {len(modifiers)} modifiers active"
    )

    # -----------------------------------------------------
    # DatasetArguments
    #
    # The calibration pipeline expects this object even
    # though we already provide the DataLoader ourselves.
    # -----------------------------------------------------

    dataset_args = DatasetArguments(
        num_calibration_samples=len(records),
        shuffle_calibration_samples=False,
        max_seq_length=1024,
        pad_to_max_length=False,
        pipeline="basic",
    )

    # -----------------------------------------------------
    # Explicitly request BASIC pipeline.
    #
    # This is important:
    #
    # basic -> normal calibration forward passes
    #
    # sequential -> torch.fx tracing
    #
    # We do NOT want sequential for Qwen3-VL.
    # -----------------------------------------------------

    pipeline = CalibrationPipeline.from_modifiers(
        modifiers,
        user="basic",
    )

    print(
        f"[pipeline] {type(pipeline).__name__} selected"
    )

    print(
        "Running calibration + quantization ..."
    )

    pipeline(
        model,
        calib_loader,
        dataset_args,
    )

    # =====================================================
    # FINALIZE
    # =====================================================

    session.finalize()

    print()
    print(
        "Calibration + quantization complete."
    )

    # =====================================================
    # SAVE
    # =====================================================

    os.makedirs(
        args.out,
        exist_ok=True,
    )

    model.save_pretrained(
        args.out,
        save_compressed=True,
        quantization_format="packed+quantized"
    )

    processor.save_pretrained(
        args.out,
    )

    print()
    print(
        f"INT8 model saved to: {args.out}"
    )


if __name__ == "__main__":
    main()