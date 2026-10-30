#!/bin/bash
#SBATCH --job-name=qwen3vl_int8
#SBATCH --output=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.out
#SBATCH --error=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --partition=gpu-a100
#SBATCH --account=research-abe
#SBATCH --chdir=/scratch/ptolemy/users/ba974/weedAI/plant-VQA
# =============================================================================
# INT8 (W8A8) quantization of merged Qwen3-VL-2B. Uses the quant-only .venvQuant.
# Everything is already installed; this job has NO internet and needs none.
#   sbatch scripts/quantize_int8.bash
#
# IMPORTANT: set MIN_PX / MAX_PX below to EXACTLY match your train.yaml, so the
# calibration vision budget equals what the model was trained / will be served at.
# =============================================================================

set -euo pipefail

SCRATCH_DIR="/scratch/ptolemy/users/ba974"
QENV="$SCRATCH_DIR/weedAI/.venvQuant"
MERGED="$SCRATCH_DIR/weedAI/plant-VQA/outputs/qwen3vl/merged"
OUT="$SCRATCH_DIR/weedAI/plant-VQA/outputs/qwen3vl/merged-int8"

# --- MUST MATCH train.yaml ---
MIN_PX=128        # <-- set to your train.yaml min_pixels_factor
MAX_PX=256        # <-- set to your train.yaml max_pixels_factor
CALIB=256         # number of calibration images

# offline: model + data are on scratch, nothing is downloaded
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module purge
module load python/3.10.8
module load cuda/12.1.1
source "$QENV/bin/activate"

mkdir -p logs
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-<none>}"
echo "vision budget: min=$MIN_PX max=$MAX_PX  (must equal train.yaml)"
nvidia-smi -L || { echo "FATAL: no GPU"; exit 1; }

[ -d "$MERGED" ] || { echo "FATAL: merged model not found at $MERGED"; exit 1; }

python quantize/quantize_int8.py \
  --model   "$MERGED" \
  --val_csv "$SCRATCH_DIR/weedAI/plant-VQA/data/val/val.csv" \
  --images  "$SCRATCH_DIR/weedAI/plant-VQA/data/val/images" \
  --out     "$OUT" \
  --calib   "$CALIB" \
  --min_pixels_factor "$MIN_PX" \
  --max_pixels_factor "$MAX_PX"

echo ""
echo "=== INT8 done -> $OUT ==="
du -sh "$OUT" || true
echo "compare against bf16 merged:"