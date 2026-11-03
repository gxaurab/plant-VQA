#!/bin/bash
#SBATCH --job-name=qwen3vl_awq_w4a16
#SBATCH --output=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.out
#SBATCH --error=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --partition=gpu-a100
#SBATCH --account=research-abe
#SBATCH --chdir=/scratch/ptolemy/users/ba974/weedAI/plant-VQA
# =============================================================================
# AWQ W4A16 quantization of merged Qwen3-VL-2B.
# Uses .venvWeedAI (has llmcompressor 0.8.1 + AWQ + replace_modules_for_calibration).
#   sbatch scripts/quantize_awq.bash
# =============================================================================

set -euo pipefail

SCRATCH_DIR="/scratch/ptolemy/users/ba974"
PROJECT_DIR="$SCRATCH_DIR/weedAI/plant-VQA"
VENV_PATH="$SCRATCH_DIR/weedAI/.venvWeedAI"          # the env with AWQ + 0.8.1
MERGED="$PROJECT_DIR/outputs/qwen3vl/merged"
OUT="$PROJECT_DIR/outputs/qwen3vl/merged-awq-w4a16"

MIN_PX=128        # MUST match train.yaml
MAX_PX=256
CALIB=256
GROUP=128

export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module purge
module load python/3.10.8
module load cuda/12.1.1
source "$VENV_PATH/bin/activate"

mkdir -p logs
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-<none>}"
nvidia-smi -L || { echo "FATAL: no GPU"; exit 1; }
[ -d "$MERGED" ] || { echo "FATAL: merged model not found at $MERGED"; exit 1; }

python quantize/quantize_awq_w4a16.py \
  --model     "$MERGED" \
  --train_csv "$PROJECT_DIR/data/train.csv" \
  --images    "$PROJECT_DIR/data/train/images" \
  --out       "$OUT" \
  --calib     "$CALIB" \
  --group_size "$GROUP" \
  --min_pixels_factor "$MIN_PX" \
  --max_pixels_factor "$MAX_PX"

echo ""
echo "=== AWQ W4A16 done -> $OUT ==="
du -sh "$OUT" || true
echo "vs bf16 merged:"
du -sh "$MERGED" || true
echo "quant config check:"
grep -A3 quantization_config "$OUT/config.json" || echo "!! NO quantization_config"