#!/bin/bash
#SBATCH --job-name=qwen3vl_w8a16
#SBATCH --output=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.out
#SBATCH --error=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100
#SBATCH --account=research-abe
#SBATCH --chdir=/scratch/ptolemy/users/ba974/weedAI/plant-VQA
# =============================================================================
# W8A16 weight-only INT8 (RTN). No calibration data needed -> fast, crash-free.
#   sbatch scripts/quantize_w8a16.bash
# =============================================================================

set -euo pipefail

SCRATCH_DIR="/scratch/ptolemy/users/ba974"
QENV="$SCRATCH_DIR/weedAI/.venvQuant"
MERGED="$SCRATCH_DIR/weedAI/plant-VQA/outputs/qwen3vl/merged"
OUT="$SCRATCH_DIR/weedAI/plant-VQA/outputs/qwen3vl/merged-w8a16"

export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module purge
module load python/3.10.8
module load cuda/12.1.1
source "$QENV/bin/activate"

mkdir -p logs
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-<none>}"
nvidia-smi -L || { echo "FATAL: no GPU"; exit 1; }
[ -d "$MERGED" ] || { echo "FATAL: merged model not found at $MERGED"; exit 1; }

python quantize/quantize_w8a16_rtn.py \
  --model "$MERGED" \
  --out   "$OUT"

echo ""
echo "=== W8A16 done -> $OUT ==="
echo "sizes (int8 should be ~half of bf16):"
du -sh "$OUT" || true
du -sh "$MERGED" || true
echo ""
echo "quant config check (should print a block):"
grep -A3 quantization_config "$OUT/config.json" || echo "!! NO quantization_config - quantization did not persist"