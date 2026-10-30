#!/bin/bash
#SBATCH --job-name=qwen3vl_quant_w4a16
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
# Quantize the merged bf16 Qwen3-VL-2B to W4A16 (INT4, RTN, data-free).
#   sbatch scripts/quantize_w4a16.bash
# Fast: weight-only RTN does no forward pass. Uses .venvQuant (llmcompressor).
# =============================================================================

SCRATCH_DIR="/scratch/ptolemy/users/ba974"
PROJECT_DIR="$SCRATCH_DIR/weedAI/plant-VQA"
VENV_PATH="$SCRATCH_DIR/weedAI/.venvQuant"                  # env with llmcompressor

# >>> CHECK THESE TWO PATHS <<<
MERGED_BF16="$PROJECT_DIR/outputs/qwen3vl/merged-bf16"      # SOURCE: your merged bf16 model
OUT_W4A16="$PROJECT_DIR/outputs/qwen3vl/merged-w4a16"       # DEST: where INT4 model is written

[ -f "$PROJECT_DIR/.env" ] && { set -a; source "$PROJECT_DIR/.env"; set +a; }

export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module purge
module load python/3.10.8
# module load cuda/12.1.1
source "$VENV_PATH/bin/activate"

python -c "import torch; print('torch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"

mkdir -p logs
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-<none>}"
nvidia-smi -L || { echo "FATAL: no GPU"; exit 1; }
[ -d "$MERGED_BF16" ] || { echo "FATAL: merged bf16 model not found at $MERGED_BF16"; exit 1; }

python quantize_w4a16_rtn.py \
  --model "$MERGED_BF16" \
  --out   "$OUT_W4A16" \
  --group_size 128

echo "=== W4A16 quantization done. Check config.json for num_bits=4, group_size=128 ==="
ls -la "$OUT_W4A16"