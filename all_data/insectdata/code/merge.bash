#!/bin/bash
#SBATCH --job-name=qwen3vl_insect_merge
#SBATCH --output=/scratch/ptolemy/users/ba974/InsectAI/logs/%x_%j.out
#SBATCH --error=/scratch/ptolemy/users/ba974/InsectAI/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100
#SBATCH --account=research-abe
#SBATCH --chdir=/scratch/ptolemy/users/ba974/InsectAI
# =============================================================================
# Qwen3-VL-2B (IP-VQA) -- merge LoRA adapter into base. No eval (merge only).
#   sbatch scripts/merge_only.bash
# Merge does no forward pass -> cannot hit the vision rot_pos_emb CUDA error.
# =============================================================================

SCRATCH_DIR="/scratch/ptolemy/users/ba974"
PROJECT_DIR="$SCRATCH_DIR/InsectAI"
VENV_PATH="$PROJECT_DIR/.venvQwenInsect"
MODEL_DIR="$SCRATCH_DIR/weedAI/qwen3VL2BInstruct"          # local base model (offline)
ADAPTER="$PROJECT_DIR/outputs/best_adapter"
MERGED_OUT="$PROJECT_DIR/outputs/merged"

[ -f "$PROJECT_DIR/.env" ] && { set -a; source "$PROJECT_DIR/.env"; set +a; }

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
echo " host: $(hostname)   job: ${SLURM_JOB_ID:-<none>}"
nvidia-smi -L || { echo "FATAL: no GPU visible"; exit 1; }
[ -d "$MODEL_DIR" ] || { echo "FATAL: base model not found at $MODEL_DIR"; exit 1; }
[ -d "$ADAPTER" ]   || { echo "FATAL: adapter not found at $ADAPTER"; exit 1; }

# Try safetensors first; the script auto-falls-back to .bin if tied tensors error.
python merge_only.py \
  --model_id "$MODEL_DIR" \
  --adapter  "$ADAPTER" \
  --out      "$MERGED_OUT" \
  --safetensors

echo ""
echo "=== merge done -> $MERGED_OUT ==="
ls -la "$MERGED_OUT"
du -sh "$MERGED_OUT" 2>/dev/null || true