#!/bin/bash
#SBATCH --job-name=qwen3vl_plant_merge
#SBATCH --output=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.out
#SBATCH --error=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00                 # merge + verify is short
#SBATCH --partition=gpu-a100
#SBATCH --account=research-abe
#SBATCH --chdir=/scratch/ptolemy/users/ba974/weedAI/plant-VQA
# =============================================================================
# Qwen3-VL-2B -- merge LoRA adapter into base, verify, save merged checkpoint.
#   sbatch scripts/merge.bash
# Produces a standalone (no-PEFT) bf16 checkpoint for quantization + Jetson.
# =============================================================================

SCRATCH_DIR="/scratch/ptolemy/users/ba974"
PROJECT_DIR="$SCRATCH_DIR/weedAI/plant-VQA"
VENV_PATH="$SCRATCH_DIR/weedAI/.venvWeedAI"
MODEL_DIR="$SCRATCH_DIR/weedAI/qwen3VL2BInstruct"      # local base model (offline)
ADAPTER="$PROJECT_DIR/outputs/best_adapter"
MERGED_OUT="$PROJECT_DIR/outputs/qwen3vl/merged"

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
echo " host: $(hostname)   job: ${SLURM_JOB_ID:-<none>}   CVD=${CUDA_VISIBLE_DEVICES:-<none>}"
nvidia-smi -L || { echo "FATAL: no GPU visible -- did you sbatch it?"; exit 1; }

# --- merge + verify --------------------------------------------------------
# --model_id points at the LOCAL base model dir (offline), matching eval.bash.
# --adapter / --out are passed explicitly so we don't depend on config paths.
# --val_csv + --val_images are the verification slice; set to your real files.
python merge_and_verify.py \
  --config "$PROJECT_DIR/configs/train.yaml" \
  --model_id "$MODEL_DIR" \
  --adapter  "$ADAPTER" \
  --out      "$MERGED_OUT" \
  --val_csv  "$PROJECT_DIR/data/val/val.csv" \
  --val_images "$PROJECT_DIR/data/val/images" \
  --n-eval 300

echo "=== merge done. merged checkpoint written to: $MERGED_OUT ==="