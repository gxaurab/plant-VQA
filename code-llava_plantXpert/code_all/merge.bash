#!/bin/bash
#SBATCH --job-name=llava15_merge
#SBATCH --output=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.out
#SBATCH --error=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --partition=gpu-a100
#SBATCH --account=research-abe
# =============================================================================
# Merge LLaVA-1.5-7B LoRA adapter into base -> standalone FP16 HF model.
#
# GPU node requested for its larger system RAM (holds 7B FP16 + adapter on CPU).
# The merge itself is CPU-side (merge_llava.py uses device_map="cpu"); the GPU
# sits idle. Short wall-time — this is minutes, not hours.
#
#   sbatch merge_llava.bash
#   tail -f logs/llava15_merge_<jobid>.out
# =============================================================================

SCRATCH_DIR="/scratch/ptolemy/users/ba974"
VENV_PATH="$SCRATCH_DIR/weedAI/.venvLlava"
PROJECT_DIR="$SCRATCH_DIR/weedAI/plant-VQA/llava"

# --- paths (edit to match your tree) ---
BASE_DIR="$SCRATCH_DIR/weedAI/llava-1.5-7b-hf"
ADAPTER_DIR="$SCRATCH_DIR/weedAI/plant-VQA/outputs/llava-1.5-7b-plant-12k/best_adapter"
OUT_DIR="$PROJECT_DIR/outputs/llava-1.5-7b-plant-merged"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module purge
module load python/3.10.8
module load cuda/12.1.1
source "$VENV_PATH/bin/activate"

cd "$PROJECT_DIR"
mkdir -p logs

echo "=== LLaVA-1.5-7B  LoRA merge ==="
echo " host : $(hostname)   job: ${SLURM_JOB_ID:-<none>}"
nvidia-smi -L || echo "  (no GPU visible — merge is CPU-side, so this is non-fatal)"
python -c "import torch; print('torch', torch.__version__)" || exit 1

# --- inputs must exist locally (offline node) ---
test -f "$BASE_DIR/config.json"       || { echo "FATAL: no base config.json at $BASE_DIR"; exit 1; }
test -d "$ADAPTER_DIR"                || { echo "FATAL: no adapter dir at $ADAPTER_DIR"; exit 1; }
test -f "$ADAPTER_DIR/adapter_config.json" || { echo "FATAL: $ADAPTER_DIR has no adapter_config.json — wrong path?"; exit 1; }
echo " base    : $BASE_DIR"
echo " adapter : $ADAPTER_DIR"
echo " out     : $OUT_DIR"
echo "================================"

python merge_llava.py \
    --base    "$BASE_DIR" \
    --adapter "$ADAPTER_DIR" \
    --out     "$OUT_DIR"

echo
echo "=== merge done. contents: ==="
ls -la "$OUT_DIR" 2>/dev/null || echo "  (nothing written — investigate)"
echo "Next: eval the merged model (eval_llava.py)."