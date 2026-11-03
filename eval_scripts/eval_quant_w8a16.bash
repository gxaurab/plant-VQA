#!/bin/bash
#SBATCH --job-name=qwen3vl_eval_int8
#SBATCH --output=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.out
#SBATCH --error=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=gpu-a100
#SBATCH --account=research-abe
#SBATCH --chdir=/scratch/ptolemy/users/ba974/weedAI/plant-VQA
# =============================================================================
# Evaluate the QUANTIZED INT8 model. Uses .venvQuant (where it loads correctly).
#   sbatch scripts/eval_quant.bash
# =============================================================================

SCRATCH_DIR="/scratch/ptolemy/users/ba974"
PROJECT_DIR="$SCRATCH_DIR/weedAI/plant-VQA"
VENV_PATH="$SCRATCH_DIR/weedAI/.venvQuant"                 # the env that loads INT8
QUANT_MODEL="$PROJECT_DIR/outputs/qwen3vl/merged-w8a16"
ROBERTA_DIR="$SCRATCH_DIR/weedAI/roberta-large"

[ -f "$PROJECT_DIR/.env" ] && { set -a; source "$PROJECT_DIR/.env"; set +a; }

export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

module purge
module load python/3.10.8
# module load cuda/12.1.1
source "$VENV_PATH/bin/activate"

echo "CUDA_HOME=$CUDA_HOME"
which nvcc || true
python -c "import torch; print('torch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0))"

mkdir -p logs
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-<none>}"
nvidia-smi -L || { echo "FATAL: no GPU"; exit 1; }
[ -d "$QUANT_MODEL" ] || { echo "FATAL: quantized model not found at $QUANT_MODEL"; exit 1; }

python eval_quant.py \
  --model    "$QUANT_MODEL" \
  --test_csv "$PROJECT_DIR/data/test3k/test_3k.csv" \
  --images   "$PROJECT_DIR/data/test3k/images/" \
  --out      "$PROJECT_DIR/outputs/eval_int8_withbert.csv" \
  --bertscore_model "$ROBERTA_DIR" \
  --min_pixels_factor 128 \
  --max_pixels_factor 256 \
  --batch_size 8 \
  --max_new_tokens 160 
 

echo "=== INT8 eval done ==="