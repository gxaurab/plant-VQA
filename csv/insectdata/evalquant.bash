#!/bin/bash
#SBATCH --job-name=qwen3vl_insect_evalquant
#SBATCH --output=/scratch/ptolemy/users/ba974/InsectAI/logs/%x_%j.out
#SBATCH --error=/scratch/ptolemy/users/ba974/InsectAI/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --partition=gpu-a100
#SBATCH --account=research-abe
#SBATCH --chdir=/scratch/ptolemy/users/ba974/InsectAI
# =============================================================================
# Evaluate a QUANTIZED Qwen3-VL IP-VQA model (W8A16 or W4A16). Scheme-agnostic.
# Run ONCE per quantized model by swapping BITS below (w8a16 / w4a16).
#   sbatch scripts/eval_quant_merged.bash
# Use the SAME env that ran the bf16 eval_merged (torch 2.6.0+cu124) and that
# has compressed-tensors + llmcompressor installed.
# =============================================================================

SCRATCH_DIR="/scratch/ptolemy/users/ba974"
PROJECT_DIR="$SCRATCH_DIR/InsectAI"
VENV_PATH="$PROJECT_DIR/.venvQwenInsect"        # <-- the WORKING eval env (cu124 torch)
ROBERTA_DIR="$SCRATCH_DIR/weedAI/roberta-large"

# >>> SWITCH THIS between "w8a16" and "w4a16" for the two runs <<<
BITS="w8a16"
QUANT_MODEL="$PROJECT_DIR/outputs/qwen3vl/merged-${BITS}"
OUT_CSV="$PROJECT_DIR/outputs/eval_${BITS}_withbert.csv"

[ -f "$PROJECT_DIR/.env" ] && { set -a; source "$PROJECT_DIR/.env"; set +a; }

export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module purge
module load python/3.10.8
# NO cuda module -- let cu124 torch use its bundled runtime (fixes the invalid-argument bug)
source "$VENV_PATH/bin/activate"

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| GPU', torch.cuda.get_device_name(0))"

mkdir -p logs
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-<none>}  BITS=$BITS"
nvidia-smi -L || { echo "FATAL: no GPU"; exit 1; }
[ -d "$QUANT_MODEL" ] || { echo "FATAL: quantized model not found at $QUANT_MODEL"; exit 1; }

python eval_quant_merged.py \
  --model    "$QUANT_MODEL" \
  --test_csv "$PROJECT_DIR/data/test/test.csv" \
  --images   "$PROJECT_DIR/data/test/images/" \
  --out      "$OUT_CSV" \
  --bertscore_model "$ROBERTA_DIR" \
  --min_pixels_factor 128 \
  --max_pixels_factor 256 \
  --batch_size 8 \
  --max_new_tokens 160

echo "=== ${BITS} eval done -> $OUT_CSV ==="