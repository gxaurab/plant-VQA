#!/bin/bash
#SBATCH --job-name=qwen3vl_plant_eval
#SBATCH --output=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.out
#SBATCH --error=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00                # eval is short; safe to lower (e.g. 02:00:00)
#SBATCH --partition=gpu-a100
#SBATCH --account=research-abe
#SBATCH --chdir=/scratch/ptolemy/users/ba974/weedAI/plant-VQA
# =============================================================================
# Qwen3-VL-2B LoRA -- evaluate a trained adapter on a test split.
#   sbatch scripts/eval.bash
# Requires: rouge_score bert_score tqdm qwen_vl_utils  (in the venv)
# =============================================================================
 
SCRATCH_DIR="/scratch/ptolemy/users/ba974"
PROJECT_DIR="$SCRATCH_DIR/weedAI/plant-VQA"
VENV_PATH="$SCRATCH_DIR/weedAI/.venvWeedAI"
MODEL_DIR="$SCRATCH_DIR/weedAI/qwen3VL2BInstruct"      # local base model (offline)
ROBERTA_DIR="$SCRATCH_DIR/weedAI/roberta-large"        # local BERTScore model (offline)
ADAPTER="$SCRATCH_DIR/weedAI/plant-VQA/outputs/best_adapter"
 
[ -f "$PROJECT_DIR/.env" ] && { set -a; source "$PROJECT_DIR/.env"; set +a; }
 
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
 
module purge
module load python/3.10.8
module load cuda/12.1.1
source "$VENV_PATH/bin/activate"
 
mkdir -p logs
echo " host: $(hostname)   job: ${SLURM_JOB_ID:-<none>}   CVD=${CUDA_VISIBLE_DEVICES:-<none>}"
nvidia-smi -L || { echo "FATAL: no GPU visible -- did you sbatch it?"; exit 1; }
 
# --- TODO: set your actual test split + image dir below ---
python eval.py \
  --adapter "$ADAPTER" \
  --model_id "$MODEL_DIR" \
  --bertscore_model "$ROBERTA_DIR" \
  --test_csv "$PROJECT_DIR/data/test3k/test_3k.csv" \
  --images  "$PROJECT_DIR/data/test3k/images/" \
  --out "$SCRATCH_DIR/weedAI/plant-VQA/outputs/eval_test.csv" \
  --batch_size 8 \
  --max_new_tokens 160
# add --no_bertscore for a quick run that skips BERTScore entirely
 
echo "=== eval done. predictions + scores written above. ==="