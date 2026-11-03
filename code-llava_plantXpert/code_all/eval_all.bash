#!/bin/bash
#SBATCH --job-name=llava15_eval_all
#SBATCH --output=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.out
#SBATCH --error=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100
#SBATCH --account=research-abe
#SBATCH --chdir=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/llava
# =============================================================================
# Evaluate FP16 baseline + W8A16 + W4A16 LLaVA-1.5 back-to-back, IDENTICAL
# config, so the compression comparison is apples-to-apples.
#   sbatch llava/eval_llava_all.bash
# Runs in .venvLlava (the eval env: transformers 4.49 + rouge/bert_score).
# The quantized folders were CREATED in .venvQuant but are EVALUATED here --
# they're just compressed-tensors dirs on disk; the eval env loads them fine
# as long as compressed-tensors is importable here (see the guard below).
# =============================================================================

SCRATCH_DIR="/scratch/ptolemy/users/ba974"
PROJECT_DIR="$SCRATCH_DIR/weedAI/plant-VQA"
VENV_PATH="$SCRATCH_DIR/weedAI/.venvQuant"          # EVAL env
ROBERTA_DIR="$SCRATCH_DIR/weedAI/roberta-large"

TEST_CSV="$PROJECT_DIR/data/test3k/test_3k.csv"
TEST_IMAGES="$PROJECT_DIR/data/test3k/images/"

# model_dir : output_csv  (space-separated pairs)
MODELS=(
  "$PROJECT_DIR/llava/outputs/llava-1.5-7b-plant-W4A16:$PROJECT_DIR/llava/outputs/eval_w4a16.csv"
)

[ -f "$PROJECT_DIR/.env" ] && { set -a; source "$PROJECT_DIR/.env"; set +a; }
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module purge
module load python/3.10.8
source "$VENV_PATH/bin/activate"

mkdir -p logs "$PROJECT_DIR/llava/outputs"
echo "host: $(hostname)  job: ${SLURM_JOB_ID:-<none>}"
nvidia-smi -L || { echo "FATAL: no GPU"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)" || exit 1
# compressed-tensors must be importable in the EVAL env to load quantized dirs
python -c "import compressed_tensors; print('compressed_tensors', compressed_tensors.__version__)" \
  || { echo "FATAL: compressed_tensors not in eval venv -- pip install compressed-tensors into .venvLlava"; exit 1; }

for pair in "${MODELS[@]}"; do
  MODEL_DIR="${pair%%:*}"
  OUT_CSV="${pair##*:}"
  echo ""
  echo "######################################################################"
  echo "# EVAL: $MODEL_DIR"
  echo "#   -> $OUT_CSV"
  echo "######################################################################"
  if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "[skip] no model at $MODEL_DIR"
    continue
  fi
  python eval_llava.py \
    --model "$MODEL_DIR" \
    --bertscore_model "$ROBERTA_DIR" \
    --test_csv "$TEST_CSV" \
    --images "$TEST_IMAGES" \
    --out "$OUT_CSV" \
    --batch_size 16 \
    --max_new_tokens 160
done

echo ""
echo "=== ALL EVALS DONE. Summary: ==="
python - <<'PY'
import pandas as pd, os
base = "/scratch/ptolemy/users/ba974/weedAI/plant-VQA/llava/outputs"
for tag, f in [("FP16","eval_fp16.csv"),("W8A16","eval_w8a16.csv"),("W4A16","eval_w4a16.csv")]:
    p = os.path.join(base, f)
    if not os.path.exists(p):
        print(f"{tag:6s}  (missing)"); continue
    d = pd.read_csv(p)
    cols = [c for c in ["rougeL","token_f1","exact_match","bertscore_f1"] if c in d.columns]
    m = d[cols].mean()
    print(f"{tag:6s}  " + "  ".join(f"{c}={m[c]:.2f}" for c in cols) + f"  (n={len(d)})")
PY