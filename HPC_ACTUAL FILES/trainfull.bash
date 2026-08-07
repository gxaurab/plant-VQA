#!/bin/bash
#SBATCH --job-name=qwen3vl_plant_12k
#SBATCH --output=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.out
#SBATCH --error=/scratch/ptolemy/users/ba974/weedAI/plant-VQA/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --partition=gpu-a100
#SBATCH --account=research-abe
#SBATCH --chdir=/scratch/ptolemy/users/ba974/weedAI/plant-VQA
# =============================================================================
# Qwen3-VL-2B-Instruct  |  LoRA (PEFT)  |  PlantExpertVQA
# SBATCH resources (cpus=16, time=20h, gpu:1, gpu-a100, account) taken from the
# known-good reference. --chdir puts the job in plant-VQA so configs/ and logs/
# resolve, even though this script lives in weedAI/scripts/.
# All hyperparameters live in configs/train.yaml.
#   sbatch scripts/trainfull.bash
# =============================================================================
 
SCRATCH_DIR="/scratch/ptolemy/users/ba974"
PROJECT_DIR="$SCRATCH_DIR/weedAI/plant-VQA"
VENV_PATH="$SCRATCH_DIR/weedAI/.venvWeedAI"
OUTPUT_DIR="$SCRATCH_DIR/outputs/qwen3vl-2b-plant-12k"
 
# load .env if present
[ -f "$PROJECT_DIR/.env" ] && { set -a; source "$PROJECT_DIR/.env"; set +a; }
 
# offline (GPU node has no internet) -- needed for local model/dataset loading
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
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'sees', torch.cuda.device_count(), 'GPU(s)')" || exit 1
 
# all hyperparameters come from configs/train.yaml (add --resume to continue)
python train.py --config configs/train.yaml
 
echo "=== done. adapter: ==="
ls -la "$OUTPUT_DIR/best_adapter/" 2>/dev/null || echo "  (none -- check logs/*.err)"