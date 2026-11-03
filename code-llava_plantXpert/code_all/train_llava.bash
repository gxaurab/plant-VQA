#!/bin/bash
#SBATCH --job-name=llava15_plant_lora
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --partition=gpu-a100
#SBATCH --account=research-abe
# =============================================================================
# LLaVA-1.5-7B-HF  —  LoRA finetune on PlantExpertVQA (12.6k-image subset)
#
#   base   : llava-hf/llava-1.5-7b-hf   (HF LlavaForConditionalGeneration)
#   LoRA   : language backbone only; vision_tower + multi_modal_projector FROZEN
#   data   : same CSV + image folders as the Qwen/Ministral runs (no JSON convert)
#
# One self-contained sbatch, ONE A100 (--gres=gpu:1).
#
# PRE-STAGE (once, login node): model already downloaded to HF cache. Verify with
#   ls $HF_HOME/hub/models--llava-hf--llava-1.5-7b-hf
#
#   sbatch train_llava.bash
#   tail -f logs/llava15_plant_lora_<jobid>.out
# =============================================================================

SCRATCH_DIR="/scratch/ptolemy/users/ba974"
VENV_PATH="$SCRATCH_DIR/projects/plant-VLM/.venvLlava"
PROJECT_DIR="$SCRATCH_DIR/projects/plant-VLM"

CONFIG="configs/train_llava.yaml"

export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export CUDA_LAUNCH_BLOCKING=0
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

module purge
module load python/3.10.8
module load cuda/12.1.1
source "$VENV_PATH/bin/activate"

cd "$PROJECT_DIR"
mkdir -p logs

echo "=== LLaVA-1.5-7B-HF  LoRA finetune (PlantExpertVQA) ==="
echo " host : $(hostname)   job: ${SLURM_JOB_ID:-<none>}   CVD=${CUDA_VISIBLE_DEVICES:-<none>}"
nvidia-smi -L || { echo "FATAL: no GPU visible — did you sbatch it?"; exit 1; }
python -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'sees', torch.cuda.device_count(), 'GPU(s)')" || exit 1
python -c "import transformers; print('transformers', transformers.__version__)"
echo " config : $CONFIG"
echo "======================================================="

# --- GPU stats logger (same pattern as the pi05 runs) ---
GPU_CSV="logs/${SLURM_JOB_NAME}_${SLURM_JOB_ID}_gpu.csv"
echo "timestamp,gpu_index,util_gpu_pct,mem_used_mib,mem_total_mib,power_w,temp_c" > "$GPU_CSV"
( while true; do
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu \
               --format=csv,noheader,nounits >> "$GPU_CSV" 2>/dev/null
    sleep 10
  done ) &
GPU_MON_PID=$!
trap "kill $GPU_MON_PID 2>/dev/null" EXIT

# --- verify LoRA hits language tower ONLY before spending GPU hours ---
echo
echo "=== LoRA target inspection (must show ZERO vision/projector hits) ==="
python train_llava.py --config "$CONFIG" --inspect || { echo "FATAL: inspect failed"; exit 1; }
echo "===================================================================="
echo

# --- train ---
python train_llava.py --config "$CONFIG"

echo
echo "=== training done. adapter + checkpoints: ==="
ls -la "outputs/llava-1.5-7b-plant-12k/best_adapter/" 2>/dev/null || echo "  (no adapter — investigate)"
ls -la "outputs/llava-1.5-7b-plant-12k/" 2>/dev/null
echo "Next: merge adapter, then eval."
