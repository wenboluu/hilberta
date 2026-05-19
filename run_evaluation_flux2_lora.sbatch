#!/bin/bash
#SBATCH --job-name=flux2_lora_eval
#SBATCH --output=logs/flux2_lora_eval_%j.out
#SBATCH --error=logs/flux2_lora_eval_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=3:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=sz3684@nyu.edu
#SBATCH --account=torch_pr_637_tandon_advanced

# ── Arguments ──
START=${1:-0}
END=${2:-5000}
LORA_DIR=${3:-"flux2_hilberta/training/exp_output/checkpoint-1000"}
NUM_STEPS=${4:-10}
OUT_DIR=${5:-"output/flux2_lora_eval_$(basename ${LORA_DIR})_${NUM_STEPS}steps"}

echo "=== FLUX.2-klein + LoRA + HilbertA Evaluation ==="
echo "Prompt range: ${START} to ${END}"
echo "LoRA dir: ${LORA_DIR}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

cd /scratch/sz3684/HilbertA/qwen_image
mkdir -p logs

source /scratch/sz3684/miniconda3/etc/profile.d/conda.sh
conda activate hilberta

python run_evaluation_flux2_lora.py \
    --start ${START} \
    --end ${END} \
    --lora_dir ${LORA_DIR} \
    --prompt_file coco_prompts.json \
    --output_dir ${OUT_DIR} \
    --num_inference_steps ${NUM_STEPS} \
    --seed 42 \
    --height 1024 \
    --width 1024 \
    --batch_size 16
