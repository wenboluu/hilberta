#!/bin/bash
#SBATCH --job-name=flux2_clear
#SBATCH --output=logs/flux2_clear_%j.out
#SBATCH --error=logs/flux2_clear_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=3:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=sz3684@nyu.edu
#SBATCH --account=torch_pr_637_tandon_advanced

START=${1:-0}
END=${2:-5000}
NUM_STEPS=${3:-10}
WINDOW_SIZE=${4:-8}
CHECKPOINT=${5:-"CLEAR/exp_output_w8/checkpoint-3000"}
OUT_DIR=${6:-"output/flux2_clear_eval_w8_ckpt3000"}

echo "=== FLUX.2-klein CLEAR Evaluation ==="
echo "Prompt range: ${START} to ${END}"
echo "Steps: ${NUM_STEPS}, window_size=${WINDOW_SIZE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output: ${OUT_DIR}"
echo "Job ID: ${SLURM_JOB_ID}, Node: $(hostname)"
echo ""

cd /scratch/sz3684/HilbertA/qwen_image
mkdir -p logs

source /scratch/sz3684/miniconda3/etc/profile.d/conda.sh
conda activate hilberta

python run_evaluation_flux2_clear.py \
    --start ${START} \
    --end ${END} \
    --prompt_file coco_prompts.json \
    --output_dir ${OUT_DIR} \
    --num_inference_steps ${NUM_STEPS} \
    --seed 42 \
    --height 1024 \
    --width 1024 \
    --batch_size 1 \
    --window_size ${WINDOW_SIZE} \
    --checkpoint ${CHECKPOINT}
