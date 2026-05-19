#!/bin/bash
#SBATCH --job-name=clear_distill
#SBATCH --output=logs/clear_distill_%j.out
#SBATCH --error=logs/clear_distill_%j.err
#SBATCH --gres=gpu:2
#SBATCH --constraint="a100|h100|h200"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=sz3684@nyu.edu
#SBATCH --account=torch_pr_637_tandon_advanced

cd /scratch/sz3684/HilbertA/qwen_image/CLEAR
mkdir -p logs

source /scratch/sz3684/miniconda3/etc/profile.d/conda.sh
conda activate hilberta

export NUM_GPUS=2
export WINDOW_SIZE=8
export OUTPUT_DIR="./exp_output_w8"
export HF_HUB_DOWNLOAD_TIMEOUT=60
export RESUME_CKPT="${RESUME_CKPT:-}"
bash distill_flux2.sh
