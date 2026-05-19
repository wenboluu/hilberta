#!/bin/bash
#SBATCH --job-name=flux2_distill
#SBATCH --output=logs/flux2_distill_%j.out
#SBATCH --error=logs/flux2_distill_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200GB
#SBATCH --time=12:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=sz3684@nyu.edu
#SBATCH --account=torch_pr_637_tandon_advanced

cd /scratch/sz3684/HilbertA/qwen_image/flux2_hilberta/training
mkdir -p logs

source /scratch/sz3684/miniconda3/etc/profile.d/conda.sh
conda activate hilberta

export NUM_GPUS=1
export OUTPUT_DIR="./exp_output_1gpu"
export HF_HUB_DOWNLOAD_TIMEOUT=60
bash train.sh