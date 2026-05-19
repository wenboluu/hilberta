#!/bin/bash
#SBATCH --job-name=cache_prompt
#SBATCH --output=logs/cache_prompt_%j.out
#SBATCH --error=logs/cache_prompt_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|h100|h200"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64GB
#SBATCH --time=01:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=sz3684@nyu.edu
#SBATCH --account=torch_pr_637_tandon_advanced

cd /scratch/sz3684/HilbertA/qwen_image/flux2_hilberta/training
mkdir -p logs

source /scratch/sz3684/miniconda3/etc/profile.d/conda.sh
conda activate hilberta

bash cache_prompt_embeds.sh
