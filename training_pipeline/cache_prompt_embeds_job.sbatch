#!/bin/bash
#SBATCH --job-name=flux_cache_prompt
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --account=pr_133_tandon_priority
#SBATCH --output=logs/cache_prompt_%j.out
#SBATCH --error=logs/cache_prompt_%j.err

# Load shell env and conda
source ~/.bashrc
conda activate /scratch/sz3684/ICML/flux_ICML_rebuttal/penv

# Move to working directory (if needed)
cd /scratch/sz3684/reorder_local_attention/

# Run your script
bash training_pipeline/cache_prompt_embeds.sh
