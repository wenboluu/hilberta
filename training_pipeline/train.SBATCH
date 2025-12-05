#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --job-name=Flux
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=tandon_a100_1,tandon_a100_2,stake_a100_1,stake_a100_2
#SBATCH --account=pr_133_tandon_priority
#SBATCH --mem=120G
#SBATCH --requeue

#SBATCH --mail-type=ALL
#SBATCH --mail-user=sz3684

module purge

TARGET_PATH=$(dirname "$(realpath "$0")")

# Activate the environment and run the training script
conda activate /scratch/sz3684/ICML/flux_ICML_rebuttal/penv
cd $TARGET_PATH
bash train.sh