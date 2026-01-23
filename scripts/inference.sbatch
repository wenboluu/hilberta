#!/bin/bash
#SBATCH --job-name=benchmark
#SBATCH --output=/scratch/wl2707/research/hilbert_attention/reorder_local_attention/logs/%x_%j.out
#SBATCH --error=/scratch/wl2707/research/hilbert_attention/reorder_local_attention/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --account=pr_133_tandon_priority

# Load the necessary modules
module purge
module load anaconda3/2020.07
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh
conda activate /scratch/wl2707/research/hilbert_attention/hilbert_env
export PATH=/scratch/wl2707/research/hilbert_attention/hilbert_env/bin:$PATH
export PYTHONPATH=/scratch/wl2707/research/hilbert_attention/hilbert_env:$PYTHONPATH
conda init bash

# Go to the working directory
cd /scratch/wl2707/research/hilbert_attention/reorder_local_attention

# Ensure partition index is passed as an argument
if [ -z "$1" ]; then
    echo "Error: Partition index not provided."
    exit 1
fi

partition=$1
echo "Starting benchmarking partition=$partition"
python run_flux.py --partition $partition

echo "Job finished successfully!"
