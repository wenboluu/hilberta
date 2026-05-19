#!/bin/bash
#SBATCH --job-name=flux2_sparge
#SBATCH --output=logs/flux2_sparge_%j.out
#SBATCH --error=logs/flux2_sparge_%j.err
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=3:00:00
#SBATCH --mail-type=BEGIN
#SBATCH --mail-user=sz3684@nyu.edu
#SBATCH --account=torch_pr_637_tandon_advanced

START=${1:-0}
END=${2:-5000}
NUM_STEPS=${3:-10}
SIMTHRESHD1=${4:--0.8}
CDFTHRESHD=${5:-0.06}
PVTHRESHD=${6:-0.0}
OUT_DIR=${7:-"output/flux2_sparge_eval"}

echo "=== FLUX.2-klein SpargeAttn Evaluation ==="
echo "Prompt range: ${START} to ${END}"
echo "Steps: ${NUM_STEPS}, simthreshd1=${SIMTHRESHD1}, cdfthreshd=${CDFTHRESHD}, pvthreshd=${PVTHRESHD}"
echo "Output: ${OUT_DIR}"
echo "Job ID: ${SLURM_JOB_ID}, Node: $(hostname)"
echo ""

cd /scratch/sz3684/HilbertA/qwen_image
mkdir -p logs

source /scratch/sz3684/miniconda3/etc/profile.d/conda.sh
conda activate hilberta

export CUDA_HOME=$CONDA_PREFIX
export CPATH=$CONDA_PREFIX/targets/x86_64-linux/include:$CPATH

python run_evaluation_flux2_sparge.py \
    --start ${START} \
    --end ${END} \
    --prompt_file coco_prompts.json \
    --output_dir ${OUT_DIR} \
    --num_inference_steps ${NUM_STEPS} \
    --seed 42 \
    --height 1024 \
    --width 1024 \
    --batch_size 1 \
    --simthreshd1 ${SIMTHRESHD1} \
    --cdfthreshd ${CDFTHRESHD} \
    --pvthreshd ${PVTHRESHD}
