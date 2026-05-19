#!/bin/bash
# Usage: bash submit_flux2_eval.sh <num_gpus> [total_samples] [num_steps] [output_dir]
# Example: bash submit_flux2_eval.sh 5
#          bash submit_flux2_eval.sh 5 5000 10 output/flux2_eval_10steps

mkdir -p logs

NUM_GPUS=${1:?Usage: bash submit_flux2_eval.sh <num_gpus> [total_samples] [num_steps] [output_dir]}
TOTAL=${2:-5000}
NUM_STEPS=${3:-4}
OUT_DIR=${4:-"output/flux2_eval"}
PER_GPU=$((TOTAL / NUM_GPUS))

echo "Submitting ${NUM_GPUS} jobs, ${PER_GPU} samples each, ${TOTAL} total"
echo "Inference steps: ${NUM_STEPS}, Output: ${OUT_DIR}"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    START=$((i * PER_GPU))
    END=$(( (i + 1) * PER_GPU ))
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        END=${TOTAL}
    fi
    echo "Job $((i+1)): samples ${START}-${END}"
    sbatch run_evaluation_flux2.sbatch ${START} ${END} ${NUM_STEPS} ${OUT_DIR}
done
