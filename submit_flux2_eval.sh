#!/bin/bash
# Usage: bash submit_flux2_eval.sh <num_gpus> [total_samples]
# Example: bash submit_flux2_eval.sh 5        # 5 GPUs, 5000 samples
#          bash submit_flux2_eval.sh 10 5000   # 10 GPUs, 5000 samples

mkdir -p logs

NUM_GPUS=${1:?Usage: bash submit_flux2_eval.sh <num_gpus> [total_samples]}
TOTAL=${2:-5000}
PER_GPU=$((TOTAL / NUM_GPUS))

echo "Submitting ${NUM_GPUS} jobs, ${PER_GPU} samples each, ${TOTAL} total"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    START=$((i * PER_GPU))
    END=$(( (i + 1) * PER_GPU ))
    # Last job picks up any remainder
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        END=${TOTAL}
    fi
    echo "Job $((i+1)): samples ${START}-${END}"
    sbatch run_evaluation_flux2.sbatch ${START} ${END}
done
