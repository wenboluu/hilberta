#!/bin/bash
# Usage: bash submit_flux2_sparge_eval.sh <num_gpus> [total_samples] [num_steps] [simthreshd1] [cdfthreshd] [pvthreshd] [output_dir]
# Example: bash submit_flux2_sparge_eval.sh 5 5000 10 -0.8 0.06 0.0 output/flux2_sparge_eval

mkdir -p logs

NUM_GPUS=${1:?Usage: bash submit_flux2_sparge_eval.sh <num_gpus> [total] [steps] [simthreshd1] [cdfthreshd] [pvthreshd] [output_dir]}
TOTAL=${2:-5000}
NUM_STEPS=${3:-10}
SIMTHRESHD1=${4:--0.8}
CDFTHRESHD=${5:-0.06}
PVTHRESHD=${6:-0.0}
OUT_DIR=${7:-"output/flux2_sparge_eval"}
PER_GPU=$((TOTAL / NUM_GPUS))

echo "Submitting ${NUM_GPUS} jobs, ${PER_GPU} samples each, ${TOTAL} total"
echo "Config: steps=${NUM_STEPS}, simthreshd1=${SIMTHRESHD1}, cdfthreshd=${CDFTHRESHD}, pvthreshd=${PVTHRESHD}"
echo "Output: ${OUT_DIR}"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    START=$((i * PER_GPU))
    END=$(( (i + 1) * PER_GPU ))
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        END=${TOTAL}
    fi
    echo "Job $((i+1)): samples ${START}-${END}"
    sbatch run_evaluation_flux2_sparge.sbatch ${START} ${END} ${NUM_STEPS} ${SIMTHRESHD1} ${CDFTHRESHD} ${PVTHRESHD} ${OUT_DIR}
done
