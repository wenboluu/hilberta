#!/bin/bash
# Usage: bash submit_flux2_clear_eval.sh <num_gpus> [total] [steps] [window_size] [checkpoint] [output_dir]
# Example: bash submit_flux2_clear_eval.sh 5 5000 10 8 CLEAR/exp_output_w8/checkpoint-3000 output/flux2_clear_eval_w8_ckpt3000

mkdir -p logs

NUM_GPUS=${1:?Usage: bash submit_flux2_clear_eval.sh <num_gpus> [total] [steps] [window_size] [checkpoint] [output_dir]}
TOTAL=${2:-5000}
NUM_STEPS=${3:-10}
WINDOW_SIZE=${4:-8}
CHECKPOINT=${5:-"CLEAR/exp_output_w8/checkpoint-3000"}
OUT_DIR=${6:-"output/flux2_clear_eval_w8_ckpt3000"}
PER_GPU=$((TOTAL / NUM_GPUS))

echo "Submitting ${NUM_GPUS} jobs, ${PER_GPU} samples each, ${TOTAL} total"
echo "Config: steps=${NUM_STEPS}, window_size=${WINDOW_SIZE}, checkpoint=${CHECKPOINT}"
echo "Output: ${OUT_DIR}"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    START=$((i * PER_GPU))
    END=$(( (i + 1) * PER_GPU ))
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        END=${TOTAL}
    fi
    echo "Job $((i+1)): samples ${START}-${END}"
    sbatch run_evaluation_flux2_clear.sbatch ${START} ${END} ${NUM_STEPS} ${WINDOW_SIZE} ${CHECKPOINT} ${OUT_DIR}
done
