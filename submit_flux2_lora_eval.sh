#!/bin/bash
# Usage: bash submit_flux2_lora_eval.sh <num_gpus> [total_samples] [lora_dir] [num_steps] [output_dir]
# Example: bash submit_flux2_lora_eval.sh 5
#          bash submit_flux2_lora_eval.sh 5 5000 flux2_hilberta/training/exp_output/checkpoint-1000 10
#          bash submit_flux2_lora_eval.sh 5 5000 flux2_hilberta/training/exp_output/checkpoint-2500 10 output/flux2_lora_eval_corrected

mkdir -p logs

NUM_GPUS=${1:?Usage: bash submit_flux2_lora_eval.sh <num_gpus> [total_samples] [lora_dir] [num_steps] [output_dir]}
TOTAL=${2:-5000}
LORA_DIR=${3:-"flux2_hilberta/training/exp_output"}
NUM_STEPS=${4:-10}
OUT_DIR=${5:-""}
PER_GPU=$((TOTAL / NUM_GPUS))

echo "Submitting ${NUM_GPUS} jobs, ${PER_GPU} samples each, ${TOTAL} total"
echo "LoRA dir: ${LORA_DIR}"
echo "Inference steps: ${NUM_STEPS}"

for i in $(seq 0 $((NUM_GPUS - 1))); do
    START=$((i * PER_GPU))
    END=$(( (i + 1) * PER_GPU ))
    if [ $i -eq $((NUM_GPUS - 1)) ]; then
        END=${TOTAL}
    fi
    echo "Job $((i+1)): samples ${START}-${END}"
    if [ -n "${OUT_DIR}" ]; then
        sbatch run_evaluation_flux2_lora.sbatch ${START} ${END} ${LORA_DIR} ${NUM_STEPS} ${OUT_DIR}
    else
        sbatch run_evaluation_flux2_lora.sbatch ${START} ${END} ${LORA_DIR} ${NUM_STEPS}
    fi
done
