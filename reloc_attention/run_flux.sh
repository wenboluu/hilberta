#!/bin/bash

# Set error handling
set -e

# Activate conda environment (if you're using conda)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate diffusion

# Respect existing CUDA device selection; do not hardcode GPU index
: "${CUDA_VISIBLE_DEVICES:=}"

# Set the base directory
BASE_DIR="/scratch/sz3684/HilbertA/reorder_local_attention/reloc_attention/"

# Default config path
CONFIG_PATH="${BASE_DIR}/config.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Check if mask_list directory exists and create masks if needed
# if [ ! -d "${BASE_DIR}/mask_list" ] || [ -z "$(ls -A ${BASE_DIR}/mask_list 2>/dev/null)" ]; then
#     echo "mask_list directory not found or empty, creating masks..."
#     "${BASE_DIR}/penv/bin/python" "${BASE_DIR}/create_mask.py"
# else
#     echo "mask_list directory exists and contains files, skipping mask creation"
# fi

# Always create masks
echo "Creating masks..."
"${BASE_DIR}/penv/bin/python" "${BASE_DIR}/pattern_utils.py"
"${BASE_DIR}/penv/bin/python" "${BASE_DIR}/create_mask.py"

# Run the script with timestamp
echo "Starting run_flux.py"
echo "Using config file: $CONFIG_PATH"

# Run the Python script with the project environment's Python
"${BASE_DIR}/penv/bin/python" "${BASE_DIR}/run_flux.py" \
    --config "$CONFIG_PATH"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "run_flux.py completed successfully"
else
    echo "Error: run_flux.py failed"
    exit 1
fi 