#!/bin/bash

# Set error handling
set -e

# Activate conda environment (if you're using conda)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate diffusion

# Respect existing CUDA device selection; do not hardcode GPU index
: "${CUDA_VISIBLE_DEVICES:=}"

# Set the base directory
BASE_DIR="/home/sz3684/diffusion/reorder_local_attention/HilbertA/reloc_attention"

# Default config path
CONFIG_PATH="${BASE_DIR}/config.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Determine curve type and mask dir from config
CURVE_TYPE="$(${HOME}/miniconda3/envs/flux_img_editing/bin/python -c 'import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print((cfg.get("curve_type") or "hilbert").lower())' "$CONFIG_PATH")"
MASK_DIR="${BASE_DIR}/mask_${CURVE_TYPE}"

# Check if mask_{curve_type} directory exists and create masks if needed
if [ ! -d "${MASK_DIR}" ] || [ -z "$(ls -A ${MASK_DIR} 2>/dev/null)" ]; then
    echo "Mask dir ${MASK_DIR} (curve=${CURVE_TYPE}) not found or empty, creating masks..."
    ${HOME}/miniconda3/envs/flux_img_editing/bin/python "${BASE_DIR}/create_mask.py"
else
    echo "Mask dir ${MASK_DIR} (curve=${CURVE_TYPE}) exists and contains files, skipping mask creation"
fi

# # Always create masks
# echo "Creating masks..."
# ~/miniconda3/envs/flux_img_editing/bin/python "${BASE_DIR}/pattern_utils.py"
# ~/miniconda3/envs/flux_img_editing/bin/python "${BASE_DIR}/create_mask.py"

# Run the script with timestamp
echo "Starting run_flux.py"
echo "Using config file: $CONFIG_PATH"

# Run the Python script with the project environment's Python
~/miniconda3/envs/flux_img_editing/bin/python "${BASE_DIR}/run_flux.py" \
    --config "$CONFIG_PATH"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "run_flux.py completed successfully"
else
    echo "Error: run_flux.py failed"
    exit 1
fi 