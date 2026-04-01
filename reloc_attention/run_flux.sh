#!/bin/bash

# Set error handling
set -e

# Activate conda environment (if you're using conda)
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate diffusion

# Respect existing CUDA device selection; if not set, use GPU 0 by default
if [ -z "${CUDA_VISIBLE_DEVICES+x}" ]; then
    export CUDA_VISIBLE_DEVICES=0
    echo "CUDA_VISIBLE_DEVICES not set, defaulting to GPU 0"
fi

# Set the base directory to the script's location
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Default config path
CONFIG_PATH="${BASE_DIR}/config.yaml"

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found at $CONFIG_PATH"
    exit 1
fi

# Use local penv Python if available, otherwise fall back to system python
if [ -f "${BASE_DIR}/penv/bin/python" ]; then
    PYTHON="${BASE_DIR}/penv/bin/python"
else
    PYTHON="python"
fi

# Determine curve type and mask dir from config
CURVE_TYPE="$($PYTHON -c 'import yaml,sys; cfg=yaml.safe_load(open(sys.argv[1])); print((cfg.get("curve_type") or "hilbert").lower())' "$CONFIG_PATH")"
MASK_DIR="${BASE_DIR}/mask_${CURVE_TYPE}"

# Check if mask_{curve_type} directory exists and create masks if needed
if [ ! -d "${MASK_DIR}" ] || [ -z "$(ls -A ${MASK_DIR} 2>/dev/null)" ]; then
    echo "Mask dir ${MASK_DIR} (curve=${CURVE_TYPE}) not found or empty, creating masks..."
    $PYTHON "${BASE_DIR}/create_mask.py"
else
    echo "Mask dir ${MASK_DIR} (curve=${CURVE_TYPE}) exists and contains files, skipping mask creation"
fi

# Run the script with timestamp
echo "Starting run_flux.py"
echo "Using config file: $CONFIG_PATH"

# Run the Python script with the project environment's Python
$PYTHON "${BASE_DIR}/run_flux.py" \
    --config "$CONFIG_PATH"

# Check if the script ran successfully
if [ $? -eq 0 ]; then
    echo "run_flux.py completed successfully"
else
    echo "Error: run_flux.py failed"
    exit 1
fi 