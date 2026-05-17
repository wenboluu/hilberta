#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source /scratch/sz3684/miniconda3/etc/profile.d/conda.sh
conda activate hilberta

echo "=== FLUX.2-klein HilbertA Inference ==="
echo "Config: ${SCRIPT_DIR}/config.yaml"
echo ""

python run_flux2.py --config "${SCRIPT_DIR}/config.yaml" "$@"
