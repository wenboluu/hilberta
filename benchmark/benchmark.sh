#!/bin/bash
# Usage: bash benchmark/benchmark.sh <generated_dir> [baseline_dir]
#
# Computes FID, LPIPS, and CLIP similarity for generated images.
#   - FID: compared against COCO test2017 pre-computed stats
#   - LPIPS & CLIP: compared against baseline_dir (pair-wise, matched by filename)
#
# Examples:
#   bash benchmark/benchmark.sh output/flux2_lora_eval_checkpoint-1000_10steps output/flux2_eval_10steps
#   bash benchmark/benchmark.sh output/flux2_eval_10steps  # FID only (no baseline for LPIPS/CLIP)

set -e

GENERATED_DIR=${1:?Usage: bash benchmark/benchmark.sh <generated_dir> [baseline_dir]}
BASELINE_DIR=${2:-""}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "========================================"
echo " Benchmark: $(basename ${GENERATED_DIR})"
echo "========================================"
echo ""

# --- FID ---
echo "--- FID (vs COCO test2017 stats) ---"
python "${SCRIPT_DIR}/compute_fid.py" "${GENERATED_DIR}"
echo ""

# --- LPIPS & CLIP ---
if [ -n "${BASELINE_DIR}" ]; then
    echo "--- LPIPS & CLIP (vs ${BASELINE_DIR}) ---"
    python "${SCRIPT_DIR}/compute_lpips_clip.py" "${BASELINE_DIR}" "${GENERATED_DIR}"
else
    echo "--- LPIPS & CLIP: skipped (no baseline_dir provided) ---"
fi

echo ""
echo "========================================"
echo " Done"
echo "========================================"
