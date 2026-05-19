#!/bin/bash
export NUM_WORKERS=1
export MODEL_NAME="black-forest-labs/FLUX.2-klein-9B"
export DATA_ROOT="/scratch/sz3684/HilbertA/qwen_image/t2i_1024"
export OUTPUT_DIR="/scratch/sz3684/HilbertA/qwen_image/t2i_1024"

source /scratch/sz3684/miniconda3/etc/profile.d/conda.sh
conda activate hilberta

torchrun --nproc_per_node=$NUM_WORKERS cache_latent_codes.py \
    --data_root=$DATA_ROOT \
    --batch_size=16 \
    --num_workers=$NUM_WORKERS \
    --pretrained_model_name_or_path=$MODEL_NAME \
    --mixed_precision='bf16' \
    --output_dir=$OUTPUT_DIR \
    --resolution=1024
