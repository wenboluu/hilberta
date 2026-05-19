#!/bin/bash
EXP_NAME=$(basename "$PWD")

MODEL_PATH="black-forest-labs/FLUX.2-klein-9B"
INSTANCE_DIR="/scratch/sz3684/HilbertA/qwen_image/t2i_1024"
OUTPUT_DIR="${OUTPUT_DIR:-./exp_output}"

echo "Running experiment: $EXP_NAME"

NUM_GPUS=${NUM_GPUS:-1}

if [ "$NUM_GPUS" -gt 1 ]; then
    LAUNCH_ARGS="--multi_gpu --num_processes=$NUM_GPUS"
else
    LAUNCH_ARGS=""
fi

accelerate launch $LAUNCH_ARGS train_distill_lora.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1.0 \
  --gradient_accumulation_steps=8 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50000 \
  --checkpointing_steps=500 \
  --checkpoints_total_limit=3 \
  --gradient_checkpointing \
  --seed=42 \
  --dataloader_num_workers=2 \
  --exp_name=$EXP_NAME \
  --num_tiles=4 \
  --hilberta_method="masking" \
  --rank=4 \
  --validation_prompt="a cat sitting on a windowsill watching the rain||a beautiful mountain landscape with a lake at sunset" \
  --num_validation_images=2 \
  --validation_steps=500
