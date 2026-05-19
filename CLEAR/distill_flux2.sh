#!/bin/bash
EXP_NAME="clear-flux2-w${WINDOW_SIZE:-16}"

MODEL_PATH="black-forest-labs/FLUX.2-klein-9B"
DATA_ROOT="/scratch/sz3684/HilbertA/qwen_image/t2i_1024"
OUTPUT_DIR="${OUTPUT_DIR:-./exp_output_clear}"

echo "Running CLEAR distillation: $EXP_NAME"

accelerate launch --config_file deepspeed_config.yaml distill_flux2.py \
  --pretrained_model_name_or_path=$MODEL_PATH \
  --data_root=$DATA_ROOT \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
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
  --checkpointing_steps=3000 \
  --gradient_checkpointing \
  --seed=42 \
  --dataloader_num_workers=2 \
  --exp_name=$EXP_NAME \
  --window_size=${WINDOW_SIZE:-16} \
  --validation_prompt="a cat sitting on a windowsill watching the rain||a beautiful mountain landscape with a lake at sunset" \
  --num_validation_images=2 \
  --validation_epochs=1 \
  ${RESUME_CKPT:+--resume_from_checkpoint=$RESUME_CKPT}
