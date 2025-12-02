EXP_NAME=$(basename "$PWD")

export MODEL_PATH="/scratch/yx2432/models/FLUX.1-dev/models--black-forest-labs--FLUX.1-dev/snapshots/0ef5fff789c832c5c7f4e127f94c8b54bbcced44"
export INSTANCE_DIR="/scratch/yx2432/MLSYS/diffusion-ft/flux_ft/dataset"
export OUTPUT_DIR="exp_output"

echo "Running experiment: $EXP_NAME"


# This exp checks the smaller local mask 16(height) x 24(width)

accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_PATH  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --mixed_precision="bf16" \
  --instance_prompt="a photo of dog" \
  --resolution=1024 \
  --train_batch_size=1 \
  --guidance_scale=1 \
  --gradient_accumulation_steps=8 \
  --optimizer="prodigy" \
  --learning_rate=1. \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=50000 \
  --checkpointing_steps=1000 \
  --validation_prompt="A photo of dog in a bucket" \
  --validation_epochs=1 \
  --gradient_checkpointing \
  --seed="42" --dataloader_num_workers=2 --exp_name=$EXP_NAME