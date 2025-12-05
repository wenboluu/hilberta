export NUM_WORKERS=1
export MODEL_NAME="black-forest-labs/FLUX.1-dev"

torchrun \
  --nproc_per_node=$NUM_WORKERS \
  --master_port=29501 \
  training_pipeline/cache_prompt_embeds.py \
  --data_root="/scratch/sz3684/reorder_local_attention/dataset/clear" \
  --batch_size=256 \
  --num_worker=$NUM_WORKERS \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --mixed_precision='bf16' \
  --output_dir="/scratch/sz3684/reorder_local_attention/dataset/clear_prompt_embedding_cache" \
  --cache_dir="/scratch/sz3684/.cache"
