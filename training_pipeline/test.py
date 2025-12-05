from PIL import Image
from dataset import loader

# Dataset and DataLoaders creation:
train_dataloader = loader(train_batch_size=1, num_workers=2,
                            img_dir="/scratch/sz3684/reorder_local_attention/dataset/cache_dataset", img_size=1024, 
                            use_cached_prompt_embeds=True,
                            use_cached_latent_codes=True)
