"""
Dataset loader for FLUX.2-klein training with cached prompt embeds and latent codes.

Cached file format:
- {name}_latent_code.safetensors: keys 'mean', 'std' (post-patchify + BN normalized)
- {name}_prompt_embed.safetensors: key 'prompt_embeds' (Qwen3 multi-layer)
"""

import os
import json
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from safetensors import safe_open


def image_resize(img, max_size=512):
    w, h = img.size
    if w >= h:
        new_w = max_size
        new_h = int((max_size / w) * h)
    else:
        new_h = max_size
        new_w = int((max_size / h) * w)
    new_w = (new_w // 32) * 32
    new_h = (new_h // 32) * 32
    return img.resize((new_w, new_h))


def c_crop(image):
    width, height = image.size
    new_size = min(width, height)
    left = (width - new_size) / 2
    top = (height - new_size) / 2
    right = (width + new_size) / 2
    bottom = (height + new_size) / 2
    return image.crop((left, top, right, bottom))


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, img_size=1024, caption_type='json',
                 use_cached_prompt_embeds=False, use_cached_latent_codes=False):
        all_images = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))
        ])
        # Filter: only keep images that have both cached latent_code and prompt_embed
        if use_cached_latent_codes and use_cached_prompt_embeds:
            self.images = [
                img for img in all_images
                if os.path.exists(img[:img.rfind('.')] + '_latent_code.safetensors')
                and os.path.exists(img[:img.rfind('.')] + '_prompt_embed.safetensors')
            ]
            if len(self.images) < len(all_images):
                print(f"Dataset: {len(self.images)}/{len(all_images)} images have both cached latent_code and prompt_embed")
        else:
            self.images = all_images
        self.img_size = img_size
        self.caption_type = caption_type
        self.use_cached_prompt_embeds = use_cached_prompt_embeds
        self.use_cached_latent_codes = use_cached_latent_codes

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            batch = {}
            img_path = self.images[idx]
            stem = img_path[:img_path.rfind('.')]

            if not self.use_cached_latent_codes:
                img = Image.open(img_path).convert('RGB')
                img = c_crop(img)
                img = img.resize((self.img_size, self.img_size))
                img = torch.from_numpy((np.array(img) / 127.5) - 1).permute(2, 0, 1)
                batch['images'] = img
            else:
                with safe_open(stem + '_latent_code.safetensors', framework="pt") as f:
                    batch['latent_codes_mean'] = f.get_tensor('mean')
                    batch['latent_codes_std'] = f.get_tensor('std')

            if not self.use_cached_prompt_embeds:
                json_path = stem + '.' + self.caption_type
                if self.caption_type == "json":
                    with open(json_path) as fj:
                        prompt = json.load(fj)['prompt']
                else:
                    with open(json_path) as ft:
                        prompt = ft.read()
                batch['prompts'] = prompt
            else:
                with safe_open(stem + '_prompt_embed.safetensors', framework="pt") as f:
                    batch['prompt_embeds'] = f.get_tensor('prompt_embeds')

            return batch
        except Exception as e:
            print(f"Error loading {self.images[idx]}: {e}")
            return self.__getitem__(random.randint(0, len(self.images) - 1))


def loader(train_batch_size, num_workers, **kwargs):
    dataset = CustomImageDataset(**kwargs)
    return DataLoader(dataset, batch_size=train_batch_size, num_workers=num_workers, shuffle=True, pin_memory=True)
