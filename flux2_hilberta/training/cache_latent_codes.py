"""
Cache VAE latent codes for FLUX.2-klein training.

Encodes images with AutoencoderKLFlux2, applies patchification and batch norm,
then saves latent distribution mean/std as safetensors for fast training loading.
"""

import argparse
import os
import math

import numpy as np
import torch
import tqdm
from PIL import Image
from safetensors.torch import save_file
from diffusers import AutoencoderKLFlux2


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Cache VAE latent codes for FLUX.2-klein")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="flux2-lora")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return args


def patchify_latents(latents):
    """FLUX.2 patchification: (B, C, H, W) -> (B, 4C, H/2, W/2)"""
    batch_size, num_channels, height, width = latents.shape
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 1, 3, 5, 2, 4)
    latents = latents.reshape(batch_size, num_channels * 4, height // 2, width // 2)
    return latents


def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.mixed_precision == 'fp16':
        dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
        cache_dir=args.cache_dir,
    ).to(device, dtype)

    # Get batch norm parameters for FLUX.2 latent normalization
    has_bn = hasattr(vae, 'bn') and vae.bn is not None

    all_info = sorted([
        os.path.join(args.data_root, f)
        for f in os.listdir(args.data_root)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.bmp'))
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    work_load = math.ceil(len(all_info) / args.num_workers)
    start = work_load * args.local_rank
    end = min(work_load * (args.local_rank + 1), len(all_info))

    for idx in tqdm.tqdm(range(start, end, args.batch_size)):
        batch_files = all_info[idx:idx + args.batch_size]
        paths = [
            os.path.join(
                args.output_dir,
                os.path.splitext(os.path.basename(item))[0] + '_latent_code.safetensors'
            )
            for item in batch_files
        ]

        # Skip if all already cached
        if all(os.path.exists(p) for p in paths):
            continue

        images = []
        for item in batch_files:
            img = Image.open(item).convert('RGB')
            img = img.resize((args.resolution, args.resolution))
            img = torch.from_numpy((np.array(img) / 127.5) - 1).permute(2, 0, 1)
            images.append(img)

        with torch.no_grad():
            images = torch.stack(images, dim=0).to(device, vae.dtype)
            latent_dist = vae.encode(images).latent_dist

            # Patchify latents (FLUX.2 specific)
            means_raw = latent_dist.mean
            stds_raw = latent_dist.std

            means_patched = patchify_latents(means_raw)
            stds_patched = patchify_latents(stds_raw)

            # Apply batch norm normalization if available
            if has_bn:
                bn_mean = vae.bn.running_mean.view(1, -1, 1, 1).to(device, dtype)
                bn_std = torch.sqrt(vae.bn.running_var.view(1, -1, 1, 1) + vae.config.batch_norm_eps).to(device, dtype)
                means_patched = (means_patched - bn_mean) / bn_std
                stds_patched = stds_patched / bn_std

            means_patched = means_patched.cpu()
            stds_patched = stds_patched.cpu()

        for path, mean, std in zip(paths, means_patched.unbind(), stds_patched.unbind()):
            save_file({'mean': mean, 'std': std}, path)


if __name__ == '__main__':
    main(parse_args())
