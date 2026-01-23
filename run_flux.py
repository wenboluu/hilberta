import argparse
import json
import os
import types
from datetime import datetime

import torch
import yaml
from diffusers import FluxPipeline

from src.flux_scheduler import FluxScheduler
from src.masking_utils import customized_call, customized_forward
from src.patch import apply_patch


def generate_image(
    pipeline,
    output_folder,
    prompt,
    prompt_idx,  # Add prompt index
    seed,
    height,
    width,
    num_tiles,
    num_of_inference_steps,
):
    print(
        f"\nGenerating image for prompt: {prompt}, prompt_idx: {prompt_idx}, seed: {seed}, height: {height}, width: {width}")
    flux_scheduler = FluxScheduler(
        timesteps=num_of_inference_steps,
        dst_recompute_timesteps=recompute_step,
        attn_recompute_timesteps=recompute_step,
        merge_step=merge_step,
        config_path="./src/transformer_layer_config.yaml",
    )

    apply_patch(
        pipeline,
        num_tiles=num_tiles,
        unet_scheduler=flux_scheduler,
        height=height,
    )

    generator = torch.Generator(device=device).manual_seed(seed)

    stable_diffusion_output = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        generator=generator,
        guidance_scale=7.5,
        num_inference_steps=num_of_inference_steps,
    )

    image = stable_diffusion_output.images[0]

    os.makedirs(output_folder, exist_ok=True)
    file_name = f"{prompt_idx}.png"  # Include prompt index in file name
    image_path = os.path.join(output_folder, file_name)
    image.save(image_path, "PNG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image generation for given prompt and seed.")
    parser.add_argument("--config", type=str, default="./src/config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--partition", type=int, required=True, help="Partition index (0-19)")  # Add partition argument
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # prompt = config['prompt_list'][0]
    seed = config['seed_list'][0]
    num_tiles = config['num_tiles']
    num_of_inference_steps = config['num_of_inference_steps']
    merge_step_interval = config['merge_step_interval']
    output_folder = config['output_folder']
    device = "cuda:0"

    recompute_step = list(range(0, num_of_inference_steps))
    merge_step = list(range(1, num_of_inference_steps, merge_step_interval))

    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="/scratch/wl2707/.cache",
        local_files_only=True,
    ).to(device)

    pipeline.customized_call = types.MethodType(customized_call, pipeline)
    pipeline.transformer.forward = types.MethodType(customized_forward, pipeline.transformer)

    pipeline.load_lora_weights("./lora_weights/ckpt_1024_16/checkpoint-2817")

    pipeline.set_progress_bar_config(disable=True)

    prompts_path = "./data/coco_prompts.json"
    with open(prompts_path, 'r') as f:
        prompts = json.load(f)
    prompts = prompts[:5000]  # Limit to 5000 prompts for benchmarking

    # Divide prompts into 20 partitions
    total_partitions = 20
    partition_size = len(prompts) // total_partitions
    start_idx = args.partition * partition_size
    end_idx = start_idx + partition_size if args.partition < total_partitions - 1 else len(prompts)
    prompts_partition = prompts[start_idx:end_idx]  # Select the prompts for the given partition

    image_sizes = [(1024, 1024)]
    for height, width in image_sizes:
        for prompt_idx, prompt in enumerate(prompts_partition, start=start_idx):  # Keep track of global prompt index
            generate_image(
                pipeline=pipeline,
                output_folder=output_folder,
                prompt=prompt,
                prompt_idx=prompt_idx,  # Pass prompt index
                seed=seed,
                height=height,
                width=width,
                num_tiles=num_tiles,
                num_of_inference_steps=num_of_inference_steps,
            )
