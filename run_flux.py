import argparse
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
    seed,
    height,
    width,
    num_tiles,
    num_of_inference_steps,
):
    print(f"\nGenerating image for prompt: {prompt}, seed: {seed}, height: {height}, width: {width}")
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
    file_name = f"{datetime.now().strftime('%m-%d-%H-%M-%S')}.png"
    image_path = os.path.join(output_folder, file_name)
    image.save(image_path, "PNG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run image generation for given prompt and seed.")
    parser.add_argument("--config", type=str, default="./src/config.yaml", help="Path to configuration YAML file")
    args = parser.parse_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    prompt = config['prompt_list'][0]
    seed = config['seed_list'][0]
    num_tiles = config['num_tiles']
    num_of_inference_steps = config['num_of_inference_steps']
    merge_step_interval = config['merge_step_interval']
    output_folder = config['output_folder']
    device = "cuda:0"

    recompute_step = list(range(0, 35))
    merge_step = list(range(1, 35, merge_step_interval))

    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="/home/wl2707/.cache/huggingface/hub",
        local_files_only=True,
    ).to(device)

    pipeline.customized_call = types.MethodType(customized_call, pipeline)
    pipeline.transformer.forward = types.MethodType(customized_forward, pipeline.transformer)

    pipeline.load_lora_weights("./lora_weight/ckpt_1024_16/checkpoint-2817")

    image_sizes = [(1024, 1024)]
    for height, width in image_sizes:
        generate_image(
            pipeline=pipeline,
            output_folder=output_folder,
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_tiles=num_tiles,
            num_of_inference_steps=num_of_inference_steps,
        )
