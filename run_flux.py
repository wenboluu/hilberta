import argparse
import json
import os
import re
import types

import torch
from tqdm import tqdm
import yaml
from diffusers import FluxPipeline

from src.flux_overrides import hilberta_pipeline_call, hilberta_transformer_forward
from src.patch import apply_hilberta_patch


def generate_image(
    pipeline,
    output_folder,
    prompt,
    prompt_idx,
    seed,
    height,
    width,
    num_tiles,
    num_of_inference_steps,
    device,
    patched=True,
    save_intermediate_steps=False,
):
    print(
        f"\nGenerating image for prompt: {prompt}, prompt_idx: {prompt_idx}, "
        f"seed: {seed}, height: {height}, width: {width}, mode: {'hilberta' if patched else 'original'}"
    )

    if patched:
        apply_hilberta_patch(pipeline, num_tiles=num_tiles, height=height)

    generator = torch.Generator(device=device).manual_seed(seed)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    intermediate_image_list = []
    if save_intermediate_steps:
        intermediate_folder = os.path.join(output_folder, f"{prompt_idx}_steps")
        os.makedirs(intermediate_folder, exist_ok=True)

        def plot_latent(pipe, step, timestep, callback_kwargs):
            latents = callback_kwargs["latents"]
            latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
            latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
            image = pipe.vae.decode(latents, return_dict=False)[0]
            image = pipe.image_processor.postprocess(image, output_type="pil")
            intermediate_image_list.append(image[0])
            return callback_kwargs

        stable_diffusion_output = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=num_of_inference_steps,
            callback_on_step_end=plot_latent,
            callback_on_step_end_tensor_inputs=["latents"],
        )

        print(f"Saving {len(intermediate_image_list)} intermediate denoising steps...")
        for step, img in tqdm(enumerate(intermediate_image_list), total=len(intermediate_image_list)):
            step_file = os.path.join(intermediate_folder, f"step_{step:03d}.jpeg")
            img.save(step_file)
    else:
        stable_diffusion_output = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            generator=generator,
            guidance_scale=7.5,
            num_inference_steps=num_of_inference_steps,
        )

    end_event.record()
    torch.cuda.synchronize()
    denoise_time = start_event.elapsed_time(end_event)
    print(f"Denoise time: {denoise_time / 1000:.2f} seconds")

    image = stable_diffusion_output.images[0]

    os.makedirs(output_folder, exist_ok=True)
    truncated_prompt = prompt[:60]
    sanitized_prompt = re.sub(r'[^\w\s-]', '', truncated_prompt)
    sanitized_prompt = re.sub(r'\s+', '_', sanitized_prompt)
    file_name = f"{sanitized_prompt}.jpeg"
    image_path = os.path.join(output_folder, file_name)
    image.save(image_path, "JPEG")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLUX image generation (patched or original).")
    parser.add_argument("--config", type=str, default="./src/config.yaml", help="Path to configuration YAML file")
    parser.add_argument("--mode", type=str, default="hilberta", choices=["hilberta", "original"],
                        help="'hilberta' applies Hilbert-curve tiled attention; 'original' uses vanilla Flux")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run on")

    # Prompt source: either --prompt for ad-hoc, or --prompts-file + --partition for batch
    parser.add_argument("--prompt", type=str, nargs="+", default=None,
                        help="One or more prompts to generate (overrides --prompts-file)")
    parser.add_argument("--prompts-file", type=str, default=None,
                        help="Path to a JSON file containing a list of prompts")
    parser.add_argument("--partition", type=int, default=None,
                        help="Partition index for splitting prompts-file (0-indexed, 20 partitions)")
    parser.add_argument("--total-partitions", type=int, default=20,
                        help="Total number of partitions when using --partition")

    parser.add_argument("--lora", type=str, default=None,
                        help="Path to LoRA weights (overrides config; use 'none' to disable)")
    parser.add_argument("--save-steps", action="store_true", help="Save intermediate denoising steps")
    parser.add_argument("--regenerate-masks", action="store_true",
                        help="Regenerate Hilbert-curve attention masks and exit")
    args = parser.parse_args()

    # Handle mask regeneration early (no GPU pipeline needed)
    if args.regenerate_masks:
        from src.utils import generate_masks
        print("Regenerating Hilbert-curve attention masks...")
        generate_masks()
        print("Done.")
        exit(0)

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    seed = config['generation']['seed_list'][0]
    num_tiles = config['tiling']['num_tiles']
    num_of_inference_steps = config['generation']['num_of_inference_steps']
    output_folder = config['generation']['output_folder']
    lora_config = config.get('lora', {})

    # Resolve prompts
    if args.prompt is not None:
        # Ad-hoc prompts from command line
        prompts_partition = args.prompt
        start_idx = 0
    elif args.prompts_file is not None:
        # Batch mode from file
        with open(args.prompts_file, 'r') as f:
            prompts = json.load(f)
        if args.partition is not None:
            partition_size = len(prompts) // args.total_partitions
            start_idx = args.partition * partition_size
            end_idx = start_idx + partition_size if args.partition < args.total_partitions - 1 else len(prompts)
            prompts_partition = prompts[start_idx:end_idx]
        else:
            prompts_partition = prompts
            start_idx = 0
    else:
        parser.error("Provide either --prompt or --prompts-file")

    # Build pipeline
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="/data1/shared_hf_home/hub",
        local_files_only=True,
    ).to(args.device)

    if args.mode == "hilberta":
        pipeline.__class__.__call__ = types.MethodType(hilberta_pipeline_call, pipeline)
        pipeline.transformer.forward = types.MethodType(hilberta_transformer_forward, pipeline.transformer)

    # Resolve LoRA: CLI --lora overrides config; 'none' disables
    if args.lora is not None:
        lora_path = None if args.lora.lower() == "none" else args.lora
    elif lora_config.get('enabled', False):
        lora_path = lora_config['path']
    else:
        lora_path = None

    if lora_path is not None:
        print(f"Loading LoRA weights from: {lora_path}")
        pipeline.load_lora_weights(lora_path)

    pipeline.set_progress_bar_config(disable=False)

    # Generate
    image_sizes = [(1024, 1024)]
    for height, width in image_sizes:
        for prompt_idx, prompt in enumerate(prompts_partition, start=start_idx):
            generate_image(
                pipeline=pipeline,
                output_folder=output_folder,
                prompt=prompt,
                prompt_idx=prompt_idx,
                seed=seed,
                height=height,
                width=width,
                num_tiles=num_tiles,
                num_of_inference_steps=num_of_inference_steps,
                device=args.device,
                patched=(args.mode == "hilberta"),
                save_intermediate_steps=args.save_steps,
            )
