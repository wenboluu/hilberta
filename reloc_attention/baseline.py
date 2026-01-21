import argparse
import numpy as np
import itertools
import os
from diffusers import StableDiffusionXLPipeline, StableDiffusionPipeline, FluxPipeline
import torch
import pandas as pd
from tqdm import tqdm
import json
from PIL import PngImagePlugin  # Import PNG plugin to handle metadata
from datetime import datetime
from flux_scheduler import FluxScheduler
import gc
from patch import apply_patch

def clean_memory():
    """Utility function to clean up GPU memory after every generation."""
    torch.cuda.empty_cache()
    gc.collect()

def generate_image(
    pipeline,
    output_folder,
    index="warmup",
    ratio=0,
    prompt="default",
    random_seed=42,
    dst_selection=None,
    max_downsample=4,
    height=768,
    width=768,
    num_tiles=64,
    merge_method=None,
    toma_variant=None,
):
    print("\n")
    print(
        f"Generating image with ratio: {ratio}, merge method: {merge_method}, dst_selection: {dst_selection}, merge_once: {False}"
    )
    #********************************************************************************************************************
    flux_scheduler = FluxScheduler(
        timesteps=35,
        dst_recompute_timesteps = recompute_step,
        attn_recompute_timesteps = recompute_step,
        merge_step = merge_step,
        config_path="/home/sz3684/diffusion/reorder_local_attention/triton_version/tomesd_global/transformer_layer_config.yaml",
    )
    #********************************************************************************************************************

    generator = torch.Generator(device=device).manual_seed(random_seed)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    stable_diffusion_output = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        generator=generator,
        guidance_scale=7.5,
        num_inference_steps=num_of_inference_steps,
        max_sequence_length=512,
    )

    image = stable_diffusion_output.images[0]

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) * 1e-3

    if index == "warmup":
        return None

    os.makedirs(output_folder, exist_ok=True)

    file_name = f"{datetime.now().strftime('%m-%d-%H-%M-%S')}*{remark}*{dst_selection}.png"
    image_path = os.path.join(output_folder, file_name)

    metadata_dict = {
        "Prompt": prompt,
        "Seed": random_seed,
        "Ratio": ratio,
        "Dst_Selection": dst_selection,
        "Merge_Method": merge_method,
        "Elapsed_Time": elapsed_time,
    }

    metadata_json = json.dumps(metadata_dict)

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("Metadata", metadata_json)

    image.save(image_path, "PNG", pnginfo=metadata)

    clean_memory()
    return elapsed_time, file_name

def evaluate_dst_selection(
    pipeline,
    output_folder,
    dst_selection_list,
    prompt_list,
    seed_list,
    ratio_list,
    num_tiles,
    warm_up=True,
    toma_variant=None,
):

    if warm_up:
        for _ in range(3):
            generate_image(pipeline, output_folder)

    configurations = itertools.product(
        ratio_list, prompt_list, dst_selection_list, seed_list
    )
    results = []

    for index, (ratio, prompt, dst_selection, seed) in tqdm(
        enumerate(configurations), desc="Processing configurations"
    ):
        merge_method = "attention"
        
        torch.cuda.reset_peak_memory_stats()
        elapsed_time, file_name = generate_image(
            pipeline=pipeline,
            output_folder=output_folder,
            index=index,
            ratio=ratio,
            prompt=prompt,
            random_seed=seed,
            dst_selection=dst_selection,
            max_downsample=4,
            height=2048,
            width=2048,
            num_tiles=num_tiles,
            merge_method=merge_method,
            toma_variant=toma_variant,
        )
        current_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"Current memory: {current_memory / 1048576:.2f}MiB, Peak memory: {peak_memory / 1048576:.2f}MiB")


        results.append(elapsed_time)

if __name__ == "__main__":
    # Argument parser setup
    import argparse
    import yaml
    import torch
    import shutil
    from pathlib import Path

    if os.path.exists('/home/sz3684/diffusion/reorder_local_attention/triton_version/tensors/'):
        shutil.rmtree('/home/sz3684/diffusion/reorder_local_attention/triton_version/tensors/')

    def load_config(config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    parser = argparse.ArgumentParser(
        description="Generate images with different configurations."
    )

    parser.add_argument(
        "--config", 
        type=str,
        default="/home/sz3684/diffusion/reorder_local_attention/triton_version/tomesd_global/config.yaml",
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    # Process parameters
    ratio_list = config['ratio_list']
    num_tiles = config['num_tiles']
    prompt_list = config['prompt_list']
    seed_list = config['seed_list']
    toma_variant = config['toma_variant']
    num_of_inference_steps = config['num_of_inference_steps']
    remark = config['remark']
    merge_step_interval = config['merge_step_interval']

    recompute_step = [_ for _ in range(0, 35)]
    print('recompute_step', recompute_step)
    merge_step = [_ for _ in range(1, 35, merge_step_interval)]
    print('merge_step', merge_step)

    if toma_variant == "global_tile":
        dst_method = 'tile_wise_facility'
    elif toma_variant == "local_stripe":
        dst_method = 'local_stripe_wise_facility'
    elif toma_variant == "global_stripe":
        dst_method = 'global_stripe_wise_facility'
    elif toma_variant == "local_tile":
        dst_method = 'local_tile_wise_facility'
    elif toma_variant == 'SVD':
        dst_method = 'SVD'

    for ratio in ratio_list:
        prompt_list = prompt_list
        seed_list = seed_list
        dst_method = dst_method
        output_folder = f"/home/sz3684/diffusion/reorder_local_attention/triton_version/output/{toma_variant}/{ratio}"
        results_file_path = f"time.md"
        device = "cuda:7"

        pipeline = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.bfloat16,
            cache_dir="/home/wl2707/.cache/huggingface/hub",
            local_files_only=True,
        ).to(device)

        evaluate_dst_selection(
            pipeline=pipeline,
            output_folder=output_folder,
            dst_selection_list=[dst_method],
            prompt_list=prompt_list,
            seed_list=seed_list,
            ratio_list=[ratio],
            num_tiles=num_tiles,
            warm_up=False,
            toma_variant=toma_variant,
        )