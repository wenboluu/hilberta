import os

import argparse
import numpy as np
import itertools
import os
from diffusers import FluxPipeline
import torch
import pandas as pd
from tqdm import tqdm
import json
from PIL import PngImagePlugin  # Import PNG plugin to handle metadata
from datetime import datetime
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
    prompt="default",
    random_seed=42,
    height=768,
    width=768,
    num_tiles=64,
):
    print("\n")

    apply_patch(
        pipeline,
        num_tiles=num_tiles,
        height=height,
    )

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

    file_name = f"{datetime.now().strftime('%m-%d-%H-%M-%S')}*{remark}.png"
    image_path = os.path.join(output_folder, file_name)

    metadata_dict = {
        "Prompt": prompt,
        "Seed": random_seed,
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
    prompt_list,
    seed_list,
    num_tiles,
    warm_up=True,
):

    if warm_up:
        for _ in range(3):
            generate_image(pipeline, output_folder)

    configurations = itertools.product(
        prompt_list, seed_list
    )
    results = []

    for index, (prompt, seed) in tqdm(
        enumerate(configurations), desc="Processing configurations"
    ):
        merge_method = "attention"
        
        torch.cuda.reset_peak_memory_stats()
        elapsed_time, file_name = generate_image(
            pipeline=pipeline,
            output_folder=output_folder,
            index=index,
            prompt=prompt,
            random_seed=seed,
            height=1024,
            width=1024,
            num_tiles=num_tiles,
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
        default= None,
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    # Process parameters
    num_tiles = config['num_tiles']
    prompt_list = config['prompt_list']
    seed_list = config['seed_list']
    num_of_inference_steps = config['num_of_inference_steps']
    remark = config['remark']
    output_folder = config['output_folder']

    os.makedirs(output_folder, exist_ok=True)

    toma_variant = None
    dst_method = None
    merge_step = None
    recompute_step = None

    prompt_list = prompt_list
    seed_list = seed_list
    dst_method = dst_method
    output_folder = output_folder
    results_file_path = f"time.md"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="/scratch/sz3684/.cache/",
        local_files_only=True,
    ).to(device)

    import types
    from masking_utils import customized_call
    from masking_utils import customized_forward
    # from reorder_utils import customized_forward

    # pipeline.customized_call = types.MethodType(customized_call, pipeline)
    pipeline.transformer.forward = types.MethodType(customized_forward, pipeline.transformer)

    pipeline.load_lora_weights("/scratch/sz3684/HilbertA/reorder_local_attention/reloc_attention/lora/checkpoint-2400")

    evaluate_dst_selection(
        pipeline=pipeline,
        output_folder=output_folder,
        prompt_list=prompt_list,
        seed_list=seed_list,
        num_tiles=num_tiles,
        warm_up=False,
    )