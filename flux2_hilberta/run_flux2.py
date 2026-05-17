"""
Main inference script for FLUX.2-klein with Hilbert relocated attention.
"""

import os
import sys
import argparse
import itertools
import yaml
import gc
from datetime import datetime

import torch
from PIL import PngImagePlugin
from diffusers import Flux2KleinPipeline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from patch import apply_patch


def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()


def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_image(pipeline, output_folder, config, index="warmup", prompt="default", seed=42, height=1024, width=1024):
    print(f"\n[{index}] Generating: {prompt[:80]}...")

    generator = torch.Generator(device="cuda").manual_seed(seed)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    image = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=config.get('num_of_inference_steps', 10),
        guidance_scale=config.get('guidance_scale', 1.0),
        generator=generator,
    ).images[0]

    end_event.record()
    torch.cuda.synchronize()
    elapsed = start_event.elapsed_time(end_event) / 1000.0

    os.makedirs(output_folder, exist_ok=True)

    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("prompt", prompt)
    metadata.add_text("seed", str(seed))
    metadata.add_text("height", str(height))
    metadata.add_text("width", str(width))
    metadata.add_text("num_tiles", str(config.get('num_tiles', 4)))
    metadata.add_text("method", str(config.get('method', 'reorder_shared')))
    metadata.add_text("inference_time", f"{elapsed:.2f}s")

    filename = f"{index}_seed{seed}_{config.get('remark', 'flux2')}.png"
    filepath = os.path.join(output_folder, filename)
    image.save(filepath, pnginfo=metadata)

    print(f"  Saved: {filepath} ({elapsed:.2f}s)")
    return image, elapsed


def prepare_pipeline(config):
    model_repo = config.get('model_repo', 'black-forest-labs/FLUX.2-klein-9B')
    cache_dir = config.get('cache_dir', None)

    print(f"Loading model: {model_repo}")
    pipe = Flux2KleinPipeline.from_pretrained(
        model_repo, torch_dtype=torch.bfloat16, cache_dir=cache_dir,
    ).to("cuda")

    height = config.get('height', 1024)
    num_tiles = config.get('num_tiles', 4)

    print(f"Applying HilbertA patch: num_tiles={num_tiles}, method={config.get('method', 'reorder_shared')}")
    pipe = apply_patch(pipe, num_tiles=num_tiles, height=height)

    return pipe


def main():
    parser = argparse.ArgumentParser(description="FLUX.2-klein with Hilbert relocated attention")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    pipe = prepare_pipeline(config)

    prompt_list = config.get('prompt_list', ["A beautiful landscape"])
    seed_list = config.get('seed_list', [42])
    output_folder = config.get('output_folder', './output/')
    height = config.get('height', 1024)
    width = config.get('width', 1024)

    # Warmup
    print("\nWarmup run...")
    generate_image(pipe, output_folder, config, index="warmup", prompt=prompt_list[0], seed=0, height=height, width=width)
    clean_memory()

    # Generate
    print(f"\nGenerating {len(prompt_list) * len(seed_list)} images...")
    idx = 0
    for prompt, seed in itertools.product(prompt_list, seed_list):
        generate_image(pipe, output_folder, config, index=idx, prompt=prompt, seed=seed, height=height, width=width)
        clean_memory()
        idx += 1

    print(f"\nDone! {idx} images saved to {output_folder}")


if __name__ == "__main__":
    main()
