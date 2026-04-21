import os
import argparse
import itertools
from diffusers import FluxPipeline
import torch
from tqdm import tqdm
import json
from PIL import PngImagePlugin
from datetime import datetime
import gc

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
    height=1024,
    width=1024,
):
    print("\n")

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

    file_name = f"{datetime.now().strftime('%m-%d-%H-%M-%S')}*baseline.png"
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
        torch.cuda.reset_peak_memory_stats()
        elapsed_time, file_name = generate_image(
            pipeline=pipeline,
            output_folder=output_folder,
            index=index,
            prompt=prompt,
            random_seed=seed,
            height=1024,
            width=1024,
        )
        current_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        print(f"Current memory: {current_memory / 1048576:.2f}MiB, Peak memory: {peak_memory / 1048576:.2f}MiB")

        results.append(elapsed_time)

if __name__ == "__main__":
    import yaml

    def load_config(config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    parser = argparse.ArgumentParser(
        description="Generate images using baseline FLUX (no modifications)."
    )

    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(script_dir, 'config.yaml'),
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for model files (if not specified, uses HuggingFace default)"
    )
    args = parser.parse_args()

    # Load configuration from YAML
    config = load_config(args.config)

    # Process parameters
    prompt_list = config['prompt_list']
    seed_list = config['seed_list']
    num_of_inference_steps = config['num_of_inference_steps']
    output_folder = config.get('output_folder', './output/baseline')

    os.makedirs(output_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load baseline FLUX pipeline (no patches, no modifications)
    # NOTE: Update cache_dir via --cache-dir argument for your server
    pipeline = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        cache_dir="/scratch/sz3684/.cache/",
        local_files_only=True,
    ).to(device)

    print("Running baseline FLUX (no patches or modifications)")

    evaluate_dst_selection(
        pipeline=pipeline,
        output_folder=output_folder,
        prompt_list=prompt_list,
        seed_list=seed_list,
        warm_up=False,
    )