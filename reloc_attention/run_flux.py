import os
import argparse
import gc
import itertools
import json
from datetime import datetime
from pathlib import Path
from types import MethodType

import torch
import yaml
from diffusers import FluxPipeline
from PIL import PngImagePlugin
from tqdm import tqdm

from patch import apply_patch

MODEL_REPO = "black-forest-labs/FLUX.1-dev"

def clean_memory():
    """Release cached GPU memory after each generation."""
    torch.cuda.empty_cache()
    gc.collect()

def generate_image(
    pipeline,
    output_folder,
    *,
    prompt,
    random_seed,
    num_tiles,
    num_steps,
    device,
    remark,
    height=768,
    width=768,
    guidance_scale=7.5,
    max_sequence_length=512,
    save_image=True,
):
    """Run a single pipeline call and optionally persist the PNG output."""
    apply_patch(
        pipeline,
        num_tiles=num_tiles,
        height=height,
    )

    generator = torch.Generator(device=device).manual_seed(random_seed)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    diffusion_output = pipeline(
        prompt=prompt,
        height=height,
        width=width,
        generator=generator,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        max_sequence_length=max_sequence_length,
    )
    image = diffusion_output.images[0]

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event) * 1e-3

    file_name = None
    if save_image:
        output_folder.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%m-%d-%H-%M-%S")
        file_name = f"{timestamp}*{remark}.png"
        image_path = output_folder / file_name

        metadata = PngImagePlugin.PngInfo()
        metadata.add_text(
            "Metadata",
            json.dumps(
                {
                    "Prompt": prompt,
                    "Seed": random_seed,
                    "Elapsed_Time": elapsed_time,
                }
            ),
        )

        image.save(image_path, "PNG", pnginfo=metadata)

    clean_memory()
    return elapsed_time, file_name


def evaluate_dst_selection(
    pipeline,
    output_folder,
    prompt_list,
    seed_list,
    *,
    num_tiles,
    num_steps,
    device,
    remark,
    height=1024,
    width=1024,
    warmup_runs=0,
    profiler=None,
):
    """Generate images for every prompt/seed combination and record timing."""
    warmup_seed = seed_list[0] if seed_list else 42
    for _ in range(warmup_runs):
        generate_image(
            pipeline=pipeline,
            output_folder=output_folder,
            prompt="warmup",
            random_seed=warmup_seed,
            num_tiles=num_tiles,
            num_steps=num_steps,
            device=device,
            remark=remark,
            height=height,
            width=width,
            save_image=False,
        )
        if profiler is not None:
            profiler.step()

    configurations = itertools.product(prompt_list, seed_list)
    total = len(prompt_list) * len(seed_list)
    results = []

    for prompt, seed in tqdm(
        configurations, total=total, desc="Processing configurations"
    ):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        elapsed_time, file_name = generate_image(
            pipeline=pipeline,
            output_folder=output_folder,
            prompt=prompt,
            random_seed=seed,
            num_tiles=num_tiles,
            num_steps=num_steps,
            device=device,
            remark=remark,
            height=height,
            width=width,
        )
        if profiler is not None:
            profiler.step()
        if torch.cuda.is_available():
            current_memory = torch.cuda.memory_allocated()
            peak_memory = torch.cuda.max_memory_allocated()
            print(
                f"Current memory: {current_memory / 1048576:.2f}MiB, "
                f"Peak memory: {peak_memory / 1048576:.2f}MiB"
            )

        results.append(
            {
                "prompt": prompt,
                "seed": seed,
                "elapsed_time": elapsed_time,
                "file_name": file_name,
                "current_memory": current_memory,
                "peak_memory": peak_memory,
            }
        )

    return results


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate images with different configurations."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    return parser.parse_args()


def prepare_pipeline(device, cache_dir=None, lora_path=None, method="masking"):
    kwargs = {"torch_dtype": torch.bfloat16, "local_files_only": True}
    if cache_dir:
        kwargs["cache_dir"] = str(cache_dir)

    pipeline = FluxPipeline.from_pretrained(MODEL_REPO, **kwargs).to(device)

    # Dynamically import customized_forward based on method
    if method == "masking":
        from masking_utils import customized_forward, customized_call
    elif method == "reorder":
        from reorder_utils_sliding import customized_forward
    elif method == "reorder_shared":
        from reorder_utils_sliding_shared import customized_forward
    else:
        raise ValueError(f"Unknown method: {method}. Must be 'masking', 'reorder', or 'reorder_shared'")

    pipeline.transformer.forward = MethodType(customized_forward, pipeline.transformer)

    # For masking method, replace pipeline's class to enable step tracking
    # We must modify the class because __call__ is a special method that must be defined on the class
    from masking_utils import customized_call
    class CustomFluxPipeline(pipeline.__class__):
        def __call__(self, *args, **kwargs):
            return customized_call(self, *args, **kwargs)

    pipeline.__class__ = CustomFluxPipeline

    if lora_path:
        pipeline.load_lora_weights(str(lora_path))

    return pipeline


def main():
    args = parse_args()
    default_config_path = Path(__file__).with_name("config.yaml")
    config_path = Path(args.config) if args.config else default_config_path
    config = load_config(config_path)

    curve_type = (config.get("curve_type") or "hilbert").lower()
    output_folder = Path(config["output_folder"]) / curve_type
    output_folder.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Curve type: {curve_type}")
    print(f"Output directory: {output_folder}")

    cache_dir_value = config.get("cache_dir", None)
    cache_dir = Path(cache_dir_value) if cache_dir_value else None

    lora_path_value = config.get("lora_weight_path", None)
    lora_path = Path(lora_path_value) if lora_path_value else None

    method = config.get("method", "masking")
    print(f"Method: {method}")

    pipeline = prepare_pipeline(
        device=device,
        cache_dir=cache_dir,
        lora_path=lora_path,
        method=method,
    )

    # Optional PyTorch profiler
    enable_profiler = bool(config.get("enable_profiler", False))
    print(f"Profiler enabled: {enable_profiler}")
    if enable_profiler:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        # Create a deterministic logs directory next to this file
        logs_dir = Path(__file__).with_name("profiler_logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        # Make the profiler fire immediately to guarantee output even with few steps
        schedule = torch.profiler.schedule(wait=0, warmup=3, active=1, repeat=1)
        handler = torch.profiler.tensorboard_trace_handler(str(logs_dir))
        print(f"Starting profiler with TensorBoard logging to {logs_dir}")
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=handler,
            record_shapes=False,  
            with_stack=True,     
            profile_memory=False, 
        ) as prof:
            print("Profiler context started")
            results = evaluate_dst_selection(
                pipeline=pipeline,
                output_folder=output_folder,
                prompt_list=config["prompt_list"],
                seed_list=config["seed_list"],
                num_tiles=config["num_tiles"],
                num_steps=config["num_of_inference_steps"],
                device=device,
                remark=config["remark"],
                height=config.get("height", 1024),
                width=config.get("width", 1024),
                warmup_runs=config.get("warmup_runs", 0),
                profiler=prof,
            )
            # Ensure at least one step is recorded and export a chrome trace as a fallback
            try:
                prof.step()
            except Exception:
                pass
            try:
                prof.export_chrome_trace(str(logs_dir / "trace.json"))
            except Exception:
                pass
    else:
        prompt_list = config["prompt_list"]

        results = evaluate_dst_selection(
            pipeline=pipeline,
            output_folder=output_folder,
            prompt_list=prompt_list,
            seed_list=config["seed_list"],
            num_tiles=config["num_tiles"],
            num_steps=config["num_of_inference_steps"],
            device=device,
            remark=config["remark"],
            height=config.get("height", 1024),
            width=config.get("width", 1024),
            warmup_runs=config.get("warmup_runs", 0),
        )

    if results:
        elapsed_times = [result["elapsed_time"] for result in results if result["elapsed_time"] is not None]
        if elapsed_times:
            avg_time = sum(elapsed_times) / len(elapsed_times)
            print(f"Average generation time: {avg_time:.2f}s")


if __name__ == "__main__":
    main()