import argparse
import json
import math
import os
import time
from typing import List, Tuple

import torch
from diffusers import FluxPipeline

# Fixed generation parameters
HEIGHT = 2048
WIDTH = 2048
GUIDANCE_SCALE = 3.5
NUM_INFERENCE_STEPS = 28
MAX_SEQUENCE_LENGTH = 512
MAX_PROMPTS = 5000
SEED = 42


def load_prompts(prompts_path: str, limit: int) -> List[str]:
    with open(prompts_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("coco_prompts.json must be a JSON list of strings")
    return data[:limit]


def list_completed_indices(output_dir: str) -> set:
    if not os.path.isdir(output_dir):
        return set()
    completed = set()
    for name in os.listdir(output_dir):
        if not name.lower().endswith(".png"):
            continue
        stem = name[:-4]
        if stem.isdigit():
            try:
                completed.add(int(stem))
            except ValueError:
                pass
    return completed


def split_contiguous_sections(total: int, num_sections: int) -> List[Tuple[int, int]]:
    if total <= 0 or num_sections <= 0:
        return []
    chunk = math.ceil(total / num_sections)
    sections: List[Tuple[int, int]] = []
    for i in range(num_sections):
        start = i * chunk
        end = min(start + chunk, total)
        if start < end:
            sections.append((start, end))
    return sections


def generate_on_device(
    device_id: int,
    prompts: List[str],
    index_start: int,
    index_end: int,
    output_dir: str,
):
    device = f"cuda:{device_id}"
    torch.cuda.set_device(device_id)
    # NOTE: Modify cache_dir below for your server's cache location
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        cache_dir="/data2/wl2707/",  # TODO: Update this path for your server
    ).to(device)

    os.makedirs(output_dir, exist_ok=True)

    for idx in range(index_start, index_end):
        out_path = os.path.join(output_dir, f"{idx:06d}.png")
        if os.path.exists(out_path):
            continue

        prompt = prompts[idx]
        try:
            generator = torch.Generator(device=device).manual_seed(SEED)
            image = pipe(
                prompt,
                height=HEIGHT,
                width=WIDTH,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=NUM_INFERENCE_STEPS,
                max_sequence_length=MAX_SEQUENCE_LENGTH,
                generator=generator,
            ).images[0]
            image.save(out_path)
        except Exception as e:
            print(f"[Device {device_id}] Error at index {idx}: {e}")
            continue


def main():
    # Get script directory for relative paths
    script_dir = os.path.dirname(os.path.abspath(__file__))

    parser = argparse.ArgumentParser(description="Parallel multi-GPU FLUX generation with resume support")
    parser.add_argument("--no-resume", action="store_true", help="Disable resume; regenerate even if files exist")
    parser.add_argument("--gpus", type=str, default="", help="Comma-separated CUDA device indices to use, e.g. '0,1,3'")
    parser.add_argument("--prompts", type=str, default=os.path.join(script_dir, "coco_prompts.json"),
                        help="Path to prompts JSON file")
    parser.add_argument("--output-dir", type=str, default=os.path.join(script_dir, "flux_2048"),
                        help="Output directory for generated images")

    args = parser.parse_args()

    prompts_path = args.prompts
    output_dir = args.output_dir

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for multi-GPU generation")

    # Determine devices to use
    all_device_count = torch.cuda.device_count()
    if args.gpus.strip():
        device_ids = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    else:
        device_ids = list(range(all_device_count))
    # Basic validation
    for did in device_ids:
        if did < 0 or did >= all_device_count:
            raise ValueError(f"GPU index {did} is out of range [0, {all_device_count - 1}]")

    print(f"Using CUDA device(s): {device_ids}")

    prompts = load_prompts(prompts_path, MAX_PROMPTS)
    limit = len(prompts)
    print(f"Loaded {limit} prompt(s)")

    os.makedirs(output_dir, exist_ok=True)

    # Determine which indices are already completed to accelerate resume
    completed = list_completed_indices(output_dir) if not args.no_resume else set()
    if completed:
        print(f"Found {len(completed)} completed indices; existing files will be skipped")

    # Pre-create simple manifest for traceability
    manifest_path = os.path.join(output_dir, "manifest.json")
    try:
        if not os.path.exists(manifest_path):
            with open(manifest_path, "w") as mf:
                json.dump({"prompts_path": os.path.abspath(prompts_path), "total": limit}, mf)
    except Exception:
        pass

    # Split first `limit` indices into contiguous sections per device
    sections = split_contiguous_sections(limit, len(device_ids))
    print("Sections:", sections)

    # Spawn one process per device
    from multiprocessing import Process

    processes: List[Process] = []
    for device_id, (start, end) in zip(device_ids, sections):
        # Each worker will check and skip existing files to support resume
        p = Process(
            target=generate_on_device,
            args=(device_id, prompts, start, end, output_dir),
        )
        p.start()
        processes.append(p)

    # Join
    for p in processes:
        p.join()

    print("All processes finished.")


if __name__ == "__main__":
    # Ensure CUDA is initialized in spawned subprocesses, not via fork
    import multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # Start method may have been set already (e.g., in interactive sessions)
        pass
    main()