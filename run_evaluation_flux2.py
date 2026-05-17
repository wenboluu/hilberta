import os
import json
import argparse
import torch
from diffusers import Flux2KleinPipeline


def main():
    parser = argparse.ArgumentParser(description="FLUX.2-klein evaluation on COCO prompts")
    parser.add_argument("--start", type=int, default=0, help="Start index (inclusive)")
    parser.add_argument("--end", type=int, default=5000, help="End index (exclusive)")
    parser.add_argument("--prompt_file", type=str, default="coco_prompts.json")
    parser.add_argument("--output_dir", type=str, default="output/flux2_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=4)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    # Load prompts
    with open(args.prompt_file, "r") as f:
        prompts = json.load(f)

    assert args.end <= len(prompts), f"End index {args.end} exceeds prompt count {len(prompts)}"

    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    device = "cuda"
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16
    ).to(device)

    # Generate images in batches
    for batch_start in range(args.start, args.end, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.end)

        # Collect prompts and indices that need generation (resume support)
        batch_indices = []
        batch_prompts = []
        for i in range(batch_start, batch_end):
            out_path = os.path.join(args.output_dir, f"{i:05d}.png")
            if os.path.exists(out_path):
                print(f"[{i}] Already exists, skipping.")
            else:
                batch_indices.append(i)
                batch_prompts.append(prompts[i])

        if not batch_prompts:
            continue

        generators = [torch.Generator(device=device).manual_seed(args.seed) for _ in batch_prompts]

        images = pipe(
            prompt=batch_prompts,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            generator=generators,
        ).images

        for idx, img in zip(batch_indices, images):
            out_path = os.path.join(args.output_dir, f"{idx:05d}.png")
            img.save(out_path)
            print(f"[{idx}] Saved: {out_path}")


if __name__ == "__main__":
    main()
