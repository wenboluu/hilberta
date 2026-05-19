"""
CLEAR Local Window Attention — FLUX.2-klein batch evaluation.

Loads a DeepSpeed checkpoint, extracts trained attention weights,
applies local window attention processors, generates images.
"""

import os
import sys
import json
import argparse
import torch
from diffusers import Flux2KleinPipeline

# Add CLEAR dir to path
clear_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CLEAR")
sys.path.insert(0, clear_dir)
from attention_processor_flux2 import (
    LocalFlexFlux2AttnProcessor,
    LocalFlexFlux2SingleAttnProcessor,
    init_local_mask_flex,
)


def load_attn_weights_from_deepspeed_ckpt(ckpt_dir):
    """Extract attention weights from DeepSpeed ZeRO-2 checkpoint."""
    model_file = os.path.join(ckpt_dir, "pytorch_model", "mp_rank_00_model_states.pt")
    sd = torch.load(model_file, map_location="cpu")
    # DeepSpeed wraps in 'module' key
    full_sd = sd.get("module", sd)
    # Filter to attention weights only
    attn_sd = {}
    for k, v in full_sd.items():
        if '.attn.to_q.' in k or '.attn.to_k.' in k or \
           '.attn.to_v.' in k or '.attn.to_out.' in k:
            attn_sd[k] = v
    return attn_sd


def main():
    parser = argparse.ArgumentParser(description="CLEAR FLUX.2-klein evaluation")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=5000)
    parser.add_argument("--prompt_file", type=str, default="coco_prompts.json")
    parser.add_argument("--output_dir", type=str, default="output/flux2_clear_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to DeepSpeed checkpoint dir (e.g. exp_output_w8/checkpoint-3000)")
    args = parser.parse_args()

    with open(args.prompt_file, "r") as f:
        prompts = json.load(f)

    assert args.end <= len(prompts), f"End index {args.end} exceeds prompt count {len(prompts)}"
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda"

    # 1. Load pipeline
    print("Loading Flux2KleinPipeline...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16
    ).to(device)

    # 2. Initialize local attention mask
    img_h = args.height // 16
    img_w = args.width // 16
    init_local_mask_flex(img_h, img_w, text_length=512,
                         window_size=args.window_size, device=device)

    # 3. Set local attention processors
    count = 0
    for name, module in pipe.transformer.named_modules():
        if hasattr(module, 'processor') and hasattr(module, 'heads'):
            if 'single' in name:
                module.processor = LocalFlexFlux2SingleAttnProcessor()
            else:
                module.processor = LocalFlexFlux2AttnProcessor()
            count += 1
    print(f"Set {count} local attention processors (window_size={args.window_size})")

    # 4. Load trained attention weights from checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    attn_sd = load_attn_weights_from_deepspeed_ckpt(args.checkpoint)
    missing, unexpected = pipe.transformer.load_state_dict(attn_sd, strict=False)
    print(f"Loaded {len(attn_sd)} attention weight tensors, {len(missing)} missing, {len(unexpected)} unexpected")

    # 5. Generate images
    for batch_start in range(args.start, args.end, args.batch_size):
        batch_end = min(batch_start + args.batch_size, args.end)

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
