"""
CLEAR Local Window Attention — FLUX.2-klein Inference

Smoke test to verify the attention processor + mask + pipeline work end-to-end.

Usage:
  python inference_flux2.py --window_size 8
  python inference_flux2.py --window_size 16 --checkpoint path/to/attn_weights.safetensors
"""

import argparse
import torch
from diffusers import Flux2KleinPipeline
from attention_processor_flux2 import (
    LocalFlexFlux2AttnProcessor,
    LocalFlexFlux2SingleAttnProcessor,
    init_local_mask_flex,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained attention weights (.safetensors)")
    parser.add_argument("--prompt", type=str, default="A cat sitting on a windowsill watching the rain.")
    parser.add_argument("--output", type=str, default="clear_flux2_test.png")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda"
    dtype = torch.bfloat16

    # 1. Load pipeline
    print("Loading Flux2KleinPipeline...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype
    ).to(device)

    # 2. Initialize local attention mask
    img_h = args.height // 16
    img_w = args.width // 16
    text_length = 512
    print(f"Initializing local mask: {img_h}x{img_w}, window_size={args.window_size}")
    init_local_mask_flex(img_h, img_w, text_length=text_length,
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
    print(f"Set {count} local attention processors")

    # 4. Optionally load trained weights
    if args.checkpoint:
        from safetensors.torch import load_file
        state_dict = load_file(args.checkpoint)
        missing, unexpected = pipe.transformer.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint: {len(state_dict)} keys, {len(missing)} missing, {len(unexpected)} unexpected")

    # 5. Generate image
    print(f"Generating: '{args.prompt}'")
    generator = torch.Generator(device=device).manual_seed(args.seed)
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=1.0,
        generator=generator,
    ).images[0]

    image.save(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
