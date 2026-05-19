import os
import sys
import json
import argparse
import torch
from safetensors.torch import load_file
from diffusers import Flux2KleinPipeline
from peft import LoraConfig, set_peft_model_state_dict

# Add flux2_hilberta to path
flux2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux2_hilberta")
sys.path.insert(0, flux2_dir)


def precompute_masks(device, num_tiles=4, sliding_cycle=4, seq_img=4096, seq_txt=512):
    """Pre-compute all Hilbert tile masks and store in GPU cache."""
    from customized_attention_processor import _mask_cache, _full_mask_cache
    from masking_utils import create_hilbert_tile_mask

    image_size = seq_img  # 4096 for 1024x1024
    L = seq_img + seq_txt

    for i in range(sliding_cycle):
        offset = (image_size // num_tiles) // sliding_cycle * i

        # Hilbert mask for image tokens
        hilbert_key = f'{seq_img}_{offset}_{num_tiles}'
        if hilbert_key not in _mask_cache:
            _mask_cache[hilbert_key] = create_hilbert_tile_mask(
                torch.empty(1, seq_img, device=device), num_tiles, offset
            )

        # Full mask for double-stream (text + image)
        full_key = f'{L}_{seq_txt}_{offset}_{num_tiles}'
        if full_key not in _full_mask_cache:
            full_mask = torch.zeros(L, L, dtype=torch.bfloat16, device=device)
            full_mask[-seq_img:, -seq_img:] = _mask_cache[hilbert_key].to(device, torch.bfloat16)
            _full_mask_cache[full_key] = full_mask

        # Full mask for single-stream (same layout, keyed with num_txt)
        single_full_key = f'{L}_{seq_txt}_{offset}_{num_tiles}'
        if single_full_key not in _full_mask_cache:
            full_mask = torch.zeros(L, L, dtype=torch.bfloat16, device=device)
            full_mask[-seq_img:, -seq_img:] = _mask_cache[hilbert_key].to(device, torch.bfloat16)
            _full_mask_cache[single_full_key] = full_mask

    print(f"Pre-computed {len(_mask_cache)} hilbert masks and {len(_full_mask_cache)} full masks on {device}")


def main():
    parser = argparse.ArgumentParser(description="FLUX.2-klein + LoRA + HilbertA evaluation")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=5000)
    parser.add_argument("--prompt_file", type=str, default="coco_prompts.json")
    parser.add_argument("--output_dir", type=str, default="output/flux2_lora_eval")
    parser.add_argument("--lora_dir", type=str, required=True,
                        help="Path to LoRA output dir (picks latest checkpoint or final weights)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_tiles", type=int, default=4)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank (must match training)")
    args = parser.parse_args()

    # Load prompts
    with open(args.prompt_file, "r") as f:
        prompts = json.load(f)
    assert args.end <= len(prompts), f"End index {args.end} exceeds prompt count {len(prompts)}"

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda"

    # --- Load pipeline ---
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16
    ).to(device)

    # --- Load LoRA weights ---
    lora_path = args.lora_dir
    if os.path.exists(os.path.join(lora_path, "pytorch_lora_weights.safetensors")):
        # Final saved weights — use pipe.load_lora_weights directly
        pipe.load_lora_weights(lora_path)
        print(f"Loaded final LoRA weights from {lora_path}")
    else:
        # Accelerate checkpoint (model.safetensors) — need to add adapter first
        if not os.path.exists(os.path.join(lora_path, "model.safetensors")):
            # Maybe lora_path is the parent dir, find latest checkpoint
            checkpoints = sorted(
                [d for d in os.listdir(lora_path) if d.startswith("checkpoint-")],
                key=lambda x: int(x.split("-")[1])
            )
            if checkpoints:
                lora_path = os.path.join(lora_path, checkpoints[-1])
                print(f"Using latest checkpoint: {lora_path}")
            else:
                raise FileNotFoundError(f"No LoRA weights found in {args.lora_dir}")

        # Add LoRA adapter with same config as training
        target_modules = [
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            "ff.linear_in", "ff.linear_out",
            "ff_context.linear_in", "ff_context.linear_out",
            "attn.to_qkv_mlp_proj",
        ]
        lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        pipe.transformer.add_adapter(lora_config)

        # Load checkpoint state dict
        state_dict = load_file(os.path.join(lora_path, "model.safetensors"))
        set_peft_model_state_dict(pipe.transformer, state_dict)
        print(f"Loaded LoRA from accelerate checkpoint: {lora_path}")

    # --- Apply HilbertA masking patch ---
    from patch import apply_patch
    apply_patch(pipe, num_tiles=args.num_tiles, height=args.height)
    print(f"HilbertA masking patch applied (num_tiles={args.num_tiles})")

    # --- Pre-compute masks on GPU ---
    seq_img = (args.height // 16) ** 2  # 4096 for 1024x1024
    precompute_masks(device, num_tiles=args.num_tiles, seq_img=seq_img)

    # --- Generate ---
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
