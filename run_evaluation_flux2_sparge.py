import os
import json
import argparse
import torch
from diffusers import Flux2KleinPipeline

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sparge'))
from evaluate.modify_model.modify_flux2 import set_spas_sage_attn_flux2


def initialize_all_attention_modules(transformer, simthreshd1, cdfthreshd, pvthreshd):
    import torch.nn as nn
    from spas_sage_attn.autotune import SparseAttentionMeansim

    device = next(transformer.parameters()).device
    count = 0

    for block in transformer.transformer_blocks:
        if hasattr(block.attn, 'inner_attention'):
            module = block.attn.inner_attention
            head_num = block.attn.heads
            module.head_num = head_num
            module.is_sparse = nn.Parameter(torch.ones(head_num, dtype=torch.bool, device=device), requires_grad=False)
            module.cdfthreshd = nn.Parameter(torch.ones(head_num, device=device) * cdfthreshd, requires_grad=False)
            module.simthreshd1 = nn.Parameter(torch.ones(head_num, device=device) * simthreshd1, requires_grad=False)
            module.simthreshd2 = nn.Parameter(torch.zeros(head_num, device=device), requires_grad=False)
            module.pvthreshd = nn.Parameter(torch.ones(head_num, device=device) * pvthreshd, requires_grad=False)
            count += 1

    for block in transformer.single_transformer_blocks:
        if hasattr(block.attn, 'inner_attention'):
            module = block.attn.inner_attention
            head_num = block.attn.heads
            module.head_num = head_num
            module.is_sparse = nn.Parameter(torch.ones(head_num, dtype=torch.bool, device=device), requires_grad=False)
            module.cdfthreshd = nn.Parameter(torch.ones(head_num, device=device) * cdfthreshd, requires_grad=False)
            module.simthreshd1 = nn.Parameter(torch.ones(head_num, device=device) * simthreshd1, requires_grad=False)
            module.simthreshd2 = nn.Parameter(torch.zeros(head_num, device=device), requires_grad=False)
            module.pvthreshd = nn.Parameter(torch.ones(head_num, device=device) * pvthreshd, requires_grad=False)
            count += 1

    print(f"Initialized {count} attention modules with simthreshd1={simthreshd1}, cdfthreshd={cdfthreshd}, pvthreshd={pvthreshd}")


def main():
    parser = argparse.ArgumentParser(description="FLUX.2-klein evaluation with SpargeAttn")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=5000)
    parser.add_argument("--prompt_file", type=str, default="coco_prompts.json")
    parser.add_argument("--output_dir", type=str, default="output/flux2_sparge_eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--simthreshd1", type=float, default=-0.8)
    parser.add_argument("--cdfthreshd", type=float, default=0.06)
    parser.add_argument("--pvthreshd", type=float, default=0.0)
    args = parser.parse_args()

    with open(args.prompt_file, "r") as f:
        prompts = json.load(f)

    assert args.end <= len(prompts), f"End index {args.end} exceeds prompt count {len(prompts)}"

    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda"
    os.environ["TUNE_MODE"] = ""

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16
    ).to(device)

    set_spas_sage_attn_flux2(pipe.transformer)
    initialize_all_attention_modules(pipe.transformer, args.simthreshd1, args.cdfthreshd, args.pvthreshd)

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
