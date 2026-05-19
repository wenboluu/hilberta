"""
Cache Qwen3 prompt embeddings for FLUX.2-klein training.

Encodes text prompts using Qwen3ForCausalLM with multi-layer hidden state extraction
(layers 9, 18, 27), then saves as safetensors for fast training loading.

Unlike FLUX.1 which uses CLIP (pooled) + T5 (sequence), FLUX.2-klein uses a single
Qwen3 encoder that outputs stacked multi-layer hidden states.
"""

import argparse
import os
import json
import math

import torch
import tqdm
from safetensors.torch import save_file
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Cache Qwen3 prompt embeddings for FLUX.2-klein")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--data_root", type=str, required=True,
                        help="Directory containing .json files with 'prompt' field")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--output_dir", type=str, default="flux2-lora")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--text_encoder_out_layers", type=str, default="9,18,27",
                        help="Comma-separated layer indices for Qwen3 hidden states")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    return args


def encode_prompt_qwen3(text_encoder, tokenizer, prompts, device, max_sequence_length=512,
                         hidden_states_layers=(9, 18, 27)):
    """Encode prompts using Qwen3 text encoder for FLUX.2."""
    all_input_ids = []
    all_attention_masks = []

    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        inputs = tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=max_sequence_length,
        )
        all_input_ids.append(inputs["input_ids"])
        all_attention_masks.append(inputs["attention_mask"])

    input_ids = torch.cat(all_input_ids, dim=0).to(device)
    attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

    output = text_encoder(
        input_ids=input_ids, attention_mask=attention_mask,
        output_hidden_states=True, use_cache=False,
    )

    # Stack selected hidden layers: [B, num_layers, seq_len, hidden_dim]
    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=text_encoder.dtype, device=device)

    # Reshape to [B, seq_len, num_layers * hidden_dim]
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

    return prompt_embeds


def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.mixed_precision == 'fp16':
        dtype = torch.float16
    elif args.mixed_precision == 'bf16':
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        cache_dir=args.cache_dir,
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=args.revision,
        cache_dir=args.cache_dir,
    ).to(device, dtype)

    hidden_states_layers = tuple(int(x) for x in args.text_encoder_out_layers.split(","))

    # Find all .json files with prompts
    all_info = sorted([
        os.path.join(args.data_root, f)
        for f in os.listdir(args.data_root) if f.endswith('.json')
    ])

    os.makedirs(args.output_dir, exist_ok=True)

    work_load = math.ceil(len(all_info) / args.num_workers)
    start = work_load * args.local_rank
    end = min(work_load * (args.local_rank + 1), len(all_info))

    for idx in tqdm.tqdm(range(start, end, args.batch_size)):
        batch_files = all_info[idx:idx + args.batch_size]
        paths = [
            os.path.join(
                args.output_dir,
                os.path.splitext(os.path.basename(item))[0] + '_prompt_embed.safetensors'
            )
            for item in batch_files
        ]

        # Skip if all already cached
        if all(os.path.exists(p) for p in paths):
            continue

        texts = []
        for item in batch_files:
            with open(item) as f:
                texts.append(json.load(f)['prompt'])

        with torch.no_grad():
            prompt_embeds = encode_prompt_qwen3(
                text_encoder, tokenizer, texts, device,
                max_sequence_length=args.max_sequence_length,
                hidden_states_layers=hidden_states_layers,
            )
            prompt_embeds = prompt_embeds.cpu()

        for path, embed in zip(paths, prompt_embeds.unbind()):
            # Save as 'prompt_embeds' (single key, unlike FLUX.1's dual CLIP+T5)
            save_file({'prompt_embeds': embed}, path)


if __name__ == '__main__':
    main(parse_args())
