#!/usr/bin/env python
# coding=utf-8
"""
FLUX.2-klein LoRA Training with Attention Distillation.

Teacher: vanilla FLUX.2-klein transformer (frozen)
Student: FLUX.2-klein transformer with HilbertA relocated attention (LoRA trainable)
Loss: MSE(student_pred, teacher_pred)

Based on the diffusers DreamBooth LoRA training script, adapted for FLUX.2-klein
and the attention distillation pattern from train_vast.
"""

import argparse
import copy
import itertools
import logging
import math
import os
import random
import shutil
import sys
import types
import warnings
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

import diffusers
from diffusers import (
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    Flux2KleinPipeline,
    Flux2Transformer2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import is_wandb_available
from diffusers.utils.torch_utils import is_compiled_module

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def log_validation(args, accelerator, transformer, transformer_teacher, global_step, is_final_validation=False):
    """Generate validation images by loading a temporary pipeline."""
    if not args.validation_prompt:
        return

    logger.info(f"Running validation at step {global_step}... Generating {args.num_validation_images} images")

    weight_dtype = torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float16

    # Offload teacher to CPU to free GPU memory for pipeline
    transformer_teacher.to("cpu")
    torch.cuda.empty_cache()

    # Build a fresh pipeline with current student weights
    transformer_unwrapped = accelerator.unwrap_model(transformer)
    transformer_unwrapped = transformer_unwrapped._orig_mod if is_compiled_module(transformer_unwrapped) else transformer_unwrapped

    pipeline = Flux2KleinPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer_unwrapped,
        torch_dtype=weight_dtype,
        cache_dir=args.cache_dir,
    )
    pipeline.to(accelerator.device)
    pipeline.vae.to(dtype=torch.float32)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    prompts = [p.strip() for p in args.validation_prompt.split("||")]
    all_images = []
    all_captions = []
    for prompt in prompts:
        for i in range(args.num_validation_images):
            image = pipeline(
                prompt=prompt,
                num_inference_steps=28,
                guidance_scale=args.guidance_scale,
                height=args.resolution,
                width=args.resolution,
                generator=generator,
            ).images[0]
            all_images.append(image)
            all_captions.append(f"{prompt}")

    for tracker in accelerator.trackers:
        phase_name = "test" if is_final_validation else "validation"
        if tracker.name == "wandb":
            tracker.log({
                phase_name: [
                    wandb.Image(img, caption=cap) for img, cap in zip(all_images, all_captions)
                ]
            }, step=global_step)

    # Free pipeline (but keep transformer reference alive)
    del pipeline.text_encoder, pipeline.tokenizer, pipeline.vae
    del pipeline
    free_memory()

    # Move teacher back to GPU
    transformer_teacher.to(accelerator.device, dtype=weight_dtype)

    return all_images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="FLUX.2-klein LoRA training with attention distillation")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True,
                        help="Path to pretrained FLUX.2-klein model")
    parser.add_argument("--revision", type=str, default=None)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--instance_data_dir", type=str, required=True,
                        help="Folder containing training images")
    parser.add_argument("--instance_prompt", type=str, required=True,
                        help="Prompt for instance images, e.g. 'a photo of sks dog'")
    parser.add_argument("--validation_prompt", type=str, default=None,
                        help="Validation prompt(s), use '||' to separate multiple prompts")
    parser.add_argument("--num_validation_images", type=int, default=2,
                        help="Number of images per validation prompt")
    parser.add_argument("--validation_steps", type=int, default=500,
                        help="Run validation every N training steps")
    parser.add_argument("--output_dir", type=str, default="flux2-distill-lora")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--center_crop", action="store_true", default=False)
    parser.add_argument("--random_flip", action="store_true")
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--guidance_scale", type=float, default=1.0,
                        help="Guidance scale for distilled models (typically 1.0)")
    parser.add_argument("--scale_lr", action="store_true", default=False)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--rank", type=int, default=4, help="LoRA rank")
    parser.add_argument("--lora_layers", type=str, default=None,
                        help="Comma-separated list of modules for LoRA")
    parser.add_argument("--weighting_scheme", type=str, default="none",
                        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"])
    parser.add_argument("--logit_mean", type=float, default=0.0)
    parser.add_argument("--logit_std", type=float, default=1.0)
    parser.add_argument("--mode_scale", type=float, default=1.29)
    parser.add_argument("--optimizer", type=str, default="prodigy",
                        choices=["AdamW", "adamw", "prodigy"])
    parser.add_argument("--use_8bit_adam", action="store_true")
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--prodigy_beta3", type=float, default=None)
    parser.add_argument("--prodigy_decouple", type=bool, default=True)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-04)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--prodigy_use_bias_correction", type=bool, default=True)
    parser.add_argument("--prodigy_safeguard_warmup", type=bool, default=True)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--cache_latents", action="store_true", default=False)
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--exp_name", type=str, default="flux2-distill")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--text_encoder_out_layers", type=str, default="9,18,27",
                        help="Comma-separated layer indices for Qwen3 hidden states")
    # HilbertA-specific
    parser.add_argument("--num_tiles", type=int, default=4)
    parser.add_argument("--hilberta_method", type=str, default="reorder_shared",
                        choices=["masking", "reorder", "reorder_shared"])
    parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data")

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class DreamBoothDataset(Dataset):
    """Simple dataset for DreamBooth training."""

    def __init__(self, instance_data_dir, instance_prompt, size=1024, center_crop=False, repeats=1):
        self.size = size
        self.center_crop = center_crop

        self.instance_data_root = Path(instance_data_dir)
        if not self.instance_data_root.exists():
            raise ValueError(f"Instance data dir {instance_data_dir} does not exist")

        instance_images = [
            p for p in Path(instance_data_dir).iterdir()
            if p.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
        ]
        self.instance_images = instance_images * repeats
        self.instance_prompt = instance_prompt
        self.num_instance_images = len(self.instance_images)

        self.image_transforms = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size) if center_crop else transforms.RandomCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return self.num_instance_images

    def __getitem__(self, index):
        instance_image = Image.open(self.instance_images[index % self.num_instance_images])
        instance_image = exif_transpose(instance_image)
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")

        return {
            "instance_images": self.image_transforms(instance_image),
            "instance_prompt": self.instance_prompt,
        }


def collate_fn(examples):
    pixel_values = torch.stack([e["instance_images"] for e in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    prompts = [e["instance_prompt"] for e in examples]
    return {"pixel_values": pixel_values, "prompts": prompts}


def encode_prompt_qwen3(text_encoder, tokenizer, prompt, device, max_sequence_length=512,
                         hidden_states_layers=(9, 18, 27)):
    """Encode prompts using Qwen3 text encoder for FLUX.2."""
    prompt = [prompt] if isinstance(prompt, str) else prompt

    all_input_ids = []
    all_attention_masks = []
    for single_prompt in prompt:
        messages = [{"role": "user", "content": single_prompt}]
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

    with torch.no_grad():
        output = text_encoder(
            input_ids=input_ids, attention_mask=attention_mask,
            output_hidden_states=True, use_cache=False,
        )

    out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=text_encoder.dtype, device=device)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)

    return prompt_embeds


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.report_to == "wandb" and not is_wandb_available():
        raise ImportError("Install wandb: pip install wandb")

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process and args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    # ─── Load models ───
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler",
        cache_dir=args.cache_dir,
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Load models one at a time to reduce peak CPU memory.
    # Text encoder and VAE will be freed after caching, so no need to move to GPU yet.
    tokenizer = Qwen2TokenizerFast.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="tokenizer",
        cache_dir=args.cache_dir,
    )
    text_encoder = Qwen3ForCausalLM.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder",
        cache_dir=args.cache_dir,
    )

    vae = AutoencoderKLFlux2.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="vae",
        cache_dir=args.cache_dir,
    )

    # Free text encoder and VAE early (we use cached data)
    del text_encoder, tokenizer, vae
    free_memory()
    logger.info("Freed text_encoder, tokenizer, vae (using cached data)")

    # Student transformer — load and move to GPU immediately
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        cache_dir=args.cache_dir,
    )

    # Teacher transformer — load and move to GPU immediately
    transformer_teacher = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer",
        cache_dir=args.cache_dir,
    )

    # ─── Apply HilbertA patch to student ───
    flux2_hilberta_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, flux2_hilberta_dir)
    from patch import apply_patch
    from reorder_utils import customized_forward

    # Write a temporary config for the student
    import yaml
    hilberta_config = {
        "num_tiles": args.num_tiles,
        "sliding_cycle": 4,
        "method": args.hilberta_method,
        "full_attn_step": [],
        "full_attn_layer": [],
        "curve_type": "hilbert",
    }
    config_path = os.path.join(flux2_hilberta_dir, "config.yaml")
    # Only write if main process to avoid race conditions
    if accelerator.is_main_process:
        with open(config_path, 'r') as f:
            existing_config = yaml.safe_load(f)
        existing_config.update(hilberta_config)
        with open(config_path, 'w') as f:
            yaml.dump(existing_config, f)
    accelerator.wait_for_everyone()

    # Monkey-patch student forward with HilbertA reordering
    transformer.forward = types.MethodType(customized_forward, transformer)
    # Apply attention processor patches
    apply_patch(
        type('Pipe', (), {'transformer': transformer})(),  # mock pipe object
        num_tiles=args.num_tiles,
        height=args.resolution,
    )

    logger.info(f"HilbertA patch applied: method={args.hilberta_method}, num_tiles={args.num_tiles}")

    # ─── Freeze everything, then add LoRA to student ───
    transformer.requires_grad_(False)
    transformer_teacher.requires_grad_(False)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    transformer.to(accelerator.device, dtype=weight_dtype)
    transformer_teacher.to(accelerator.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # LoRA target modules for FLUX.2
    if args.lora_layers is not None:
        target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
    else:
        target_modules = [
            # Double-stream blocks
            "attn.to_k", "attn.to_q", "attn.to_v", "attn.to_out.0",
            "attn.add_k_proj", "attn.add_q_proj", "attn.add_v_proj", "attn.to_add_out",
            # Double-stream FF
            "ff.linear_in", "ff.linear_out",
            "ff_context.linear_in", "ff_context.linear_out",
            # Single-stream blocks (fused QKV+MLP input projection)
            "attn.to_qkv_mlp_proj",
            # Note: single-stream attn.to_out (Linear) shares name with
            # double-stream attn.to_out (ModuleList), causing peft error.
            # to_qkv_mlp_proj already covers input projections; to_out is
            # the fused output projection. We skip it to avoid the conflict.
        ]

    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian",
        target_modules=target_modules,
    )
    transformer.add_adapter(transformer_lora_config)

    # Upcast LoRA params to fp32 for mixed precision
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # ─── Optimizer ───
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    params_to_optimize = [{"params": transformer_lora_parameters, "lr": args.learning_rate}]

    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            import bitsandbytes as bnb
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )
    elif args.optimizer.lower() == "prodigy":
        import prodigyopt
        optimizer = prodigyopt.Prodigy(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # ─── Dataset (using cached latent codes and prompt embeds) ───
    from dataset import loader
    train_dataloader = loader(
        train_batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        img_dir=args.instance_data_dir,
        img_size=args.resolution,
        use_cached_prompt_embeds=True,
        use_cached_latent_codes=True,
    )

    vae_scale_factor = 8  # 2 ** (len(block_out_channels) - 1)

    # ─── LR Scheduler ───
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler, optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles, power=args.lr_power,
    )

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            "flux2-distill-lora", config=vars(args),
            init_kwargs={"wandb": {"name": args.exp_name}},
        )

    # ─── Training loop ───
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running training *****")
    logger.info(f"  Num batches per epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0

    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting fresh.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps), initial=initial_global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(transformer):
                with torch.inference_mode():
                    # Load cached latent codes
                    mean = batch['latent_codes_mean'].to(dtype=weight_dtype, device=accelerator.device)
                    std = batch['latent_codes_std'].to(dtype=weight_dtype, device=accelerator.device)
                    sample = torch.randn_like(mean)
                    model_input = mean + std * sample  # Already patchified + BN normalized from cache

                    # Load cached prompt embeds
                    prompt_embeds = batch['prompt_embeds'].to(dtype=weight_dtype, device=accelerator.device)

                    # Prepare latent IDs (pre-compute on first step, reuse after)
                    if not hasattr(main, '_cached_latent_ids') or main._cached_latent_ids.shape[0] != model_input.shape[0]:
                        main._cached_latent_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(accelerator.device)
                    latent_ids = main._cached_latent_ids

                    # Pack for transformer input: [B, C, H, W] -> [B, H*W, C]
                    packed_model_input = Flux2KleinPipeline._pack_latents(model_input)

                    # Prepare text IDs (pre-compute on first step, reuse after)
                    if not hasattr(main, '_cached_text_ids') or main._cached_text_ids.shape[0] != prompt_embeds.shape[0]:
                        main._cached_text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(accelerator.device)
                    text_ids = main._cached_text_ids

                    # Sample noise
                    noise = torch.randn_like(packed_model_input)
                    bsz = packed_model_input.shape[0]

                    # Sample timesteps
                    u = compute_density_for_timestep_sampling(
                        weighting_scheme=args.weighting_scheme,
                        batch_size=bsz,
                        logit_mean=args.logit_mean,
                        logit_std=args.logit_std,
                        mode_scale=args.mode_scale,
                    )
                    indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                    timesteps = noise_scheduler_copy.timesteps[indices].to(device=packed_model_input.device)

                    # Add noise (flow matching): zt = (1 - t) * x + t * z
                    sigmas = get_sigmas(timesteps, n_dim=packed_model_input.ndim, dtype=packed_model_input.dtype)
                    noisy_model_input = (1.0 - sigmas) * packed_model_input + sigmas * noise

                    # Guidance
                    if accelerator.unwrap_model(transformer).config.guidance_embeds:
                        guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
                        guidance = guidance.expand(bsz)
                    else:
                        guidance = None

                    # ─── Teacher forward (no grad) ───
                    teacher_pred = transformer_teacher(
                        hidden_states=noisy_model_input,
                        timestep=timesteps / 1000,
                        guidance=guidance,
                        encoder_hidden_states=prompt_embeds,
                        txt_ids=text_ids,
                        img_ids=latent_ids,
                        return_dict=False,
                    )[0]

                # ─── Student forward (with grad, HilbertA applied) ───
                model_pred = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_ids,
                    return_dict=False,
                )[0]

                # ─── Distillation loss: MSE(student, teacher) ───
                loss = ((model_pred.float() - teacher_pred.float()) ** 2).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                    if args.checkpoints_total_limit is not None:
                        checkpoints = os.listdir(args.output_dir)
                        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                        if len(checkpoints) >= args.checkpoints_total_limit:
                            num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                            for removing_checkpoint in checkpoints[:num_to_remove]:
                                shutil.rmtree(os.path.join(args.output_dir, removing_checkpoint))

                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

                if accelerator.is_main_process and args.validation_prompt and global_step % args.validation_steps == 0:
                    log_validation(args, accelerator, transformer, transformer_teacher, global_step)

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # Save checkpoint each epoch
        if accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
            accelerator.save_state(save_path)
            logger.info(f"Saved state to {save_path}")

    # ─── Final validation ───
    if accelerator.is_main_process and args.validation_prompt:
        log_validation(args, accelerator, transformer, transformer_teacher, global_step, is_final_validation=True)

    # ─── Save final LoRA weights ───
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer_unwrapped = accelerator.unwrap_model(transformer)
        transformer_unwrapped = transformer_unwrapped._orig_mod if is_compiled_module(transformer_unwrapped) else transformer_unwrapped
        transformer_unwrapped = transformer_unwrapped.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer_unwrapped)

        Flux2KleinPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )
        logger.info(f"LoRA weights saved to {args.output_dir}")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
