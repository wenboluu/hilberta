#!/usr/bin/env python
# coding=utf-8
"""
CLEAR Local Window Attention Distillation for FLUX.2-klein.

Closely follows CLEAR/distill.py structure, adapted for FLUX.2-klein:
- Flux2Transformer2DModel instead of FluxTransformer2DModel
- Qwen3 text encoder (cached) instead of CLIP+T5
- AutoencoderKLFlux2 (cached) instead of AutoencoderKL
- FLUX.2-specific attention processors (fused QKV, single-stream MLP)

Teacher: vanilla FLUX.2-klein (frozen, full attention)
Student: FLUX.2-klein with local window attention (trainable attention weights)
Loss: loss_fm + 0.5 * loss_distill + 0.5 * loss_attn
"""

import argparse
import copy
import logging
import math
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from safetensors.torch import save_file
from tqdm.auto import tqdm

import diffusers
from diffusers import (
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

from attention_processor_flux2 import (
    Flux2AttnProcessor,
    Flux2SingleAttnProcessor,
    LocalFlexFlux2AttnProcessor,
    LocalFlexFlux2SingleAttnProcessor,
    init_local_mask_flex,
    attn_outputs,
    attn_outputs_teacher,
)

if is_wandb_available():
    import wandb

logger = get_logger(__name__)


def log_validation(pipeline, args, accelerator, epoch, torch_dtype):
    """Generate validation images and log to wandb."""
    pipeline.to(accelerator.device, dtype=torch_dtype)
    pipeline.set_progress_bar_config(disable=True)

    generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None

    images = []
    prompts = [p.strip() for p in args.validation_prompt.split("||")]
    for prompt in prompts:
        for _ in range(args.num_validation_images):
            image = pipeline(
                prompt=prompt, num_inference_steps=28,
                guidance_scale=args.guidance_scale,
                height=args.resolution, width=args.resolution,
                generator=generator,
            ).images[0]
            images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({"validation": [wandb.Image(img) for img in images]}, step=epoch)

    del pipeline
    free_memory()
    return images


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="CLEAR distillation for FLUX.2-klein")
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True, help="Directory with cached training data")
    parser.add_argument("--output_dir", type=str, default="exp_output_clear")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=0)
    parser.add_argument("--lr_num_cycles", type=int, default=1)
    parser.add_argument("--lr_power", type=float, default=1.0)
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--max_sequence_length", type=int, default=512)
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
    parser.add_argument("--report_to", type=str, default="wandb")
    parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--checkpointing_steps", type=int, default=500)
    parser.add_argument("--checkpoints_total_limit", type=int, default=3)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--validation_prompt", type=str, default=None)
    parser.add_argument("--validation_epochs", type=int, default=1)
    parser.add_argument("--num_validation_images", type=int, default=2)
    parser.add_argument("--exp_name", type=str, default="clear-flux2-distill")
    # CLEAR-specific
    parser.add_argument("--window_size", type=int, default=16)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()
    return args


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
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

    # ─── Load scheduler ───
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="scheduler", cache_dir=args.cache_dir,
    )

    # ─── Load transformers (student + teacher) ───
    transformer = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", cache_dir=args.cache_dir,
    )
    transformer_teacher = Flux2Transformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="transformer", cache_dir=args.cache_dir,
    )

    # ─── Initialize local attention mask ───
    init_local_mask_flex(
        args.resolution // 16, args.resolution // 16,
        text_length=args.max_sequence_length,
        window_size=args.window_size, device=accelerator.device,
    )
    logger.info(f"Initialized local mask: {args.resolution//16}x{args.resolution//16}, window={args.window_size}")

    # ─── Set attention processors ───
    # FLUX.2's attn_processors property may return empty dict, so iterate named_modules
    student_count, teacher_count = 0, 0

    for name, module in transformer.named_modules():
        if hasattr(module, 'processor') and hasattr(module, 'heads'):
            is_single = 'single' in name
            distill = is_single and student_count % 4 == 0
            if is_single:
                module.processor = LocalFlexFlux2SingleAttnProcessor(distill=distill)
            else:
                module.processor = LocalFlexFlux2AttnProcessor(distill=distill)
            student_count += 1

    for name, module in transformer_teacher.named_modules():
        if hasattr(module, 'processor') and hasattr(module, 'heads'):
            is_single = 'single' in name
            distill = is_single and teacher_count % 4 == 0
            if is_single:
                module.processor = Flux2SingleAttnProcessor(distill=distill)
            else:
                module.processor = Flux2AttnProcessor(distill=distill)
            teacher_count += 1

    logger.info(f"Set processors: {student_count} student, {teacher_count} teacher")

    # ─── Freeze everything ───
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

    # ─── Unfreeze attention weights (matching original distill.py line 737-739) ───
    for _name, _param in transformer.named_parameters():
        if '.attn.to_q.' in _name or '.attn.to_k.' in _name or \
           '.attn.to_v.' in _name or '.attn.to_out.' in _name:
            _param.requires_grad = True

    trainable_count = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in transformer.parameters())
    logger.info(f"Trainable: {trainable_count:,} / {total_count:,} ({trainable_count/total_count*100:.1f}%)")

    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # Make sure trainable params are in float32 for fp16
    if args.mixed_precision == "fp16":
        cast_training_params([transformer], dtype=torch.float32)

    # ─── Optimizer (matching original distill.py line 762-825) ───
    transformer_attn_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    params_to_optimize = [{"params": transformer_attn_parameters, "lr": args.learning_rate}]

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
        if args.learning_rate <= 0.1:
            logger.warning("Learning rate is too low for Prodigy. Generally better to set around 1.0")
        optimizer = prodigyopt.Prodigy(
            params_to_optimize,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            beta3=args.prodigy_beta3,
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
            decouple=args.prodigy_decouple,
            use_bias_correction=args.prodigy_use_bias_correction,
            safeguard_warmup=args.prodigy_safeguard_warmup,
        )

    # ─── Dataset (reuse HilbertA's cached data loader) ───
    training_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "flux2_hilberta", "training")
    sys.path.insert(0, training_dir)
    from dataset import loader
    train_dataloader = loader(
        train_batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        img_dir=args.data_root,
        img_size=args.resolution,
        use_cached_prompt_embeds=True,
        use_cached_latent_codes=True,
    )

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

    # ─── Prepare with accelerator ───
    guidance_embeds = transformer.config.guidance_embeds

    transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        transformer, optimizer, train_dataloader, lr_scheduler
    )

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if accelerator.is_main_process:
        accelerator.init_trackers(
            "clear-flux2-distill", config=vars(args),
            init_kwargs={"wandb": {"name": args.exp_name}},
        )

    # ─── Training ───
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    logger.info("***** Running CLEAR distillation *****")
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
            path = args.resume_from_checkpoint
        else:
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if dirs else None

        if path is None:
            accelerator.print(f"Checkpoint '{args.resume_from_checkpoint}' not found. Starting fresh.")
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(path)
            global_step = int(path.split("-")[-1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps), initial=initial_global_step, desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()

        for _, batch in enumerate(train_dataloader):
            # Note: no accelerator.accumulate() — DeepSpeed ZeRO-2 handles
            # gradient accumulation internally via deepspeed_config.yaml

            # ─── Load cached embeddings ───
            prompt_embeds = batch['prompt_embeds'].to(dtype=weight_dtype)
            text_ids = Flux2KleinPipeline._prepare_text_ids(prompt_embeds).to(
                device=accelerator.device)

            with torch.no_grad():
                # ─── Load cached latents ───
                mean = batch['latent_codes_mean'].to(dtype=weight_dtype)
                std = batch['latent_codes_std'].to(dtype=weight_dtype)
                model_input = mean + std * torch.randn_like(mean)

                latent_ids = Flux2KleinPipeline._prepare_latent_ids(model_input).to(
                    accelerator.device)
                packed_model_input = Flux2KleinPipeline._pack_latents(model_input)

                # ─── Sample noise and timesteps ───
                noise = torch.randn_like(packed_model_input)
                bsz = packed_model_input.shape[0]

                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme, batch_size=bsz,
                    logit_mean=args.logit_mean, logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler.config.num_train_timesteps).long()
                timesteps = noise_scheduler.timesteps[indices].to(device=packed_model_input.device)

                sigmas = get_sigmas(timesteps, n_dim=packed_model_input.ndim,
                                    dtype=packed_model_input.dtype)
                noisy_model_input = (1.0 - sigmas) * packed_model_input + sigmas * noise

                # ─── Guidance ───
                if guidance_embeds:
                    guidance = torch.tensor([args.guidance_scale], device=accelerator.device).expand(bsz)
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

            # ─── Student forward (with grad) ───
            model_pred = transformer(
                hidden_states=noisy_model_input,
                timestep=timesteps / 1000,
                guidance=guidance,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_ids,
                return_dict=False,
            )[0]

            # ─── Loss (matching original distill.py lines 1064-1074) ───
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=args.weighting_scheme, sigmas=sigmas)
            target = noise - packed_model_input

            loss_fm = (weighting.float() * (model_pred.float() - target.float()) ** 2).mean()
            loss_distill = (weighting.float() * (model_pred.float() - teacher_pred.float()) ** 2).mean()

            if len(attn_outputs) > 0 and len(attn_outputs_teacher) > 0:
                loss_attn = sum([
                    (weighting.float().squeeze(-1) * (ao.float() - at.float()) ** 2).mean()
                    for ao, at in zip(attn_outputs, attn_outputs_teacher)
                ]) / len(attn_outputs)
            else:
                loss_attn = torch.tensor(0.0, device=accelerator.device)

            loss = loss_fm + loss_distill * 0.5 + loss_attn * 0.5

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            attn_outputs.clear()
            attn_outputs_teacher.clear()

            # Check if optimization step happened
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for ckpt in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, ckpt))

                if global_step % args.checkpointing_steps == 0:
                    save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)
                    logger.info(f"Saved state to {save_path}")

            logs = {
                "loss_fm": loss_fm.detach().item(),
                "loss_distill": loss_distill.detach().item(),
                "loss_attn": loss_attn.detach().item() if torch.is_tensor(loss_attn) else 0.0,
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

        # ─── End-of-epoch validation ───
        free_memory()
        if accelerator.is_main_process:
            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                pipeline = Flux2KleinPipeline.from_pretrained(
                    args.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    torch_dtype=weight_dtype,
                    cache_dir=args.cache_dir,
                )
                log_validation(pipeline, args, accelerator, epoch, weight_dtype)

    # ─── Save attention weights (matching original distill.py lines 1172-1189) ───
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        transformer_unwrapped = unwrap_model(transformer)
        transformer_unwrapped = transformer_unwrapped.to(weight_dtype)

        state_dict = {}
        for _name, _param in transformer_unwrapped.named_parameters():
            if '.attn.to_q.' in _name or '.attn.to_k.' in _name or \
               '.attn.to_v.' in _name or '.attn.to_out.' in _name:
                state_dict[_name] = _param

        save_file(state_dict, os.path.join(args.output_dir, 'attn_weights.safetensors'))
        logger.info(f"Saved {len(state_dict)} attention weight tensors to {args.output_dir}/attn_weights.safetensors")

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
