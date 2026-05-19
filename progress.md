# Progress Log

## v0.11 — SpargeAttn FLUX.2 Adaptation and Speed Benchmarking

**Date:** 2026-03-28

**Changes:**
- Adapted SpargeAttn (block-sparse attention with INT8 quantization) for FLUX.2-klein
  - Created `sparge/evaluate/modify_model/modify_flux2.py` with FLUX.2-specific processors (`SageAttnFlux2AttnProcessor`, `SageAttnFlux2SingleAttnProcessor`)
  - Created `sparge/evaluate/flux2_inference.py` for single-GPU inference with tuning/manual hyperparams/sparsity logging
- Created SpargeAttn batch evaluation pipeline: `run_evaluation_flux2_sparge.py` + `.sbatch` + `submit_flux2_sparge_eval.sh`
  - Configurable sparsity hyperparameters (simthreshd1, cdfthreshd, pvthreshd) via CLI
- Created unified speed benchmark (`benchmark/benchmark_speed.py`) with two modes:
  - `--mode kernel`: Isolates attention kernels (SDPA vs HilbertA Triton vs SpargeAttn), uses real Q/K/V captured from pipeline
  - `--mode e2e`: Full pipeline timing (baseline vs HilbertA reorder vs SpargeAttn)
- Added GROUPS to Triton kernel autotuner key for correct per-tile-count tuning

**Known Issues:**
- HilbertA Triton kernel `reorder_shared` with 16 tiles crashes due to group_size=240 not being divisible by any valid BLOCK_M (64/128). Use `reorder` (no center) for 16t, or `masking` method.
- SpargeAttn kernel is slower than SDPA at 1024x1024 due to block selection overhead (INT8 quantization + CDF block selection). Designed for longer sequences.

## v0.10 — Seam Attention, Mask Correction, and Benchmark Suite

**Date:** 2026-03-28

**Changes:**
- Added `masking_seam` method: OR's a seam mask onto Hilbert tile masks, allowing boundary tokens at tile edges to cross-attend across tiles (reduces seam artifacts)
  - `_compute_seam_mask()` pre-computes seam positions for 64x64 and 128x128 grids
  - Supports both double-stream and single-stream processors
- Fixed masking shared region: removed experimental Hilbert center shared indices (256 extra tokens per tile center), now correctly uses only text (512) + spatial center 16x16 (256) = 768 shared tokens, consistent with `reorder_shared`
- Created LoRA evaluation pipeline: `run_evaluation_flux2_lora.py` + `.sbatch` + `submit_flux2_lora_eval.sh`
  - Loads LoRA weights (supports both final and accelerate checkpoint format)
  - Applies HilbertA masking patch with pre-computed GPU masks
  - Configurable output directory (5th arg) for A/B comparison
- Created unified benchmark suite (`benchmark/`): `benchmark.sh` runs FID (vs COCO test2017 stats), LPIPS, and CLIP similarity in one command
- Streamlined `benchmark_vs_sdpa.py` output for concise kernel timing results
- Made eval scripts parameterized: configurable inference steps and output directory

## v0.9 — LoRA Distillation Training Pipeline

**Date:** 2026-03-27

**Changes:**
- Added attention distillation LoRA training (`flux2_hilberta/training/`)
  - Teacher (vanilla FLUX.2) vs Student (FLUX.2 + HilbertA masking + LoRA), MSE loss
  - Validation image generation during training logged to wandb
  - CPU memory optimization for multi-GPU DDP training
- Pre-computed COCO test2017 InceptionV3 statistics for FID

## v0.8 — Multi-Model Support: FLUX.2-klein and Qwen-Image Adaptations

**Date:** 2026-03-25

**Changes:**
- Created `flux2_hilberta/` — full HilbertA adaptation for FLUX.2-klein-9B
  - Custom attention processors for both double-stream (`Flux2HilbertAttnProcessor`) and single-stream (`Flux2HilbertSingleAttnProcessor`) blocks
  - Adapted reorder utilities for FLUX.2's shared modulation architecture and separate RoPE computation
  - Fixed `num_txt_tokens` bug where single-stream blocks had text token count set to 0, breaking Triton sparse attention
- Created FLUX.2-klein batch evaluation pipeline
  - `run_evaluation_flux2.py` with batching support, resume capability, configurable prompt intervals
  - `run_evaluation_flux2.sbatch` SLURM job script for NYU HPC
  - `submit_flux2_eval.sh` auto-submitter for multi-GPU parallelism
- Added hierarchical `AGENTS.md` documentation across the codebase
- Updated `reloc_attention/requirement.txt` for latest diffusers compatibility

**Known Issues:**
- `flux2_hilberta` image quality needs verification after `num_txt_tokens` fix

## v0.7 — Benchmark Scripts

**Date:** 2026-03 (prior)

**Changes:**
- Added benchmark scripts for Triton kernel performance comparison

## v0.6 — Triton Kernel Optimizations

**Date:** 2026-03 (prior)

**Changes:**
- Optimized Triton kernel performance
- Fixed sliding window offset for non-zero offsets
- Enabled step-wise full attention control
- Enabled Morton curve attention
- Max distance sharing
- Reorder operations with/without sharing
- Kernel fusion and pipeline parallelization
