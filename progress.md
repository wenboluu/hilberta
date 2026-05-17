# Progress Log

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
