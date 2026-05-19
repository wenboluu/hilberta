# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of **Relocated Attention (HilbertA)** for diffusion models. It optimizes attention computation by:
1. Reordering image tokens using Hilbert curves to preserve spatial locality
2. Applying local attention patterns with pre-computed masks
3. Using Triton GPU kernels for efficient computation
4. Implementing a sliding window mechanism with configurable offsets

The approach splits attention into shared (text) and image-specific parts, enabling efficient high-resolution image generation. Supports multiple model backends: FLUX.1 and FLUX.2-klein.

## Key Commands

### Running Inference
```bash
# FLUX.1 (original)
cd reloc_attention && bash run_flux.sh

# FLUX.2-klein
cd flux2_hilberta && bash run_flux2.sh
```

### FLUX.2 Batch Evaluation
```bash
# Submit 5 GPU jobs, 1000 images each, batch_size=16
bash submit_flux2_eval.sh 5

# Or manually
sbatch run_evaluation_flux2.sbatch 0 1000
```

### Testing Triton Kernels
```bash
cd reloc_attention/triton_code/
python test_clean.py
python test_simple_equivalence.py
python benchmark.py
```

### Environment
- Conda environment: `hilberta` (`conda activate hilberta`)
- Dependencies: `reloc_attention/requirement.txt`

## Architecture

### Model Adaptations

Each model has its own directory with adapted versions of the core components:

| Component | FLUX.1 (`reloc_attention/`) | FLUX.2 (`flux2_hilberta/`) |
|-----------|---------------------------|---------------------------|
| Block types | FluxTransformerBlock + FluxSingleTransformerBlock | Flux2TransformerBlock + Flux2SingleTransformerBlock |
| RoPE format | Real (cos, sin) concatenated [text+image] | Real (cos, sin) computed separately then concatenated |
| Modulation | Inside blocks (norm1) | Outside blocks (shared modulation modules) |
| Text tokens | Fixed 512 | Fixed 512 |
| Triton kernels | Local | Symlink to reloc_attention/triton_code |

### Shared Components (model-agnostic)
- **Triton kernels** (`reloc_attention/triton_code/`): Operate on generic Q/K/V tensors
- **Hilbert curve math** (`utils.py` in each dir): `get_hilbert_flat_indices()`, `get_inverse_hilbert_indices()`
- **Tiling primitives**: `tile()`, `untile()`, `hilbert_tile()`, `hilbert_untile()`

### Core Pipeline Flow
1. **Configuration** (`config.yaml`): num_tiles, sliding_cycle, method, prompts
2. **Patching** (`patch.py`): Replaces transformer blocks' attention processors
3. **Reordering** (`reorder_utils.py`): Hilbert curve reordering of image tokens + RoPE
4. **Attention** (`customized_attention_processor.py`): Sparse attention via Triton kernels
5. **Customized forward** (in `reorder_utils.py`): Replaces transformer's forward method

### Configuration Parameters

Each `config.yaml` controls:
- `num_tiles`: 4 or 16 (spatial partitioning)
- `sliding_cycle`: Number of offset patterns (typically 4)
- `method`: `"reorder_shared"` (default), `"reorder"`, `"masking"`, or `"masking_seam"`
- `full_attn_step` / `full_attn_layer`: Steps/layers using full attention

### Image Size Support
- **1024x1024**: 4096 tokens (64x64 grid) — all models
- **2048x2048**: 16384 tokens (128x128 grid) — all models

### Sliding Window Mechanism
Each transformer block gets an offset: `offset = (image_size // num_tiles) // sliding_cycle * (layer_idx % sliding_cycle)`

## Evaluation Pipeline

- `run_evaluation_flux2.py`: Baseline batch generation (no HilbertA patch)
- `run_evaluation_flux2_lora.py`: LoRA + HilbertA masking batch generation, supports accelerate checkpoint loading
- `run_evaluation_flux2_sparge.py`: SpargeAttn batch generation with configurable sparsity hyperparameters
- `run_evaluation_flux2.sbatch` / `run_evaluation_flux2_lora.sbatch` / `run_evaluation_flux2_sparge.sbatch`: SLURM job scripts (A100, 8 CPU, 32GB)
- `submit_flux2_eval.sh` / `submit_flux2_lora_eval.sh` / `submit_flux2_sparge_eval.sh`: Auto-submits N jobs with evenly divided ranges

### Quality Benchmarking
- `benchmark/benchmark.sh <generated_dir> [baseline_dir]`: Runs FID (vs COCO test2017 stats), LPIPS, and CLIP similarity
- `benchmark/compute_fid.py`, `benchmark/compute_lpips_clip.py`: Individual metric scripts

### Speed Benchmarking
- `benchmark/benchmark_speed.py --mode kernel`: Kernel-only timing (captures real Q/K/V from pipeline, benchmarks SDPA vs HilbertA Triton vs SpargeAttn)
- `benchmark/benchmark_speed.py --mode e2e`: End-to-end pipeline timing (baseline vs HilbertA reorder vs SpargeAttn)

## SpargeAttn Integration

SpargeAttn (block-sparse attention with INT8 Q/K quantization) is integrated as a comparison baseline:
- `sparge/`: Cloned SpargeAttn repo with FLUX.2 adaptation in `evaluate/modify_model/modify_flux2.py`
- Build: `cd sparge && pip install --no-build-isolation -e .` (requires GPU node, CUDA 12.6 via conda)
- Key hyperparameters: `simthreshd1` (block homogeneity gate), `cdfthreshd` (CDF block budget), `pvthreshd` (runtime pruning)
- On A100 (SM 8.0): uses sageattn1 (INT8 Q/K + FP16 values); SM 8.9+ uses sageattn2 (FP8 values)

## File Organization

```
├── reloc_attention/              # FLUX.1 implementation (original)
│   ├── patch.py, customized_attention_processor.py, reorder_utils*.py
│   ├── triton_code/              # Shared Triton kernels
│   └── config.yaml
├── flux2_hilberta/               # FLUX.2-klein adaptation
│   ├── patch.py, customized_attention_processor.py, reorder_utils.py
│   ├── masking_utils.py          # Hilbert tile mask creation
│   ├── training/                 # LoRA distillation training
│   ├── triton_code -> ../reloc_attention/triton_code
│   └── config.yaml
├── sparge/                       # SpargeAttn baseline comparison
│   └── evaluate/modify_model/modify_flux2.py  # FLUX.2 adaptation
├── run_evaluation_flux2.py       # Baseline batch eval
├── run_evaluation_flux2_lora.py  # LoRA + HilbertA batch eval
├── run_evaluation_flux2_sparge.py # SpargeAttn batch eval
├── submit_flux2_eval.sh          # Multi-GPU submitter (baseline)
├── submit_flux2_lora_eval.sh     # Multi-GPU submitter (LoRA)
├── submit_flux2_sparge_eval.sh   # Multi-GPU submitter (SpargeAttn)
├── benchmark/                    # FID, LPIPS, CLIP, speed benchmarks
└── coco_prompts.json             # Evaluation prompts
```

## Important Implementation Notes

### Attention Processor Integration
Custom processors are injected via `patch.py`:
```python
module.attn.processor = CustomAttnProcessor()
module.attn.processor._tome_info = {"args": {"offset": offset, ...}}
```

### FLUX.2 Single-Stream Blocks
Single-stream blocks need `num_txt_tokens` set on their processors before execution so the Triton kernel correctly identifies shared (text) vs local (image) regions. This is set dynamically in `customized_forward`.

## Commit Style
Use conventional prefixes: `[FEAT]`, `[FIX]`, `[PERF]`, `[DOCS]`
