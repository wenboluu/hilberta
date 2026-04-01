# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a research implementation of **Relocated Attention** for the FLUX diffusion model. It optimizes attention computation by:
1. Reordering image tokens using Hilbert curves to preserve spatial locality
2. Applying local attention patterns with pre-computed masks
3. Using Triton GPU kernels for efficient computation
4. Implementing a sliding window mechanism with configurable offsets

The approach splits attention into shared (text) and image-specific parts, enabling efficient high-resolution image generation.

## Key Commands

### Running Inference
```bash
# Main execution (uses config.yaml)
./reloc_attention/run_flux.sh

# Direct Python execution
reloc_attention/penv/bin/python reloc_attention/run_flux.py --config reloc_attention/config.yaml
```

### Testing Triton Kernels
```bash
cd reloc_attention/triton_code/

# Test kernel accuracy
../penv/bin/python test_kernel_accuracy.py

# Benchmark kernel performance
../penv/bin/python benchmark.py

# Test parallel implementation
../penv/bin/python test_parallel.py
```

### Environment Setup
The project uses a local Python environment at `reloc_attention/penv/`. Dependencies are in `reloc_attention/requirement.txt`.

## Architecture

### Core Pipeline Flow
1. **Configuration** ([config.yaml](reloc_attention/config.yaml)): Defines num_tiles (4 or 16), sliding_cycle, prompts, and model parameters
2. **Patching** ([patch.py](reloc_attention/patch.py)): Replaces standard FLUX transformer blocks with custom attention processors
3. **Mask Generation** ([create_mask.py](reloc_attention/create_mask.py)): Pre-computes attention masks based on Hilbert curve patterns (stored in `mask_list/`)
4. **Reordering** ([reorder_utils.py](reloc_attention/reorder_utils.py)): Applies Hilbert curve transformations to tokens and rotary embeddings
5. **Attention** ([customized_attention_processor.py](reloc_attention/customized_attention_processor.py)): Custom attention with Triton kernels

### Critical Components

**reorder_utils.py** - Token reordering functions:
- `tile()` / `untile()`: Convert between spatial and tiled representations
- `hilbert_tile()` / `hilbert_untile()`: Apply Hilbert curve reordering
- `apply_hilbert_reorder()`: Main entry point for reordering pipeline

**patch.py** - Model patching:
- `apply_patch()`: Injects custom attention processors into FLUX transformer
- Creates sliding window info with offsets: `offset = (image_size/num_tiles)/sliding_cycle * i`
- Applies to both FluxTransformerBlock and FluxSingleTransformerBlock

**customized_attention_processor.py** - Attention implementation:
- Loads pre-computed masks from `mask_list/` at module import
- Implements `FluxAttnProcessor2_0_for_transformerblock_global`
- Supports both PyTorch and Triton implementations (switchable in code)

**triton_code/** - GPU kernels:
- `reloc_triton_kernel_bf16.py`: Main BF16 Triton kernel
- `reloc_triton_kernel_parallel.py`: Parallel computation variant
- `original_code.py`: Reference PyTorch implementation

### Configuration Parameters

The [config.yaml](reloc_attention/config.yaml) controls:
- `num_tiles`: 4 or 16 (determines spatial partitioning)
- `sliding_cycle`: Number of offset patterns (typically 4)
- `prompt_list`: Text prompts for generation
- `seed_list`: Random seeds
- `num_of_inference_steps`: Denoising steps (default 28)
- `full_attn_step` / `full_attn_layer`: Which steps/layers use full attention instead of local

### Image Size Support
- **1024x1024**: Uses 4096 tokens (64×64 grid)
- **2048x2048**: Uses 16384 tokens (128×128 grid)

Masks are pre-computed for: `{4096, 16384} × {4, 16 tiles} × {multiple offsets}`

## Current Development Branch

Branch: `triton` (based on git status)

Recent work focuses on Triton kernel optimizations:
- Max distance sharing
- Reorder operations with/without sharing
- Kernel fusion
- Pipeline parallelization

## Important Implementation Notes

### Mask System
- Masks are pre-loaded at module import in customized_attention_processor.py
- Mask key format: `{image_size}_{offset}_{num_tiles}` (e.g., "4096_256_4")
- Must run `create_mask.py` if mask_list/ is empty (automatically handled by run_flux.sh)

### Hilbert Curve Order
The Hilbert curve utilities in [utils.py](reloc_attention/utils.py) and [pattern_utils.py](reloc_attention/pattern_utils.py) handle:
- Forward mapping: spatial → Hilbert order
- Inverse mapping: Hilbert → spatial order
- Closed Hilbert curves for periodic patterns

### Sliding Window Mechanism
Each transformer block gets assigned an offset based on its layer index:
```python
offset = (image_size // num_tiles) // sliding_cycle * (layer_idx % sliding_cycle)
```
This creates different attention patterns across layers.

### Attention Processor Integration
The custom attention processor is injected via:
```python
module.attn.processor = FluxAttnProcessor2_0_for_transformerblock_global()
module.attn.processor._tome_info = info_dict  # Contains offset
```

## File Organization

```
reloc_attention/
├── run_flux.py              # Main inference script
├── run_flux.sh              # Bash wrapper with environment setup
├── config.yaml              # Configuration parameters
├── patch.py                 # Model patching logic
├── customized_attention_processor.py  # Custom attention implementation
├── reorder_utils.py         # Hilbert curve reordering
├── pattern_utils.py         # Hilbert pattern generation
├── masking_utils.py         # Mask computation utilities
├── create_mask.py           # Mask pre-computation script
├── utils.py                 # Shared utilities
├── mask_list/               # Pre-computed attention masks
├── triton_code/             # Triton kernel implementations
│   ├── reloc_triton_kernel_bf16.py
│   ├── reloc_triton_kernel_parallel.py
│   ├── benchmark.py
│   └── test_*.py
├── output/                  # Generated images with metadata
└── penv/                    # Local Python environment
```

## Working with Indices

Pre-computed Hilbert indices are stored as `.pt` files:
- `indices_4096_256.pt`: For 64×64 grids
- `indices_16384_1024.pt`: For 128×128 grids

These can be regenerated using utilities in `recompute_indices/` if needed.