# HilbertA — Relocated Attention for Diffusion Models

A research implementation of **Relocated Attention** using Hilbert space-filling curves for efficient local attention in diffusion transformers. Supports multiple model backends.

## Supported Models

| Model | Directory | Status |
|-------|-----------|--------|
| FLUX.1 | `reloc_attention/` | Stable |
| FLUX.2-klein-9B | `flux2_hilberta/` | In development |

## How It Works

1. **Hilbert Reordering**: Image tokens are reordered using Hilbert curves to preserve spatial locality
2. **Local Attention**: Tokens are split into tiles; each tile attends only within itself + a shared region (text + center)
3. **Triton Kernels**: Sparse attention is computed via custom Triton GPU kernels (Flash Attention v2 based)
4. **Sliding Window**: Per-layer offsets create diverse attention patterns across transformer layers

## Quick Start

### FLUX.1 (original)
```bash
cd reloc_attention && bash run_flux.sh
```

### FLUX.2-klein
```bash
cd flux2_hilberta && bash run_flux2.sh
```

### FLUX.2 Evaluation (batch generation on COCO prompts)
```bash
# Generate 5000 images across 5 GPUs
bash submit_flux2_eval.sh 5
```

## Project Structure

```
├── reloc_attention/          # FLUX.1 relocated attention (original)
├── flux2_hilberta/           # FLUX.2-klein adaptation
├── run_evaluation_flux2.py   # Batch evaluation script (COCO prompts)
├── run_evaluation_flux2.sbatch  # SLURM job script
├── submit_flux2_eval.sh      # Multi-GPU job submitter
├── run_flux2.py              # FLUX.2-klein quick test script
├── run_qwen_img.py           # Qwen-Image quick test script
└── coco_prompts.json         # COCO evaluation prompts
```

## Environment

- Python 3.10+, PyTorch 2.7+, Triton 3.3+
- Conda environment: `hilberta`
- Dependencies: `reloc_attention/requirement.txt`

## Attention Methods

Configured via `config.yaml` → `method`:

- **`reorder_shared`** (default): Hilbert reorder with center region as shared global attention
- **`reorder`**: Simple Hilbert reorder, all image tokens local
- **`masking`**: Pre-computed boolean attention masks
