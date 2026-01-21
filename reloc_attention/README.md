# Relocated Attention Mechanism

A high-performance attention mechanism implementation for large-scale image generation, optimized for the FLUX model.

## Overview

The relocated attention mechanism improves efficiency by:
1. Splitting attention computation into shared and image-specific parts
2. Using Triton kernels for optimized GPU computation

## Implementation

### Core Components
- `customized_attention_processor.py`: Main attention processor
- `triton_code/`: Optimized Triton kernels
  - `reloc_triton_kernel_bf16.py`: BF16 precision kernel
  - `original_code.py`: Reference implementation

### Supporting Files
- `mask_list/`: Pre-computed attention masks
- `create_mask.py`: Mask generation script
- `config.yaml`: Model parameters
- `run_flux.sh`: Execution script

## Quick Start
1. Run inference:
```bash
./run_flux.sh
```

2. Configure parameters in `config.yaml`:
```yaml
num_tiles: 4  # or 16
prompt_list: ["your prompt"]
seed_list: [42]
num_of_inference_steps: 28
output_folder: "output"
```

## Directory Structure
```
reloc_attention/
├── customized_attention_processor.py  # Main attention implementation
├── triton_code/                      # Optimized Triton kernels
├── mask_list/                        # Pre-computed attention masks
├── create_mask.py                    # Mask generation script
├── config.yaml                       # Configuration file
├── run_flux.sh                       # Execution script
└── run_flux.py                       # Main execution script
```

## Testing

To test different implementations:
1. Open `customized_attention_processor.py`
2. Uncomment the desired implementation section:
   - PyTorch implementation
   - Triton implementation