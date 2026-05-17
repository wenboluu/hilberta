<!-- Parent: ../AGENTS.md -->
<!-- Generated: 2026-03-24 | Updated: 2026-03-24 -->

# reloc_attention

## Purpose
Core implementation of Relocated Attention for FLUX. Contains the full inference pipeline: model patching, token reordering via space-filling curves, attention mask generation, custom attention processors, and multiple reorder strategy variants (basic, sliding window, shared sliding window).

## Key Files

| File | Description |
|------|-------------|
| `run_flux.py` | Main inference script — loads FLUX pipeline, applies patches, generates images |
| `run_flux.sh` | Bash launcher with environment setup (creates masks if missing, activates penv) |
| `config.yaml` | Central configuration: `num_tiles`, `sliding_cycle`, `curve_type`, `method`, prompts, seeds |
| `patch.py` | Model patching — replaces FLUX transformer blocks with custom `ToMeBlock` classes that use relocated attention; computes per-layer sliding window offsets |
| `customized_attention_processor.py` | `FluxAttnProcessor2_0_for_transformerblock_global` — custom attention processor supporting masking, reorder, and reorder_shared methods; loads config at module import |
| `reorder_utils.py` | Token reordering with shared text context — `tile()`, `untile()`, `hilbert_tile()`, `hilbert_untile()`, `apply_hilbert_reorder()` |
| `reorder_utils_sliding.py` | Sliding window variant of reorder — adds offset-based Hilbert index shifting |
| `reorder_utils_sliding_shared.py` | Shared sliding window variant — combines sliding offsets with LRU-cached index computation |
| `masking_utils.py` | Mask-based approach — `create_hilbert_tile_mask()` generates attention masks from Hilbert/Morton indices |
| `create_mask.py` | Pre-computes attention mask `.pt` files for all size/offset/tile combinations |
| `pattern_utils.py` | Closed Hilbert curve mapping utilities — forward/inverse mappings for periodic patterns |
| `utils.py` | Shared utilities: `isinstance_str()`, `init_generator()`, `get_hilbert_flat_indices()`, `get_inverse_hilbert_indices()`, `get_morton_flat_indices()` |
| `baseline.py` | Baseline FLUX inference without relocated attention (for comparison) |
| `original_pipe.py` | Original unmodified pipeline reference |
| `original_pipeline_parallel.py` | Original pipeline with parallelization support |
| `dev_log.md` | Development log and notes |
| `requirement.txt` | Python dependencies (torch, triton, diffusers, etc.) |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `triton_code/` | Triton GPU kernel implementations, benchmarks, and tests (see `triton_code/AGENTS.md`) |
| `mask_list/` | Pre-computed attention mask tensors (`.pt` files); key format: `{image_size}_{offset}_{num_tiles}` |
| `output/` | Generated images with metadata |
| `penv/` | Local Python virtual environment |

## For AI Agents

### Working In This Directory
- **Three attention methods** exist (set via `config.yaml` → `method`):
  - `masking`: Pre-computed boolean masks applied to standard SDPA
  - `reorder`: Token reordering via Hilbert curve, then local attention per tile
  - `reorder_shared`: Reorder with shared text tokens across all tiles (current default)
- **Module import side effects**: `customized_attention_processor.py` reads `config.yaml` and loads mask files at import time — be aware of working directory assumptions
- **Three reorder_utils variants**: `reorder_utils.py` (basic shared), `reorder_utils_sliding.py` (sliding), `reorder_utils_sliding_shared.py` (sliding + shared with caching). The active one is imported based on the method
- **Sliding window offsets** are computed in `patch.py`: `offset = (image_size // num_tiles) // sliding_cycle * (layer_idx % sliding_cycle)`
- **Encoder hidden states** (text tokens) are averaged across tiles after attention in `patch.py` `ToMeBlock.forward()`

### Testing Requirements
- Run kernel accuracy tests from `triton_code/` before changing attention logic
- After modifying reorder functions, verify round-trip: `reorder → attention → unreorder` should match baseline
- Use `baseline.py` to generate reference outputs for comparison
- Always test with both `num_tiles=4` and `num_tiles=16`

### Common Patterns
- Hilbert indices are computed via `get_hilbert_flat_indices(p)` where `p` is the grid exponent (6 for 64x64, 7 for 128x128)
- Tile operations use `torch.as_strided` for zero-copy reshaping
- `_tome_info` dict is attached to attention processors to pass per-layer config (offset, step, etc.)
- RoPE (rotary positional embeddings) are reordered alongside tokens

## Dependencies

### Internal
- `triton_code/` — GPU kernels called from attention processors
- `mask_list/` — pre-computed masks loaded by `customized_attention_processor.py`

### External
- `diffusers` — FLUX pipeline, `Attention`, `AttnProcessor2_0`, model outputs
- `torch` — tensor operations, SDPA, profiling
- `hilbertcurve` — Hilbert curve index computation
- `numpy` — array utilities for index manipulation
- `yaml` — configuration loading

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
