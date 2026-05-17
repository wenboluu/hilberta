<!-- Generated: 2026-03-24 | Updated: 2026-03-24 -->

# qwen_image (Relocated Attention for FLUX)

## Purpose
Research implementation of **Relocated Attention** for the FLUX diffusion model. Optimizes attention computation by reordering image tokens using space-filling curves (Hilbert/Morton) to exploit spatial locality, enabling efficient high-resolution image generation with local attention patterns and Triton GPU kernels.

## Key Files

| File | Description |
|------|-------------|
| `CLAUDE.md` | Project documentation and development guide for Claude Code |
| `diff.py` | CLI utility to compute pixel-wise difference images between two PNG files |
| `imageNet1k_class.py` | ImageNet-1K class label dictionary (synset ID → human-readable name) |
| `morton_vis.ipynb` | Jupyter notebook for visualizing Morton curve patterns |

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `reloc_attention/` | Core implementation: attention processors, reordering, masking, inference pipeline (see `reloc_attention/AGENTS.md`) |

## For AI Agents

### Working In This Directory
- Root-level files are utilities and data assets; the main implementation lives in `reloc_attention/`
- The local Python environment is at `reloc_attention/penv/` — use `reloc_attention/penv/bin/python` to run scripts
- Configuration is in `reloc_attention/config.yaml`; always check it before modifying attention behavior
- Pre-computed masks live in `reloc_attention/mask_list/` (or `mask_hilbert/`, `mask_morton/`) — do not check in regenerated masks without matching config entries

### Testing Requirements
- Run `reloc_attention/penv/bin/python reloc_attention/triton_code/test_clean.py` for kernel correctness
- Use `torch.allclose` for numerical parity checks against PyTorch reference
- Ensure reproducible seeds with `torch.manual_seed`

### Common Patterns
- PyTorch style tensor naming: `q`, `k`, `v` for query/key/value
- Python 3.10+, 4-space indentation
- Triton kernels use BF16 precision by default
- Commit messages use conventional prefixes: `[FEAT]`, `[FIX]`, `[PERF]`, `[DOC]`

## Dependencies

### External
- `torch` 2.7+ — deep learning framework
- `triton` 3.3+ — GPU kernel compiler
- `diffusers` — Hugging Face diffusion pipeline (FLUX)
- `hilbertcurve` — Hilbert curve index computation
- `PIL/Pillow` — image I/O

<!-- MANUAL: Any manually added notes below this line are preserved on regeneration -->
