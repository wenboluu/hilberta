# Repository Guidelines

## Project Structure & Module Organization
- Core implementation sits in `reloc_attention/` with Triton kernels under `triton_code/` and attention processors in `customized_attention_processor.py`.
- Supporting assets such as prebuilt masks live in `reloc_attention/mask_list/`; configuration and launch scripts (`config.yaml`, `run_flux.py`, `run_flux.sh`) also reside here.
- Integration samples (`diff.py`, `run_flux_editing.py`, `run_flux_inpainting.py`) are top-level; GPU tests are in `reloc_attention/test/`.

## Build, Test, and Development Commands
- `pip install -r reloc_attention/requirement.txt` installs dependencies including Torch 2.7 and Triton 3.3.
- `bash reloc_attention/run_flux.sh` launches the standard FLUX inference flow using `config.yaml`.
- `python reloc_attention/test/test_kernel.py` recompiles the Triton kernel and verifies numerical parity with the PyTorch reference.

## Coding Style & Naming Conventions
- Python 3.10+, 4-space indentation; follow PyTorch style for tensor naming (`q`, `k`, `v`) and keep module-level constants uppercase.
- Place Triton kernels in `reloc_attention/triton_code/` with descriptive snake_case filenames (e.g., `reloc_triton_kernel_bf16.py`).
- Prefer explicit imports from local modules and keep device-specific logic gated by `torch.device` checks.

## Testing Guidelines
- Use the provided Triton vs. PyTorch comparison script in `reloc_attention/test/`; add fast targeted unit checks when touching masking or grouping logic.
- Name new tests `test_<feature>.py` and ensure reproducible seeds with `torch.manual_seed`.
- Aim to validate both correctness (via `torch.allclose`) and performance-sensitive paths when introducing new kernels.

## Commit & Pull Request Guidelines
- Follow existing conventional prefixing (`[FEAT]`, `[FIX]`, `[DOC]`) based on recent history; keep messages under 72 characters in the first line.
- Include a short rationale plus impact or benchmarking notes in the body when altering Triton code or configs.
- Pull requests should summarize architecture changes, list tested commands, and attach sample outputs (image paths from `reloc_attention/output/` when relevant).

## Security & Configuration Tips
- Mask files are CUDA tensors; never check in regenerated assets without matching `config.yaml` entries.
- Validate any new kernel flags against supported CUDA 12.6 stack and keep cached artifacts disabled during review (`TRITON_DISABLE_CACHE=1`).
