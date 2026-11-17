# Guide for reader agents

Use this file to help a reader install, run, reproduce, or extend the public FLUX implementation. Follow the reader's requested scope. Detailed commands and data formats are in `docs/REPRODUCE.md`.

## Supported setup

- Linux, Python 3.10; PyTorch 2.9.0 / CUDA 12.6, Triton 3.5.0, Diffusers 0.36.0. Use `requirements-cu126.txt` as constraints; do not upgrade dependencies just to resolve an unrelated task.
- Base model: FLUX.1-dev, with revision pinned in `hilberta/runtime.py`. Readers obtain model access and authenticate locally with Hugging Face; never put credentials in code or commits.
- Default: composite Hilbert, 4 tiles, reciprocal K1, no shared image region. Faster alternative: 16 tiles / K3. Configurations cover 1024 and 2048.
- K limits external selections before reciprocal closure; the final number of external neighbors can exceed K. Text queries remain global, and image queries attend to text.
- Inference: merged rank-32 LoRA, paired Triton attention, persistent layout across sampling. Training: Flex64, per-layer packing/unpacking, frozen base weights and trainable FP32 rank/alpha-32 LoRA.
- The paper covers more than this release. Do not assume FLUX.2, video, or every illustrated paper variant is implemented here.

## Find the implementation

| Files | Responsibility |
|---|---|
| `hilberta/geometry.py`, `plan.py`, `bridge_k.py` | Ordering, four-phase schedule, reciprocal bridge selection |
| `hilberta/paired_kernel.py`, `paired_metadata.py` | Default inference kernel and contiguous block metadata |
| `hilberta/flux_adapter.py`, `sampling_layout.py` | Model integration and persistent sampling layout |
| `hilberta/flex_sparse.py`, `sparse_kernel.py` | Differentiable backend and alternative sparse kernel |
| `hilberta/cache.py`, `train.py`, `runtime.py` | Teacher trajectories, DDP, adapter/checkpoint loading |
| `hilberta/generate.py`, `evaluate.py`, `benchmark.py` | Generation, image metrics, resident kernel benchmark |
| `configs/`, `tests/` | Runnable configurations and correctness checks |
| `results/`, `docs/RESULTS.md` | Published aggregates, metric definitions, comparison scope |

## Run a reader's experiment

1. Follow README installation; run `python -m pip check`. Each CLI supports `--help`.
2. Inspect the chosen configuration and adapter metadata. Do not mix adapters with a different tile count, bridge K, or generation resolution.
3. For a baseline use `hilberta.generate --method full`. An untrained sparse run requires `--allow-untrained` and is not evidence of the reported trained quality.
4. For training, use the guide's manifest → teacher cache → DDP → adapter workflow. Keep prompt IDs disjoint across train/eval and preserve the same evaluation prompts, seeds, scheduler, and sampling parameters across methods.
5. Before GPU-intensive jobs, inspect available memory and active jobs; select the devices the reader authorized. Do not interrupt other processes or start full-model training as a smoke test.
6. Keep generated images, caches, credentials, and checkpoints out of git. Output directories must be new; cache generation has no automatic resume. Inspect incomplete shards instead of silently mixing them into a new cache.
7. Training `--resume` requires the same world size, effective batch configuration, seed, cache signature, and configuration. `--init-adapter` starts a fresh optimizer. `last.pt` is recovery state; `adapter_step*.pt` is for inference.

## Correctness and checks

Do not change token identity, RoPE association, attention topology, or softmax normalization as a performance optimization. Base and extra attention must share the same normalizer; independent attention outputs cannot be added or averaged. Keep text connectivity and the four-phase schedule intact unless the reader explicitly requests a different method.

```bash
python -m pip install -c requirements-cu126.txt -e '.[dev]'
ruff check hilberta tests
ruff format --check hilberta tests
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest tests -q
```

These CPU tests do not download FLUX. For kernel changes, also run the relevant target-GPU correctness checks before benchmarking. Changes to packing, positions, or the sampling loop need model/sampling equivalence checks; CPU tests alone do not establish BF16 full-model parity. For documentation-only changes, check links and assets; GPU work is unnecessary.

## Report results accurately

- Trained release adapters and exact historical prompt manifests are not bundled. Portable cache/train/evaluation code still needs full production parity validation. Do not claim exact reproduction from sample prompts or untrained outputs.
- FID/KID use real COCO images. LPIPS/PSNR/SSIM use matched Full FLUX outputs. CLIP measures image/text alignment.
- Separate resident attention-kernel latency, denoiser forward latency, and 28-step denoising time. Record hardware, resolution, dtype, batch size, warmup, and timing boundaries. Reported denoising time excludes VAE decode and file writing.
- A fixed tile count does not imply linear complexity. Do not infer latency from sparse-entry counts, and do not invent missing competitor kernel or 4096 quality measurements.
- Keep result data unchanged unless replacing it with traceable measurements. State what was run and what remains unverified.

## Documentation and assets

Keep the README compact. Put operational instructions in `docs/REPRODUCE.md` and result definitions in `docs/RESULTS.md`. Do not add private handovers, agent transcripts, server paths, or development diaries to the public repository. `AGENTS.md` is the single public agent guide.

The README pipeline is the authors' paper figure, not a newly drawn diagram; its source PDF and provenance are under `docs/assets/`. Website redesign is a separate task. Do not alter `website/index.html`, styles, scripts, or demo assets during a code/documentation task unless requested.
