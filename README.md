# HilbertA

Composite Hilbert ordering, sliding tiles, and reciprocal block bridges for sparse attention in FLUX.1-dev.

[Paper](https://openreview.net/forum?id=yhMt5V4FNb) · [Project page & results](https://wenboluu.github.io/hilberta/) · [Reproduction guide](docs/REPRODUCE.md) · [Validation status](docs/IMPLEMENTATION.md)

This release contains the FLUX inference kernels, model integration, LoRA training tools, and evaluation code. It supports 1024 and 2048 image generation; 4096 has performance measurements only. Video experiments are outside this release.

**Trained LoRA weights are not yet published on Hugging Face or bundled here.** You can run full-attention generation, untrained sparse smoke tests, or train your own adapter. Reproducing the reported quality requires the matching trained adapter and evaluation manifest; neither is bundled. The portable training/evaluation runners still need production parity checks against the historical runs.

## Which configuration should I use?

| Configuration | Use |
|---|---|
| **4 tiles / K1 / no shared image region** | **Default.** Best fidelity to Full FLUX among our evaluated sparse configurations. |
| 16 tiles / K3 / no shared image region | Faster alternative with lower teacher fidelity. |

The default combines **4 tiles, K1, no shared image region, and reciprocal block bridges**. Both configurations keep global text attention. K is the number of selected external blocks **before** reciprocal closure. LoRA rank and alpha are 32.

- **Inference:** paired Triton kernel, merged LoRA, persistent image layout across the entire sampling loop. Reorder once; restore raster order before decoding.
- **Training:** Flex64 (FlexAttention with 64-token blocks). This backend supports autograd and packs/unpacks each layer.
- **Reference:** explicit masks and small CPU model tests. The runtime does not silently switch backends.

## Installation

Linux, Python 3.10, an NVIDIA GPU with BF16 support, and a driver compatible with CUDA 12.6. Reported benchmarks use H100 80 GB. The pinned stack is PyTorch 2.9.0, Triton 3.5.0, and Diffusers 0.36.0.

```bash
git clone --branch main https://github.com/wenboluu/hilberta.git
cd hilberta
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -c requirements-cu126.txt -e '.[test,eval]'
python -m pip check
```

Accept the [FLUX.1-dev model terms](https://huggingface.co/black-forest-labs/FLUX.1-dev), then run `hf auth login`. Model weights download separately. The model revision is pinned in [runtime.py](hilberta/runtime.py).

## Generate images

The example prompts are smoke-test inputs, not the paper evaluation set. Each output directory must be new.

```bash
# Original full-attention FLUX.
python -m hilberta.generate --method full \
  --prompts configs/prompts.example.jsonl --output outputs/full

# Untrained sparse smoke test. This does not reproduce the reported quality.
python -m hilberta.generate --allow-untrained \
  --prompts configs/prompts.example.jsonl --output outputs/untrained

# Use an adapter saved by hilberta.train, when available.
python -m hilberta.generate --adapter checkpoints/adapter_step001792.pt \
  --prompts configs/prompts.example.jsonl --output outputs/hilberta
```

For 16 tiles, add `--tiles 16 --k 3` and use its matching adapter. For 2048, add `--resolution 2048` and use the matching 2048 adapter. The default protocol is 28 steps, guidance 7.5, seed 2707, with the seed reset for each prompt.

## Train, evaluate, and benchmark

The [reproduction guide](docs/REPRODUCE.md) provides complete commands and the prompt-manifest format.

| Task | Entry point |
|---|---|
| Cache full-attention teacher trajectories | `python -m hilberta.cache --help` |
| Train rank-32 LoRA with DDP; resume checkpoints | `python -m hilberta.train --help` |
| FID, KID, CLIP, LPIPS, PSNR, SSIM | `python -m hilberta.evaluate --help` |
| Kernel correctness and warmed latency | `python -m hilberta.benchmark --help` |

Four [configurations](configs/) cover 1024/2048 × 4-tile K1/16-tile K3. Training samples cached latent/velocity pairs; it does not run an online teacher. No dataset or teacher cache is bundled.

```bash
python -m hilberta.benchmark --resolution 1024 --tiles 4 --k 1 \
  --output outputs/kernel-1024-t4.json
```

The benchmark checks sampled query rows against SDPA before timing. Its resident-QKV kernel time excludes packing and is distinct from denoiser or image-generation latency.

## Reported results

2,048 COCO validation prompts; matched initial latents, seed 2707, 28 steps, guidance 7.5. FID compares with real COCO images; LPIPS compares with matched Full FLUX outputs. Time is **28-step denoising**, batch 1, one H100 80 GB, cached text; VAE decoding and file writing are excluded.

### 1024 × 1024

| Method | FID ↓ | LPIPS ↓ | Time ↓ | Speedup ↑ |
|---|---:|---:|---:|---:|
| Full FLUX | 49.93 | — | 5.82 s | — |
| **HilbertA 4-tile K1** | 50.28 | 0.376 | 5.10 s | 1.14× |
| HilbertA 16-tile K3 | 49.79 | 0.414 | 4.96 s | 1.17× |
| CLEAR local16 | 49.99 | 0.476 | 5.52 s | 1.05× |
| SpargeAttn top-k 0.25 | 53.68 | 0.602 | 5.40 s | 1.08× |

### 2048 × 2048

| Method | FID ↓ | LPIPS ↓ | Time ↓ | Speedup ↑ |
|---|---:|---:|---:|---:|
| Full FLUX | 52.40 | — | 31.90 s | — |
| **HilbertA 4-tile K1** | 53.17 | 0.376 | 19.83 s | 1.61× |
| HilbertA 16-tile K3 | 52.54 | 0.422 | 17.65 s | 1.81× |
| CLEAR local16 | 57.23 | 0.629 | 21.60 s | 1.48× |
| SpargeAttn top-k 0.25 | 54.14 | 0.542 | 20.11 s | 1.59× |

At 2048, our LoRAs were adapted at 2048; CLEAR uses its official 1024 checkpoint; SpargeAttn is training-free. Competitor configurations were selected for speed among those tested. These results do not establish superiority on every metric.

The [project page](https://wenboluu.github.io/hilberta/) contains all quality metrics and 1024/2048/4096 scaling results. [Aggregate JSON](results/image_results.json) retains full precision and the [source commit](https://github.com/wenboluu/hilberta-dev/commit/f9ab0b272dd26a005cf9ab8fe6869e24100a0738); [benchmark CSV](results/efficiency_all_rows.csv) contains the detailed timings. These are historical measurements, not a new benchmark of the portable runners.

## Repository layout

| Path | Contents |
|---|---|
| `hilberta/geometry.py`, `plan.py`, `bridge_k.py` | Ordering, tile schedule, and reciprocal bridge topology |
| `hilberta/paired_kernel.py`, `paired_metadata.py` | Default inference kernel and block metadata |
| `hilberta/flux_adapter.py`, `sampling_layout.py` | FLUX integration and persistent sampling layout |
| `hilberta/flex_sparse.py`, `sparse_kernel.py` | Differentiable training backend and alternative sparse kernel |
| `configs/` | Four training configurations and example prompts |
| `tests/` | Topology, kernel-contract, packing, model integration, and portable-runner checks |
| `docs/` | Reproduction instructions, validation scope, and dependency notices |
| `results/` | Aggregate measurements; no images or checkpoints |
| `website/` | Static project page, published results, and archived reports |

This is the public release repository. The initial snapshot was exported from the [development `main` commit](https://github.com/wenboluu/hilberta-dev/commit/c0a337d4845fc0e9704f758f33dd344774930f64). Historical branches and experiments remain in [`hilberta-dev`](https://github.com/wenboluu/hilberta-dev); use this repository's `main` for installation.

## Development checks

```bash
python -m pip install -c requirements-cu126.txt -e '.[dev]'
ruff check hilberta tests
ruff format --check hilberta tests
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest tests -q
```

CPU tests use small models and do not download FLUX. See [CPU-only installation](docs/REPRODUCE.md#cpu-tests). Full-model GPU training capacity, numerical parity, and convergence require separate validation, as listed in [implementation status](docs/IMPLEMENTATION.md).

## License

Our code is [MIT licensed](LICENSE). FLUX weights, datasets, and dependencies retain their own terms; see [third-party notices](docs/THIRD_PARTY.md). The [paper page](https://openreview.net/forum?id=yhMt5V4FNb) provides publication and citation information.
