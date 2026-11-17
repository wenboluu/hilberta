# Reported FLUX results

## Protocol and scope

FLUX.1-dev; 2,048 COCO validation prompts; matched initial latents; seed 2707; 28 steps; guidance 7.5; up to 512 text tokens. FID/KID compare with real COCO images. LPIPS (AlexNet at 512 pixels), PSNR, and SSIM compare with matched Full FLUX outputs. CLIP is ViT-L/14 image/text cosine. KID uses 100 subsets of 1,000 images; subset standard deviation is not a confidence interval. The evaluation set overlaps earlier diagnostic captions and is not claimed entirely unseen.

HilbertA uses composite Hilbert ordering, four sliding phases, reciprocal block bridges, no shared image region, and global text attention. Its rank/alpha-32 LoRA configurations are 4-tile K1 and 16-tile K3. Quality at 1024 used Flex64; quality at 2048 used the paired kernel. Performance uses merged BF16 LoRA and the paired kernel. These are reported measurements, not new validations of the portable runners.

CLEAR uses its official local16 checkpoint; SpargeAttn uses top-k 0.25 with INT8 Q/K and FP8 V. Configurations were selected for speed among those tested, not from an exhaustive quality–speed sweep. At 2048, our adapters were adapted at 2048; CLEAR uses its official 1024-trained checkpoint and SpargeAttn is training-free. The 2048 timing rows reuse the original benchmark rather than a new timing of the final 2048 adapters.

## 1024 × 1024 quality

| Method | FID ↓ | KID ×10³ ↓ (mean ± std) | CLIP ↑ | LPIPS ↓ | PSNR ↑ (dB) | SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Full FLUX | 49.93 | 10.78 ± 0.82 | 0.2612 | — | — | — |
| HilbertA 4-tile K1 | 50.28 | 11.26 ± 0.88 | 0.2611 | 0.376 | 15.63 | 0.665 |
| HilbertA 16-tile K3 | 49.79 | 10.66 ± 0.89 | 0.2610 | 0.414 | 15.09 | 0.645 |
| CLEAR local16 | 49.99 | 10.10 ± 0.88 | 0.2608 | 0.476 | 13.83 | 0.604 |
| SpargeAttn top-k 0.25 | 53.68 | 9.26 ± 0.48 | 0.2466 | 0.602 | 12.35 | 0.540 |

## 1024 × 1024 denoising time

One H100 80 GB, batch 1, cached text, 28 steps. VAE decode and file writing are excluded.

| Method | Seconds / image ↓ | Speedup ↑ |
|---|---:|---:|
| Full FLUX | 5.82 | 1.00× |
| HilbertA 4-tile K1 | 5.10 | 1.14× |
| HilbertA 16-tile K3 | 4.96 | 1.17× |
| CLEAR local16 | 5.52 | 1.05× |
| SpargeAttn top-k 0.25 | 5.40 | 1.08× |

## 2048 × 2048 quality

| Method | FID ↓ | KID ×10³ ↓ (mean ± std) | CLIP ↑ | LPIPS ↓ | PSNR ↑ (dB) | SSIM ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Full FLUX | 52.40 | 11.88 ± 0.78 | 0.2579 | — | — | — |
| HilbertA 4-tile K1 | 53.17 | 12.45 ± 0.80 | 0.2592 | 0.376 | 15.11 | 0.742 |
| HilbertA 16-tile K3 | 52.54 | 12.08 ± 0.84 | 0.2601 | 0.422 | 14.39 | 0.724 |
| CLEAR local16 | 57.23 | 9.24 ± 0.49 | 0.2491 | 0.629 | 11.14 | 0.595 |
| SpargeAttn top-k 0.25 | 54.14 | 10.55 ± 0.51 | 0.2577 | 0.542 | 12.33 | 0.640 |

## 2048 × 2048 denoising time

One H100 80 GB, batch 1, cached text, 28 steps. VAE decode and file writing are excluded.

| Method | Seconds / image ↓ | Speedup ↑ |
|---|---:|---:|
| Full FLUX | 31.90 | 1.00× |
| HilbertA 4-tile K1 | 19.83 | 1.61× |
| HilbertA 16-tile K3 | 17.65 | 1.81× |
| CLEAR local16 | 21.60 | 1.48× |
| SpargeAttn top-k 0.25 | 20.11 | 1.59× |

## Resolution scaling

Fixed tile counts. Single H100 80 GB, batch 1. Kernel times are phase-weighted across 57 layers using resident Q/K/V and metadata; packing is excluded. Denoiser times measure one transformer forward. These scopes are distinct from full generation time. Fixed tile counts do not imply linear complexity.

### Attention kernel (ms / layer)

| Method | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| Full FLUX | 0.814 | 10.192 | 153.467 |
| HilbertA 4-tile K1 | 0.318 | 2.463 | 30.380 |
| HilbertA 16-tile K3 | 0.258 | 1.169 | 9.557 |

### Denoiser forward (ms)

| Method | 1024 | 2048 | 4096 |
|---|---:|---:|---:|
| Full FLUX | 208.473 | 1150.413 | 11128.352 |
| HilbertA 4-tile K1 | 179.918 | 712.444 | 3937.594 |
| HilbertA 16-tile K3 | 177.075 | 639.542 | 2757.367 |
| CLEAR local16 | 197.583 | 772.875 | OOM |
| SpargeAttn top-k 0.25 | 193.498 | 713.819 | 3591.791 |

CLEAR ran out of memory constructing its 4096 mask. Competitor kernels were not measured separately. There is no 4096 quality evaluation. Warmup and compilation are excluded; the CSV retains detailed repeats and timing scopes.

## Training recipe

Frozen BF16 FLUX.1-dev with FP32 LoRA, rank/alpha 32, dropout 0. Teacher supervision uses 2,048 COCO train captions × 2 seeds × 28 trajectory points = 114,688 records. One epoch at global batch 64 gives 1,792 optimizer updates. Training uses Flex64, gradient checkpointing, and AdamW; the 1024 learning rate is 5e-5 and the 2048 adaptation learning rate is 2e-5, with 100 warmup updates. The 2048 variants initialize from their matching 1024 adapters.

These describe the reported protocol. The public runner's operational commands, resume restrictions, and production-validation limits are in the reproduction guide. Exact historical prompt manifests and release adapters are not bundled.

## Data files

- `results/image_results.json`: full-precision aggregates. Rows follow `names`. Each quality row contains FID, KID mean, KID subset std, CLIP, LPIPS, PSNR, SSIM, and seconds/image. Fidelity nulls identify the Full FLUX reference. Kernel columns are 1024/2048/4096 for Full/4-tile/16-tile; denoiser rows follow all methods.
- `results/efficiency_all_rows.csv`: detailed measurements, including additional resolutions, batch sizes, timing scopes, and variability.
- `source_commit` identifies measurement provenance; it does not imply that the public CLI reproduced those runs.

See [reproduction instructions](REPRODUCE.md) and the [project page](https://wenboluu.github.io/hilberta/).
