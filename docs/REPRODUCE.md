# Reproduction guide

Use the README's pinned CUDA environment. These are executable interfaces for new runs, not a claim that the release runners have already reproduced the published metrics. The exact historical caption manifests and trained adapters are not yet bundled. Freeze your train/eval manifests before generating data; never select prompts based on model outputs.

## CPU tests

For a machine without a GPU, install the matching CPU torch wheels before the package:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 python -m pytest tests -q
```

Linux Triton is retained for kernel module imports; tests do not launch CUDA kernels. CPU tests exercise small native Diffusers models, masks, positional/layout behavior and scheduler output consistency. They do not test BF16 GPU arithmetic or H100 training capacity.

## Prompt manifests

Each JSONL row has a unique filename-safe `id` and a nonempty `prompt`:

```json
{"id":"000001","prompt":"A small boat on a calm blue sea under a clear sky."}
```

The example file is a smoke-test input, not an evaluation set. Reported training used 2,048 COCO train captions × two seeds × 28 timesteps = 114,688 supervised records. Reported evaluation used 2,048 COCO val captions and seed 2707. Obtain COCO under its terms; keep train/eval image IDs separate. Exact paper prompt membership is required to reproduce the table, not just the same sample count.

## Teacher cache

Full FLUX supplies cached latent inputs and velocity targets at each denoising step. Text embeddings are stored once per caption within a shard; each trajectory retains 28 records. This avoids teacher forwards during LoRA updates.

```bash
python -m hilberta.cache --prompts data/train.jsonl --output cache/1024 --resolution 1024 --seeds 2707 2708 --steps 28
```

For independent workers use the same manifest/options and output root, with distinct `--shard i --shards N --device cuda:i`. Each worker writes only `shard-iii`; existing shard directories are rejected. A completed shard contains `index.json` and `protocol.json`. Failed partial shards must be inspected and removed or regenerated into a fresh root; automatic cache resume is not implemented. Do not mix incomplete and completed production without checking the expected record count. Model revision is pinned in `hilberta/runtime.py`; optional `--model` and `--revision` must match between cache and train. Training also strictly checks cached resolution, sampling step count and guidance against its config; this release uses 28 steps and guidance 7.5.

## LoRA training

Frozen BF16 FLUX; trainable FP32 rank/alpha-32 LoRA parameters. Flex64 attention; gradient checkpointing; AdamW, betas (0.9, 0.999), eps 1e-8, weight decay 1e-4, gradient clipping 1.0. LR has 100-update linear warmup then stays constant. Per-record loss is mean squared error between sparse and cached full-model velocity predictions. There is no online teacher or image reconstruction loss.

```bash
# Eight GPUs: microbatch 8 × eight ranks × accum 1 = global batch 64.
torchrun --standalone --nproc_per_node=8 -m hilberta.train   --cache cache/1024 --config configs/flux_1024_t4k1.json   --microbatch 8 --accum 1 --output outputs/train-t4k1
```

If memory is insufficient, use `--microbatch 4 --accum 2` or `--microbatch 1 --accum 8`. The required global batch stays 64. Capacity has not been established for this new runner. One epoch visits every cached timestep record once; 114,688 records produce 1,792 optimizer updates. All ranks must use the same options. `metrics.jsonl` logs loss, gradient norm and learning rate; this minimal runner has no automatic dev evaluation or W&B integration.

Use `configs/flux_1024_t16k3.json` for the 16-tile variant. For 2048, generate a separate 2048 cache, select the corresponding 2048 config and initialize from the matching 1024 adapter:

```bash
torchrun --standalone --nproc_per_node=8 -m hilberta.train   --cache cache/2048 --config configs/flux_2048_t4k1.json   --init-adapter outputs/train-t4k1/adapter_step001792.pt   --microbatch 1 --accum 8 --output outputs/train-2048-t4k1
```

`--init-adapter` starts a new optimizer, whereas `--resume outputs/train-t4k1/last.pt` restores optimizer, adapter, RNG and update position. Resume requires identical world size, batch/accumulation, seed, config and cache signature. It cannot continue with an arbitrary new GPU count. Checkpoints save every 500 updates by default and at the final update; change with `--save-every`. `last.pt` is the recovery file, `adapter_step*.pt` is the inference adapter. After interruption, progress after the most recent checkpoint is repeated.

## Evaluation

Generate Full FLUX and sparse images from the identical frozen manifest, seed and sampling parameters using `hilberta.generate`. Prepare matched COCO reference images as RGB PNGs named by manifest ID. The three directories must have exactly the same PNG filenames. Do not compare PSNR/SSIM against COCO: those metrics compare to the matched Full FLUX image.

```bash
python -m hilberta.evaluate --images outputs/hilberta --teacher outputs/full   --real data/coco-val-png --prompts data/eval.jsonl --output outputs/metrics.json
```

This computes FID/KID vs real images, CLIP ViT-L/14 image-text cosine, LPIPS AlexNet at 512 pixels, and PSNR/SSIM vs Full FLUX at native resolution. KID uses 100 subsets of 1,000 images, RNG seed 2707; use at least 1,000 samples with the default subset size. Report raw KID (no ×1,000); subset standard deviation is not a confidence interval. JSON contains aggregate metrics and per-image values. Image decoding/resizing and evaluation-library versions can affect values: the new evaluation entrypoint needs parity checks against the historical evaluator before attributing small numerical differences to a model.

## Resident kernel benchmark

```bash
python -m hilberta.benchmark --resolution 1024 --tiles 4 --k 1 --output outputs/kernel-1024-t4.json
python -m hilberta.benchmark --resolution 2048 --tiles 16 --k 3 --output outputs/kernel-2048-t16.json
python -m hilberta.benchmark --resolution 4096 --tiles 16 --k 3 --output outputs/kernel-4096-t16.json
```

Repeat each configuration you need. The CLI checks selected global/image query rows against an independent SDPA mask oracle in all four phases before timing. It reports warmed median/p10/p90, full flash-SDPA time, and phase-weighted paired-kernel time (15/14/14/14 of 57 layers). This check samples rows; it is not exhaustive whole-model correctness.

These timings use resident synthetic BF16 Q/K/V, batch 1, 24 heads, head dimension 128. Metadata construction and external packing are excluded. They are neither denoiser latency nor 28-step generation latency. The release does not yet include portable reproduction of every historical end-to-end/competitor benchmark. Do not compare these kernel times directly to competitor end-to-end times. A fixed tile count does not imply linear asymptotic complexity.
