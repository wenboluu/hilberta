# FLUX sparse attention, stage 3: 2048 x 2048 LoRA fine-tuning and evaluation (Full FLUX / ours 4-tile K1 / ours 16-tile K3 / CLEAR local16 / SpargeAttn topk0.25)

Every number is **measured** unless a cell says MISSING / n/a; missing inputs are reported, never invented. No plots; image comparisons are HTML pages. 
Protocol: the same 2,048 COCO val2017 prompts and real-image reference as stage 2; seed-2707 initial latents built at 2048 (`initial_latent(seed, device, 2048, 2048)`), 28 steps, guidance 7.5, true CFG 1, max_sequence_length 512, cached text embeddings. Competitors are the configurations chosen in stage 2 (no re-screening), run with their official code at 2048.

## 0. Headline (read this first)

**Quality at 2048x2048** (2,048 COCO val prompts, 28 steps, guidance 7.5, true CFG 1, seed-2707 initial latents at 2048; FID/KID vs the same real COCO set as stage 2, LPIPS-Alex at512 / PSNR / SSIM paired vs Full FLUX at 2048): (metrics_summary resolution = 2048)

| method | FID | KID mean (std) | LPIPS mean | CLIP cos | PSNR dB | SSIM |
|---|---:|---:|---:|---:|---:|---:|
| Full FLUX | 52.40 | 0.0119 (0.0008) | ref | 0.2579 | ref | ref |
| ours 4-tile K1 (2048 LoRA) | 53.17 | 0.0125 (0.0008) | 0.376 | 0.2592 | 15.11 | 0.742 |
| ours 16-tile K3 (2048 LoRA) | 52.54 | 0.0121 (0.0008) | 0.422 | 0.2601 | 14.39 | 0.724 |
| CLEAR local16 (official) | 57.23 | 0.0092 (0.0005) | 0.629 | 0.2491 | 11.14 | 0.595 |
| SpargeAttn topk0.25 (official, training-free) | 54.14 | 0.0106 (0.0005) | 0.542 | 0.2577 | 12.33 | 0.640 |

KID std is the spread over 100 subsets, not a confidence interval. CLEAR uses its official 1024-trained checkpoint and SpargeAttn is training-free (both run at 2048 unchanged); ours use LoRAs fine-tuned at 2048 (section 2).

**2048 training runs** (details in section 2):

| run | updates done / planned | micro x world x accum = global batch | lr | warmup | wall (h, sum of update times) | final train MSE | mean MSE last 50 |
|---|---:|---|---:|---:|---:|---:|---:|
| ours 4-tile K1 @2048 (`training_2048_t4k1/`) | 1792 / 1792 | 4 x 8 x 2 = 64 | 2e-05 | 100 | 13.91 | 0.037333 | 0.035670 |
| ours 16-tile K3 @2048 (`training_2048_t16k3/`) | 1792 / 1792 | 4 x 8 x 2 = 64 | 2e-05 | 100 | 12.27 | 0.063364 | 0.056867 |

**Speed at 2048, batch 1** from `kernel_bench/summary.json` (stage-2 bench, by reference; no new timing in stage 3); e2e = 28-step pipeline call (VAE excluded), denoiser = one transformer forward; speedup vs Full FLUX at the same point:

| method | e2e 2048 b1 (s/image) | denoiser 2048 b1 (ms) |
|---|---:|---:|
| Full FLUX (native flash SDPA) | 31.90 | 1150 |
| ours 4-tile K1 (paired kernel, merged LoRA) | 19.83 (1.61x) | 712 (1.61x) |
| ours 16-tile K3 (paired kernel, merged LoRA) | 17.65 (1.81x) | 640 (1.80x) |
| CLEAR local16 (official) | 21.60 (1.48x) | 773 (1.49x) |
| SpargeAttn topk0.25 (official) | 20.11 (1.59x) | 714 (1.61x) |

- The stage-2 2048 rows of ours were timed with a merged LoRA in the BF16 weights; merging the 2048-trained adapters gives the same compute, so these rows apply to the stage-3 adapters unchanged (speed is adapter-independent; the quality above is what changed).

## 1. Environment, commits and generation protocols

- Stage-3 source commit (`source_commit_stage3.txt`): `45e2bb3 (hilberta branch stage3-2048; pod code tree md5 bcd23f788cdfed1d969503139aba87e1 matches the commit)`
- Metrics environment: {"python": "3.11.2", "torch": "2.9.1", "torch_fidelity": "0.4.0", "lpips": "0.1.4", "transformers": "4.55.2", "numpy": "1.26.4", "Pillow": "12.3.0", "scipy": "1.17.1", "scikit_image": null}
- Generation protocol `clear_local16`: resolution [2048, 2048]; backend torch FlexAttention (official processors, torch.compile), shared global block mask; steps 28; guidance 7.5; true_cfg 1.0; batch 1; adapter - (- updates); manifest sha `5a2d7bf1f5151a4bc805a1b8000e5041510f29e8ebeab3428fce2994c6222fcc`
- Generation protocol `full_flux`: resolution [2048, 2048]; backend full_native_flash; steps 28; guidance 7.5; true_cfg 1.0; batch 1; adapter - (- updates); manifest sha `5a2d7bf1f5151a4bc805a1b8000e5041510f29e8ebeab3428fce2994c6222fcc`
- Generation protocol `ours16tile_2048`: resolution 2048; backend paired; steps 28; guidance 7.5; true_cfg 1.0; batch 1; adapter d2d6acb2ccb3e4a527ba02e001cbede66d0b1df9be9a2429228717fec3543b27 (1792 updates); manifest sha `5a2d7bf1f5151a4bc805a1b8000e5041510f29e8ebeab3428fce2994c6222fcc`
- Generation protocol `ours4tile_2048`: resolution 2048; backend paired; steps 28; guidance 7.5; true_cfg 1.0; batch 1; adapter 27cd7d53d07720309af27658be9e2c9acd5a2d4b8a50d23f0581a34a7d798a41 (1792 updates); manifest sha `5a2d7bf1f5151a4bc805a1b8000e5041510f29e8ebeab3428fce2994c6222fcc`
- Generation protocol `sparge_topk0.25`: resolution [2048, 2048]; backend spas_sage2_attn_meansim_topk_cuda; steps 28; guidance 7.5; true_cfg 1.0; batch 1; adapter - (- updates); manifest sha `5a2d7bf1f5151a4bc805a1b8000e5041510f29e8ebeab3428fce2994c6222fcc`

## 2. Training at 2048 (actual batch / updates / wall time / final MSE)

### Run C: 4-tile K1 at 2048 (`training_2048_t4k1/`)

- Resolution 2048 (grid 128); topology t4k1: tiles 4, bridge block64_k1_reciprocal (K=1); backend FlexAttention64 packed QKV
- Initialisation / adapter / optimizer state: initialization = `original FLUX plus LoRA loaded from /tmp/formal/flux/experiment/training/checkpoints/adapter_step001792.pt (1024 t4k1 adapter, 1792 updates); fresh AdamW state, fresh warmup/LR schedule, update counter 0`; optimizer_reset = `True`
- Optimizer adamw betas(.9,.999) eps1e-8 wd1e-4; lr 2e-05 (as specified); warmup 100 (as specified); constant after warmup; grad clip 1.0; precision BF16 base, FP32 LoRA
- **Actual batch:** microbatch 4 x 8 ranks x accum 2 = global batch 64 (config: 4 x 8 x 2 = 64); planned updates 1792 over 114688 records
- complete.json: {"complete": true, "completed_updates": 1792, "processed_records": 114688, "records": 114688}
- metrics.jsonl: **1792 updates done** (1792 rows); wall 13.91 h (sum of update_seconds); train MSE first 0.117443 -> final 0.037333 (mean of the last 50 updates 0.035670)
- **Final checkpoint:** `training_2048_t4k1/checkpoints/adapter_step001792.pt`

### Run D: 16-tile K3 at 2048 (`training_2048_t16k3/`)

- Resolution 2048 (grid 128); topology t16k3: tiles 16, bridge block64_k3_reciprocal (K=3); backend FlexAttention64 packed QKV
- Initialisation / adapter / optimizer state: initialization = `original FLUX plus LoRA loaded from /tmp/formal/flux/experiment/training_16tile_k3/checkpoints/adapter_step001792.pt (1024 t16k3 adapter, 1792 updates); fresh AdamW state, fresh warmup/LR schedule, update counter 0`; optimizer_reset = `True`
- Optimizer adamw betas(.9,.999) eps1e-8 wd1e-4; lr 2e-05 (as specified); warmup 100 (as specified); constant after warmup; grad clip 1.0; precision BF16 base, FP32 LoRA
- **Actual batch:** microbatch 4 x 8 ranks x accum 2 = global batch 64 (config: 4 x 8 x 2 = 64); planned updates 1792 over 114688 records
- complete.json: {"complete": true, "completed_updates": 1792, "processed_records": 114688, "records": 114688}
- metrics.jsonl: **1792 updates done** (1792 rows); wall 12.27 h (sum of update_seconds); train MSE first 0.162146 -> final 0.063364 (mean of the last 50 updates 0.056867)
- **Final checkpoint:** `training_2048_t16k3/checkpoints/adapter_step001792.pt`

Specified for both runs: initialise from the corresponding 1024 adapter, lr 2e-5, warmup 100, fresh optimizer state (no 1024 optimizer moments). The values above are what the run directories record; any difference is flagged.

## 3. Quality at 2048 (2,048 COCO val prompts, matched initial latents)

| Method | What it is | FID (aux.) | KID mean +- std | LPIPS mean / median / p90 | CLIP cos | PSNR mean / median (dB) | PSNR inf | SSIM mean / median |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Full FLUX | native flash SDPA full attention, no LoRA (baseline; official PyTorch/diffusers kernel) | 52.398 | 0.011882 +- 0.000784 | reference | 0.2579 | reference | reference | reference |
| clear_local16 | CLEAR official code + official checkpoint, config local16  (**competitor official kernel/path**) | 57.229 | 0.009243 +- 0.000488 | 0.6291 / 0.6391 / 0.7135 | 0.2491 | 11.14 / 10.93 | 0 | 0.5953 / 0.5945 |
| ours16tile_2048 | 16-tile K3 reciprocal, LoRA r32 fine-tuned at 2048 (our method; generation backend as recorded in its protocol.json, section 1) | 52.536 | 0.012077 +- 0.000836 | 0.4221 / 0.4225 / 0.5832 | 0.2601 | 14.39 / 14.07 | 0 | 0.7238 / 0.7306 |
| ours4tile_2048 | 4-tile K1 reciprocal, LoRA r32 fine-tuned at 2048 (our method; generation backend as recorded in its protocol.json, section 1) | 53.167 | 0.012454 +- 0.000801 | 0.3759 / 0.3697 / 0.5481 | 0.2592 | 15.11 / 14.74 | 0 | 0.7419 / 0.7497 |
| sparge_topk0.25 | SpargeAttn training-free spas_sage2_attn_meansim_topk_cuda, config topk0.25  (**competitor official kernel**) | 54.141 | 0.010551 +- 0.000505 | 0.5418 / 0.5491 / 0.7081 | 0.2577 | 12.33 / 12.11 | 0 | 0.6401 / 0.6458 |

- metrics_summary resolution: 2048; eval root `/tmp/formal/flux/experiment/evaluation_2048`; real reference `/tmp/formal/flux/experiment/evaluation/coco_real` (same COCO photos and cached Inception features as stage 2, sha-verified).
- FID/KID vs real COCO: torch-fidelity 0.4.0, torch-fidelity inception-v3-compat 2048 pool; KID 100 x 1000, seed 2707, raw KID; KID std is the spread over subsets, not a CI. Real half-vs-half FID 58.171 (n=1024). Inception resizes every image to 299, so 2048 and 1024 numbers use the same feature space but are different generated sets.
- LPIPS: {"metric": "LPIPS-Alex at512", "resize": "bilinear antialias 2048->512 on [-1,1] float", "implementation": {"package": "lpips", "version": "0.1.4", "net": "alex", "lpips_version": "0.1", "torch": "2.9.1"}}; CLIP: {"metric": "CLIP cosine", "model": "openai/clip-vit-large-patch14", "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41", "transformers": "4.55.2", "note": "L2-normalised image/text embeddings, mean cosine, no scaling (not CLIPScore x100 / x2.5)"}; COCO real CLIP 0.2568.
- PSNR/SSIM vs Full FLUX on the 2048 originals: PNG -> uint8 RGB -> float32 / 255 in [0,1]; no resize; implementation {"code": "experiments/coco1024/metrics.py ssim()/psnr() (torch)", "torch": "2.9.1", "numpy": "1.26.4", "Pillow": "12.3.0", "device": "cuda:0", "crosscheck": "scikit-image not installed in this environment; only our torch implementation was used"}.
- Per-image values: `evaluation_2048/metrics_stage3/per_image_metrics.jsonl`. Missing metric blocks: none.

- 64-sample comparison page: `evaluation_2048/comparisons/index_stage3.html` (present), same frozen subset as stage 1/2.

## 4. Direct cross-resolution before 2048 training (`evaluation_2048/crossres_pre/`)

The 10 frozen showcase prompts generated at 2048 by every method **before** any 2048 training: ours with the 1024 adapters, CLEAR with its official checkpoint, SpargeAttn training-free. Visual only (no metrics, n=10).

| method dir | images | resolution | adapter / config |
|---|---:|---|---|
| `clear_local16` | 10/10 | [2048, 2048] | local16 |
| `full_flux` | 10/10 | [2048, 2048] | - |
| `ours16tile_1024lora` | 10/10 | 2048 | /tmp/formal/flux/experiment/training_16tile_k3/checkpoints/adapter_step001792.pt |
| `ours4tile_1024lora` | 10/10 | 2048 | /tmp/formal/flux/experiment/training/checkpoints/adapter_step001792.pt |
| `sparge_topk0.25` | 10/10 | [2048, 2048] | 0.25 |

- Page: `evaluation_2048/showcase10/crossres_pre.html` (present).

## 5. Speed at 2048

No new timing in stage 3 (`kernel_bench_2048/` absent). The 2048 rows of the stage-2 bench (`kernel_bench/summary.json`, same H100 harness; ours with merged LoRA, whose cost does not depend on the adapter values) apply unchanged; see `report_stage2.md` section 4 for the full table. Rows at 2048:

| stage | backend | topology | batch | phase | median_ms | p10_ms | p90_ms | speedup_vs_full_native_flash | peak_allocated_gib | status |
|---|---|---|---|---|---|---|---|---|---|---|
| attention | flex64_per_call | t4k1 | 1 | weighted | 3.259 | 3.215 | 3.330 | 3.127 | 4.817 | ok |
| attention | flex64_per_call | t16k3 | 1 | weighted | 1.771 | 1.766 | 1.778 | 5.755 | 4.817 | ok |
| attention | persistent_paired | t4k1 | 1 | weighted | 2.474 | 2.449 | 2.528 | 4.120 | 0.483 | ok |
| attention | persistent_paired | t16k3 | 1 | weighted | 1.164 | 1.158 | 1.248 | 8.753 | 0.483 | ok |
| attention | flex64_per_call | t4k1 | 2 | weighted | 7.003 | 6.901 | 7.170 | 2.996 | 5.104 | ok |
| attention | flex64_per_call | t16k3 | 2 | weighted | 3.933 | 3.922 | 3.948 | 5.335 | 5.104 | ok |
| attention | persistent_paired | t4k1 | 2 | weighted | 4.965 | 4.921 | 5.042 | 4.225 | 0.960 | ok |
| attention | persistent_paired | t16k3 | 2 | weighted | 2.372 | 2.336 | 2.459 | 8.844 | 0.960 | ok |
| attention | flex64_per_call | t4k1 | 4 | weighted | 14.095 | 13.887 | 14.381 | 2.967 | 5.684 | ok |
| attention | flex64_per_call | t16k3 | 4 | weighted | 7.692 | 7.660 | 7.855 | 5.437 | 5.684 | ok |
| attention | persistent_paired | t4k1 | 4 | weighted | 9.662 | 9.553 | 9.769 | 4.328 | 1.919 | ok |
| attention | persistent_paired | t16k3 | 4 | weighted | 4.520 | 4.462 | 4.581 | 9.252 | 1.919 | ok |
| attention | flex64_per_call | t4k1 | 8 | weighted | 28.117 | 27.697 | 28.494 | 2.867 | 6.966 | ok |
| attention | flex64_per_call | t16k3 | 8 | weighted | 15.225 | 15.100 | 15.558 | 5.295 | 6.966 | ok |
| attention | persistent_paired | t4k1 | 8 | weighted | 19.167 | 19.039 | 19.229 | 4.206 | 3.837 | ok |
| attention | persistent_paired | t16k3 | 8 | weighted | 8.905 | 8.821 | 9.030 | 9.053 | 3.836 | ok |
| denoiser | clear_local16 | None | 1 | None | 772.875 | 771.443 | 775.548 | 1.488 | 25.516 | ok |
| denoiser | clear_local16_superseded | None | 1 | None | None | None | None |  | None | oom |
| denoiser | full_native_flash | None | 1 | None | 1150.413 | 1145.047 | 1155.002 | 1.000 | 23.871 | ok |
| denoiser | paired_t16k3 | t16k3 | 1 | None | 639.542 | 632.565 | 642.296 | 1.799 | 23.873 | ok |
| denoiser | paired_t4k1 | t4k1 | 1 | None | 712.444 | 707.665 | 714.404 | 1.615 | 23.874 | ok |
| denoiser | sparge_topk0.25 | None | 1 | None | 713.819 | 713.425 | 714.199 | 1.612 | 23.870 | ok |
| denoiser | clear_local16 | None | 2 | None | 1556.672 | 1552.541 | 1562.275 | 1.478 | 25.522 | ok |
| denoiser | clear_local16_superseded | None | 2 | None | None | None | None |  | None | skipped_after_oom |
| denoiser | full_native_flash | None | 2 | None | 2300.399 | 2299.201 | 2303.612 | 1.000 | 25.519 | ok |
| denoiser | paired_t16k3 | t16k3 | 2 | None | 1285.050 | 1281.888 | 1287.299 | 1.790 | 25.524 | ok |
| denoiser | paired_t4k1 | t4k1 | 2 | None | 1434.425 | 1430.916 | 1437.572 | 1.604 | 25.524 | ok |
| denoiser | sparge_topk0.25 | None | 2 | None | 1429.849 | 1427.311 | 1431.474 | 1.609 | 25.521 | ok |
| denoiser | clear_local16 | None | 4 | None | 3152.136 | 3144.994 | 3156.011 | 1.466 | 28.820 | ok |
| denoiser | clear_local16_superseded | None | 4 | None | None | None | None |  | None | skipped_after_oom |
| denoiser | full_native_flash | None | 4 | None | 4619.669 | 4618.781 | 4621.105 | 1.000 | 28.818 | ok |
| denoiser | paired_t16k3 | t16k3 | 4 | None | 2561.972 | 2553.401 | 2564.130 | 1.803 | 28.827 | ok |
| denoiser | paired_t4k1 | t4k1 | 4 | None | 2854.865 | 2847.361 | 2857.177 | 1.618 | 28.827 | ok |
| denoiser | sparge_topk0.25 | None | 4 | None | 2879.280 | 2866.667 | 2891.285 | 1.604 | 28.819 | ok |
| denoiser | clear_local16 | None | 8 | None | 6323.836 | 6315.673 | 6338.291 | 1.473 | 35.418 | ok |
| denoiser | clear_local16_superseded | None | 8 | None | None | None | None |  | None | skipped_after_oom |
| denoiser | full_native_flash | None | 8 | None | 9314.813 | 9305.657 | 9329.762 | 1.000 | 35.416 | ok |
| denoiser | paired_t16k3 | t16k3 | 8 | None | 5118.901 | 5113.201 | 5124.867 | 1.820 | 35.432 | ok |
| denoiser | paired_t4k1 | t4k1 | 8 | None | 5739.725 | 5717.669 | 5741.940 | 1.623 | 35.433 | ok |
| denoiser | sparge_topk0.25 | None | 8 | None | 5783.090 | 5776.102 | 5801.874 | 1.611 | 35.418 | ok |
| generation | clear_local16 | None | 1 | None | 21597.354 | 21541.611 | 21597.354 | 1.477 | 24.038 | ok |
| generation | full_native_flash | None | 1 | None | 31901.742 | 31880.961 | 31901.742 | 1.000 | 24.038 | ok |
| generation | ours_t16k3 | t16k3 | 1 | None | 17645.447 | 17643.211 | 17645.447 | 1.808 | 24.039 | ok |
| generation | ours_t4k1 | t4k1 | 1 | None | 19826.000 | 19807.842 | 19826.000 | 1.609 | 24.039 | ok |
| generation | sparge_topk0.25 | None | 1 | None | 20114.354 | 20092.668 | 20114.354 | 1.586 | 24.038 | ok |
| generation | clear_local16 | None | 2 | None | 43445.613 | 43363.977 | 43445.613 | 1.469 | 25.690 | ok |
| generation | full_native_flash | None | 2 | None | 63833.473 | 63826.379 | 63833.473 | 1.000 | 25.690 | ok |
| generation | ours_t16k3 | t16k3 | 2 | None | 35314.895 | 35243.992 | 35314.895 | 1.808 | 25.690 | ok |
| generation | ours_t4k1 | t4k1 | 2 | None | 39555.258 | 39421.258 | 39555.258 | 1.614 | 25.691 | ok |
| generation | sparge_topk0.25 | None | 2 | None | 40361.852 | 40308.285 | 40361.852 | 1.582 | 25.690 | ok |
| generation | clear_local16 | None | 4 | None | 87651.516 | 87576.477 | 87651.516 | 1.456 | 29.001 | ok |
| generation | full_native_flash | None | 4 | None | 127582.055 | 127551.891 | 127582.055 | 1.000 | 29.000 | ok |
| generation | ours_t16k3 | t16k3 | 4 | None | 70068.023 | 69990.281 | 70068.023 | 1.821 | 29.001 | ok |
| generation | ours_t4k1 | t4k1 | 4 | None | 78292.688 | 78238.625 | 78292.688 | 1.630 | 29.001 | ok |
| generation | sparge_topk0.25 | None | 4 | None | 81195.219 | 80869.984 | 81195.219 | 1.571 | 29.001 | ok |
| generation | clear_local16 | None | 8 | None | 174774.594 | 174052.438 | 174774.594 | 1.450 | 35.622 | ok |
| generation | full_native_flash | None | 8 | None | 253499.250 | 253292.297 | 253499.250 | 1.000 | 35.621 | ok |
| generation | ours_t16k3 | t16k3 | 8 | None | 139190.484 | 139098.406 | 139190.484 | 1.821 | 35.622 | ok |
| generation | ours_t4k1 | t4k1 | 8 | None | 156787.891 | 156487.078 | 156787.891 | 1.617 | 35.622 | ok |
| generation | sparge_topk0.25 | None | 8 | None | 164232.391 | 164158.844 | 164232.391 | 1.544 | 35.621 | ok |
| kernel | full_native_flash | None | 1 | weighted | 10.192 | 9.926 | 10.466 | 1.000 | 0.393 | ok |
| kernel | paired | t16k3 | 1 | 0 | 1.168 | 1.149 | 1.184 |  | 0.418 | ok |
| kernel | paired | t4k1 | 1 | 0 | 2.453 | 2.395 | 2.508 |  | 0.418 | ok |
| kernel | paired | t16k3 | 1 | 1 | 1.161 | 1.159 | 1.209 |  | 0.418 | ok |
| kernel | paired | t4k1 | 1 | 1 | 2.464 | 2.408 | 2.505 |  | 0.418 | ok |
| kernel | paired | t16k3 | 1 | 2 | 1.173 | 1.154 | 1.192 |  | 0.418 | ok |
| kernel | paired | t4k1 | 1 | 2 | 2.467 | 2.409 | 2.527 |  | 0.418 | ok |
| kernel | paired | t16k3 | 1 | 3 | 1.161 | 1.159 | 1.219 |  | 0.418 | ok |
| kernel | paired | t4k1 | 1 | 3 | 2.473 | 2.413 | 2.512 |  | 0.418 | ok |
| kernel | paired | t16k3 | 1 | weighted | 1.169 | 1.155 | 1.197 | 8.715 | 0.418 | ok |
| kernel | paired | t4k1 | 1 | weighted | 2.463 | 2.415 | 2.507 | 4.138 | 0.418 | ok |
| kernel | full_native_flash | None | 2 | weighted | 20.979 | 20.668 | 21.723 | 1.000 | 0.777 | ok |
| kernel | paired | t16k3 | 2 | 0 | 2.392 | 2.339 | 2.478 |  | 0.780 | ok |
| kernel | paired | t4k1 | 2 | 0 | 5.073 | 4.958 | 5.288 |  | 0.780 | ok |
| kernel | paired | t16k3 | 2 | 1 | 2.429 | 2.338 | 2.472 |  | 0.780 | ok |
| kernel | paired | t4k1 | 2 | 1 | 5.096 | 5.052 | 5.104 |  | 0.780 | ok |
| kernel | paired | t16k3 | 2 | 2 | 2.422 | 2.328 | 2.461 |  | 0.780 | ok |
| kernel | paired | t4k1 | 2 | 2 | 5.093 | 4.913 | 5.243 |  | 0.780 | ok |
| kernel | paired | t16k3 | 2 | 3 | 2.429 | 2.320 | 2.455 |  | 0.780 | ok |
| kernel | paired | t4k1 | 2 | 3 | 5.095 | 5.028 | 5.193 |  | 0.780 | ok |
| kernel | paired | t16k3 | 2 | weighted | 2.419 | 2.335 | 2.459 | 8.673 | 0.780 | ok |
| kernel | paired | t4k1 | 2 | weighted | 5.090 | 5.032 | 5.175 | 4.122 | 0.780 | ok |
| kernel | full_native_flash | None | 4 | weighted | 41.819 | 40.619 | 43.042 | 1.000 | 1.553 | ok |
| kernel | paired | t16k3 | 4 | 0 | 4.482 | 4.407 | 4.655 |  | 1.559 | ok |
| kernel | paired | t4k1 | 4 | 0 | 9.804 | 9.687 | 10.183 |  | 1.559 | ok |
| kernel | paired | t16k3 | 4 | 1 | 4.525 | 4.449 | 4.670 |  | 1.559 | ok |
| kernel | paired | t4k1 | 4 | 1 | 9.773 | 9.516 | 10.148 |  | 1.559 | ok |
| kernel | paired | t16k3 | 4 | 2 | 4.502 | 4.424 | 4.583 |  | 1.559 | ok |
| kernel | paired | t4k1 | 4 | 2 | 9.753 | 9.678 | 10.082 |  | 1.559 | ok |
| kernel | paired | t16k3 | 4 | 3 | 4.547 | 4.447 | 4.641 |  | 1.559 | ok |
| kernel | paired | t4k1 | 4 | 3 | 9.785 | 9.701 | 10.041 |  | 1.559 | ok |
| kernel | paired | t16k3 | 4 | weighted | 4.514 | 4.434 | 4.626 | 9.265 | 1.559 | ok |
| kernel | paired | t4k1 | 4 | weighted | 9.795 | 9.646 | 10.115 | 4.270 | 1.559 | ok |
| kernel | full_native_flash | None | 8 | weighted | 80.617 | 80.291 | 82.020 | 1.000 | 3.106 | ok |
| kernel | paired | t16k3 | 8 | 0 | 8.733 | 8.658 | 8.882 |  | 3.118 | ok |
| kernel | paired | t4k1 | 8 | 0 | 18.957 | 18.835 | 19.058 |  | 3.118 | ok |
| kernel | paired | t16k3 | 8 | 1 | 8.817 | 8.795 | 8.898 |  | 3.118 | ok |
| kernel | paired | t4k1 | 8 | 1 | 18.972 | 18.888 | 19.055 |  | 3.118 | ok |
| kernel | paired | t16k3 | 8 | 2 | 8.777 | 8.697 | 8.919 |  | 3.118 | ok |
| kernel | paired | t4k1 | 8 | 2 | 18.921 | 18.767 | 18.993 |  | 3.118 | ok |
| kernel | paired | t16k3 | 8 | 3 | 8.815 | 8.744 | 8.989 |  | 3.118 | ok |
| kernel | paired | t4k1 | 8 | 3 | 18.980 | 18.874 | 19.046 |  | 3.118 | ok |
| kernel | paired | t16k3 | 8 | weighted | 8.784 | 8.749 | 8.883 | 9.177 | 3.118 | ok |
| kernel | paired | t4k1 | 8 | weighted | 18.958 | 18.849 | 19.031 | 4.253 | 3.118 | ok |

## 6. Showcase at 2048 (`evaluation_2048/showcase10/index.html`)

- Frozen stage-2 selection reused (`evaluation/showcase10/selection.json`, seed 2707, 10 prompts; not reselected). Columns: COCO real | Full FLUX | ours4tile_2048 | ours16tile_2048 | CLEAR | Sparge.
- Page present; column availability: coco_real=10/10, full_flux=10/10, ours4tile_2048=10/10, ours16tile_2048=10/10, clear_local16=10/10, sparge_topk0.25=10/10

## 7. Reproduction commands (pod; `source /tmp/formal/flux/env.sh`)

```bash
bash experiments/competitors/t2048_competitors.sh crossres 6 7          # CLEAR / Sparge on the 10 showcase prompts at 2048 (pre-training cross-res set)
bash experiments/competitors/t2048_competitors.sh eval 0 1 2 3 4 5 6 7  # CLEAR local16 + Sparge topk0.25, 2048 prompts at 2048, 8 shards
bash experiments/coco1024/scripts/stage3_eval.sh --gpus 0,1            # metrics (2 GPUs) + showcase + crossres page + compare page + this report + HDFS mirror
python -m experiments.coco1024.metrics --exp $EXP --eval-root $EXP/evaluation_2048 --resolution 2048 --out metrics_stage3 --methods auto --stage all --inception-weights $F/weights/weights-inception-2015-12-05-6726825d.pth
python -m experiments.coco1024.report --exp $EXP --stage3
```

## 8. Explicit status list

- Methods evaluated at 2048: full_flux, clear_local16, ours16tile_2048, ours4tile_2048, sparge_topk0.25; expected full_flux, ours4tile_2048, ours16tile_2048, clear_local16, sparge_topk0.25; missing: none
- Training runs: training_2048_t4k1: complete, training_2048_t16k3: complete
- CLEAR at 4096 is not attempted (its official mask build OOMs, stage 2). Literature numbers are not placed next to measured ones.
