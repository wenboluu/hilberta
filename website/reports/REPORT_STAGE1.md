# FLUX sparse attention (4 tiles, reciprocal K1, Flex64): COCO 1024 training and evaluation report

Every number below is **measured** in this run unless the line is tagged *estimate* or appears in the Literature table.

## 1. Setup

| Item | Setting |
|---|---|
| Base model | black-forest-labs/FLUX.1-dev |
| Base revision | 3de623fc3c33e44ffbe2bad470d0f45bccf2eb21 |
| Resolution | 1024 x 1024 |
| Image tokens | 4,096 (64 x 64) |
| Max text tokens | 512 |
| Ordering | composite Hilbert |
| Tiles | 4 (1,024 image tokens each) |
| Sliding offsets | 0, 256, 512, 768 |
| Phase | layer index mod 4 |
| Bridge | geometric K1 + reciprocal closure |
| Shared image region | none |
| Text connectivity | bidirectional global |
| Student backend | FlexAttention, mask block 64 |
| Adapter | LoRA r32 alpha32 dropout0 (74,711,040 params) |
| Base weights | frozen BF16 |
| Adapter weights | FP32 |
| Parallelism | 8-GPU DDP |
| Teacher seeds | 2707, 2708 |
| Teacher steps | 28 |
| Guidance | 7.5 |
| True CFG | 1.0 |
| Training length | 1 epoch |

- Commit SHA: `9c35fcc24edf5ff27b10dc43f20d33e7fa76ab48` (hilberta branch coco1024-formal; pod checkout has no .git, SHA recorded from the devbox working copy)
- Software: {"torch": "2.9.1", "diffusers": "0.36.0", "transformers": "4.55.2", "peft": "0.17.1", "accelerate": "1.14.0", "triton": "3.5.1", "numpy": "1.26.4", "Pillow": "12.3.0", "safetensors": "0.8.0", "torch-fidelity": "0.4.0", "lpips": "0.1.4", "scipy": "1.17.1"}; CUDA 12.9; GPUs ['NVIDIA H100 80GB HBM3', 'NVIDIA H100 80GB HBM3', 'NVIDIA H100 80GB HBM3', 'NVIDIA H100 80GB HBM3', 'NVIDIA H100 80GB HBM3', 'NVIDIA H100 80GB HBM3', 'NVIDIA H100 80GB HBM3', 'NVIDIA H100 80GB HBM3']
- Attention backend (training): FlexAttention64 packed QKV; flex kernel options: ada
- image_mask_sha256: `04864d8e08df1c8fc11e19de7349f2fb6bff1a82f257bbdda81ca32e8614ec9c`; lora_targets_sha256: `0017a3b06fae39759257930d6c2da2cf872ee39d5d5edfae8586145120f1c1f7`; manifest_sha256: `509b174550d284ba0122e84566e0674e4de379e896ef835e50004fcaae48af13`
- Source hashes: `plan.py`=08375ea2c64e, `bridge_k.py`=c2913c94266e, `flex_sparse.py`=1c7b86e11adf, `reciprocal.py`=89dd9c85d09b, `common.py`=5b5de21572f3, `produce.py`=5175c9c283f7, `sampler.py`=93f997073d88, `data.py`=b8d9ae4b5627, `train.py`=156414ca9af3, `generate.py`=da5934764d1f, `proposed_lora_parameters.json`=0017a3b06fae

## 2. Data

- Selection: seed 2707, 2048 prompts per split; algorithm: experiments.pixart_ordering.protocol.make_splits: seeded shuffle of image ids, one seeded caption per image, reject empty/exact/near-duplicate (Jaccard or SequenceMatcher >= threshold) against already selected captions
- Annotation sha256: train `4b62086319480e0739ef390d04084515defb9c213ff13605a036061e33314317`, val `afe3b30e403dd7f228e2373023abbd60042a6e10ec6874d3652df034d289ebb9`
- Manifest sha256: training `509b174550d284ba0122e84566e0674e4de379e896ef835e50004fcaae48af13`, evaluation `5a2d7bf1f5151a4bc805a1b8000e5041510f29e8ebeab3428fce2994c6222fcc`
- Checks: train/eval image-id overlap 0, identical caption texts 0, evaluation images downloaded 2048
- Historical contact: Earlier experiments in this repo used CC3M for training; data/coco_prompts.json (20k captions, no ids) and two hardcoded COCO captions were used for diagnostics. The evaluation set is therefore not claimed unseen. Train hits: {"hardcoded_caption_ids": [], "coco_prompts_json_text_overlap": [2613, 27746, 31664, 62306, 81078, 85016, 87422, 88359, 112016, 114101, 116010, 126146, 139678, 145653, 145713, 167238, 182311, 207670, 214534, 217710, 225769, 233040, 242948, 250340, 268561, 268885, 285030, 308150, 312914, 338391, 350866, 360673, 363805, 397382, 412046, 429839, 434866, 447401, 451599, 468214, 474538, 498347, 508699, 521982, 546754, 563276, 563871, 568853, 586546, 587716, 588597, 597622, 621639, 622603, 634280, 640210, 642205, 647383, 651482, 665523, 676986, 679802, 681096, 694405, 722027, 725246, 736701, 740390, 740993, 751797, 788699, 792272, 798743, 815337, 825831, 829709]}. Evaluation hits: {"hardcoded_caption_ids": [313586], "coco_prompts_json_text_overlap": [58848, 61271, 137601, 180800, 214152, 266021, 322692, 407185, 421613, 432605, 469683, 500143, 541184, 596147, 608832, 610609, 644395, 657475, 693791, 719730]}. The evaluation set is **not** claimed unseen.

## 3. Teacher cache

- Prompts 2048, trajectories valid 4096/4096, records valid **114688** (required 114,688), complete=True
- Bytes 124750043136 (116.2 GiB), mean trajectory 29.0 MiB
- Teacher backend: native_sdpa_flash; steps 28; guidance 7.5; dtype bfloat16; manifest_sha256 `509b174550d284ba0122e84566e0674e4de379e896ef835e50004fcaae48af13`

## 4. Pre-training checks

- `model.json`: output_relative_l2=0.0055, grad_relative_l2=0.2154, grad_cosine=0.9827, batched_vs_single_relative_l2=0.0093, frozen_unchanged=True, lora_changed=True, only_lora_trainable=True
- `plan.json`: {"mask_sha256": "04864d8e08df1c8fc11e19de7349f2fb6bff1a82f257bbdda81ca32e8614ec9c", "grid_size": 64, "tile_count": 4, "text_tokens": 512, "shared_tokens": 0, "block_size": 64, "bridge": true, "full": 
- `synthetic.json`: all gates passed

| microbatch | global batch | median update s | records/s | peak allocated GiB | peak reserved GiB | status |
|---:|---:|---:|---:|---:|---:|---|
| 8 | 64 | 6.64 | 9.63 | 44.7 | 46.4 | OK |
| 16 | 128 | 13.37 | 9.58 | 66.1 | 70.2 | OK |
- Chosen microbatch: 8; resume drill: {"resume_exit": 0, "updates_logged": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], "contiguous": true, "coverage": {"logged": 768, "distinct": 768, "duplicates": 0, "complete": true}, "passed": true}

## 5. Training

- Final batch: microbatch 8 x 8 ranks = global batch 64, accumulation 1
- Updates: 1792 of 1792 planned (114688 records, epoch fraction 1.0000)
- Wall-clock (sum of update_seconds): 3.32 h; median update 6.65 s; median records/s 9.62
- Peak GPU memory (max over ranks and updates): allocated 44.7 GiB, reserved 46.4 GiB
- Final train MSE 0.029450; first 0.162967; grad norm final 0.0401; lr 5.00e-05
- Checkpointing: recovery every 5 updates (durable), retained adapters every 100 updates + final
- complete.json: {"complete": true, "completed_updates": 1792, "processed_records": 114688, "records": 114688}
- Record coverage (records/rank*.jsonl): {"logged": 114688, "distinct": 114688, "duplicates": 0, "complete": true}
- Logging: local metrics.jsonl only; no W&B run (user decision; spec §10 W&B run ID therefore absent from checkpoints)

## 6. Evaluation (2,048 COCO val prompts, seed 2707, matched initial latents)

- Protocol `full_flux`: backend native_sdpa_flash (full attention, no LoRA); steps 28; guidance 7.5; true_cfg 1.0; adapter_sha256 -; adapter_updates -
- Protocol `ours`: backend FlexAttention64 packed QKV (per-layer pack/unpack); steps 28; guidance 7.5; true_cfg 1.0; adapter_sha256 fb4bc94aa94f7dfe267db6f6ebedd0895b31e7d18687e551483fde5cbae8f6d8; adapter_updates 1792

| Set vs real COCO | FID (auxiliary) | KID mean | KID std over subsets |
|---|---:|---:|---:|
| full_flux | 49.933 | 0.010783 | 0.000822 |
| ours | 50.285 | 0.011262 | 0.000883 |
- Reference: real half vs half FID 58.171 (n=1024, finite-sample reference only, not a model result)
- KID: 100 subsets x 1000, seed 2707, shared draws; **std is not a confidence interval**; raw KID, no x1000 scaling. KID is the primary distribution metric; FID at n=2,048 is finite-sample biased.
- Implementation: torch-fidelity 0.4.0, torch-fidelity inception-v3-compat 2048 pool, weights sha256 `6726825d0af5f729cebd5821db510b11b1cfad8faad88a03f1befd49fb9129b2`

- **LPIPS-Alex at512** (ours vs full FLUX, same prompt and initial latent; bilinear antialias 1024->512 on [-1,1] float; {"package": "lpips", "version": "0.1.4", "net": "alex", "lpips_version": "0.1"}): mean 0.3755, median 0.3739, p90 0.5327, n=2048. Lower = closer to Full FLUX, not a quality claim.

- **CLIP cosine** (openai/clip-vit-large-patch14 @ 32bd64288804d66eefd0ccbe215aa642df71cc41, transformers 4.55.2; L2-normalised image/text embeddings, mean cosine, no scaling (not CLIPScore x100 / x2.5)): coco_real 0.2568, full_flux 0.2612, ours 0.2611

- Visual comparison page: `evaluation/comparisons/index.html` (present); subset frozen before metrics (`comparisons/subset.json`).

### Generation batch-size benchmark (16 eval prompts, one GPU each, nvidia-smi dmon SM% every 2 s)

| method | batch | steady s/image | speedup vs batch 1 | SM% busy mean |
|---|---:|---:|---:|---:|
| full | 1 | 6.27 | 1.00 | 91.1 |
| full | 2 | 6.06 | 1.03 | 91.2 |
| full | 4 | 5.90 | 1.06 | 90.5 |
| full | 8 | 5.79 | 1.08 | 91.7 |
| ours | 1 | 7.88 | 1.00 | 92.1 |
| ours | 2 | 7.65 | 1.03 | 93.0 |
| ours | 4 | 7.43 | 1.06 | 91.6 |
| ours | 8 | 7.34 | 1.07 | 90.6 |
- The 2,048-image sets above were generated at batch 1 (protocol.json `batch`); batching gives 3-8% at 1024^2 because SM is already ~91% busy at batch 1.

## 7. Measured / Estimated / Literature

- **Measured:** everything in sections 2–6 that is present above (manifests, cache audit, gates, capacity table, training log, metrics).
- **Estimated:** none are reported as results. Planning-time throughput and memory projections were superseded by the capacity table and the training log.
- **Literature:** no published competing method was run under this protocol in this round; no literature numbers are placed next to measured ones.

| Method | Source | Dataset / split | n | Resolution | Metric impl. | Reported value |
|---|---|---|---|---|---|---|
| (none) | | | | | | |
