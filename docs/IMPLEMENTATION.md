# Implementation and validation status

## What is included

- Composite Hilbert ordering and a four-phase sliding tile schedule.
- Geometry-derived reciprocal block bridges: K limits external selections before reciprocal closure. Text queries attend to all tokens; all image queries attend to text. The released configs have no shared image region.
- Paired Triton inference with exact common-normalizer attention, metadata construction and persistent sampling layout.
- Flex64 differentiable attention for training. This path packs/unpacks each layer and is not persistent-layout training.
- CPU topology/metadata, contract, Diffusers adapter and complete tiny sampling tests, plus Flex mask/packing and gradient checks and portable data tests.
- Block-level adjacency during bridge-plan setup avoids an image-token N×N temporary mask; the explicit-mask CPU oracle still verifies the resulting topology.
- New portable cache, DDP LoRA training, generation, evaluation and resident kernel benchmark entrypoints.

The kernels/adapters derive from the historical research branch. The portable CLIs are a cleaned release interface, not the original remote training code used to produce every report. CPU correctness does not establish full-model BF16 gradient equivalence, full-scale memory capacity, throughput, or training convergence for these new runners.

## Release boundaries

| Item | Status |
|---|---|
| Reported 1024/2048 quality and H100 timings | Historical results; machine-readable aggregates included |
| Published rank-32 LoRA weights | Pending; do not substitute untrained sparse outputs |
| Exact historical train/eval prompt manifests | Not bundled; sample prompt file is only for smoke tests |
| Portable cache/train/evaluate runner GPU production validation | Pending |
| Tiny CPU model/layout/sampling correctness | Automated tests included; check CI for this revision |
| GPU kernel numerical checks | Benchmark includes sampled-row checks; must run on target device |
| Competitor kernels/checkpoints | Not vendored; historical comparison settings documented in results |
| 4096 image quality | Not evaluated; performance measurements only |
| Wan video | Research archive only, outside this release |

Do not describe the archive's measured speedups as a fresh benchmark of this commit. Keep model revision, dataset IDs, prompt manifest hash, adapter checkpoint, configuration, environment and timing scope with every new result. Do not change topology/backend during an epoch.

## Reproducibility gates before claiming the paper numbers

1. Obtain the matching LoRA weights and exact evaluation manifest.
2. Run target-GPU kernel correctness checks and small same-seed full-pipeline checks.
3. Validate cache Euler replay and record coverage; compare the new runner's losses/gradients with the historical implementation on the same records before production training.
4. Verify evaluation preprocessing against the historical evaluator, then run the complete frozen protocol.
5. Measure attention, denoiser and generation scopes separately. State whether text encoding, reordering and VAE decode are included.

## Aggregate result schema

`results/image_results.json` records the source research commit. Rows follow `names`. Each `quality[resolution]` row contains FID, KID mean, KID subset std, CLIP cosine, LPIPS mean, PSNR mean, SSIM mean, and batch-1 seconds/image. Null fidelity fields denote the Full FLUX reference. Kernel rows cover Full/4-tile/16-tile and columns 1024/2048/4096. Denoiser rows follow all method names with the same resolution columns; CLEAR's 4096 null means OOM in the tested path, not a zero latency.

## Release preparation checks (2026-11-10)

The final CPU suite passed 70 tests with 90 reported subtests, including
Flex mask/packing and gradient checks and portable data tests. A synthetic
1024 / 4-tile K1 paired-kernel smoke check passed all four phases on an
RTX 6000 Ada GPU; maximum sampled-row relative L2 error was 0.001245.
These checks support extraction correctness. They are not new H100 performance
measurements or full FLUX generation/training validation. Subsequent CI runs
are tracked separately.

## Website validation

The separate `gh-pages` source is static and self-contained. Chromium checks passed at desktop (1440px) and mobile (390px): no page errors, no external asset requests, no document overflow; resolution/phase/bridge/scaling controls work. Its block-level diagram was checked against all four actual K1 plan phases. No model server is deployed.

## Repository cleanup checks (2026-11-11)

The code and tests now use one Ruff format, enforced in CI alongside undefined-name and unused-code checks. Kernel and model-integration ASTs are unchanged; code changes remove unused imports/an unused setup tensor and make the package export explicit. The CPU suite passed 70 tests and 90 subtests after cleanup. No new GPU performance or training claims were made. The README now links directly to the supported entrypoints and the project page, without requiring a historical research branch.
