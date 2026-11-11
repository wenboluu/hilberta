# HilbertA project page

Self-contained static GitHub Pages site. No framework, external fonts, CDN, tracking, model runtime, API keys, or build dependencies. CSS, JavaScript, SVG and datasets are local. External links are navigation only.

Preview: `python -m http.server 8000 --directory /path/to/project-page`, then open `http://localhost:8000`. Serve over HTTP; browser file:// fetching restrictions prevent the interactive data from loading otherwise.

GitHub Actions publishes this folder as the root of the Pages artifact. All asset paths are relative and work under `/hilberta/`. The initial website was migrated from the [development `gh-pages` commit](https://github.com/wenboluu/hilberta-dev/commit/19c8eb8c4d2da30e213ef98c76b865a97e2f2952).

## Content

- `data/results.json`: stage-report metrics; source commit is recorded.
- `data/efficiency_all_rows.csv`: detailed benchmark records.
- `data/demo.json`: exact 64-token block connectivity for composite Hilbert, four phases, 4 tiles K1, no shared image region.
- `reports/`: verbatim source reports (historical paths are provenance, not current reproduction instructions).
- `assets/site.js`: browser-only phase, resolution and chart controls.

Default recommendation: 4-tile K1 (teacher fidelity). 16-tile K3 is the speed option. Trained weights are forthcoming; no download link is fabricated.

Measured scope is stated beside tables/charts. Generation excludes VAE decode. Kernel timing excludes external packing. 4096 quality is not evaluated; CLEAR/Sparge standalone kernel timings are unavailable. No generated examples are invented.

## Diagram validation

`python tools/generate_demo.py` regenerates the geometry with the Python standard library. Diagram centers, all four phase offsets and every reciprocal bridge pair were verified against the existing `bridge_k_plan(4,1,False,grid_size=64)` implementation on CPU.

## Research-page redesign (2026-11-10)

Autoplay method animation uses exact stored geometry. Four stages: Hilbert traversal, contiguous sequence, four sliding phases, reciprocal bridges. Pauses offscreen/in background; reduced-motion starts paused. Pause and manual stage controls are available. Quality is split into two compact tables; KID is explicitly scaled ×1000. Bold values are best among sparse methods based on unrounded source values. Default quality preset remains 4-tile K1. Performance charts are separate from quality and from generation timing.

## Picture and attention view

`assets/patch-demo.jpg` is a downsampled Full FLUX image from the project's existing evaluation assets (prompt ID `5b8c27a58c5da3cdc592`: geraniums on the window sill of an old house). It illustrates patch movement only; it is not a new quality result. The source was `comparison_six_variants_20prompts_v1/full/5b8c27a58c5da3cdc592.png`.

The image is cropped into 8×8 patches matching the actual 64-token blocks. The attention matrix uses sequence indices on both axes. Its image-only entry counts are 4096 (full), 1024 (tile), and 1054 / 1072 / 1064 / 1072 (tile + reciprocal bridge, by phase). These counts are not measured speedups. Global text connections remain active in the real model and are omitted from this diagram.

Denoising time is shown in two compact tables, one per resolution; speedup uses unrounded measurements. Quality controls also select the corresponding kernel readout. The page uses one slate-blue visual palette, subdued source-image colors, and line styles for competitor series.

The picture is preloaded and decoded before SVG crop construction or autoplay. Initial block positions are assigned before mounting. Each reorder gets a 900 ms settling interval before the corresponding attention view appears. Autoplay requires 90% of the method figure to be visible, and pauses in a background tab. Phase controls reserve constant space; mobile controls precede the diagram.

Independent review and Chromium checks cover delayed image loading, a complete automatic loop, block/curve alignment, all phases, desktop/mobile layout, reduced motion and local-only runtime requests.
