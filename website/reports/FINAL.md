# HilbertA 交接实验：最终汇总

更新：2026-11-08。本文汇总 `flux-training-handoff` 上全部已完成实验的结果。表中数字都是实测值，原始数字和细节见各阶段报告：

| 报告 | 内容 |
|---|---|
| [`docs/REPORT_STAGE1.md`](REPORT_STAGE1.md) | FLUX 1024，4-tile K1 训练与 COCO-1024 评估 |
| [`docs/REPORT_STAGE2.md`](REPORT_STAGE2.md) | FLUX 1024，16-tile K3 训练；kernel/denoiser/e2e 速度；与 CLEAR、SpargeAttn 对比 |
| [`docs/REPORT_STAGE3.md`](REPORT_STAGE3.md) | FLUX 2048，两版 LoRA 续训与 2048 评估 |
| [`video/wan21/REPORT_TRAIN_ZH.md`](../video/wan21/REPORT_TRAIN_ZH.md) | Wan2.1-T2V-1.3B 六方法 LoRA，8×H100 训练、dev 和 28 段视频指标 |

所有实验都在 8×H100 80GB 上完成。权重、cache、生成的图片和视频都保存在 HDFS `zsy/flux_sparse/` 和 `zsy/wan21_sparse/` 下，不在 GitHub 上。HuggingFace 曾尝试公开发布，没有成功，仓库已删除。

---

## 1. 结论

1. **FLUX 质量与 Full FLUX 基本持平。** 两版 HilbertA 的 FID/KID/CLIP 都和 Full FLUX.1-dev 在同一水平。与 Full FLUX 成对比较时，LPIPS、PSNR、SSIM 都明显好于 CLEAR 和 SpargeAttn，在 1024 和 2048 两个分辨率都成立。
2. **FLUX 速度随分辨率增加而提高。** 在 1024 下，attention 只占 denoiser 时间的约 22%，所以端到端只快 1.14–1.17×。到 2048，端到端快 1.61–1.81×。到 4096，denoiser 快 2.8–4.0×；kernel 单独测在 2048 快 4.1–8.7×，在 4096 快 5.1–16×。在所有测到的分辨率上，16-tile K3 都是最快的方法。
3. **4-tile K1 和 16-tile K3 各有所长。** 4-tile 更贴近 Full FLUX（LPIPS 更低，PSNR/SSIM 更高）。16-tile 的 FID/KID 略好，速度也更快。
4. **2048 续训有效。** 从 1024 LoRA 接着训一个 epoch 后，2048 的 FID/KID 与 Full FLUX 的差距在 0.1–0.8 FID 以内，LPIPS 和 1024 时基本一样。CLEAR 和 SpargeAttn 直接跑在 2048 上，质量明显下降。
5. **Wan 视频：只有 288_bridge 生成的画面基本连贯。** 六种 tile 稀疏方法都能把 dev 误差降低 64–75%。但没有跨 tile 连接的 5 种方法（四种 base 排序，加上 halo）生成的视频里，能看到明显的拼块和重复主体。只有 288_bridge 的画面基本连贯，它的 dev MSE 和对 teacher 的 LPIPS 也都是最好的。如果只比较排序方式，tube 和 844 好于纯 3D Hilbert。

---

## 2. FLUX.1-dev（HilbertA LoRA）

### 2.1 设置

四版 LoRA 共用以下设置：

| 项目 | 设置 |
|---|---|
| Base | FLUX.1-dev，revision `3de623f`，BF16，冻结 |
| LoRA | rank 32，alpha 32，dropout 0，304 个 Linear 模块，共 74.7M 参数，FP32 |
| Attention | composite Hilbert 排序，4 个 phase 循环滑动（`layer % 4`），64-token block 间 reciprocal bridge，无 shared region，text 全局可见 |
| Teacher | 原始 full-attention FLUX；2,048 条 COCO train caption × 2 seeds × 28 steps = 114,688 records；guidance 7.5 |
| 训练 | 1 epoch = 1,792 updates，global batch 64，AdamW，wd 1e-4，grad clip 1.0，warmup 100 后 LR 恒定 |
| 评估 | 2,048 条 COCO val caption，每条 1 张图，seed 2707，各方法初始 latent 相同，28 steps，guidance 7.5 |

指标说明：
- FID/KID：和真实 COCO 图像比较。用 torch-fidelity，KID 取 100 个 subset × 每个 1000 张。
- LPIPS（AlexNet，缩到 512）、PSNR、SSIM：和同一初始 latent 生成的 Full FLUX 图成对比较。
- CLIP：ViT-L/14 计算的图文 cosine 相似度。

| LoRA | 分辨率 | tiles / bridge | 初始化 | LR | 训练耗时 | train MSE（首 → 末） |
|---|---|---|---|---|---|---|
| 4-tile K1 @1024 | 1024 | 4 / K=1 | zero LoRA | 5e-5 | 3.32 h | 0.163 → 0.029 |
| 16-tile K3 @1024 | 1024 | 16 / K=3 | zero LoRA | 5e-5 | 3.19 h | – → 0.048 |
| 4-tile K1 @2048 | 2048 | 4 / K=1 | 4-tile K1 @1024 | 2e-5，optimizer 重置 | 13.91 h | 0.117 → 0.037 |
| 16-tile K3 @2048 | 2048 | 16 / K=3 | 16-tile K3 @1024 | 2e-5，optimizer 重置 | 12.27 h | 0.162 → 0.063 |

四版都跑完了全部 1,792 个 update，每条 record 恰好用过一次，没有 NaN。

### 2.2 1024×1024 质量与速度

| 方法 | FID | KID | CLIP | LPIPS↓ | PSNR↑ | SSIM↑ | e2e s/img（b1） |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full FLUX | 49.93 | 0.0108 | 0.2612 | – | – | – | 5.82 |
| **HilbertA 4-tile K1** | 50.29 | 0.0113 | 0.2611 | **0.375** | **15.63** | **0.665** | 5.10（1.14×） |
| **HilbertA 16-tile K3** | **49.79** | 0.0107 | 0.2610 | 0.414 | 15.09 | 0.645 | **4.96（1.17×）** |
| CLEAR local16 | 49.99 | 0.0101 | 0.2608 | 0.477 | 13.83 | 0.604 | 5.52（1.05×） |
| SpargeAttn top-k 0.25 | 53.68 | 0.0093 | 0.2466 | 0.602 | 12.35 | 0.540 | 5.40（1.08×） |

### 2.3 2048×2048 质量与速度

| 方法 | FID | KID | CLIP | LPIPS↓ | PSNR↑ | SSIM↑ | e2e s/img（b1） |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full FLUX | 52.40 | 0.0119 | 0.2579 | – | – | – | 31.90 |
| **HilbertA 4-tile K1** | 53.17 | 0.0125 | 0.2592 | **0.376** | **15.11** | **0.742** | 19.83（1.61×） |
| **HilbertA 16-tile K3** | 52.54 | 0.0121 | **0.2601** | 0.422 | 14.39 | 0.724 | **17.65（1.81×）** |
| CLEAR local16 | 57.23 | 0.0092 | 0.2491 | 0.629 | 11.14 | 0.595 | 21.60（1.48×） |
| SpargeAttn top-k 0.25 | 54.14 | 0.0106 | 0.2577 | 0.542 | 12.33 | 0.640 | 20.11（1.59×） |

注意，2048 这一档的条件并不对等：
- **HilbertA：** 用 2048 续训过的 LoRA。
- **CLEAR：** 用它官方在 1024 训练的 checkpoint，直接跑 2048。
- **SpargeAttn：** 本身不需要训练，也是直接跑 2048。

### 2.4 速度分解（batch 1，单张 H100）

| 方法 | denoiser 1024 | denoiser 2048 | denoiser 4096 | kernel 1024 | kernel 2048 | kernel 4096 |
|---|---:|---:|---:|---:|---:|---:|
| Full FLUX（native flash SDPA） | 208 ms | 1150 ms | 11128 ms | 0.814 ms | 10.19 ms | 153.5 ms |
| HilbertA 4-tile K1 | 1.16× | 1.61× | 2.83× | 2.56× | 4.14× | 5.05× |
| HilbertA 16-tile K3 | 1.18× | 1.80× | **4.04×** | 3.16× | **8.72×** | **16.06×** |
| CLEAR local16 | 1.06× | 1.49× | OOM | – | – | – |
| SpargeAttn top-k 0.25 | 1.08× | 1.61× | 3.10× | – | – | – |

测速条件：
- **同一 harness：** 所有方法都在同一套测速代码里跑。CLEAR 和 SpargeAttn 用各自官方代码和权重，在各自的环境里运行。
- **LoRA 已 merge：** 测速时我们的 LoRA 已经 merge 进 BF16 权重。merge 前后的输出相对 L2 差 0.005。
- **kernel 的口径：** kernel 时间是 57 层按 phase 加权后的单层延迟。
- **生成图用的是另一个 backend：** 评估用的图由 Flex64 训练 backend 生成，测速用的是 paired 推理 kernel。两者输出的相对 L2 差不超过 0.016，在 0.02 的门限内。
- **SpargeAttn 有额外量化：** 它的 Q/K 是 INT8、V 是 FP8；我们和 CLEAR 都是 BF16。

### 2.5 其他数字

- **1024 生成时用 batch 的收益：** batch 1 → 8，Full FLUX 快 8%，ours 快 7%。原因是 batch 1 时 GPU 的 SM 利用率已经约 91%，再加 batch 提升有限（stage 1）。
- **2048 直接用 1024 LoRA：** 在 2048 训练之前，用 1024 的 LoRA 直接生成了 10 张 2048 的图，只做了目测对比，没有算指标（stage 3 §4）。
- **评估集不是完全没见过的数据：** COCO 评估集和仓库早期诊断时用过的 caption 有少量重叠：1 条硬编码的 caption，另有 20 条 caption 文本重叠。所以这里不声称评估集是"未见过"的数据（stage 1 §2）。

---

## 3. Wan2.1-T2V-1.3B 六方法 LoRA

### 3.1 设置

| 项目 | 设置 |
|---|---|
| Base | Wan2.1-T2V-1.3B-Diffusers，revision `0fad780`，冻结 |
| 视频 | 480×832，81 帧，16 fps，每段 32,760 个 image token |
| Attention | tile 8192，4 个 phase 偏移（0/2048/4096/6144），无 shared；text cross-attention 保持 dense |
| LoRA | 每种方法一套，rank 32，alpha 32，覆盖全部 300 个 Linear |
| 数据 | OpenVid-1M：2,048 段 train（每段 4 条 record）和 64 段 dev（每段 8 条 record），train/dev 按原始视频来源分开 |
| Teacher | 原始 full-attention Wan，在真实视频加噪后的输入上做一次 conditional forward |
| 训练 | 8 卡 DDP，6 种方法合成一个 batch 一起算，每种方法的 global batch 为 8；共 2,048 updates（2 epoch）；LR 1e-4 恒定；每个 adapter 单独 clip 到 1 |
| 速度 | 每个 update 13.37 s，总训练 8.3 h；teacher cache 共 93 GB，建 cache 用了 43 min |

各方法的含义：
- **288_base：** 2×8×8 的分层 Hilbert 排序。
- **hilbert3d：** 纯 3D Hilbert 排序。
- **844_base：** 8×4×4 的分层 Hilbert 排序。
- **tube：** 空间上用 Hilbert 排序，同一位置在时间上的 token 连续排列。
- **288_halo：** 288 排序，每个 tile 在序列上前后各多看 205 个 token。
- **288_bridge：** 288 排序，在时空相邻、但被 tile 边界切开的 64-token block 之间加 reciprocal 连接。额外的 attention 计算约为 base 的 5%。

### 3.2 Dev（512 条 record）与生成（4 个 prompt × seed 2707）

| 方法 | dev MSE step0 | dev MSE step2048 | 2048/0 | NSE | ViCLIP↑ | LPIPS→teacher↓ | denoise s |
|---|---:|---:|---:|---:|---:|---:|---:|
| full attention | – | – | – | – | 0.289 | 0 | 70.5 |
| 288_base | 0.0726 | 0.0251 | 0.35 | 0.0162 | 0.289 | 0.633 | 61.3 |
| hilbert3d | 0.0528 | 0.0154 | 0.29 | 0.0100 | 0.279 | 0.663 | 61.3 |
| 844_base | 0.0489 | 0.0129 | 0.26 | 0.0084 | 0.279 | 0.627 | 61.4 |
| tube | 0.0479 | 0.0121 | 0.25 | 0.0078 | 0.290 | 0.614 | 61.2 |
| 288_halo | 0.0664 | 0.0237 | 0.36 | 0.0154 | 0.299 | 0.645 | 62.9 |
| **288_bridge** | **0.0372** | **0.0097** | 0.26 | **0.0063** | 0.282 | **0.519** | 61.8 |

- **dev 一直在下降：** 10 次 dev 评估中，每种方法的误差都是单调下降，没有回升。各 σ 档的比值和 VBench 四项见 Wan 报告。
- **看视频时最明显的现象：** 除 bridge 外，其余方法的视频里都有明显的竖直或矩形接缝，以及重复出现的主体，例如两只猫、两个人。VBench 的 consistency 指标衡量的是帧与帧之间的稳定，所以没有反映出这个问题。逐帧对比图在 `video/wan21/report_data/compare/`。
- **样本量很小：** 只有 4 个视频，结果只能用来筛选机制，不能作为可靠的质量排名。
- **速度：** sparse 方法每层都要重新 pack 和 unpack Q/K/V，没有用 persistent layout。所以 denoise 只比 full attention 快约 13%，不能直接换算成理论上的加速比。

### 3.3 检查（未通过的项照实记录）

- **H100 smoke：** 全部通过。cross-batch gate 在 H100 上也通过了，而它在原来的 Ada 机器上是失败的。
- **DDP 同步：** 我预设的门限是 1e-4，**实测没有过**。DDP 梯度与参考梯度的相对 L2 差是 0.0016–0.0026。作为对照，同一条 record 重复 backward 两次本身就差 0.0015–0.0019，两者同一量级。各卡之间的梯度和参数 bitwise 一致。导出再加载后 bitwise 一致。
- **resume：** 数据读取顺序完全一致；loss 的差异和上面的 backward 噪声在同一量级。

---

## 4. 产物位置

| 内容 | 位置 |
|---|---|
| FLUX 四版 LoRA（step 1792） | HDFS `zsy/flux_sparse/experiment/{training, training_16tile_k3, training_2048_t4k1, training_2048_t16k3}/checkpoints/adapter_step001792.pt` |
| FLUX 生成图，每种方法 2,048 张 | HDFS `zsy/flux_sparse/experiment/evaluation*/`；对比页为 `evaluation/comparisons/index.html` 和 `evaluation_2048/comparisons/index_stage3.html` |
| FLUX 逐图指标 | HDFS `evaluation*/per_image_metrics.jsonl`、`evaluation_2048/metrics_stage3/` |
| Wan 六套 adapter，以及 final 和 last 状态 | HDFS `zsy/wan21_sparse/train/` |
| Wan 28 段视频、逐帧数据、指标 | HDFS `zsy/wan21_sparse/generation/`；指标原始文件在本仓库 `video/wan21/report_data/` |
| Wan 数据集和 cache | HDFS `zsy/wan21_sparse/{data, cache}`，manifest 在本仓库 `report_data/manifest/` |
| 代码 | FLUX：`experiments/coco1024/`、`experiments/flux_kernel/`；Wan：`video/wan21/code/{prod,data,eval}`、`video/wan21/scripts/` |
