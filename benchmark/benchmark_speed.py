"""
Speed Benchmark: HilbertA vs SpargeAttn vs SDPA (baseline)

Two modes:
  --mode kernel   Benchmark attention kernels with random Q/K/V
  --mode e2e      Full FLUX.2-klein inference pipeline timing

Usage:
  python benchmark/benchmark_speed.py --mode kernel
  python benchmark/benchmark_speed.py --mode e2e
"""

import gc
import os
import sys
import math
import time
import argparse
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Timing utilities ──

def cuda_timer(fn, num_warmup=10, num_iters=100):
    """Run fn, return median time in ms."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))

    times.sort()
    return times[len(times) // 2]


def wall_timer(fn, num_warmup=2, num_iters=5):
    """Run fn, return median wall-clock time in seconds."""
    for _ in range(num_warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(num_iters):
        torch.cuda.synchronize()
        t0 = time.time()
        fn()
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)

    times.sort()
    return times[len(times) // 2]


# ── Kernel benchmark ──

def capture_real_qkv(device="cuda:0"):
    """Run one inference step and capture Q/K/V from a double-stream block."""
    from diffusers import Flux2KleinPipeline
    from diffusers.models.embeddings import apply_rotary_emb as rope

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16
    ).to(device)

    captured = {}
    orig_processor = pipe.transformer.transformer_blocks[0].attn.processor

    class CaptureProcessor:
        def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                     attention_mask=None, image_rotary_emb=None, **kwargs):
            if hasattr(attn, 'fused_projections') and attn.fused_projections:
                query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
            else:
                query, key, value = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)

            query = query.unflatten(-1, (attn.heads, -1))
            key = key.unflatten(-1, (attn.heads, -1))
            value = value.unflatten(-1, (attn.heads, -1))
            query, key = attn.norm_q(query), attn.norm_k(key)

            if encoder_hidden_states is not None:
                if hasattr(attn, 'fused_projections') and attn.fused_projections and hasattr(attn, 'to_added_qkv'):
                    eq, ek, ev = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
                else:
                    eq, ek, ev = attn.add_q_proj(encoder_hidden_states), attn.add_k_proj(encoder_hidden_states), attn.add_v_proj(encoder_hidden_states)
                eq, ek, ev = eq.unflatten(-1, (attn.heads, -1)), ek.unflatten(-1, (attn.heads, -1)), ev.unflatten(-1, (attn.heads, -1))
                eq, ek = attn.norm_added_q(eq), attn.norm_added_k(ek)
                query, key, value = torch.cat([eq, query], dim=1), torch.cat([ek, key], dim=1), torch.cat([ev, value], dim=1)

            if image_rotary_emb is not None:
                query, key = rope(query, image_rotary_emb, sequence_dim=1), rope(key, image_rotary_emb, sequence_dim=1)

            if not captured:
                captured['q'] = query.transpose(1, 2).detach().clone()
                captured['k'] = key.transpose(1, 2).detach().clone()
                captured['v'] = value.transpose(1, 2).detach().clone()

            out = F.scaled_dot_product_attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
            out = out.transpose(1, 2).flatten(2, 3)

            if encoder_hidden_states is not None:
                txt_len = encoder_hidden_states.shape[1]
                enc_out, hs_out = out[:, :txt_len], out[:, txt_len:]
                return attn.to_out[1](attn.to_out[0](hs_out)), attn.to_add_out(enc_out)
            return attn.to_out[1](attn.to_out[0](out))

    pipe.transformer.transformer_blocks[0].attn.processor = CaptureProcessor()
    print("  Running one inference to capture real Q/K/V...")
    pipe(prompt="A beautiful landscape with mountains and a lake.",
         height=1024, width=1024, guidance_scale=1.0, num_inference_steps=10,
         generator=torch.Generator(device=device).manual_seed(42))

    q, k, v = captured['q'], captured['k'], captured['v']
    del pipe, captured
    torch.cuda.empty_cache()
    return q, k, v


def benchmark_kernel(args):
    device = "cuda:0"

    q, k, v = capture_real_qkv(device)
    B, H, L, D = q.shape
    seq_txt = 512
    seq_img = L - seq_txt
    sm_scale = 1.0 / math.sqrt(D)

    print(f"  Captured Q/K/V: shape={list(q.shape)}, dtype={q.dtype}\n")

    results = []

    # 1. SDPA baseline
    sdpa_ms = cuda_timer(lambda: F.scaled_dot_product_attention(q, k, v, scale=sm_scale))
    results.append(("SDPA (baseline)", sdpa_ms, 0.0))
    print(f"  SDPA: {sdpa_ms:.3f}ms")

    # 2. HilbertA Triton kernel
    sys.path.insert(0, os.path.join(ROOT, "reloc_attention", "triton_code"))
    from reloc_triton_kernel_parallel_sliding import attention as triton_attn

    # 4t: reorder_shared (text + center as shared)
    center_size = 256
    N_CTX_shared_4t = seq_txt + center_size
    N_CTX_4t = L - N_CTX_shared_4t
    try:
        ms = cuda_timer(lambda: triton_attn(
            q, k, v, N_CTX_shared_4t, N_CTX_4t, False, sm_scale, 4, 0, False
        ))
        local_group = N_CTX_4t // 4
        sparse_pairs = N_CTX_shared_4t * L + N_CTX_4t * (N_CTX_shared_4t + local_group)
        sparsity = 1.0 - sparse_pairs / (L * L)
        results.append(("HilbertA-4t-shared", ms, sparsity))
        print(f"  HilbertA-4t-shared: {ms:.3f}ms")
    except RuntimeError as e:
        print(f"  HilbertA-4t-shared: SKIPPED ({e})")

    # 16t: reorder (text only as shared, no center)
    N_CTX_shared_16t = seq_txt
    N_CTX_16t = L - N_CTX_shared_16t
    try:
        ms = cuda_timer(lambda: triton_attn(
            q, k, v, N_CTX_shared_16t, N_CTX_16t, False, sm_scale, 16, 0, False
        ))
        local_group = N_CTX_16t // 16
        sparse_pairs = N_CTX_shared_16t * L + N_CTX_16t * (N_CTX_shared_16t + local_group)
        sparsity = 1.0 - sparse_pairs / (L * L)
        results.append(("HilbertA-16t", ms, sparsity))
        print(f"  HilbertA-16t: {ms:.3f}ms")
    except RuntimeError as e:
        print(f"  HilbertA-16t: SKIPPED ({e})")

    # 3. SpargeAttn
    from spas_sage_attn import spas_sage_attn_meansim_cuda
    sparge_ms = cuda_timer(lambda: spas_sage_attn_meansim_cuda(
        q, k, v, simthreshd1=-0.3, cdfthreshd=0.85, pvthreshd=0.0,
        tensor_layout="HND", attention_sink=True, return_sparsity=False,
    ))
    # Get sparsity from one call
    _, sparge_sparsity = spas_sage_attn_meansim_cuda(
        q, k, v, simthreshd1=-0.3, cdfthreshd=0.85, pvthreshd=0.0,
        tensor_layout="HND", attention_sink=True, return_sparsity=True,
    )
    results.append(("SpargeAttn", sparge_ms, sparge_sparsity))
    print(f"  SpargeAttn: {sparge_ms:.3f}ms")

    # Summary
    print(f"\n{'Method':<20} {'Time(ms)':>10} {'Speedup':>9} {'Sparsity':>10}")
    print("-" * 52)
    for name, ms, sp in results:
        speedup = sdpa_ms / ms
        print(f"{name:<20} {ms:>9.3f}ms {speedup:>8.2f}x {sp*100:>9.1f}%")


# ── End-to-end benchmark ──

def benchmark_e2e(args):
    from diffusers import Flux2KleinPipeline

    device = "cuda:0"
    prompt = "A group of men and woman playing a game of frisbee."
    gen_kwargs = dict(
        prompt=prompt, height=1024, width=1024,
        guidance_scale=1.0, num_inference_steps=30,
        generator=torch.Generator(device=device).manual_seed(42),
    )

    results = []

    # 1. Baseline
    print("  Loading baseline...")
    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16
    ).to(device)

    baseline_s = wall_timer(lambda: pipe(**gen_kwargs))
    results.append(("Baseline", baseline_s))
    print(f"  Baseline: {baseline_s:.2f}s")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # 2. HilbertA 4t (reorder_shared) and 16t (reorder)
    flux2_dir = os.path.join(ROOT, "flux2_hilberta")
    sys.path.insert(0, flux2_dir)
    config_path = os.path.join(flux2_dir, "config.yaml")
    import yaml, importlib

    for num_tiles, method, label in [(4, "reorder_shared", "HilbertA-4t"), (16, "reorder", "HilbertA-16t")]:
        print(f"  Loading {label} ({method})...")

        # Update config.yaml
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        orig_method, orig_tiles = cfg['method'], cfg['num_tiles']
        cfg['method'] = method
        cfg['num_tiles'] = num_tiles
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # Reload modules to pick up new config
        import customized_attention_processor as cap
        import patch as hilbert_patch
        import reorder_utils as ru
        importlib.reload(cap)
        importlib.reload(hilbert_patch)
        importlib.reload(ru)

        pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16
        ).to(device)
        hilbert_patch.apply_patch(pipe, num_tiles=num_tiles, height=1024)

        s = wall_timer(lambda: pipe(**gen_kwargs))
        results.append((label, s))
        print(f"  {label}: {s:.2f}s")
        del pipe
        torch.cuda.empty_cache()

        # Restore config
        cfg['method'] = orig_method
        cfg['num_tiles'] = orig_tiles
        with open(config_path, 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

    # 3. SpargeAttn
    print("  Loading SpargeAttn...")
    sys.path.insert(0, os.path.join(ROOT, "sparge"))
    from evaluate.modify_model.modify_flux2 import set_spas_sage_attn_flux2

    pipe = Flux2KleinPipeline.from_pretrained(
        "black-forest-labs/FLUX.2-klein-9B", torch_dtype=torch.bfloat16
    ).to(device)

    os.environ["TUNE_MODE"] = ""
    set_spas_sage_attn_flux2(pipe.transformer)

    # Initialize hyperparams
    import torch.nn as nn
    from spas_sage_attn.autotune import SparseAttentionMeansim
    for block in list(pipe.transformer.transformer_blocks) + list(pipe.transformer.single_transformer_blocks):
        if hasattr(block.attn, 'inner_attention'):
            m = block.attn.inner_attention
            h = block.attn.heads
            m.head_num = h
            m.is_sparse = nn.Parameter(torch.ones(h, dtype=torch.bool, device=device), requires_grad=False)
            m.cdfthreshd = nn.Parameter(torch.ones(h, device=device) * 0.85, requires_grad=False)
            m.simthreshd1 = nn.Parameter(torch.ones(h, device=device) * -0.3, requires_grad=False)
            m.simthreshd2 = nn.Parameter(torch.zeros(h, device=device), requires_grad=False)
            m.pvthreshd = nn.Parameter(torch.ones(h, device=device) * 0.0, requires_grad=False)

    sparge_s = wall_timer(lambda: pipe(**gen_kwargs))
    results.append(("SpargeAttn", sparge_s))
    print(f"  SpargeAttn: {sparge_s:.2f}s")
    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    # Summary
    print(f"\n{'Method':<20} {'Time(s)':>10} {'Speedup':>9}")
    print("-" * 42)
    for name, s in results:
        speedup = results[0][1] / s
        print(f"{name:<20} {s:>9.2f}s {speedup:>8.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["kernel", "e2e"], required=True)
    args = parser.parse_args()

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"GPU: {gpu}\n")

    if args.mode == "kernel":
        print("=== Kernel Benchmark (1024x1024) ===")
        benchmark_kernel(args)
    else:
        print("=== End-to-End Benchmark (1024x1024, 10 steps) ===")
        benchmark_e2e(args)
