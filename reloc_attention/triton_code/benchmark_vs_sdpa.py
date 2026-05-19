"""
Benchmark: Triton Sparse Attention vs PyTorch SDPA (Flash Attention)
"""

import os
import torch
import torch.nn.functional as F
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from reloc_triton_kernel_parallel_sliding import attention as triton_attention


def benchmark_attention(
    batch_size=1, num_heads=24, head_dim=128,
    image_size=4096, text_length=512, num_tiles=4,
    center_region_size=256, offset=0,
    num_warmup=10, num_iters=100, dtype=torch.bfloat16,
):
    device = "cuda:0"
    total_length = text_length + image_size
    N_CTX_shared = text_length + center_region_size
    N_CTX = image_size - center_region_size
    GROUPS = num_tiles
    local_group_size = N_CTX // GROUPS

    # Theoretical sparsity
    full_pairs = total_length * total_length
    sparse_pairs = N_CTX_shared * total_length + N_CTX * (N_CTX_shared + local_group_size)
    sparsity = 1.0 - sparse_pairs / full_pairs
    theoretical_speedup = full_pairs / sparse_pairs

    # Create inputs
    q = torch.randn(batch_size, num_heads, total_length, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, num_heads, total_length, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, num_heads, total_length, head_dim, device=device, dtype=dtype)
    sm_scale = 1.0 / math.sqrt(head_dim)

    # Warmup
    for _ in range(num_warmup):
        triton_attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, offset, False)
        F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
    torch.cuda.synchronize()

    # Benchmark Triton
    triton_times = []
    for _ in range(num_iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        triton_attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, offset, False)
        e.record()
        torch.cuda.synchronize()
        triton_times.append(s.elapsed_time(e))

    # Benchmark SDPA
    sdpa_times = []
    for _ in range(num_iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
        e.record()
        torch.cuda.synchronize()
        sdpa_times.append(s.elapsed_time(e))

    triton_med = sorted(triton_times)[len(triton_times) // 2]
    sdpa_med = sorted(sdpa_times)[len(sdpa_times) // 2]
    actual_speedup = sdpa_med / triton_med

    return {
        "triton_ms": triton_med,
        "sdpa_ms": sdpa_med,
        "actual_speedup": actual_speedup,
        "theoretical_speedup": theoretical_speedup,
        "efficiency": actual_speedup / theoretical_speedup * 100,
        "sparsity": sparsity,
        "group_size": local_group_size,
    }


def main():
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    print(f"GPU: {gpu}\n")

    configs = [
        ("1024x1024, 4t, off=0",   dict(image_size=4096,  num_tiles=4,  center_region_size=256,  offset=0)),
        ("1024x1024, 4t, off=256", dict(image_size=4096,  num_tiles=4,  center_region_size=256,  offset=256)),
        ("1024x1024, 16t",         dict(image_size=4096,  num_tiles=16, center_region_size=256,  offset=0)),
        ("2048x2048, 4t",          dict(image_size=16384, num_tiles=4,  center_region_size=1024, offset=0)),
        ("2048x2048, 16t",         dict(image_size=16384, num_tiles=16, center_region_size=1024, offset=0)),
    ]

    results = []
    for name, kwargs in configs:
        try:
            r = benchmark_attention(**kwargs)
            results.append((name, r))
            print(f"  {name}: triton={r['triton_ms']:.3f}ms  sdpa={r['sdpa_ms']:.3f}ms  speedup={r['actual_speedup']:.2f}x  (theory={r['theoretical_speedup']:.2f}x)")
        except RuntimeError as e:
            print(f"  {name}: SKIPPED ({e})")

    # Summary table
    print(f"\n{'Config':<25} {'Sparsity':>9} {'Triton':>9} {'SDPA':>9} {'Speedup':>9} {'Theory':>9} {'Effic':>8}")
    print("-" * 82)
    for name, r in results:
        print(f"{name:<25} {r['sparsity']*100:>8.1f}% {r['triton_ms']:>8.3f}ms {r['sdpa_ms']:>8.3f}ms {r['actual_speedup']:>8.2f}x {r['theoretical_speedup']:>8.2f}x {r['efficiency']:>7.1f}%")


if __name__ == "__main__":
    main()
