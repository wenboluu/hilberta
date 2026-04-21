"""
Benchmark: Triton Sparse Attention with N_CTX_shared = 0

Compare:
1. n_shared = 0 (pure local attention, no shared region)
2. n_shared = 768 (current config for 1024x1024)
3. Full SDPA

This isolates the impact of shared region on efficiency.
"""

import os
import torch
import torch.nn.functional as F
import math

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from reloc_triton_kernel_parallel_sliding import attention as triton_attention


def benchmark_config(
    n_shared: int,
    n_seq: int,
    num_groups: int,
    num_heads: int = 24,
    head_dim: int = 128,
    num_warmup: int = 10,
    num_iters: int = 100,
    dtype=torch.bfloat16,
):
    device = "cuda:0"
    total_len = n_shared + n_seq
    sm_scale = 1.0 / math.sqrt(head_dim)
    local_group_size = n_seq // num_groups

    # Create tensors
    q = torch.randn(1, num_heads, total_len, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, num_heads, total_len, head_dim, device=device, dtype=dtype)
    v = torch.randn(1, num_heads, total_len, head_dim, device=device, dtype=dtype)

    # Warmup
    for _ in range(num_warmup):
        _ = triton_attention(q, k, v, n_shared, n_seq, False, sm_scale, num_groups, 0, False)
        _ = F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
    torch.cuda.synchronize()

    # Benchmark Triton
    triton_times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = triton_attention(q, k, v, n_shared, n_seq, False, sm_scale, num_groups, 0, False)
        end.record()
        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))
    triton_median = sorted(triton_times)[len(triton_times)//2]

    # Benchmark SDPA
    sdpa_times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _ = F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
        end.record()
        torch.cuda.synchronize()
        sdpa_times.append(start.elapsed_time(end))
    sdpa_median = sorted(sdpa_times)[len(sdpa_times)//2]

    # Calculate theoretical speedup
    full_pairs = total_len * total_len
    if n_shared == 0:
        # Pure local: each query attends to local_group_size keys
        sparse_pairs = n_seq * local_group_size
    else:
        # Shared + local
        sparse_pairs = n_shared * total_len + n_seq * (n_shared + local_group_size)

    theoretical_speedup = full_pairs / sparse_pairs
    actual_speedup = sdpa_median / triton_median
    efficiency = actual_speedup / theoretical_speedup * 100

    return {
        "n_shared": n_shared,
        "n_seq": n_seq,
        "total_len": total_len,
        "num_groups": num_groups,
        "local_group_size": local_group_size,
        "triton_ms": triton_median,
        "sdpa_ms": sdpa_median,
        "theoretical_speedup": theoretical_speedup,
        "actual_speedup": actual_speedup,
        "efficiency": efficiency,
        "sparsity": 1 - sparse_pairs / full_pairs,
    }


def run_benchmark():
    print("="*90)
    print("BENCHMARK: Impact of Shared Region Size")
    print("="*90)

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    results = []

    # Test 1: 1024x1024, 4 tiles
    print("\n### 1024x1024 equivalent, 4 tiles ###")

    # n_shared = 0 (pure local)
    print("\nTesting n_shared=0 (pure local attention)...")
    r1 = benchmark_config(n_shared=0, n_seq=4608, num_groups=4)
    results.append(("1024² pure local (shared=0)", r1))

    # n_shared = 768 (current config)
    print("Testing n_shared=768 (current config)...")
    r2 = benchmark_config(n_shared=768, n_seq=3840, num_groups=4)
    results.append(("1024² with shared=768", r2))

    # Test 2: 1024x1024, 16 tiles
    print("\n### 1024x1024 equivalent, 16 tiles ###")

    # n_shared = 0
    print("\nTesting n_shared=0 (pure local attention)...")
    r3 = benchmark_config(n_shared=0, n_seq=4608, num_groups=16)
    results.append(("1024² pure local 16t (shared=0)", r3))

    # n_shared = 768
    print("Testing n_shared=768...")
    r4 = benchmark_config(n_shared=768, n_seq=3840, num_groups=16)
    results.append(("1024² with shared=768, 16t", r4))

    # Test 3: 2048x2048, 4 tiles
    print("\n### 2048x2048 equivalent, 4 tiles ###")

    # n_shared = 0
    print("\nTesting n_shared=0...")
    r5 = benchmark_config(n_shared=0, n_seq=16896, num_groups=4)
    results.append(("2048² pure local (shared=0)", r5))

    # n_shared = 1536
    print("Testing n_shared=1536 (current config)...")
    r6 = benchmark_config(n_shared=1536, n_seq=15360, num_groups=4)
    results.append(("2048² with shared=1536", r6))

    # Test 4: 2048x2048, 16 tiles
    print("\n### 2048x2048 equivalent, 16 tiles ###")

    # n_shared = 0
    print("\nTesting n_shared=0...")
    r7 = benchmark_config(n_shared=0, n_seq=16896, num_groups=16)
    results.append(("2048² pure local 16t (shared=0)", r7))

    # n_shared = 1536
    print("Testing n_shared=1536...")
    r8 = benchmark_config(n_shared=1536, n_seq=15360, num_groups=16)
    results.append(("2048² with shared=1536, 16t", r8))

    # Summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    print(f"{'Configuration':<35} {'Shared':>7} {'SeqLen':>7} {'Triton':>10} {'SDPA':>10} {'Theory':>8} {'Actual':>8} {'Effic':>7}")
    print("-"*90)
    for name, r in results:
        print(f"{name:<35} {r['n_shared']:>7} {r['total_len']:>7} {r['triton_ms']:>9.3f}ms {r['sdpa_ms']:>9.3f}ms {r['theoretical_speedup']:>7.2f}x {r['actual_speedup']:>7.2f}x {r['efficiency']:>6.1f}%")
    print("="*90)

    # Analysis
    print("\n" + "="*90)
    print("ANALYSIS: Shared Region Impact")
    print("="*90)

    # Compare pure local vs with shared
    print("\n1024x1024, 4 tiles:")
    pure = results[0][1]
    shared = results[1][1]
    print(f"  Pure local:  {pure['triton_ms']:.3f}ms, {pure['actual_speedup']:.2f}x speedup, {pure['efficiency']:.1f}% efficiency")
    print(f"  With shared: {shared['triton_ms']:.3f}ms, {shared['actual_speedup']:.2f}x speedup, {shared['efficiency']:.1f}% efficiency")
    print(f"  Shared region adds: {shared['triton_ms'] - pure['triton_ms']:.3f}ms (+{(shared['triton_ms']/pure['triton_ms'] - 1)*100:.1f}%)")

    print("\n1024x1024, 16 tiles:")
    pure = results[2][1]
    shared = results[3][1]
    print(f"  Pure local:  {pure['triton_ms']:.3f}ms, {pure['actual_speedup']:.2f}x speedup, {pure['efficiency']:.1f}% efficiency")
    print(f"  With shared: {shared['triton_ms']:.3f}ms, {shared['actual_speedup']:.2f}x speedup, {shared['efficiency']:.1f}% efficiency")
    print(f"  Shared region adds: {shared['triton_ms'] - pure['triton_ms']:.3f}ms (+{(shared['triton_ms']/pure['triton_ms'] - 1)*100:.1f}%)")

    print("\n2048x2048, 4 tiles:")
    pure = results[4][1]
    shared = results[5][1]
    print(f"  Pure local:  {pure['triton_ms']:.3f}ms, {pure['actual_speedup']:.2f}x speedup, {pure['efficiency']:.1f}% efficiency")
    print(f"  With shared: {shared['triton_ms']:.3f}ms, {shared['actual_speedup']:.2f}x speedup, {shared['efficiency']:.1f}% efficiency")
    print(f"  Shared region adds: {shared['triton_ms'] - pure['triton_ms']:.3f}ms (+{(shared['triton_ms']/pure['triton_ms'] - 1)*100:.1f}%)")

    print("\n2048x2048, 16 tiles:")
    pure = results[6][1]
    shared = results[7][1]
    print(f"  Pure local:  {pure['triton_ms']:.3f}ms, {pure['actual_speedup']:.2f}x speedup, {pure['efficiency']:.1f}% efficiency")
    print(f"  With shared: {shared['triton_ms']:.3f}ms, {shared['actual_speedup']:.2f}x speedup, {shared['efficiency']:.1f}% efficiency")
    print(f"  Shared region adds: {shared['triton_ms'] - pure['triton_ms']:.3f}ms (+{(shared['triton_ms']/pure['triton_ms'] - 1)*100:.1f}%)")

    print("\n" + "="*90)


if __name__ == "__main__":
    run_benchmark()
