"""
Benchmark: Triton Sparse Attention Kernel vs PyTorch F.scaled_dot_product_attention

This script compares the performance of:
1. Your Triton sparse attention kernel (with sparsity from reorder_shared method)
2. Standard PyTorch SDPA (full attention)

The goal is to measure whether the sparsity translates into actual speedup.
"""

import os
import sys
import torch
import torch.nn.functional as F
import math
import time
from torch.utils.benchmark import Timer
import triton

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import your Triton kernel
from reloc_triton_kernel_parallel_sliding import attention as triton_attention

def benchmark_attention(
    batch_size: int = 1,
    num_heads: int = 24,
    head_dim: int = 128,
    image_size: int = 4096,  # 4096 for 1024x1024, 16384 for 2048x2048
    text_length: int = 512,
    num_tiles: int = 4,
    center_region_size: int = 256,  # 16x16=256 for 1024, 32x32=1024 for 2048
    offset: int = 0,
    num_warmup: int = 10,
    num_iters: int = 100,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Benchmark Triton sparse attention vs SDPA.

    For reorder_shared method:
    - N_CTX_shared = text_length + center_region_size (tokens that attend to everything)
    - N_CTX = image_size - center_region_size (tokens with local attention only)
    - GROUPS = num_tiles (number of local attention groups)

    Sparsity calculation:
    - Full attention FLOPs: O(L^2) where L = text_length + image_size
    - Sparse attention FLOPs:
        - Shared tokens (N_CTX_shared): attend to all L tokens
        - Local tokens (N_CTX): attend to N_CTX_shared + (N_CTX / GROUPS) tokens each
    """
    device = "cuda:0"

    # Total sequence length
    total_length = text_length + image_size

    # For reorder_shared method
    N_CTX_shared = text_length + center_region_size
    N_CTX = image_size - center_region_size
    GROUPS = num_tiles

    # Calculate theoretical sparsity
    full_attention_pairs = total_length * total_length
    # Shared tokens attend to all, local tokens attend to shared + local_group
    local_group_size = N_CTX // GROUPS
    sparse_attention_pairs = (
        N_CTX_shared * total_length +  # shared tokens attend to all
        N_CTX * (N_CTX_shared + local_group_size)  # local tokens attend to shared + local group
    )
    sparsity = 1.0 - (sparse_attention_pairs / full_attention_pairs)
    theoretical_speedup = full_attention_pairs / sparse_attention_pairs

    # Kernel efficiency analysis
    BLOCK_M = 128  # Typical Triton block size
    BLOCK_N = 64
    num_blocks_per_group = math.ceil(local_group_size / BLOCK_M)
    wasted_in_last_block = (num_blocks_per_group * BLOCK_M - local_group_size) if local_group_size % BLOCK_M != 0 else 0
    block_utilization = local_group_size / (num_blocks_per_group * BLOCK_M) if num_blocks_per_group > 0 else 0

    # KV iterations per query block
    kv_iters_shared = math.ceil(N_CTX_shared / BLOCK_N)
    kv_iters_local = math.ceil(local_group_size / BLOCK_N)
    total_kv_iters = kv_iters_shared + kv_iters_local

    print(f"\n{'='*70}")
    print(f"Configuration:")
    print(f"  Image size: {image_size} tokens ({int(math.sqrt(image_size))}x{int(math.sqrt(image_size))})")
    print(f"  Text length: {text_length}")
    print(f"  Total sequence length: {total_length}")
    print(f"  Num tiles (GROUPS): {num_tiles}")
    print(f"  Center region: {center_region_size} tokens")
    print(f"  N_CTX_shared: {N_CTX_shared}")
    print(f"  N_CTX (local): {N_CTX}")
    print(f"  Local group size: {local_group_size}")
    print(f"  Offset: {offset}")
    print(f"  Batch size: {batch_size}, Heads: {num_heads}, Head dim: {head_dim}")
    print(f"  Dtype: {dtype}")
    print(f"\nTheoretical Analysis:")
    print(f"  Full attention pairs: {full_attention_pairs:,}")
    print(f"  Sparse attention pairs: {sparse_attention_pairs:,}")
    print(f"  Sparsity: {sparsity*100:.2f}%")
    print(f"  Theoretical speedup: {theoretical_speedup:.2f}x")
    print(f"\nKernel Efficiency Analysis (assuming BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}):")
    print(f"  Local group size: {local_group_size} tokens")
    print(f"  Blocks per group: {num_blocks_per_group}")
    print(f"  Block utilization: {block_utilization*100:.1f}%")
    print(f"  Wasted slots in last block: {wasted_in_last_block}")
    print(f"  KV iterations (shared): {kv_iters_shared}")
    print(f"  KV iterations (local): {kv_iters_local}")
    print(f"  Total KV iterations per query block: {total_kv_iters}")
    if local_group_size < BLOCK_M:
        print(f"  WARNING: Group size ({local_group_size}) < BLOCK_M ({BLOCK_M}) - inefficient!")
    print(f"{'='*70}")

    # Create input tensors
    query = torch.randn(batch_size, num_heads, total_length, head_dim, device=device, dtype=dtype)
    key = torch.randn(batch_size, num_heads, total_length, head_dim, device=device, dtype=dtype)
    value = torch.randn(batch_size, num_heads, total_length, head_dim, device=device, dtype=dtype)

    sm_scale = 1.0 / math.sqrt(head_dim)

    # Warmup
    print("\nWarming up...")
    for _ in range(num_warmup):
        # Triton sparse attention
        _ = triton_attention(query, key, value, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, offset, False)
        # SDPA full attention
        _ = F.scaled_dot_product_attention(query, key, value, scale=sm_scale)
    torch.cuda.synchronize()

    # Benchmark Triton sparse attention
    print(f"\nBenchmarking Triton sparse attention ({num_iters} iterations)...")
    torch.cuda.synchronize()

    triton_times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = triton_attention(query, key, value, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, offset, False)
        end.record()

        torch.cuda.synchronize()
        triton_times.append(start.elapsed_time(end))

    triton_median = sorted(triton_times)[len(triton_times) // 2]
    triton_mean = sum(triton_times) / len(triton_times)
    triton_min = min(triton_times)
    triton_max = max(triton_times)

    # Benchmark SDPA full attention
    print(f"Benchmarking SDPA full attention ({num_iters} iterations)...")
    torch.cuda.synchronize()

    sdpa_times = []
    for _ in range(num_iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = F.scaled_dot_product_attention(query, key, value, scale=sm_scale)
        end.record()

        torch.cuda.synchronize()
        sdpa_times.append(start.elapsed_time(end))

    sdpa_median = sorted(sdpa_times)[len(sdpa_times) // 2]
    sdpa_mean = sum(sdpa_times) / len(sdpa_times)
    sdpa_min = min(sdpa_times)
    sdpa_max = max(sdpa_times)

    # Calculate actual speedup
    actual_speedup = sdpa_median / triton_median
    efficiency = actual_speedup / theoretical_speedup * 100

    # Print results
    print(f"\n{'='*70}")
    print(f"Results (times in ms):")
    print(f"{'='*70}")
    print(f"  Triton Sparse Attention:")
    print(f"    Median: {triton_median:.4f} ms")
    print(f"    Mean:   {triton_mean:.4f} ms")
    print(f"    Min:    {triton_min:.4f} ms")
    print(f"    Max:    {triton_max:.4f} ms")
    print(f"\n  SDPA Full Attention:")
    print(f"    Median: {sdpa_median:.4f} ms")
    print(f"    Mean:   {sdpa_mean:.4f} ms")
    print(f"    Min:    {sdpa_min:.4f} ms")
    print(f"    Max:    {sdpa_max:.4f} ms")
    print(f"\n{'='*70}")
    print(f"Speedup Analysis:")
    print(f"{'='*70}")
    print(f"  Theoretical speedup: {theoretical_speedup:.2f}x")
    print(f"  Actual speedup:      {actual_speedup:.2f}x")
    print(f"  Efficiency:          {efficiency:.1f}%")

    if actual_speedup < 1.0:
        print(f"\n  WARNING: Triton kernel is {1/actual_speedup:.2f}x SLOWER than SDPA!")
        print(f"  Possible reasons:")
        print(f"    - Kernel launch overhead dominates at this sequence length")
        print(f"    - Memory access patterns not optimal")
        print(f"    - SDPA uses Flash Attention which is highly optimized")
        print(f"    - Sparsity ratio ({sparsity*100:.1f}%) may not be enough to overcome overhead")
    elif actual_speedup < theoretical_speedup * 0.5:
        print(f"\n  NOTE: Achieving {efficiency:.0f}% of theoretical speedup")
        print(f"  Room for optimization in kernel implementation")
    else:
        print(f"\n  GOOD: Achieving {efficiency:.0f}% of theoretical speedup!")

    print(f"{'='*70}\n")

    return {
        "triton_median_ms": triton_median,
        "sdpa_median_ms": sdpa_median,
        "actual_speedup": actual_speedup,
        "theoretical_speedup": theoretical_speedup,
        "efficiency": efficiency,
        "sparsity": sparsity,
        "local_group_size": local_group_size,
        "block_utilization": block_utilization,
        "kv_iters_local": kv_iters_local,
        "kv_iters_shared": kv_iters_shared,
    }


def run_all_benchmarks():
    """Run benchmarks for different configurations."""

    print("\n" + "="*70)
    print("BENCHMARK: Triton Sparse Attention vs PyTorch SDPA")
    print("="*70)

    # Get GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU: {gpu_name}")

    results = []

    # Configuration 1: 1024x1024 image, 4 tiles
    print("\n" + "#"*70)
    print("# Test 1: 1024x1024 image (4096 tokens), 4 tiles")
    print("#"*70)
    r1 = benchmark_attention(
        image_size=4096,
        text_length=512,
        num_tiles=4,
        center_region_size=256,  # 16x16
        offset=0,
    )
    results.append(("1024x1024, 4 tiles, offset=0", r1))

    # Configuration 2: 1024x1024 image, 4 tiles, with offset
    print("\n" + "#"*70)
    print("# Test 2: 1024x1024 image (4096 tokens), 4 tiles, offset=256")
    print("#"*70)
    r2 = benchmark_attention(
        image_size=4096,
        text_length=512,
        num_tiles=4,
        center_region_size=256,
        offset=256,
    )
    results.append(("1024x1024, 4 tiles, offset=256", r2))

    # Configuration 3: 1024x1024 image, 16 tiles (more sparsity)
    print("\n" + "#"*70)
    print("# Test 3: 1024x1024 image (4096 tokens), 16 tiles (more sparsity)")
    print("#"*70)
    r3 = benchmark_attention(
        image_size=4096,
        text_length=512,
        num_tiles=16,
        center_region_size=256,
        offset=0,
    )
    results.append(("1024x1024, 16 tiles", r3))

    # Configuration 4: 2048x2048 image (if memory allows)
    print("\n" + "#"*70)
    print("# Test 4: 2048x2048 image (16384 tokens), 4 tiles")
    print("#"*70)
    try:
        r4 = benchmark_attention(
            image_size=16384,
            text_length=512,
            num_tiles=4,
            center_region_size=1024,  # 32x32
            offset=0,
        )
        results.append(("2048x2048, 4 tiles", r4))
    except RuntimeError as e:
        print(f"Skipped due to: {e}")

    # Configuration 5: 2048x2048 image, 16 tiles
    print("\n" + "#"*70)
    print("# Test 5: 2048x2048 image (16384 tokens), 16 tiles (more sparsity)")
    print("#"*70)
    try:
        r5 = benchmark_attention(
            image_size=16384,
            text_length=512,
            num_tiles=16,
            center_region_size=1024,
            offset=0,
        )
        results.append(("2048x2048, 16 tiles", r5))
    except RuntimeError as e:
        print(f"Skipped due to: {e}")

    # Summary table
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    print(f"{'Configuration':<30} {'GroupSize':>10} {'BlkUtil':>8} {'Sparsity':>10} {'Theory':>8} {'Actual':>8} {'Effic':>8}")
    print("-"*100)
    for name, r in results:
        print(f"{name:<30} {r['local_group_size']:>10} {r['block_utilization']*100:>7.1f}% {r['sparsity']*100:>9.1f}% {r['theoretical_speedup']:>7.2f}x {r['actual_speedup']:>7.2f}x {r['efficiency']:>7.1f}%")
    print("="*100)

    # Analysis
    print("\nANALYSIS:")
    print("-"*100)
    print("Key insight: When group_size is small relative to BLOCK_M (128), efficiency drops significantly.")
    print("")
    for name, r in results:
        if r['local_group_size'] < 256:
            print(f"  {name}: group_size={r['local_group_size']} is small -> only {r['block_utilization']*100:.0f}% block utilization")
    print("")
    print("The shared region (N_CTX_shared) dominates compute for small images:")
    for name, r in results:
        shared_ratio = r['kv_iters_shared'] / (r['kv_iters_shared'] + r['kv_iters_local'])
        print(f"  {name}: {shared_ratio*100:.0f}% of KV iterations are for shared tokens")
    print("="*100)


if __name__ == "__main__":
    run_all_benchmarks()
