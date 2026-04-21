"""
Calculate FLOPs for sparse attention vs full attention.
"""

def calc_attention_flops(num_queries, num_keys, head_dim, num_heads):
    """
    Attention FLOPs per batch:
    - QK^T: 2 * Q * K * D (matmul)
    - softmax: ~5 * Q * K (exp, sum, div - negligible compared to matmul)
    - P @ V: 2 * Q * K * D (matmul)
    Total ≈ 4 * Q * K * D * H
    """
    return 4 * num_queries * num_keys * head_dim * num_heads


def analyze_flops(
    n_shared,       # N_CTX_shared
    n_seq,          # N_CTX (image tokens)
    num_groups,     # GROUPS
    head_dim=128,
    num_heads=24,
):
    total_len = n_shared + n_seq
    local_group_size = n_seq // num_groups

    # Full attention FLOPs
    full_flops = calc_attention_flops(total_len, total_len, head_dim, num_heads)

    # Sparse attention FLOPs
    # 1. Shared region queries (n_shared) attend to ALL tokens (total_len)
    shared_query_flops = calc_attention_flops(n_shared, total_len, head_dim, num_heads)

    # 2. Image region queries (n_seq) attend to shared + local group
    #    Each query attends to (n_shared + local_group_size) keys
    image_query_flops = calc_attention_flops(n_seq, n_shared + local_group_size, head_dim, num_heads)

    sparse_flops = shared_query_flops + image_query_flops

    # Breakdown
    print(f"\nConfiguration: n_shared={n_shared}, n_seq={n_seq}, groups={num_groups}")
    print(f"  local_group_size = {local_group_size}")
    print(f"  total_len = {total_len}")
    print()
    print(f"Full Attention FLOPs:   {full_flops:>15,}")
    print(f"Sparse Attention FLOPs: {sparse_flops:>15,}")
    print(f"  - Shared queries:     {shared_query_flops:>15,} ({shared_query_flops/sparse_flops*100:.1f}%)")
    print(f"  - Image queries:      {image_query_flops:>15,} ({image_query_flops/sparse_flops*100:.1f}%)")
    print()
    print(f"FLOPs Reduction: {(1 - sparse_flops/full_flops)*100:.1f}%")
    print(f"Theoretical Speedup: {full_flops/sparse_flops:.2f}x")

    return {
        "full_flops": full_flops,
        "sparse_flops": sparse_flops,
        "shared_query_flops": shared_query_flops,
        "image_query_flops": image_query_flops,
        "speedup": full_flops / sparse_flops,
    }


if __name__ == "__main__":
    print("="*60)
    print("FLOPs Analysis: 4 tiles vs 16 tiles")
    print("="*60)

    # 1024x1024 image
    print("\n### 1024x1024 Image (n_seq=3840, n_shared=768) ###")
    r4 = analyze_flops(n_shared=768, n_seq=3840, num_groups=4)
    r16 = analyze_flops(n_shared=768, n_seq=3840, num_groups=16)

    print("\n" + "-"*60)
    print("Comparison (1024x1024):")
    print(f"  4 tiles:  {r4['sparse_flops']:,} FLOPs, {r4['speedup']:.2f}x speedup")
    print(f"  16 tiles: {r16['sparse_flops']:,} FLOPs, {r16['speedup']:.2f}x speedup")
    print(f"  16 tiles uses {r16['sparse_flops']/r4['sparse_flops']*100:.1f}% of 4 tiles FLOPs")

    # 2048x2048 image
    print("\n\n### 2048x2048 Image (n_seq=15360, n_shared=1536) ###")
    r4_2k = analyze_flops(n_shared=1536, n_seq=15360, num_groups=4)
    r16_2k = analyze_flops(n_shared=1536, n_seq=15360, num_groups=16)

    print("\n" + "-"*60)
    print("Comparison (2048x2048):")
    print(f"  4 tiles:  {r4_2k['sparse_flops']:,} FLOPs, {r4_2k['speedup']:.2f}x speedup")
    print(f"  16 tiles: {r16_2k['sparse_flops']:,} FLOPs, {r16_2k['speedup']:.2f}x speedup")
    print(f"  16 tiles uses {r16_2k['sparse_flops']/r4_2k['sparse_flops']*100:.1f}% of 4 tiles FLOPs")
