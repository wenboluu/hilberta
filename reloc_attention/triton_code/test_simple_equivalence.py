"""
Simplified test: Just check if reorder with proper recover gives back the same result.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reloc_triton_kernel_parallel_sliding import attention as triton_attention
from reorder_utils_sliding_shared import _get_index_on_device


def test_reorder_recover_cycle():
    """
    Test if reorder + kernel + recover gives correct results.

    The key question: Should reorder method with offset in reordering use OFFSET=0 in kernel?
    """
    print("=" * 80)
    print("Test: Reorder + Attention + Recover Cycle")
    print("=" * 80)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cpu':
        print("CUDA not available")
        return

    B, H = 1, 2
    N_CTX_shared = 512
    N_CTX = 4096
    D = 128
    GROUPS = 4
    OFFSET = 256

    torch.manual_seed(42)

    # Original data in spatial order
    q_spatial = torch.randn(B, H, N_CTX_shared + N_CTX, D, device=device, dtype=torch.bfloat16)
    k_spatial = torch.randn(B, H, N_CTX_shared + N_CTX, D, device=device, dtype=torch.bfloat16)
    v_spatial = torch.randn(B, H, N_CTX_shared + N_CTX, D, device=device, dtype=torch.bfloat16)

    sm_scale = 1.0 / (D ** 0.5)

    # Get reordering indices (with offset applied during reordering)
    hilbert_index_base = _get_index_on_device(N_CTX, torch.device('cpu'))
    hilbert_index_with_offset = torch.remainder(hilbert_index_base - OFFSET, N_CTX)
    hilbert_sorted_positions = torch.argsort(hilbert_index_with_offset).to(device)

    print(f"\nReordering with OFFSET={OFFSET} applied in the reordering step")
    print(f"hilbert_sorted_positions[i] = spatial position of token at Hilbert position i")

    # Reorder
    q_hilbert = q_spatial.clone()
    k_hilbert = k_spatial.clone()
    v_hilbert = v_spatial.clone()

    img_indices = hilbert_sorted_positions + N_CTX_shared
    q_hilbert[:, :, N_CTX_shared:, :] = q_spatial[:, :, img_indices, :]
    k_hilbert[:, :, N_CTX_shared:, :] = k_spatial[:, :, img_indices, :]
    v_hilbert[:, :, N_CTX_shared:, :] = v_spatial[:, :, img_indices, :]

    print(f"\nTokens reordered to Hilbert order")

    # Test 1: Kernel with OFFSET=0
    print(f"\n### Test 1: Kernel with OFFSET=0 ###")
    out_hilbert_0 = triton_attention(q_hilbert, k_hilbert, v_hilbert, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, 0)

    # Recover to spatial order
    inverse_indices = torch.zeros_like(hilbert_sorted_positions)
    inverse_indices[hilbert_sorted_positions] = torch.arange(N_CTX, device=device)
    unreorder_img_indices = inverse_indices + N_CTX_shared

    out_recovered_0 = out_hilbert_0.clone()
    out_recovered_0[:, :, N_CTX_shared:, :] = out_hilbert_0[:, :, unreorder_img_indices, :]

    print(f"Output (recovered to spatial): mean = {out_recovered_0.float().mean():.6f}")

    # Test 2: Kernel with OFFSET=256
    print(f"\n### Test 2: Kernel with OFFSET={OFFSET} ###")
    out_hilbert_offset = triton_attention(q_hilbert, k_hilbert, v_hilbert, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, OFFSET)

    out_recovered_offset = out_hilbert_offset.clone()
    out_recovered_offset[:, :, N_CTX_shared:, :] = out_hilbert_offset[:, :, unreorder_img_indices, :]

    print(f"Output (recovered to spatial): mean = {out_recovered_offset.float().mean():.6f}")

    # Test 3: For comparison - reorder with offset=0, kernel with offset=256
    print(f"\n### Test 3: Reorder OFFSET=0, Kernel OFFSET={OFFSET} ###")

    hilbert_sorted_no_offset = torch.argsort(hilbert_index_base).to(device)

    q_hilbert_v3 = q_spatial.clone()
    k_hilbert_v3 = k_spatial.clone()
    v_hilbert_v3 = v_spatial.clone()

    img_indices_v3 = hilbert_sorted_no_offset + N_CTX_shared
    q_hilbert_v3[:, :, N_CTX_shared:, :] = q_spatial[:, :, img_indices_v3, :]
    k_hilbert_v3[:, :, N_CTX_shared:, :] = k_spatial[:, :, img_indices_v3, :]
    v_hilbert_v3[:, :, N_CTX_shared:, :] = v_spatial[:, :, img_indices_v3, :]

    out_hilbert_v3 = triton_attention(q_hilbert_v3, k_hilbert_v3, v_hilbert_v3, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, OFFSET)

    inverse_indices_v3 = torch.zeros_like(hilbert_sorted_no_offset)
    inverse_indices_v3[hilbert_sorted_no_offset] = torch.arange(N_CTX, device=device)
    unreorder_img_indices_v3 = inverse_indices_v3 + N_CTX_shared

    out_recovered_v3 = out_hilbert_v3.clone()
    out_recovered_v3[:, :, N_CTX_shared:, :] = out_hilbert_v3[:, :, unreorder_img_indices_v3, :]

    print(f"Output (recovered to spatial): mean = {out_recovered_v3.float().mean():.6f}")

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    diff_0_vs_offset = (out_recovered_0.float() - out_recovered_offset.float()).abs()
    diff_0_vs_v3 = (out_recovered_0.float() - out_recovered_v3.float()).abs()

    print(f"\nTest1 (reorder={OFFSET}, kernel=0) vs Test2 (reorder={OFFSET}, kernel={OFFSET}):")
    print(f"  Mean diff: {diff_0_vs_offset.mean():.6f}")

    print(f"\nTest1 (reorder={OFFSET}, kernel=0) vs Test3 (reorder=0, kernel={OFFSET}):")
    print(f"  Mean diff: {diff_0_vs_v3.mean():.6f}")

    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    print("\nThe question is: which combination matches the masking method?")
    print("\nMasking method:")
    print("  - Tokens in spatial order")
    print("  - Mask created with Hilbert indices rotated by OFFSET")
    print("  - Grouping is: Hilbert positions after rotation divided into tiles")
    print("\nReorder method should match:")
    print("  - Reorder tokens with offset applied: hilbert_index - OFFSET")
    print("  - Kernel groups by consecutive positions (which are consecutive in rotated Hilbert space)")
    print("  - Kernel OFFSET=0 (because offset already applied in reordering)")
    print("\nSo Test1 should be correct!")

    print(f"\nTo verify, let's check a specific position's attention pattern...")

if __name__ == "__main__":
    test_reorder_recover_cycle()
