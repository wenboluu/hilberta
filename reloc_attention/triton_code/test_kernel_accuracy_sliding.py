import os
import sys
import torch
import torch.nn.functional as F
from reloc_triton_kernel_parallel_sliding import attention

os.environ["TRITON_DISABLE_CACHE"] = "1"

# ========== Configuration ==========
Z, H, N_CTX_shared, N_CTX, D_HEAD = 1, 24, 512, 4096, 128  # assume 64 is the global (e.g., text) tokens
GROUPS = 4
assert N_CTX % GROUPS == 0
group_size = N_CTX // GROUPS

dtype = torch.bfloat16
device = torch.device("cuda")

torch.manual_seed(0)
q = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)
k = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)
v = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)

sm_scale = 1.0 / (D_HEAD ** 0.5)

# ========== Triton implementation ==========
out_triton = attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, OFFSET=256, USE_TMA=False)
# ========== Triton implementation ==========


def build_group_sliding_mask(n_shared: int, n_seq: int, groups: int, offset: int, device, dtype):
    m_total = n_shared + n_seq
    mask = torch.full((m_total, m_total), float('-inf'), device=device, dtype=dtype)
    group_size = n_seq // groups
    off = offset % n_seq

    # Shared query rows: allow all keys (shared + all image)
    mask[:n_shared, :] = 0.0

    # Image query rows: allow shared keys and shifted group keys
    for r in range(n_shared, m_total):
        g = (r - n_shared) // group_size
        # allow shared
        mask[r, :n_shared] = 0.0
        # shifted group window
        group_start = g * group_size
        src_start = (group_start + off) % n_seq
        right = min(group_size, n_seq - src_start)
        # first contiguous slice
        mask[r, n_shared + src_start : n_shared + src_start + right] = 0.0
        # wrapped remainder
        rem = group_size - right
        if rem > 0:
            mask[r, n_shared : n_shared + rem] = 0.0
    return mask

# ========== PyTorch reference with kernel-equivalent group sliding mask ==========
offset = 256
attn_mask = build_group_sliding_mask(N_CTX_shared, N_CTX, GROUPS, offset, device=q.device, dtype=q.dtype)
output_sdpa = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
output_sdpa = output_sdpa.to(q.dtype)

# ========== Compare outputs ==========
max_diff = (out_triton - output_sdpa).abs().max().item()
total_diff = (out_triton - output_sdpa).abs().sum().item()
print(f"[Test GROUPS={GROUPS}] Max diff between Triton and Torch reference: {max_diff:.6f}")
print(f"Total absolute difference: {total_diff:.6f}")

# Assert correctness within tolerance
assert torch.allclose(out_triton, output_sdpa, atol=1e-2), "Mismatch in outputs!"
print("✅ Test passed.")
