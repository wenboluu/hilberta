import os
import torch
import torch.nn.functional as F
from reloc_triton_kernel import attention

# Disable Triton kernel cache to ensure recompile on each run
os.environ["TRITON_DISABLE_CACHE"] = "1"

# ========== Configuration ==========
Z, H, N_CTX_shared, N_CTX, D_HEAD = 1, 24, 16, 1024, 128  # assume 64 is the global (e.g., text) tokens
GROUPS = 4
assert N_CTX % GROUPS == 0
group_size = N_CTX // GROUPS

dtype = torch.float16
device = torch.device("cuda")

# ========== Construct random input ==========
torch.manual_seed(0)
q = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)
k = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)
v = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)
# v[...] = 0.05

# use arange for q k v for debugging
# q = torch.arange(0, (N_CTX_shared + N_CTX) * D_HEAD, device=device, dtype=dtype).reshape(Z, H, N_CTX_shared + N_CTX, D_HEAD)
# k = torch.arange(0, (N_CTX_shared + N_CTX) * D_HEAD, device=device, dtype=dtype).reshape(Z, H, N_CTX_shared + N_CTX, D_HEAD)
# v = torch.arange(0, (N_CTX_shared + N_CTX) * D_HEAD, device=device, dtype=dtype).reshape(Z, H, N_CTX_shared + N_CTX, D_HEAD)

sm_scale = 1.0 / (D_HEAD ** 0.5)

# ========== Triton implementation ==========
out_triton = attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False)[:,:,N_CTX_shared:,:]

# ========== PyTorch reference with group-based masking ==========
# 1) Compute group index for each position
idx = torch.arange(N_CTX, device=device)
group_idx = idx // group_size  # Shape: [N_CTX]

# 2) Create a mask that disables attention across groups
#    mask[i, j] = True if i and j are from different groups
sub_mask = (group_idx.unsqueeze(0) != group_idx.unsqueeze(1))  # Shape: [N_CTX, N_CTX]
mask = torch.ones(N_CTX_shared + N_CTX, N_CTX_shared + N_CTX, device=device, dtype=torch.bool)
mask[N_CTX_shared:, N_CTX_shared:] = sub_mask
mask[N_CTX_shared:, :N_CTX_shared] = False

# 3) Compute full attention scores
attn_scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale  # [Z, H, N_CTX, N_CTX]

# 4) Mask out scores across different groups
attn_scores = attn_scores.masked_fill(mask[None, None, :, :], float('-inf'))

print(out_triton)
print("triton shape", out_triton.shape)
print(attn_scores)

# 5) Softmax and weighted sum
attn_probs = F.softmax(attn_scores, dim=-1)  # [Z, H, N_CTX, N_CTX]
out_ref = torch.matmul(attn_probs, v)[:,:,N_CTX_shared:,:]        # [Z, H, N_CTX, D_HEAD]
print(out_ref)
print("ref shape", out_ref.shape)

# import pdb; pdb.set_trace()

# ========== Compare outputs ==========
max_diff = (out_triton - out_ref).abs().max().item()
total_diff = (out_triton - out_ref).abs().sum().item()

print(f"[Test GROUPS={GROUPS}] Max diff between Triton and Torch reference: {max_diff:.6f}")
print(f"Total absolute difference: {total_diff:.6f}")

# Assert correctness within tolerance
assert torch.allclose(out_triton, out_ref, atol=1e-2), "Mismatch in outputs!"
print("✅ Test passed.")
