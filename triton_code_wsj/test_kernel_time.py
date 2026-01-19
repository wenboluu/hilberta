import os
import torch
import torch.nn.functional as F
from reloc_triton_kernel import attention
import time

# Disable Triton kernel cache to ensure recompile on each run
os.environ["TRITON_DISABLE_CACHE"] = "1"

# ========== Configuration ==========
Z, H, N_CTX_shared, N_CTX, D_HEAD = 1, 24, 256, 4096, 128  # assume 64 is the global (e.g., text) tokens
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

# ========== Performance Testing ==========
NUM_ITERATIONS = 100
torch.cuda.synchronize()

# Warm up
for _ in range(10):
    _ = attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False)

# Measure Triton kernel total time
torch.cuda.synchronize()
start_time = time.perf_counter()
for _ in range(NUM_ITERATIONS):
    out_triton = attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False)
    torch.cuda.synchronize()
triton_total_time = time.perf_counter() - start_time

# 1) Compute group index for each position
idx = torch.arange(N_CTX, device=device)
group_idx = idx // group_size  # Shape: [N_CTX]

# 2) Create a mask that disables attention across groups
sub_mask = (group_idx.unsqueeze(0) != group_idx.unsqueeze(1))  # Shape: [N_CTX, N_CTX]
mask = torch.ones(N_CTX_shared + N_CTX, N_CTX_shared + N_CTX, device=device, dtype=torch.bool)
mask[N_CTX_shared:, N_CTX_shared:] = sub_mask
mask[N_CTX_shared:, :N_CTX_shared] = False

# Measure PyTorch reference total time
torch.cuda.synchronize()
start_time = time.perf_counter()
for _ in range(NUM_ITERATIONS):

    # 3) Use scaled_dot_product_attention with the mask
    out_ref = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask[None, None, :, :],
        scale=sm_scale,
        dropout_p=0.0
    )[:,:,N_CTX_shared:,:]
    torch.cuda.synchronize()
pytorch_total_time = time.perf_counter() - start_time

# ========== Compare outputs ==========
out_triton = attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False)[:,:,N_CTX_shared:,:]
max_diff = (out_triton - out_ref).abs().max().item()
total_diff = (out_triton - out_ref).abs().sum().item()

print(f"\nPerformance Results (total time for {NUM_ITERATIONS} iterations):")
print(f"Triton kernel total time: {triton_total_time*1000:.2f} ms")
print(f"PyTorch reference total time: {pytorch_total_time*1000:.2f} ms")
print(f"Total time difference: {(pytorch_total_time - triton_total_time)*1000:.2f} ms")
print(f"Speedup: {pytorch_total_time/triton_total_time:.2f}x")

print(f"\n[Test GROUPS={GROUPS}] Max diff between Triton and Torch reference: {max_diff:.6f}")
print(f"Total absolute difference: {total_diff:.6f}")

# Assert correctness within tolerance
assert torch.allclose(out_triton, out_ref, atol=1e-2), "Mismatch in outputs!"
print("✅ Test passed.")
