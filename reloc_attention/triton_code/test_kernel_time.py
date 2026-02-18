import os
import torch
import torch.nn.functional as F
from reloc_triton_kernel_parallel import attention
import time
import numpy as np

# Disable Triton kernel cache to ensure recompile on each run
os.environ["TRITON_DISABLE_CACHE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# ========== Configuration ==========
Z, H, N_CTX_shared, N_CTX, D_HEAD = 1, 24, 512, 4096, 128  # assume 64 is the global (e.g., text) tokens
GROUPS = 4
assert N_CTX % GROUPS == 0
group_size = N_CTX // GROUPS

dtype = torch.bfloat16
device = torch.device("cuda:0")

torch.manual_seed(0)
q = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)
k = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)
v = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)

assert q.is_cuda and k.is_cuda and v.is_cuda

sm_scale = 1.0 / (D_HEAD ** 0.5)

#add warm up before timing
for i in range(2000):
    attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False)

# ========== Triton implementation ==========
triton_time_list = []
for i in range(3000):
    start_time = time.time()
    out_triton = attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False)[:,:,N_CTX_shared:,:]
    torch.cuda.synchronize()
    end_time = time.time()
    triton_time_list.append(end_time - start_time)
    # check if the output is nan
    if torch.isnan(out_triton).any():
        print("Output is nan")
        break



# ========== PyTorch reference with group-based masking ==========
idx = torch.arange(N_CTX, device=device)
group_idx = idx // group_size  # Shape: [N_CTX]
sub_mask = (group_idx.unsqueeze(0) != group_idx.unsqueeze(1))  # Shape: [N_CTX, N_CTX]
mask = torch.ones(N_CTX_shared + N_CTX, N_CTX_shared + N_CTX, device=device, dtype=torch.bool)
mask[N_CTX_shared:, N_CTX_shared:] = sub_mask
mask[N_CTX_shared:, :N_CTX_shared] = False

mask = mask.logical_not()

torch_time_list = []
for j in range(3000):
    start_time = time.time()
    out_ref = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask[None, None, :, :],
        dropout_p=0.0
    )[:,:,N_CTX_shared:,:]
    torch.cuda.synchronize()
    end_time = time.time()
    torch_time_list.append(end_time - start_time)

max_diff = (out_triton - out_ref).abs().max().item()
total_diff = (out_triton - out_ref).abs().sum().item()

median_triton_time = np.median(triton_time_list)
median_torch_time = np.median(torch_time_list)
speedup = (median_torch_time - median_triton_time) / median_torch_time * 100

print(f"Median Triton time: {median_triton_time*1000:.3f} ms")
print(f"Median PyTorch time: {median_torch_time*1000:.3f} ms") 
print(f"Triton is {speedup:.2f}% faster than PyTorch")

# Assert correctness within tolerance
assert torch.allclose(out_triton, out_ref, atol=1e-2), "Mismatch in outputs!"
print("✅ Test passed.")
