import os
import torch
import torch.nn.functional as F
from reloc_triton_kernel import attention
import time

# ========== Environment Setup ==========
os.environ["TRITON_DISABLE_CACHE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ========== Configuration ==========

s1 = torch.cuda.Stream()
s2 = torch.cuda.Stream()

# Model dimensions
Z, H = 1, 24                    # Batch size and number of heads
N_CTX_shared = 256             # Number of shared context tokens
N_CTX = 256                   # Number of local context tokens
D_HEAD = 64                   # Dimension of each attention head
GROUPS = 4                     # Number of groups for local attention
assert N_CTX % GROUPS == 0
group_size = N_CTX // GROUPS

# Device and dtype setup
dtype = torch.float16
device = torch.device("cuda:0")  # This will use GPU 2 due to CUDA_VISIBLE_DEVICES=2
sm_scale = 1.0 / (D_HEAD ** 0.5)

# ========== Input Generation ==========
torch.manual_seed(0)
q = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)
k = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)
v = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)

q_prime = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)
k_prime = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)
v_prime = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)

# Verify tensors are on GPU
assert q.is_cuda and k.is_cuda and v.is_cuda
assert q_prime.is_cuda and k_prime.is_cuda and v_prime.is_cuda

# ========== Create Attention Mask ==========
idx = torch.arange(N_CTX, device=device)
group_idx = idx // group_size
sub_mask = (group_idx.unsqueeze(0) != group_idx.unsqueeze(1))

# Create full attention mask
mask = torch.ones(N_CTX_shared + N_CTX, N_CTX_shared + N_CTX, device=device, dtype=torch.bool)
mask[N_CTX_shared:, N_CTX_shared:] = sub_mask
mask[N_CTX_shared:, :N_CTX_shared] = False
mask = mask.logical_not()

mask_1 = mask.clone()
mask_2 = mask.clone()


# warm up
for i in range(1000):
    out_ref = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=mask_1[None, None, :, :],
        dropout_p=0.0
    )[:,:,N_CTX_shared:,:]
torch.cuda.synchronize()
# ========== Compute Attention ==========
# Create events for timing
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Start timing
start_event.record()

# Launch operations in parallel
for i in range(2000):
    # Launch first operation in stream 1
    # s1.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s1):
        pytorch_output = F.scaled_dot_product_attention(
            q_prime, k_prime, v_prime,
            attn_mask=mask_2[None, None, :, :],
            dropout_p=0.0
        )[:,:,N_CTX_shared:,:]

    # Launch second operation in stream 2
    # s2.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s2):
        triton_output = attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False)[:,:,N_CTX_shared:,:]
# Record end time
end_event.record()

# Wait for all operations to complete
torch.cuda.synchronize()
parallel_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
print(f"Time taken with parallel streams: {parallel_time} seconds")

# Compare with sequential execution
# start_event.record()
# for i in range(2000):
#     out_triton = attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False)[:,:,N_CTX_shared:,:]
#     out_ref = F.scaled_dot_product_attention(
#         q_prime, k_prime, v_prime,
#         attn_mask=mask[None, None, :, :],
#         dropout_p=0.0
#     )[:,:,N_CTX_shared:,:]
# end_event.record()

# # Wait for events to complete
# end_event.synchronize()
# sequential_time = start_event.elapsed_time(end_event) / 1000.0  # Convert to seconds
# print(f"Time taken with sequential execution: {sequential_time} seconds")

# # Calculate speedup
# speedup = (sequential_time - parallel_time) / sequential_time * 100
# print(f"Parallel execution is {speedup:.2f}% faster than sequential execution")
