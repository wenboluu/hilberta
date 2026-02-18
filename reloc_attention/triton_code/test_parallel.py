import os
import torch
import torch.nn.functional as F
from reloc_triton_kernel import attention
from reloc_triton_kernel_parallel import attention as parallel_attention

# ========== Environment ==========
os.environ["TRITON_DISABLE_CACHE"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

# ========== Configuration ==========
Z, H = 1, 24
N_CTX_shared, N_CTX = 512, 4096
D_HEAD = 128
GROUPS = 4
assert N_CTX % GROUPS == 0
device = torch.device("cuda")
dtype = torch.float16
sm_scale = 1.0 / (D_HEAD ** 0.5)

# ========== Inputs & Mask ==========
torch.manual_seed(0)
q  = torch.randn(Z, H, N_CTX_shared + N_CTX, D_HEAD, device=device, dtype=dtype)
k  = torch.randn_like(q)
v  = torch.randn_like(q)
q2 = torch.randn_like(q)
k2 = torch.randn_like(q)
v2 = torch.randn_like(q)

# build inverted mask for scaled_dot_product_attention
group_size = N_CTX // GROUPS
idx = torch.arange(N_CTX, device=device)
sub_mask = (idx.unsqueeze(0) != idx.unsqueeze(1))
mask = torch.ones(N_CTX_shared+N_CTX, N_CTX_shared+N_CTX, device=device, dtype=torch.bool)
mask[N_CTX_shared:, N_CTX_shared:] = sub_mask
mask[N_CTX_shared:, :N_CTX_shared] = False
mask = mask.logical_not()  # True → allow, False→block

# ========== Warmup ==========
for _ in range(500):
    _ = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None,None], dropout_p=0.0)
    _ = attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False)
    _ = parallel_attention(q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False)
torch.cuda.synchronize()

# ========== Helper: measure one fn  ==========
def benchmark(fn, *args, iters=2000):
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0  # → seconds

# ========== 1) Sequential ==========
def run_sequential():
    # 1a) PyTorch reference
    t1 = benchmark(lambda: F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask[None,None], dropout_p=0.0))
    # 1b) Triton attention
    t2 = benchmark(lambda: attention(
        q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False))
    return t1, t2

# ========== 2) CUDA Streams ==========
def run_streams():
    s1 = torch.cuda.Stream()
    s2 = torch.cuda.Stream()
    # note: we need to synchronize each stream at the end of the loop
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(2000):
        with torch.cuda.stream(s1):
            F.scaled_dot_product_attention(
                q2, k2, v2, attn_mask=mask[None,None], dropout_p=0.0)
        with torch.cuda.stream(s2):
            attention(
                q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False)
    # 等待两条 stream 完全跑完
    torch.cuda.current_stream().wait_stream(s1)
    torch.cuda.current_stream().wait_stream(s2)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0

# ========== 3) Parallel‑attention ==========
def run_parallel_attention():
    return benchmark(lambda: parallel_attention(
        q, k, v, N_CTX_shared, N_CTX, False, sm_scale, GROUPS, False))

# ========== Execute & Report ==========
t1_ref, t1_triton = run_sequential()
t2_streamed = run_streams()
t3_parallel = run_parallel_attention()

print(f"1) Sequential  PyTorch: {t1_ref:.3f}s,  Triton: {t1_triton:.3f}s")
print(f"2) Streamed   PyTorch+Triton overlap: {t2_streamed:.3f}s")
print(f"3) Parallel   parallel_attention: {t3_parallel:.3f}s")