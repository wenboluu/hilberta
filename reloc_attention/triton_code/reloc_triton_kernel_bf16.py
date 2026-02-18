"""
Fused Attention
===============

This is a Triton implementation of the Flash Attention v2 algorithm from Tri Dao (https://tridao.me/publications/flash2/flash2.pdf)

Credits: OpenAI kernel team

Extra Credits:

* Original flash attention paper (https://arxiv.org/abs/2205.14135)
* Rabe and Staats (https://arxiv.org/pdf/2112.05682v2.pdf)

"""

import pytest
import torch
import pdb

import triton
import triton.language as tl

try:
    from triton.tools.tensor_descriptor import TensorDescriptor
    HAS_TENSOR_DESC = True
except ModuleNotFoundError:
    HAS_TENSOR_DESC = False

# DEVICE = triton.runtime.driver.active.get_active_torch_device()

if torch.cuda.is_available():
    DEVICE = torch.device("cuda", torch.cuda.current_device())
else:
    DEVICE = torch.device("cpu")# DEVICE = "cuda"


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


def supports_tma():
    return HAS_TENSOR_DESC and is_cuda() and torch.cuda.get_device_capability()[0] >= 9


@triton.jit
def _attn_fwd_inner(acc, l_i, m_i, q,  #
                    K_block_ptr, V_block_ptr,  #
                    start_m, qk_scale,  #
                    BLOCK_M: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK_N: tl.constexpr,  #
                    STAGE: tl.constexpr, offs_m: tl.constexpr, offs_n: tl.constexpr,  #
                    N_CTX: tl.constexpr, fp8_v: tl.constexpr, group_start: tl.constexpr, group_end: tl.constexpr):

    for rel_n in range(0, group_end - group_start, BLOCK_N):
        rel_n = tl.multiple_of(rel_n, BLOCK_N)

        if rel_n + BLOCK_N > group_end - group_start:
            tl.device_print("rel_n =", rel_n)
            tl.device_print("group_end - group_start =", group_end - group_start)
            tl.device_print("BLOCK_N =", BLOCK_N)
            tl.device_print("group_start =", group_start)
            tl.device_print("group_end =", group_end)
            tl.device_print("rel_n + BLOCK_N =", rel_n + BLOCK_N)

        K_block_ptr_cur = tl.advance(K_block_ptr, (0, rel_n))
        V_block_ptr_cur = tl.advance(V_block_ptr, (rel_n, 0))
        
        k = tl.load(K_block_ptr_cur, boundary_check = (0,1))
        qk = tl.dot(q, k)
        
        m_ij = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
        qk = qk * qk_scale - m_ij[:, None]
        p = tl.math.exp2(qk)
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + l_ij
        # -- update output accumulator --
        acc = acc * alpha[:, None]

        # update acc
        v = tl.load(V_block_ptr_cur, boundary_check = (0,1))
        # pdb.set_trace()
        if fp8_v:
            p = p.to(tl.float8e5)
        else:
            p = p.to(tl.float16)
        # p = p.to(v.dtype)
        # pdb.set_trace()
        acc = tl.dot(p, v, acc)
        # update m_i and l_i

        m_i = m_ij
    return acc, l_i, m_i

# We don't run auto-tuning every time to keep the tutorial fast. Keeping
# the code below and commenting out the equivalent parameters is convenient for
# re-tuning.
configs = [
    triton.Config({'BLOCK_M': BM, 'BLOCK_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64]\
    for s in ([1] if is_hip() else [3, 4, 7])\
    for w in [4, 8]\
]


def keep(conf):
    BLOCK_M = conf.kwargs["BLOCK_M"]
    BLOCK_N = conf.kwargs["BLOCK_N"]
    if BLOCK_M * BLOCK_N < 128 * 128 and conf.num_warps == 8:
        return False
    return True


@triton.autotune(list(filter(keep, configs)), key=["N_CTX", "HEAD_DIM"])
@triton.jit
def _attn_fwd(Q, K, V, sm_scale, M, Out,  #
              stride_qz, stride_qh, stride_qm, stride_qk,  #
              stride_kz, stride_kh, stride_kn, stride_kk,  #
              stride_vz, stride_vh, stride_vk, stride_vn,  #
              stride_oz, stride_oh, stride_om, stride_on,  #
              Z, H, N_CTX_shared, N_CTX,  #
              HEAD_DIM: tl.constexpr,  #
              BLOCK_M: tl.constexpr,  #
              BLOCK_N: tl.constexpr,  #
              STAGE: tl.constexpr,  #
              GROUPS: tl.constexpr  #
              ):
    # Print configuration at runtime
    # tl.device_print("Config: BLOCK_M=", tl.full([1], BLOCK_M, dtype=tl.int32))
    # tl.device_print("BLOCK_N=", tl.full([1], BLOCK_N, dtype=tl.int32))
    
    tl.static_assert(BLOCK_N <= HEAD_DIM)

    # pdb.set_trace()
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh
    qvk_offset_after_shared = qvk_offset + N_CTX_shared * stride_qm

    # Calculate the group index 
    GROUP_SIZE = N_CTX // GROUPS
    GROUP_ID = (start_m * BLOCK_M) // GROUP_SIZE
    GROUP_START = GROUP_ID * GROUP_SIZE
    GROUP_END = GROUP_START + GROUP_SIZE
    
    # tl.device_print('N_CTX', N_CTX)
    # tl.device_print('GROUPS', GROUPS)
    # tl.device_print("BLOCK_M=", BLOCK_M) 

    assert BLOCK_M <= GROUP_SIZE, "BLOCK_M must be <= group_size"
    assert GROUP_SIZE % BLOCK_M == 0, "group_size must be divisible by BLOCK_M"
    assert GROUP_SIZE % BLOCK_N == 0, "group_size must be divisible by BLOCK_N"

    # tl.device_print("group_id =", group_id)
    # block pointers
    Q_block_ptr = tl.make_block_ptr(
        base=Q + qvk_offset_after_shared,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_qm, stride_qk),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    V_block_ptr_shared = tl.make_block_ptr(
        base=V + qvk_offset,
        shape=(N_CTX_shared, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr_shared = tl.make_block_ptr(
        base=K + qvk_offset,
        shape=(HEAD_DIM, N_CTX_shared),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    V_block_ptr = tl.make_block_ptr(
        base=V + qvk_offset_after_shared,
        shape=(GROUP_SIZE, HEAD_DIM),
        strides=(stride_vk, stride_vn),
        offsets=(0, 0),
        block_shape=(BLOCK_N, HEAD_DIM),
        order=(1, 0),
    )
    K_block_ptr = tl.make_block_ptr(
        base=K + qvk_offset_after_shared,
        shape=(HEAD_DIM, GROUP_SIZE),
        strides=(stride_kk, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_N),
        order=(0, 1),
    )
    O_block_ptr = tl.make_block_ptr(
        base=Out + qvk_offset_after_shared,
        shape=(N_CTX, HEAD_DIM),
        strides=(stride_om, stride_on),
        offsets=(start_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, HEAD_DIM),
        order=(1, 0),
    )
    
    # Add boundary checks for block pointers
    assert GROUP_START + GROUP_SIZE <= N_CTX, "Group block exceeds sequence length"
    assert GROUP_START * HEAD_DIM + GROUP_SIZE * HEAD_DIM <= N_CTX * HEAD_DIM, "Group block exceeds tensor size"

    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    # load scales
    qk_scale = sm_scale
    qk_scale *= 1.44269504  # 1/log(2)
    # load q: it will stay in SRAM throughout
    q = tl.load(Q_block_ptr, boundary_check = (0,1))
    # pdb.set_trace()

    # stage 1: off-band
    # For causal = True, STAGE = 3 and _attn_fwd_inner gets 1 as its STAGE
    # For causal = False, STAGE = 1, and _attn_fwd_inner gets 3 as its STAGE
    if STAGE & 1:
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr_shared, V_block_ptr_shared,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5, 0, N_CTX_shared,  #
                                        )
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        4 - STAGE, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5, GROUP_START, GROUP_END,  #
                                        )
    # stage 2: on-band
    if STAGE & 2:
        # barrier makes it easier for compielr to schedule the
        # two loops independently
        acc, l_i, m_i = _attn_fwd_inner(acc, l_i, m_i, q, K_block_ptr, V_block_ptr,  #
                                        start_m, qk_scale,  #
                                        BLOCK_M, HEAD_DIM, BLOCK_N,  #
                                        2, offs_m, offs_n, N_CTX, V.dtype.element_ty == tl.float8e5, GROUP_START, GROUP_END,  #
                                        )
    # epilogue
    m_i += tl.math.log2(l_i)
    acc = acc / l_i[:, None]
    m_ptrs = M + off_hz * N_CTX + offs_m
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, acc.to(Out.type.element_ty))

class _attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, n_shared, n_seq, causal, sm_scale, num_groups, USE_TMA=True):
        # shape constraints
        HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
        # when v is in float8_e5m2 it is transposed.
        HEAD_DIM_V = v.shape[-1]
        assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
        assert HEAD_DIM_K in {16, 32, 64, 128, 256}
        o = torch.empty_like(q)
        stage = 3 if causal else 1
        extra_kern_args = {}
        # Tuning for AMD target
        if is_hip():
            waves_per_eu = 3 if HEAD_DIM_K <= 64 else 2
            extra_kern_args = {"waves_per_eu": waves_per_eu, "allow_flush_denorm": True}

        M = torch.empty((q.shape[0], q.shape[1], n_seq), device=q.device, dtype=torch.float32) # Maximize value for each position 
        grid = lambda args: (triton.cdiv(n_seq, args["BLOCK_M"]), q.shape[0] * q.shape[1], 1)
        ctx.grid = grid
        _attn_fwd[grid](
            q, k, v, sm_scale, M, o,  #
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),  
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),  
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),  
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),  
            q.shape[0], q.shape[1],  
            N_CTX_shared=n_shared,
            N_CTX=n_seq,  
            HEAD_DIM=HEAD_DIM_K,  
            STAGE=stage,  
            GROUPS=num_groups,
            **extra_kern_args,
        )

        # ctx.save_for_backward(q, k, v, o, M)
        ctx.sm_scale = sm_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return o


attention = _attention.apply


