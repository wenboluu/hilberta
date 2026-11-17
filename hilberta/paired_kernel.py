"""Experimental exact Q128 sparse attention. Pair topology is precomputed on CPU.

This module is isolated from the stable kernel until correctness and measured
end-to-end improvement justify adoption. Different query halves share K/V loads
but preserve their own original masks and softmax denominators.
"""

import math
from numbers import Integral, Real
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl
from .sparse_kernel import _tensor_byte_span, _non_overlapping_strides


@triton.jit
def _paired_fwd(
    Q,
    K,
    V,
    O,
    INDICES,
    FLAGS,
    COUNTS,
    QB: tl.constexpr,
    QH: tl.constexpr,
    QN: tl.constexpr,
    KB: tl.constexpr,
    KH: tl.constexpr,
    KN: tl.constexpr,
    VB: tl.constexpr,
    VH: tl.constexpr,
    VN: tl.constexpr,
    OB: tl.constexpr,
    OH: tl.constexpr,
    ON: tl.constexpr,
    HEADS: tl.constexpr,
    N: tl.constexpr,
    D: tl.constexpr,
    G: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    ALL_QUERIES: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    pid = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // HEADS, bh % HEADS
    if ALL_QUERIES:
        qfirst = pid * 128
        isglobal = qfirst < G
        row = (qfirst - G) // 128
        count = tl.load(COUNTS + row, mask=~isglobal, other=0)
        count = tl.where(isglobal, N // BLOCK_N, count)
    else:
        qfirst = G + pid * 128
        row = pid
        count = tl.load(COUNTS + row)
    qi = qfirst + tl.arange(0, 128)
    dd = tl.arange(0, D)
    kk = tl.arange(0, BLOCK_N)
    shifts = (tl.arange(0, 128) // 64)[:, None] + (2 * (kk // 64))[None, :]
    q = tl.load(Q + batch * QB + head * QH + qi[:, None] * QN + dd[None, :])
    maximum = tl.full((128,), -float("inf"), tl.float32)
    normalizer = tl.zeros((128,), tl.float32)
    accumulator = tl.zeros((128, D), tl.float32)
    for entry in tl.range(0, count, num_stages=PIPELINE_STAGES):
        if ALL_QUERIES:
            block = tl.load(INDICES + row * MAX_BLOCKS + entry, mask=~isglobal, other=0)
            flags = tl.load(FLAGS + row * MAX_BLOCKS + entry, mask=~isglobal, other=0)
            block = tl.where(isglobal, entry, block)
            flags = tl.where(isglobal, (1 << (2 * (BLOCK_N // 64))) - 1, flags)
        else:
            block = tl.load(INDICES + row * MAX_BLOCKS + entry)
            flags = tl.load(FLAGS + row * MAX_BLOCKS + entry)
        ki = block * BLOCK_N + kk
        k = tl.load(K + batch * KB + head * KH + ki[None, :] * KN + dd[:, None])
        v = tl.load(V + batch * VB + head * VH + ki[:, None] * VN + dd[None, :])
        scores = tl.dot(q, k) * SCALE_LOG2
        allowed = ((flags >> shifts) & 1) != 0
        scores = tl.where(allowed, scores, -float("inf"))
        newmax = tl.maximum(maximum, tl.max(scores, axis=1))
        safe_max = tl.where(newmax == -float("inf"), 0.0, newmax)
        alpha = tl.exp2(maximum - safe_max)
        prob = tl.exp2(scores - safe_max[:, None])
        normalizer = normalizer * alpha + tl.sum(prob, axis=1)
        accumulator = accumulator * alpha[:, None]
        accumulator = tl.dot(prob.to(v.dtype), v, accumulator)
        maximum = newmax
    out = accumulator / normalizer[:, None]
    tl.store(O + batch * OB + head * OH + qi[:, None] * ON + dd[None, :], out)


def paired_attention(
    q, k, v, metadata, *, global_strategy="flash", num_warps=4, num_stages=2, scale=None, out=None
):
    """Exact inference; pair metadata must be validated on CPU before transfer.

    Input checks inspect host metadata only. They do not transfer CUDA values,
    synchronize, construct masks, or pack Q/K/V. Out-of-range flags/indices in
    manually fabricated metadata are outside this API's setup-time contract.
    """
    if q.ndim != 4 or k.shape != q.shape or v.shape != q.shape or any(x <= 0 for x in q.shape):
        raise ValueError("Equal nonempty B,H,N,D Q/K/V required")
    if q.shape[-1] not in (64, 128) or any(x.stride(-1) != 1 for x in (q, k, v)):
        raise ValueError("D64 or128 must be contiguous")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q/K/V must share BF16/FP16 dtype")
    if not q.is_cuda or k.device != q.device or v.device != q.device:
        raise ValueError("Q/K/V must be on the same CUDA device")
    if any(any(s <= 0 for s in x.stride()[:3]) for x in (q, k, v)):
        raise ValueError("Positive Q/K/V outer strides required")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise ValueError("Inference only; disable autograd")
    for name in ("tokens", "global_tokens", "key_block_size"):
        value = getattr(metadata, name)
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError("Integer metadata shape fields required")
    N, G = metadata.tokens, metadata.global_tokens
    if (
        q.shape[-2] != N
        or G % 128
        or N % 128
        or not 0 <= G < N
        or metadata.key_block_size not in (64, 128)
    ):
        raise ValueError("Metadata/token shape mismatch or invalid128-token alignment")
    rows = (N - G) // 128
    if (
        metadata.kv_indices.ndim != 2
        or metadata.kv_indices.shape[0] != rows
        or metadata.kv_indices.shape[1] < 1
    ):
        raise ValueError("Paired index shape must match ordinary query pairs")
    if metadata.kv_flags.shape != metadata.kv_indices.shape or metadata.kv_counts.shape != (rows,):
        raise ValueError("Paired flag/count shape mismatch")
    for x in (metadata.kv_indices, metadata.kv_flags, metadata.kv_counts):
        if x.device != q.device or x.dtype != torch.int32 or not x.is_contiguous():
            raise ValueError("Cached CUDA contiguous int32 metadata required")
    if global_strategy not in ("flash", "triton"):
        raise ValueError("Invalid global strategy")
    if any(isinstance(x, bool) or not isinstance(x, Integral) for x in (num_warps, num_stages)):
        raise TypeError("Integer launch tuning required")
    if num_warps not in (4, 8) or num_stages not in (1, 2, 3, 4):
        raise ValueError("Invalid launch tuning")
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, Real)):
        raise TypeError("scale must be a real scalar, never a tensor")
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Positive finite scale required")
    if out is not None:
        if (
            out.shape != q.shape
            or out.dtype != q.dtype
            or out.device != q.device
            or out.stride(-1) != 1
        ):
            raise ValueError("Output must match Q shape/dtype/device with contiguous D")
        if any(s <= 0 for s in out.stride()) or not _non_overlapping_strides(out):
            raise ValueError("Positive non-overlapping output strides required")
        if torch.is_grad_enabled() and out.requires_grad:
            raise ValueError("Inference output cannot require gradients")
        begin, end = _tensor_byte_span(out)
        for x in (q, k, v, metadata.kv_indices, metadata.kv_flags, metadata.kv_counts):
            first, last = _tensor_byte_span(x)
            if begin < last and first < end:
                raise ValueError("Output cannot alias inputs or metadata")
    else:
        out = torch.empty_like(q)
    if global_strategy == "flash" and G:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            go = F.scaled_dot_product_attention(
                q[:, :, :G], k, v, dropout_p=0.0, is_causal=False, scale=scale
            )
        out[:, :, :G].copy_(go)
    all_queries = global_strategy == "triton"
    count = N if all_queries else N - G
    _paired_fwd[(count // 128, q.shape[0] * q.shape[1])](
        q,
        k,
        v,
        out,
        metadata.kv_indices,
        metadata.kv_flags,
        metadata.kv_counts,
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        *out.stride()[:3],
        q.shape[1],
        N,
        q.shape[3],
        G,
        metadata.kv_indices.shape[1],
        metadata.key_block_size,
        scale * 1.4426950408889634,
        all_queries,
        int(num_stages),
        num_warps=int(num_warps),
        num_stages=int(num_stages),
    )
    return out
