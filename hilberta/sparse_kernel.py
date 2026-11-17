"""Exact block-sparse Flux self-attention on a persistent token layout.

Q/K/V use [batch, heads, tokens, head_dim] views. Physical token order is a
fully connected global prefix followed by ordinary 64-token query blocks.
Every ordinary metadata row lists all allowed physical K/V BLOCK IDs once,
including the global prefix. Shared/base/bridge edges therefore enter ONE
online softmax. No per-query gather, dense mask, packing, or output recovery
is performed here; positional handling and initial/final packing belong to
the native model adapter.

The implementation is inference-only. BF16/FP16 tensor-core products use FP32
softmax/output accumulators. Numerical agreement and latency must be measured;
entry counts are not speed claims.
"""

from __future__ import annotations

import math
from numbers import Integral, Real

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import triton
import triton.language as tl


@triton.jit
def _block_sparse_fwd(
    Q,
    K,
    V,
    O,
    INDICES,
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
    TOKENS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    GLOBAL: tl.constexpr,
    MAX_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SCALE_LOG2: tl.constexpr,
    ALL_QUERIES: tl.constexpr,
    PIPELINE_STAGES: tl.constexpr,
):
    query_program = tl.program_id(0)
    bh = tl.program_id(1)
    batch, head = bh // HEADS, bh % HEADS
    if ALL_QUERIES:
        first_q = query_program * BLOCK_M
        is_global = first_q < GLOBAL
        ordinary_row = (first_q - GLOBAL) // BLOCK_N
        count = tl.load(COUNTS + ordinary_row, mask=~is_global, other=0)
        count = tl.where(is_global, TOKENS // BLOCK_N, count)
    else:
        first_q = GLOBAL + query_program * BLOCK_M
        ordinary_row = (first_q - GLOBAL) // BLOCK_N
        count = tl.load(COUNTS + ordinary_row)
    qids = first_q + tl.arange(0, BLOCK_M)
    dims = tl.arange(0, HEAD_DIM)
    key_offsets = tl.arange(0, BLOCK_N)
    q = tl.load(Q + batch * QB + head * QH + qids[:, None] * QN + dims[None, :])
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    normalizer = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), tl.float32)
    for entry in tl.range(0, count, num_stages=PIPELINE_STAGES):
        if ALL_QUERIES:
            block = tl.load(INDICES + ordinary_row * MAX_BLOCKS + entry, mask=~is_global, other=0)
            block = tl.where(is_global, entry, block)
        else:
            block = tl.load(INDICES + ordinary_row * MAX_BLOCKS + entry)
        kids = block * BLOCK_N + key_offsets
        k = tl.load(K + batch * KB + head * KH + dims[:, None] + kids[None, :] * KN)
        v = tl.load(V + batch * VB + head * VH + kids[:, None] * VN + dims[None, :])
        scores = tl.dot(q, k) * SCALE_LOG2
        updated_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        alpha = tl.exp2(maximum - updated_maximum)
        probability = tl.exp2(scores - updated_maximum[:, None])
        normalizer = normalizer * alpha + tl.sum(probability, axis=1)
        accumulator = accumulator * alpha[:, None]
        accumulator = tl.dot(probability.to(v.dtype), v, accumulator)
        maximum = updated_maximum
    output = accumulator / normalizer[:, None]
    tl.store(O + batch * OB + head * OH + qids[:, None] * ON + dims[None, :], output)


def validate_metadata(
    kv_indices, kv_counts, *, tokens, global_tokens=768, block_size=64, require_global_prefix=True
):
    """Run once on CPU metadata before launch; never called in the hot path.

    Returns independent logical entry counts. Duplicate K/V blocks are rejected:
    processing one twice would change the softmax denominator and topology.
    Padded entries beyond each count are ignored and may contain -1.
    """
    if block_size != 64:
        raise ValueError("This kernel currently requires 64-token physical blocks")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (tokens, global_tokens)
    ):
        raise TypeError("tokens and global_tokens must be integers")
    if not (0 <= global_tokens < tokens) or tokens % block_size or global_tokens % block_size:
        raise ValueError(
            "tokens/global prefix must be block-aligned with at least one ordinary block"
        )
    if kv_indices.dtype != torch.int32 or kv_counts.dtype != torch.int32:
        raise TypeError("Block metadata must use int32")
    if kv_indices.device.type != "cpu" or kv_counts.device.type != "cpu":
        raise ValueError("Validate CPU metadata before transferring/caching it on GPU")
    rows = (tokens - global_tokens) // block_size
    if kv_indices.ndim != 2 or kv_indices.shape[0] != rows or kv_indices.shape[1] < 1:
        raise ValueError("kv_indices must be [ordinary_query_blocks, max_KV_blocks]")
    if kv_counts.shape != (rows,):
        raise ValueError("kv_counts must have one entry per ordinary query block")
    if not kv_indices.is_contiguous() or not kv_counts.is_contiguous():
        raise ValueError("Metadata must be contiguous")
    nblocks, global_blocks = tokens // block_size, global_tokens // block_size
    counts = kv_counts.tolist()
    for row, count in enumerate(counts):
        if not 1 <= count <= kv_indices.shape[1]:
            raise ValueError(f"Invalid K/V count in ordinary query block {row}")
        ids = kv_indices[row, :count].tolist()
        if min(ids) < 0 or max(ids) >= nblocks or len(set(ids)) != count:
            raise ValueError(f"Out-of-range or duplicate K/V blocks in row {row}")
        if require_global_prefix and not set(range(global_blocks)).issubset(ids):
            raise ValueError(f"Row {row} misses global-prefix keys")
    ordinary_entries = sum(counts) * block_size * block_size
    return {
        "tokens": tokens,
        "global_tokens": global_tokens,
        "block_size": block_size,
        "ordinary_query_blocks": rows,
        "min_kv_blocks": min(counts),
        "max_kv_blocks": max(counts),
        "global_directed_entries": global_tokens * tokens,
        "ordinary_directed_entries": ordinary_entries,
        "total_directed_entries": global_tokens * tokens + ordinary_entries,
    }


def explicit_mask(kv_indices, kv_counts, *, tokens, global_tokens=768, block_size=64):
    """CPU correctness oracle only; the production kernel never calls this."""
    validate_metadata(
        kv_indices, kv_counts, tokens=tokens, global_tokens=global_tokens, block_size=block_size
    )
    mask = torch.zeros((tokens, tokens), dtype=torch.bool)
    mask[:global_tokens, :] = True
    for row, count in enumerate(kv_counts.tolist()):
        queries = slice(global_tokens + row * block_size, global_tokens + (row + 1) * block_size)
        for block in kv_indices[row, :count].tolist():
            mask[queries, block * block_size : (block + 1) * block_size] = True
    return mask


def _tensor_byte_span(tensor):
    # data_ptr, sizes, strides and element_size are host-side tensor metadata.
    # A bounding interval conservatively rejects aliases even for pitched views.
    begin = tensor.data_ptr()
    extent = 1 + sum((size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride()))
    return begin, begin + extent * tensor.element_size()


def _non_overlapping_strides(tensor):
    extent = 1
    for stride, size in sorted(
        (stride, size) for stride, size in zip(tensor.stride(), tensor.shape) if size > 1
    ):
        if stride < extent:
            return False
        extent += (size - 1) * stride
    return True


def _check_inputs(q, k, v, kv_indices, kv_counts, global_tokens, block_size, out):
    if (
        q.ndim != 4
        or k.shape != q.shape
        or v.shape != q.shape
        or any(size <= 0 for size in q.shape)
    ):
        raise ValueError("Q/K/V must have equal, nonempty [B,H,N,D] shapes")
    if q.dtype not in (torch.float16, torch.bfloat16) or k.dtype != q.dtype or v.dtype != q.dtype:
        raise TypeError("Q/K/V must share FP16 or BF16 dtype")
    if not q.is_cuda or k.device != q.device or v.device != q.device:
        raise ValueError("Q/K/V must be on one CUDA device")
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q, k, v)):
        raise ValueError("sparse_attention is inference-only; disable autograd before calling")
    if q.shape[-1] not in (64, 128) or any(x.stride(-1) != 1 for x in (q, k, v)):
        raise ValueError("D must be64 or128 and contiguous; outer strides may differ")
    if any(any(stride <= 0 for stride in x.stride()[:3]) for x in (q, k, v)):
        raise ValueError("Q/K/V outer strides must be positive")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (global_tokens, block_size)
    ):
        raise TypeError("global_tokens and block_size must be integers")
    n = q.shape[-2]
    if (
        block_size != 64
        or n % block_size
        or global_tokens % block_size
        or not 0 <= global_tokens < n
    ):
        raise ValueError("N and global_tokens must align to64, with ordinary queries remaining")
    rows = (n - global_tokens) // block_size
    if (
        kv_indices.ndim != 2
        or kv_indices.shape[0] != rows
        or kv_indices.shape[1] < 1
        or kv_counts.shape != (rows,)
    ):
        raise ValueError("Metadata shape does not match the ordinary query partition")
    if any(
        x.device != q.device or x.dtype != torch.int32 or not x.is_contiguous()
        for x in (kv_indices, kv_counts)
    ):
        raise ValueError("Metadata must be cached contiguous CUDA int32 tensors on the Q device")
    if out is not None:
        if (
            out.shape != q.shape
            or out.dtype != q.dtype
            or out.device != q.device
            or out.stride(-1) != 1
        ):
            raise ValueError(
                "Provided output must match Q shape/dtype/device with contiguous last dimension"
            )
        if any(stride <= 0 for stride in out.stride()) or not _non_overlapping_strides(out):
            raise ValueError("Provided output must have positive, non-overlapping strides")
        if torch.is_grad_enabled() and out.requires_grad:
            raise ValueError("An inference output cannot require gradients")
        out_lo, out_hi = _tensor_byte_span(out)
        for tensor in (q, k, v, kv_indices, kv_counts):
            in_lo, in_hi = _tensor_byte_span(tensor)
            if out_lo < in_hi and in_lo < out_hi:
                raise ValueError("Provided output must not alias Q/K/V or metadata storage")


def sparse_attention(
    q,
    k,
    v,
    kv_indices,
    kv_counts,
    global_tokens=768,
    block_size=64,
    *,
    global_strategy="flash",
    query_block_size=64,
    num_warps=4,
    num_stages=2,
    scale=None,
    out=None,
):
    """Return the exact mask's attention in the original Q shape/dtype.

    Grad-enabled inputs and aliased/overlapping provided outputs are rejected.
    Metadata must be validated once with ``validate_metadata`` and transferred
    once by the model adapter. This launch performs no metadata .item(), CPU
    transfer, per-token gather, dense mask construction, or topology selection.

    ``global_strategy='flash'`` uses native Flash SDPA for global-prefix queries
    and this fused sparse kernel for ordinary queries. ``'triton'`` covers both
    partitions in one custom launch, where global queries visit all K/V blocks.
    Both strategies apply a single normalizer per query. The default is only an
    initial implementation choice; benchmark either strategy before speed claims.
    Output allocation preserves dense Q storage order (e.g. native BNHD viewed
    as BHND), avoiding an attention-output transpose/reshape copy in the adapter.
    """
    _check_inputs(q, k, v, kv_indices, kv_counts, global_tokens, block_size, out)
    if global_strategy not in ("flash", "triton"):
        raise ValueError("global_strategy must be flash or triton")
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (query_block_size, num_warps, num_stages)
    ):
        raise TypeError("Kernel tuning arguments must be integers")
    if (
        query_block_size not in (32, 64)
        or num_warps not in (4, 8)
        or num_stages not in (1, 2, 3, 4)
    ):
        raise ValueError("Supported tuning: query_block_size32/64, warps4/8, stages1..4")
    if scale is not None and (isinstance(scale, bool) or not isinstance(scale, Real)):
        raise TypeError("scale must be a real Python/NumPy scalar, never a tensor")
    global_tokens, block_size = int(global_tokens), int(block_size)
    query_block_size, num_warps, num_stages = map(int, (query_block_size, num_warps, num_stages))
    scale = q.shape[-1] ** -0.5 if scale is None else float(scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("Softmax scale must be positive and finite")
    if out is None:
        out = torch.empty_like(q)
    if global_strategy == "flash" and global_tokens:
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            global_output = F.scaled_dot_product_attention(
                q[:, :, :global_tokens], k, v, dropout_p=0.0, is_causal=False, scale=scale
            )
        out[:, :, :global_tokens].copy_(global_output)
    all_queries = global_strategy == "triton"
    query_count = q.shape[-2] if all_queries else q.shape[-2] - global_tokens
    grid = (query_count // query_block_size, q.shape[0] * q.shape[1])
    _block_sparse_fwd[grid](
        q,
        k,
        v,
        out,
        kv_indices,
        kv_counts,
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        *out.stride()[:3],
        q.shape[1],
        q.shape[2],
        q.shape[3],
        global_tokens,
        kv_indices.shape[1],
        query_block_size,
        block_size,
        scale * 1.4426950408889634,
        all_queries,
        num_stages,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    return out
