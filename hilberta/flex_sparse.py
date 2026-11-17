"""FlexAttention block-sparse attention for a registered block plan; training-capable.

Isolated from the inference Triton kernel. The block mask is compiled from the
plan's 64-token block adjacency (physical layout: text prefix, then image blocks
in plan.image_order). Forward and backward come from torch.compile'd
flex_attention. Tile sizes are set explicitly because inductor's default
forward configuration for BF16/D128 needs 112 KB of shared memory and Ada
(SM89) has 99 KB. The processor permutes Q/K/V into the packed order per
attention call and restores the native order afterwards, so the surrounding
native model (RoPE, projections, residuals) is untouched and autograd flows.
"""

from __future__ import annotations
import contextlib
import functools
import inspect
import types
import torch
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor

ADA_KERNEL_OPTIONS = {
    64: dict(
        fwd_BLOCK_M=64,
        fwd_BLOCK_N=64,
        fwd_num_stages=2,
        fwd_num_warps=4,
        bwd_BLOCK_M1=64,
        bwd_BLOCK_N1=64,
        bwd_BLOCK_M2=64,
        bwd_BLOCK_N2=64,
        bwd_num_stages=2,
        bwd_num_warps=4,
    ),
    128: dict(
        fwd_BLOCK_M=128,
        fwd_BLOCK_N=64,
        fwd_num_stages=2,
        fwd_num_warps=8,
        bwd_BLOCK_M1=64,
        bwd_BLOCK_N1=64,
        bwd_BLOCK_M2=64,
        bwd_BLOCK_N2=64,
        bwd_num_stages=2,
        bwd_num_warps=4,
    ),
}

torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
_compiled = torch.compile(flex_attention, dynamic=False, fullgraph=True)


def block_adjacency(plan, phase):
    """Bool [blocks, blocks] in the plan's physical layout; global rows see everything."""
    p = plan.phases[phase]
    nb, g = plan.total_tokens // plan.block_size, plan.global_tokens // plan.block_size
    adj = torch.zeros((nb, nb), dtype=torch.bool)
    adj[:g] = True
    for row, count in enumerate(p.kv_counts.tolist()):
        adj[g + row, p.kv_indices[row, :count].long()] = True
    return adj


def build_block_mask(adj, device, block_size=64, mask_block=64):
    """BlockMask over tokens from a block adjacency; mask_block 64 gives full blocks only."""
    if mask_block not in ADA_KERNEL_OPTIONS or mask_block % block_size:
        raise ValueError("mask_block must be 64 or 128 and a multiple of the plan block size")
    n = adj.shape[0] * block_size
    adj = adj.to(device)

    def mask_mod(b, h, q, kv):
        return adj[q // block_size, kv // block_size]

    return create_block_mask(mask_mod, None, None, n, n, device=device, BLOCK_SIZE=mask_block)


def flex_sparse_attention(q, k, v, block_mask, mask_block=64, scale=None):
    """q/k/v [B,H,N,D]; returns [B,H,N,D]. Differentiable."""
    return _compiled(
        q, k, v, block_mask=block_mask, scale=scale, kernel_options=ADA_KERNEL_OPTIONS[mask_block]
    )


class FlexPlanBank:
    """Immutable per-plan metadata; block masks are built once per device."""

    def __init__(self, plan, mask_block=64):
        if plan.block_size != 64 or len(plan.phases) != 4:
            raise ValueError("Plan must use 64-token blocks and four phases")
        self.plan, self.mask_block = plan, mask_block
        self.text_tokens, self.total_tokens = int(plan.text_tokens), int(plan.total_tokens)
        self.adjacency = [block_adjacency(plan, i) for i in range(4)]
        image = plan.image_order.to(torch.long)
        self.index = torch.cat([torch.arange(self.text_tokens), self.text_tokens + image])
        self.inverse = torch.argsort(self.index)
        self.image_order, self.image_inverse = image, torch.argsort(image)
        self.in_forward, self.entry_reorders, self.exit_restores = False, 0, 0
        self._device = {}

    def on_device(self, device):
        key = str(device)
        if key not in self._device:
            self._device[key] = dict(
                index=self.index.to(device),
                inverse=self.inverse.to(device),
                image_order=self.image_order.to(device),
                image_inverse=self.image_inverse.to(device),
                masks=[
                    build_block_mask(a, device, self.plan.block_size, self.mask_block)
                    for a in self.adjacency
                ],
            )
        return self._device[key]

    def sparsity(self):
        return (
            [float(m.sparsity()) for m in next(iter(self._device.values()))["masks"]]
            if self._device
            else None
        )


def _packed_dispatch(bank, phase, query, key, value, attn_mask=None, **kwargs):
    """per_call layout: gather Q/K/V into the packed order, attend, scatter the output back."""
    if attn_mask is not None:
        raise ValueError("The Flex plan processor cannot combine an external attention mask")
    if query.shape[1] != bank.total_tokens:
        raise ValueError(
            f"Sequence length {query.shape[1]} does not match the plan ({bank.total_tokens})"
        )
    dev = bank.on_device(query.device)
    q, k, v = [x.index_select(1, dev["index"]).transpose(1, 2) for x in (query, key, value)]
    out = flex_sparse_attention(q, k, v, dev["masks"][phase], bank.mask_block)
    return out.transpose(1, 2).index_select(1, dev["inverse"])


def _persistent_dispatch(bank, phase, query, key, value, attn_mask=None, **kwargs):
    """persistent layout: tokens already arrive packed (entry hook); no per-layer gather."""
    if attn_mask is not None:
        raise ValueError("The Flex plan processor cannot combine an external attention mask")
    if query.shape[1] != bank.total_tokens:
        raise ValueError(
            f"Sequence length {query.shape[1]} does not match the plan ({bank.total_tokens})"
        )
    # Gradient checkpointing re-runs blocks during backward, after the exit hook has fired,
    # so the live in_forward flag cannot be the guard; require that the entry hook ran at all.
    if bank.entry_reorders == 0:
        raise RuntimeError(
            "persistent layout requires calling the whole Transformer (entry hook never ran)"
        )
    dev = bank.on_device(query.device)
    out = flex_sparse_attention(
        query.transpose(1, 2),
        key.transpose(1, 2),
        value.transpose(1, 2),
        dev["masks"][phase],
        bank.mask_block,
    )
    return out.transpose(1, 2)


class _PersistentHooks:
    """Reorder image tokens and their position ids once at Transformer entry; restore once at exit.

    index_select is differentiable, so training gradients flow to the raster inputs unchanged.
    Text tokens are untouched; RoPE is computed by the native code from the reordered img_ids,
    so every token keeps its own position. Gradient checkpointing recomputes nothing extra here.
    """

    def __init__(self, model, bank):
        self.bank, self.signature = bank, inspect.signature(model.forward)
        self.image_tokens = bank.total_tokens - bank.text_tokens

    def before(self, model, args, kwargs):
        if self.bank.in_forward:
            raise RuntimeError(
                "Reentrant Transformer forwards are unsupported in persistent layout"
            )
        bound = self.signature.bind(*args, **kwargs)
        values = bound.arguments
        hidden, ids = values.get("hidden_states"), values.get("img_ids")
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or hidden.shape[1] != self.image_tokens
        ):
            raise ValueError("hidden_states image-token count does not match the plan")
        if not isinstance(ids, torch.Tensor) or ids.shape[-2:] != (self.image_tokens, 3):
            raise ValueError("img_ids must have final shape [image_tokens, 3]")
        text = values.get("encoder_hidden_states")
        if not isinstance(text, torch.Tensor) or text.shape[1] != self.bank.text_tokens:
            raise ValueError("encoder_hidden_states text-token count does not match the plan")
        order = self.bank.on_device(hidden.device)["image_order"]
        values["hidden_states"] = hidden.index_select(1, order)
        values["img_ids"] = ids.index_select(
            ids.ndim - 2, self.bank.on_device(ids.device)["image_order"]
        )
        self.bank.in_forward = True
        self.bank.entry_reorders += 1
        return bound.args, bound.kwargs

    def after(self, model, args, output):
        active, self.bank.in_forward = self.bank.in_forward, False
        if output is None or not active:
            return output
        value = output.sample if isinstance(output, Transformer2DModelOutput) else output[0]
        if value.shape[1] != self.image_tokens:
            raise ValueError("Transformer output does not have one row per image token")
        restored = value.index_select(1, self.bank.on_device(value.device)["image_inverse"])
        self.bank.exit_restores += 1
        return (
            Transformer2DModelOutput(sample=restored)
            if isinstance(output, Transformer2DModelOutput)
            else (restored,) + tuple(output[1:])
        )


class FlexPlanProcessor:
    """Native FluxAttnProcessor.__call__ with only dispatch_attention_fn replaced."""

    def __init__(self, bank, phase, counter, layout="per_call"):
        native = FluxAttnProcessor.__call__
        dispatch = _persistent_dispatch if layout == "persistent" else _packed_dispatch
        namespace = dict(
            native.__globals__, dispatch_attention_fn=functools.partial(dispatch, bank, phase)
        )
        self._native = types.FunctionType(
            native.__code__, namespace, native.__name__, native.__defaults__, native.__closure__
        )
        self._inner, self.phase, self.counter = FluxAttnProcessor(), phase, counter

    def __call__(self, *args, **kwargs):
        self.counter[self.phase] += 1
        return self._native(self._inner, *args, **kwargs)


@contextlib.contextmanager
def flex_plan_attention(model, plan, mask_block=64, *, bank=None, layout="per_call"):
    """Run every joint/single block's attention through the plan's block mask (phase = layer % 4).

    layout='per_call': each attention gathers Q/K/V into the packed order and scatters the
    output back (no model hooks; any partial forward works).
    layout='persistent': image tokens and img_ids are reordered once at Transformer entry
    and restored once at exit by forward hooks; attention runs directly on the packed stream.
    Call the whole Transformer in this mode. Both layouts work in training and inference and
    leave public inputs/outputs in native raster order. A cached bank (same plan and
    mask_block) may be passed to reuse compiled block masks across calls. Yields the bank.
    """
    if layout not in ("per_call", "persistent"):
        raise ValueError("layout must be 'per_call' or 'persistent'")
    blocks = list(model.transformer_blocks) + list(model.single_transformer_blocks)
    originals = [b.attn.processor for b in blocks]
    if any(
        type(p) is not FluxAttnProcessor or p._attention_backend not in (None, "native")
        for p in originals
    ):
        raise ValueError("Only the native, non-parallel FluxAttnProcessor is supported")
    if bank is None:
        bank = FlexPlanBank(plan, mask_block)
    elif bank.plan is not plan or bank.mask_block != mask_block:
        raise ValueError("Cached bank must match plan and mask block")
    bank.calls, bank.layout, bank.in_forward = [0, 0, 0, 0], layout, False
    handles = []
    try:
        for i, b in enumerate(blocks):
            b.attn.set_processor(FlexPlanProcessor(bank, i % 4, bank.calls, layout))
        if layout == "persistent":
            hooks = _PersistentHooks(model, bank)
            handles.append(model.register_forward_pre_hook(hooks.before, with_kwargs=True))
            handles.append(model.register_forward_hook(hooks.after, prepend=True, always_call=True))
        yield bank
    finally:
        for h in handles:
            h.remove()
        for b, p in zip(blocks, originals):
            b.attn.set_processor(p)
        bank.in_forward = False
