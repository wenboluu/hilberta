"""Native Diffusers 0.36 Flux image-only masks for frozen-model diagnostics.

Only the image/image quadrant is restricted. Text is a fully connected prefix
and remains dynamically updated by native Flux joint and single blocks. Native
Q/K RMSNorm, RoPE, projections and block operations are delegated unchanged.
This is a dense-mask fidelity reference, not a sparse kernel or speed claim.
Contexts are for single-threaded evaluation; no training or context parallelism.
"""

from __future__ import annotations

import contextlib
import hashlib

import torch
from diffusers.models.attention_dispatch import attention_backend as dispatch_backend
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor
from torch.nn.attention import SDPBackend, sdpa_kernel

from hilberta.geometry import hilbert_order
from tests.helpers import tile_labels

_BACKENDS = {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION}
_ACTIVE = "_hilbert_flux_mask_context_active"


def build_flux_image_masks(tiles=4, grid=64, legacy_shared_region=False):
    """Return CPU bool [4, grid**2, grid**2] in raster token identities.

    The original composite Hilbert ordering uses the middle half followed by
    its spatial inversion, and four offsets separated by tile_width / 4.
    ``legacy_shared_region=True`` is solely an old-Flux diagnostic control:
    for grid64 it adds global row AND column access for the center16x16 tokens,
    exactly matching the original 4096-token mask. It defaults to False.
    """
    if tiles not in (4, 16):
        raise ValueError("Flux pilot tile count must be 4 or 16")
    h = hilbert_order(grid)
    n, quarter = h.numel(), h.numel() // 4
    order = torch.cat((h[quarter : 3 * quarter], n - 1 - h[quarter : 3 * quarter]))
    labels = torch.stack([tile_labels(order, tiles, phase) for phase in range(4)])
    masks = labels[:, :, None] == labels[:, None, :]
    if legacy_shared_region:
        if grid != 64:
            raise ValueError("Exact legacy center256 control requires a 64x64 token grid")
        ids = torch.arange(n).reshape(grid, grid)[24:40, 24:40].flatten()
        masks[:, ids, :] = True
        masks[:, :, ids] = True
    return masks


@contextlib.contextmanager
def flux_attention_backend(backend="efficient"):
    """Use native Diffusers dispatch and exactly one PyTorch SDPA backend.

    Apply the same context to the native full-attention control. No fallback,
    autocast, precision conversion, or TF32 flag changes are introduced here.
    """
    if backend not in _BACKENDS:
        raise ValueError("backend must be math (CPU correctness) or efficient (CUDA)")
    with dispatch_backend("native"), sdpa_kernel([_BACKENDS[backend]]):
        yield


class _MaskBank:
    def __init__(self, masks, info):
        self.masks = masks
        self.info = info
        self.cache = {}

    def joint_mask(self, phase, text_tokens, device):
        # One allocation per phase/prefix/device, shared across all 57 blocks.
        # Lazy device placement supports sequential CPU offload as well.
        key = (phase, text_tokens, str(device))
        if key not in self.cache:
            n = self.masks.shape[-1]
            mask = torch.ones((text_tokens + n, text_tokens + n), dtype=torch.bool, device=device)
            mask[text_tokens:, text_tokens:] = self.masks[phase].to(device)
            self.cache[key] = mask
            self.info["prepared_mask_count"] = len(self.cache)
        return self.cache[key]


class FluxImageMaskProcessor:
    """Delegate a native Flux processor, injecting a shared explicit mask."""

    def __init__(self, original, bank, layer, phase, calls, stream, backend):
        self.original, self.bank = original, bank
        self.layer, self.phase, self.calls = layer, phase, calls
        self.stream, self.backend = stream, backend

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        if attention_mask is not None:
            raise ValueError("An existing attention mask cannot be combined implicitly")
        if hidden_states.ndim != 3:
            raise ValueError("Flux hidden states must have shape [batch, tokens, channels]")
        if self.backend == "efficient" and hidden_states.device.type != "cuda":
            raise ValueError("efficient requires CUDA; select math for CPU checks")
        n = self.bank.masks.shape[-1]
        if self.stream == "joint":
            if encoder_hidden_states is None or hidden_states.shape[1] != n:
                raise ValueError("Joint Flux blocks require image tokens and separate text states")
            text_tokens = encoder_hidden_states.shape[1]
        else:
            if encoder_hidden_states is not None:
                raise ValueError(
                    "Single Flux attention expects an already concatenated text prefix"
                )
            text_tokens = hidden_states.shape[1] - n
        if text_tokens <= 0:
            raise ValueError("Flux attention must retain a nonempty text prefix")
        mask = self.bank.joint_mask(self.phase, text_tokens, hidden_states.device)
        self.calls[self.layer] += 1
        return self.original(
            attn,
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=mask,
            image_rotary_emb=image_rotary_emb,
        )


@contextlib.contextmanager
def flux_mask_attention(model, masks, backend="efficient"):
    """Mask native Flux blocks and yield call-count/protocol metadata.

    Usage::
        masks = build_flux_image_masks(tiles=4)
        with flux_mask_attention(pipe.transformer, masks) as info:
            result = pipe(...)

    Layer indices continue across joint and single stacks: joint0..18 followed
    by single19..56 for the native 19+38 model. Phase is global_index % 4 on
    every denoiser call; it does not reset at the single stack or advance with
    denoising time. Model weights may be FP32/BF16/FP16; dtype is unchanged.
    Mask clones, processors and backend scopes are restored/freed on exit.
    """
    if getattr(model, _ACTIVE, False):
        raise RuntimeError("Nested Flux mask contexts on the same model are unsupported")
    if backend not in _BACKENDS:
        raise ValueError("backend must be math or efficient")
    if not isinstance(masks, torch.Tensor) or masks.dtype != torch.bool:
        raise TypeError("masks must be a bool tensor [4, N, N]")
    if (
        masks.ndim != 3
        or masks.shape[0] != 4
        or masks.shape[1] != masks.shape[2]
        or not masks.shape[1]
    ):
        raise ValueError("masks must have nonempty shape [4, N, N]")
    if not bool(masks.any(-1).all()):
        raise ValueError("Every image query needs at least one image key")
    joint = list(model.transformer_blocks)
    single = list(model.single_transformer_blocks)
    if not joint or not single:
        raise ValueError("Both native Flux joint and single stacks are required")
    blocks = joint + single
    originals = [b.attn.processor for b in blocks]
    for i, (block, processor) in enumerate(zip(blocks, originals)):
        if type(processor) is not FluxAttnProcessor:
            raise ValueError("Only native Diffusers FluxAttnProcessor is supported")
        if (
            processor._attention_backend not in (None, "native")
            or processor._parallel_config is not None
        ):
            raise ValueError("Only native, non-parallel attention dispatch is supported")
        if (block.attn.added_kv_proj_dim is not None) != (i < len(joint)):
            raise ValueError("Flux block stream and attention projection type disagree")
    immutable = masks.detach().to(device="cpu", dtype=torch.bool).clone()
    calls = [0] * len(blocks)
    phase_assignment = [i % 4 for i in range(len(blocks))]
    info = {
        "backend": backend,
        "dispatch": "native",
        "fallback_enabled": False,
        "image_tokens": immutable.shape[-1],
        "phases": 4,
        "joint_layers": len(joint),
        "single_layers": len(single),
        "layer_phases": phase_assignment,
        "layer_calls": calls,
        "text_connectivity": "native dynamic prefix; all text rows and columns allowed",
        "layout": "native raster image identities",
        "precision": "unchanged",
        "image_mask_sha256": hashlib.sha256(immutable.numpy().tobytes()).hexdigest(),
        "prepared_mask_count": 0,
    }
    bank = _MaskBank(immutable, info)
    setattr(model, _ACTIVE, True)
    try:
        for i, (block, original) in enumerate(zip(blocks, originals)):
            block.attn.set_processor(
                FluxImageMaskProcessor(
                    original,
                    bank,
                    i,
                    i % 4,
                    calls,
                    "joint" if i < len(joint) else "single",
                    backend,
                )
            )
        with flux_attention_backend(backend):
            yield info
    finally:
        for block, original in zip(blocks, originals):
            block.attn.set_processor(original)
        bank.cache.clear()
        delattr(model, _ACTIVE)
