"""Persistent-layout native Flux attention for a registered block-sparse plan.

Reorder image states and their positional IDs at Transformer entry; preserve
that order through all native joint/single blocks; inverse once at prediction
exit. Text and shared-image projections remain distinct. No per-layer image
permutation, rolling, or QKV packing is introduced. Inference only.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect

import torch
import torch.nn.functional as F
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.transformers.transformer_flux import FluxAttnProcessor, _get_qkv_projections
from torch.nn.attention import SDPBackend, sdpa_kernel

_ACTIVE = "_flux_kernel_persistent_layout"
_SDP_BACKENDS = {"math": SDPBackend.MATH, "efficient": SDPBackend.EFFICIENT_ATTENTION}


class _PlanBank:
    """Validate and copy immutable CPU metadata once; cache device copies."""

    def __init__(self, plan, info):
        self.image_tokens = int(plan.image_tokens)
        self.text_tokens = int(plan.text_tokens)
        self.shared_tokens = int(plan.shared_tokens)
        self.global_tokens = int(plan.global_tokens)
        self.total_tokens = int(plan.total_tokens)
        self.block_size = int(plan.block_size)
        self.info = info
        if (
            self.image_tokens <= 0
            or self.text_tokens <= 0
            or not 0 <= self.shared_tokens <= self.image_tokens
        ):
            raise ValueError(
                "Plan requires nonempty image/text tokens and a valid shared-image count"
            )
        if (
            self.global_tokens != self.text_tokens + self.shared_tokens
            or self.total_tokens != self.text_tokens + self.image_tokens
        ):
            raise ValueError("Plan global/total token counts are inconsistent")
        if (
            self.block_size <= 0
            or self.total_tokens % self.block_size
            or self.global_tokens % self.block_size
        ):
            raise ValueError("Total tokens and global prefix must be block-aligned")
        self.order = plan.image_order.detach().to(device="cpu", dtype=torch.int64).clone()
        self.inverse = plan.inverse_image_order.detach().to(device="cpu", dtype=torch.int64).clone()
        expected = torch.arange(self.image_tokens)
        if self.order.shape != expected.shape or not torch.equal(
            self.order.sort().values, expected
        ):
            raise ValueError("image_order must contain every image-token identity exactly once")
        if self.inverse.shape != expected.shape or not torch.equal(
            self.inverse[self.order], expected
        ):
            raise ValueError("inverse_image_order is not the inverse of image_order")
        if len(plan.phases) != 4:
            raise ValueError("Exactly four phase metadata tables are required")
        self.phases = []
        query_blocks = (self.total_tokens - self.global_tokens) // self.block_size
        total_blocks = self.total_tokens // self.block_size
        global_blocks = self.global_tokens // self.block_size
        for phase in plan.phases:
            ids = phase.kv_indices.detach().to(device="cpu", dtype=torch.int32).clone()
            counts = phase.kv_counts.detach().to(device="cpu", dtype=torch.int32).clone()
            if ids.ndim != 2 or ids.shape[0] != query_blocks or counts.shape != (query_blocks,):
                raise ValueError("KV metadata must have one row per ordinary physical query block")
            if bool(((counts <= 0) | (counts > ids.shape[1])).any()):
                raise ValueError("KV block counts are outside their metadata width")
            for row, count in zip(ids, counts.tolist()):
                active = row[:count]
                if (
                    bool(((active < 0) | (active >= total_blocks)).any())
                    or active.unique().numel() != count
                ):
                    raise ValueError("Active KV block IDs must be valid and unique")
                if not torch.isin(torch.arange(global_blocks, dtype=torch.int32), active).all():
                    raise ValueError(
                        "Every ordinary query must retain every text/shared global key block"
                    )
            self.phases.append((ids.contiguous(), counts.contiguous()))
        self.paired_phases = None
        if info["backend"] == "paired":
            from .paired_metadata import pair_metadata

            if self.block_size != 64:
                raise ValueError("Paired backend requires 64-token physical blocks")
            self.paired_phases = [
                pair_metadata(
                    ids,
                    counts,
                    tokens=self.total_tokens,
                    global_tokens=self.global_tokens,
                    key_block_size=64,
                )
                for ids, counts in self.phases
            ]
        self.device_cache, self.mask_cache = {}, {}
        self.order_hash = hashlib.sha256(self.order.numpy().tobytes()).hexdigest()

    def on_device(self, device):
        key = str(device)
        if key not in self.device_cache:
            self.device_cache[key] = {
                "order": self.order.to(device),
                "inverse": self.inverse.to(device),
                "phases": [(ids.to(device), counts.to(device)) for ids, counts in self.phases],
            }
            if self.paired_phases is not None:
                self.device_cache[key]["paired_phases"] = [p.to(device) for p in self.paired_phases]

            self.info["metadata_device_copies"] = len(self.device_cache)
        return self.device_cache[key]

    def dense_mask(self, phase, device):
        key = (phase, str(device))
        if key not in self.mask_cache:
            # Correctness reference only: construct directly from the validated
            # sparse metadata, not from another topology implementation.
            mask = torch.zeros((self.total_tokens, self.total_tokens), dtype=torch.bool)
            mask[: self.global_tokens, :] = True
            ids, counts = self.phases[phase]
            for row, count in enumerate(counts.tolist()):
                query_start = self.global_tokens + row * self.block_size
                for block in ids[row, :count].tolist():
                    mask[
                        query_start : query_start + self.block_size,
                        block * self.block_size : (block + 1) * self.block_size,
                    ] = True
            self.mask_cache[key] = mask.to(device)
            self.info["reference_masks_prepared"] = len(self.mask_cache)
        return self.mask_cache[key]

    def clear(self):
        self.device_cache.clear()
        self.mask_cache.clear()


class _Runtime:
    def __init__(self, model, bank, info):
        self.bank, self.info = bank, info
        self.signature = inspect.signature(model.forward)
        self.in_forward = False
        self.expected_output_batch = None

    def before(self, model, args, kwargs):
        if self.in_forward:
            raise RuntimeError(
                "Reentrant Transformer forwards are unsupported in persistent-layout context"
            )
        if torch.is_grad_enabled():
            raise RuntimeError(
                "Persistent sparse Flux is inference-only; use torch.no_grad or inference_mode"
            )
        bound = self.signature.bind(*args, **kwargs)
        values = bound.arguments
        hidden = values.get("hidden_states")
        text = values.get("encoder_hidden_states")
        ids = values.get("img_ids")
        text_ids = values.get("txt_ids")
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 3
            or hidden.shape[1] != self.bank.image_tokens
        ):
            raise ValueError("hidden_states image-token count does not match the plan")
        if (
            not isinstance(text, torch.Tensor)
            or text.ndim != 3
            or text.shape[:2] != (hidden.shape[0], self.bank.text_tokens)
        ):
            raise ValueError("encoder_hidden_states batch/text-token count does not match the plan")
        if (
            not isinstance(ids, torch.Tensor)
            or ids.ndim not in (2, 3)
            or ids.shape[-2:] != (self.bank.image_tokens, 3)
        ):
            raise ValueError(
                "img_ids must match image token identities with final shape [image_tokens,3]"
            )
        if (
            not isinstance(text_ids, torch.Tensor)
            or text_ids.ndim not in (2, 3)
            or text_ids.shape[-2:] != (self.bank.text_tokens, 3)
        ):
            raise ValueError("txt_ids must match the native text prefix")
        if (
            values.get("controlnet_block_samples") is not None
            or values.get("controlnet_single_block_samples") is not None
        ):
            raise ValueError("ControlNet residuals require a separate layout-aware implementation")
        extra = values.get("joint_attention_kwargs")
        if extra is not None and (not isinstance(extra, dict) or set(extra) - {"scale"}):
            raise ValueError(
                "Only native scalar LoRA scale is supported in joint_attention_kwargs; no IP-adapter/extra masks"
            )
        metadata = self.bank.on_device(hidden.device)
        if self.info["layout_scope"] == "transformer":
            values["hidden_states"] = hidden.index_select(1, metadata["order"])
            values["img_ids"] = ids.index_select(
                ids.ndim - 2, self.bank.on_device(ids.device)["order"]
            )
            self.info["image_reorder_calls"] += 1
            self.info["image_id_reorder_calls"] += 1
            self.info["image_layout_gather_read_write_bytes"] += (
                2 * hidden.numel() * hidden.element_size()
            )
        self.expected_output_batch = hidden.shape[0]
        self.in_forward = True
        self.info["transformer_calls"] += 1
        return bound.args, bound.kwargs

    def after(self, model, args, output):
        # always_call clears state after a native forward exception. No output
        # recovery is possible or counted in that case.
        active = self.in_forward
        self.in_forward = False
        if output is None or not active:
            return output
        if isinstance(output, Transformer2DModelOutput):
            value = output.sample
        elif isinstance(output, tuple) and len(output) == 1:
            value = output[0]
        else:
            raise TypeError("Expected native Flux Transformer2DModelOutput or one-element tuple")
        if value.ndim != 3 or value.shape[:2] != (
            self.expected_output_batch,
            self.bank.image_tokens,
        ):
            raise ValueError("Transformer output shape does not match image layout")
        if self.info["layout_scope"] == "sampling":
            return output
        restored = value.index_select(1, self.bank.on_device(value.device)["inverse"])
        self.info["image_inverse_calls"] += 1
        self.info["image_layout_gather_read_write_bytes"] += (
            2 * value.numel() * value.element_size()
        )
        return (
            Transformer2DModelOutput(sample=restored)
            if isinstance(output, Transformer2DModelOutput)
            else (restored,)
        )


class PersistentFluxProcessor:
    """Native Diffusers 0.36 projections/norm/RoPE with one replaced attention op."""

    def __init__(self, runtime, layer, backend, kernel_options):
        self.runtime, self.layer, self.phase = runtime, layer, layer % 4
        self.backend, self.kernel_options = backend, dict(kernel_options)

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        image_rotary_emb=None,
    ):
        if not self.runtime.in_forward:
            raise RuntimeError(
                "Call the whole Transformer so the persistent image layout is established"
            )
        if attention_mask is not None:
            raise ValueError(
                "External attention masks cannot be combined implicitly with the registered plan"
            )
        bank = self.runtime.bank
        # This order and these exact native helpers match FluxAttnProcessor.
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )
        query = attn.norm_q(query.unflatten(-1, (attn.heads, -1)))
        key = attn.norm_k(key.unflatten(-1, (attn.heads, -1)))
        value = value.unflatten(-1, (attn.heads, -1))
        if attn.added_kv_proj_dim is not None:
            encoder_query = attn.norm_added_q(encoder_query.unflatten(-1, (attn.heads, -1)))
            encoder_key = attn.norm_added_k(encoder_key.unflatten(-1, (attn.heads, -1)))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        if query.shape[1] != bank.total_tokens:
            raise ValueError("Native attention sequence length does not match the plan")
        q, k, v = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
        if self.backend == "sdpa_reference":
            result = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=bank.dense_mask(self.phase, query.device),
                dropout_p=0.0,
                is_causal=False,
            )
        elif self.backend == "paired":
            from .paired_kernel import paired_attention

            metadata = bank.on_device(query.device)["paired_phases"][self.phase]
            result = paired_attention(q, k, v, metadata, **self.kernel_options)
        else:
            from .sparse_kernel import sparse_attention

            ids, counts = bank.on_device(query.device)["phases"][self.phase]
            result = sparse_attention(
                q,
                k,
                v,
                ids,
                counts,
                global_tokens=bank.global_tokens,
                block_size=bank.block_size,
                **self.kernel_options,
            )
        self.runtime.info["layer_calls"][self.layer] += 1
        hidden_states = result.transpose(1, 2).flatten(2, 3).to(query.dtype)
        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [
                    encoder_hidden_states.shape[1],
                    hidden_states.shape[1] - encoder_hidden_states.shape[1],
                ],
                dim=1,
            )
            hidden_states = attn.to_out[1](attn.to_out[0](hidden_states))
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        return hidden_states


@contextlib.contextmanager
def persistent_flux_attention(
    model,
    plan,
    backend="sdpa_reference",
    sdpa_backend="math",
    kernel_options=None,
    *,
    layout_scope="transformer",
):
    """Reversibly run a native Flux Transformer in the plan's persistent layout.

    Pass the original native pipeline unchanged inside this inference context.
    The image order persists for one whole Transformer call (all blocks), then
    native raster predictions return to the scheduler. Metadata is moved once
    per device and cached across denoising calls. With layout_scope='sampling',
    inputs/position IDs must already be packed and predictions remain packed;
    use sample_flux_persistent to manage native pipeline entry and exit.
    This context is single-threaded
    and rejects reentrant forwards, ControlNet residuals and IP-adapter inputs.
    """
    if layout_scope not in ("transformer", "sampling"):
        raise ValueError("layout_scope must be transformer or sampling")
    if backend not in ("sdpa_reference", "sparse", "paired") or sdpa_backend not in _SDP_BACKENDS:
        raise ValueError(
            "backend must be sdpa_reference/sparse/paired; sdpa_backend must be math/efficient"
        )
    if getattr(model, _ACTIVE, False) or getattr(model, "_hilbert_flux_mask_context_active", False):
        raise RuntimeError("Nested/conflicting Flux layout or image-mask contexts are unsupported")
    if model.training or getattr(model, "gradient_checkpointing", False):
        raise ValueError("Native Flux model must be in eval mode without gradient checkpointing")
    if getattr(model, "_cache_config", None) is not None:
        raise ValueError("Transformer cache state needs a separate layout-aware implementation")
    joint, single = list(model.transformer_blocks), list(model.single_transformer_blocks)
    if not joint or not single:
        raise ValueError("Both native Flux joint and single blocks are required")
    blocks = joint + single
    originals = [block.attn.processor for block in blocks]
    for i, (block, processor) in enumerate(zip(blocks, originals)):
        if (
            type(processor) is not FluxAttnProcessor
            or processor._attention_backend not in (None, "native")
            or processor._parallel_config is not None
        ):
            raise ValueError("Only native, non-parallel Diffusers FluxAttnProcessor is supported")
        if (block.attn.added_kv_proj_dim is not None) != (i < len(joint)):
            raise ValueError("Native joint/single projection types disagree with block layout")
    info = {
        "backend": backend,
        "sdpa_backend": sdpa_backend,
        "kernel_options": dict(kernel_options or {}),
        "layout_scope": layout_scope,
        "transformer_calls": 0,
        "image_reorder_calls": 0,
        "image_inverse_calls": 0,
        "image_id_reorder_calls": 0,
        "per_layer_image_permutations": 0,
        "layer_calls": [0] * len(blocks),
        "layer_phases": [i % 4 for i in range(len(blocks))],
        "joint_layers": len(joint),
        "single_layers": len(single),
        "metadata_device_copies": 0,
        "reference_masks_prepared": 0,
        "image_layout_gather_read_write_bytes": 0,
        "layout": "native text prefix; shared image prefix; remaining image tokens; "
        + (
            "one Transformer-entry/exit gather"
            if layout_scope == "transformer"
            else "packed throughout sampling"
        ),
        "shared_tokens_keep_image_projections": True,
        "native_dynamic_text_preserved": True,
        "precision": "unchanged; native QK RMSNorm and RoPE",
        "weights_modified": False,
    }
    bank = _PlanBank(plan, info)
    info["image_order_sha256"] = bank.order_hash
    runtime = _Runtime(model, bank, info)
    handles = []
    setattr(model, _ACTIVE, True)
    try:
        for i, block in enumerate(blocks):
            block.attn.set_processor(
                PersistentFluxProcessor(runtime, i, backend, kernel_options or {})
            )
        # External pre-hooks can still observe public raster inputs. Conversely,
        # prepend recovery so external post-hooks always see raster predictions.
        handles.append(model.register_forward_pre_hook(runtime.before, with_kwargs=True))
        handles.append(model.register_forward_hook(runtime.after, prepend=True, always_call=True))
        with sdpa_kernel([_SDP_BACKENDS[sdpa_backend]]):
            yield info
    finally:
        for handle in handles:
            handle.remove()
        for block, processor in zip(blocks, originals):
            block.attn.set_processor(processor)
        bank.clear()
        delattr(model, _ACTIVE)
