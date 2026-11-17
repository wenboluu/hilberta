"""One entry/exit permutation for the entire native Flux sampling trajectory.

No pipeline loop or scheduler arithmetic is copied. Native prepare_latents is
wrapped to pack image tokens/IDs, and native unpack is wrapped to restore raster
order immediately before decoding. Latent-only returns are restored explicitly.
"""

from __future__ import annotations

import torch
from diffusers import FluxPipeline, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

from .flux_adapter import persistent_flux_attention

_ACTIVE = "_flux_sampling_layout_active"
_MISSING = object()


@torch.no_grad()
def sample_flux_persistent(
    pipe,
    plan,
    *,
    backend="paired",
    sdpa_backend="efficient",
    kernel_options=None,
    packed_callback_on_step_end=None,
    **pipeline_kwargs,
):
    """Return (native_pipeline_result, layout_info), with public raster output.

    Supports the native text-to-image FluxPipeline and deterministic FlowMatch
    Euler scheduler. Initial noise is generated in the native raster order before
    permutation, preserving seed semantics. Model hooks and the explicitly named
    packed_callback_on_step_end see PACKED image tokens during this call. Ordinary
    callback_on_step_end is rejected to prevent silently changing its contract.

    This is an inference-only, single-threaded, reversible wrapper. Native cached
    embeddings, CFG, timesteps, sigmas, VAE and output formats are retained. Extra
    adapters/caches remain subject to persistent_flux_attention's existing guards.
    """
    if type(pipe) is not FluxPipeline:
        raise TypeError("Only the native FluxPipeline is supported")
    if type(pipe.scheduler) is not FlowMatchEulerDiscreteScheduler:
        raise TypeError("Only deterministic FlowMatchEulerDiscreteScheduler is supported")
    if pipe.scheduler.config.get("stochastic_sampling", False):
        raise ValueError("Stochastic scheduler noise requires token-identity-aware RNG")
    if getattr(pipe, _ACTIVE, False):
        raise RuntimeError("Nested sampling layout is unsupported")
    if pipeline_kwargs.get("callback_on_step_end") is not None:
        raise ValueError("Use packed_callback_on_step_end: its latents are in packed image order")
    for name in (
        "ip_adapter_image",
        "ip_adapter_image_embeds",
        "negative_ip_adapter_image",
        "negative_ip_adapter_image_embeds",
    ):
        if pipeline_kwargs.get(name) is not None:
            raise ValueError("IP-adapter needs a separate layout-aware implementation")
    if kernel_options is None and backend == "paired":
        kernel_options = dict(global_strategy="triton", num_warps=8, num_stages=2)
    if packed_callback_on_step_end is not None:
        pipeline_kwargs["callback_on_step_end"] = packed_callback_on_step_end
    order = plan.image_order.detach().cpu().clone()
    inverse = plan.inverse_image_order.detach().cpu().clone()
    device_indices = {}
    boundary = dict(
        sampling_image_reorder_calls=0,
        sampling_id_reorder_calls=0,
        sampling_inverse_calls=0,
        sampling_layout_read_write_bytes=0,
        callback_layout="packed",
        public_output_layout="raster",
    )
    originals = {
        name: pipe.__dict__.get(name, _MISSING) for name in ("prepare_latents", "_unpack_latents")
    }
    prepare = pipe.prepare_latents
    unpack = pipe._unpack_latents

    def indices(device):
        key = str(device)
        if key not in device_indices:
            device_indices[key] = (order.to(device), inverse.to(device))
        return device_indices[key]

    def packed_prepare(*args, **kwargs):
        if boundary["sampling_image_reorder_calls"]:
            raise RuntimeError("Expected exactly one native prepare_latents call")
        latent, ids = prepare(*args, **kwargs)
        if latent.ndim != 3 or latent.shape[1] != plan.image_tokens:
            raise ValueError("Prepared image-token count does not match the plan")
        if ids.ndim not in (2, 3) or ids.shape[-2:] != (plan.image_tokens, 3):
            raise ValueError("Prepared image position IDs do not match the plan")
        latent = latent.index_select(1, indices(latent.device)[0])
        ids = ids.index_select(ids.ndim - 2, indices(ids.device)[0])
        boundary["sampling_image_reorder_calls"] += 1
        boundary["sampling_id_reorder_calls"] += 1
        boundary["sampling_layout_read_write_bytes"] += 2 * (
            latent.numel() * latent.element_size() + ids.numel() * ids.element_size()
        )
        return latent, ids

    def restore(latent):
        if boundary["sampling_image_reorder_calls"] != 1 or boundary["sampling_inverse_calls"]:
            raise RuntimeError("Expected exactly one sampling entry and exit permutation")
        if latent.ndim != 3 or latent.shape[1] != plan.image_tokens:
            raise ValueError("Final latent image-token count does not match the plan")
        result = latent.index_select(1, indices(latent.device)[1])
        boundary["sampling_inverse_calls"] += 1
        boundary["sampling_layout_read_write_bytes"] += 2 * latent.numel() * latent.element_size()
        return result

    def raster_unpack(latents, *args, **kwargs):
        return unpack(restore(latents), *args, **kwargs)

    setattr(pipe, _ACTIVE, True)
    try:
        with persistent_flux_attention(
            pipe.transformer,
            plan,
            backend=backend,
            sdpa_backend=sdpa_backend,
            kernel_options=kernel_options,
            layout_scope="sampling",
        ) as info:
            pipe.prepare_latents = packed_prepare
            pipe._unpack_latents = raster_unpack
            result = pipe(**pipeline_kwargs)
            if pipeline_kwargs.get("output_type") == "latent":
                if isinstance(result, FluxPipelineOutput):
                    result = FluxPipelineOutput(images=restore(result.images))
                elif isinstance(result, tuple) and len(result) == 1:
                    result = (restore(result[0]),)
                else:
                    raise TypeError("Unexpected native Flux latent return type")
            if boundary["sampling_inverse_calls"] != 1:
                raise RuntimeError("Native pipeline did not pass the sampling exit boundary")
            info.update(boundary)
            return result, info
    finally:
        for name, old in originals.items():
            if old is _MISSING:
                pipe.__dict__.pop(name, None)
            else:
                setattr(pipe, name, old)
        delattr(pipe, _ACTIVE)
