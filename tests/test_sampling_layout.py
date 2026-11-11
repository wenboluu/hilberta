"""Whole native CPU pipelines: scheduler trajectories, seeds and raster outputs."""

import numpy as np
import pytest
import torch
from diffusers import AutoencoderKL, FluxPipeline, FlowMatchEulerDiscreteScheduler

from hilberta.flux_adapter import persistent_flux_attention
from hilberta.sampling_layout import sample_flux_persistent
from tests.test_adapter import model_and_args, toy_plan


def fixture():
    model, args = model_and_args()
    vae = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 8),
        layers_per_block=1,
        latent_channels=1,
        norm_num_groups=4,
        sample_size=16,
        shift_factor=0.0,
        scaling_factor=1.0,
    ).eval()
    pipe = FluxPipeline(
        transformer=model,
        scheduler=FlowMatchEulerDiscreteScheduler(),
        vae=vae,
        text_encoder=None,
        tokenizer=None,
        text_encoder_2=None,
        tokenizer_2=None,
    )
    pipe.set_progress_bar_config(disable=True)
    kwargs = dict(
        prompt_embeds=args["encoder_hidden_states"],
        pooled_prompt_embeds=args["pooled_projections"],
        latents=args["hidden_states"],
        height=16,
        width=16,
        num_inference_steps=4,
        max_sequence_length=4,
        guidance_scale=7.5,
    )
    return pipe, toy_plan(), kwargs


@pytest.mark.parametrize("output_type", ["latent", "np"])
@pytest.mark.parametrize("return_dict", [False, True])
@pytest.mark.parametrize("true_cfg", [False, True])
def test_native_sampling_matches_per_forward_bitwise(output_type, return_dict, true_cfg):
    pipe, plan, kwargs = fixture()
    kwargs.update(output_type=output_type, return_dict=return_dict)
    if true_cfg:
        kwargs.update(
            true_cfg_scale=2.0,
            negative_prompt_embeds=kwargs["prompt_embeds"] * 0.5,
            negative_pooled_prompt_embeds=kwargs["pooled_prompt_embeds"] * 0.5,
        )
    source = kwargs["latents"].clone()
    old_states, new_states, old_predictions, new_predictions = [], [], [], []

    def callback(target):
        def capture(p, i, t, values):
            target.append(values["latents"].clone())
            return values

        return capture

    handle = pipe.transformer.register_forward_hook(
        lambda m, a, o: old_predictions.append(o[0].clone())
    )
    with torch.no_grad(), persistent_flux_attention(pipe.transformer, plan) as old_info:
        reference = pipe(**kwargs, callback_on_step_end=callback(old_states))
    handle.remove()
    sigmas = pipe.scheduler.sigmas.clone()
    handle = pipe.transformer.register_forward_hook(
        lambda m, a, o: new_predictions.append(o[0].clone())
    )
    actual, info = sample_flux_persistent(
        pipe,
        plan,
        backend="sdpa_reference",
        sdpa_backend="math",
        packed_callback_on_step_end=callback(new_states),
        **kwargs,
    )
    handle.remove()
    before = reference.images if return_dict else reference[0]
    after = actual.images if return_dict else actual[0]
    if output_type == "latent":
        assert torch.equal(before, after)
    else:
        assert np.array_equal(before, after)
    assert torch.equal(sigmas, pipe.scheduler.sigmas)
    assert len(old_states) == len(new_states) == 4
    assert all(
        torch.equal(a, b[:, plan.inverse_image_order]) for a, b in zip(old_states, new_states)
    )
    assert len(old_predictions) == len(new_predictions) == 4 * (2 if true_cfg else 1)
    assert all(
        torch.equal(a, b[:, plan.inverse_image_order])
        for a, b in zip(old_predictions, new_predictions)
    )
    assert info["transformer_calls"] == old_info["transformer_calls"] == len(old_predictions)
    assert (
        info["image_reorder_calls"]
        == info["image_inverse_calls"]
        == info["image_id_reorder_calls"]
        == 0
    )
    assert (
        info["sampling_image_reorder_calls"]
        == info["sampling_id_reorder_calls"]
        == info["sampling_inverse_calls"]
        == 1
    )
    assert info["per_layer_image_permutations"] == 0
    assert torch.equal(source, kwargs["latents"])
    assert "prepare_latents" not in pipe.__dict__ and "_unpack_latents" not in pipe.__dict__


def test_native_noise_rng_same_seed_and_repeated_calls():
    pipe, plan, kwargs = fixture()
    kwargs.pop("latents")
    kwargs["output_type"] = "latent"
    with torch.no_grad(), persistent_flux_attention(pipe.transformer, plan):
        reference = pipe(**kwargs, generator=torch.Generator().manual_seed(31)).images
    for _ in range(2):
        actual, info = sample_flux_persistent(
            pipe,
            plan,
            backend="sdpa_reference",
            sdpa_backend="math",
            generator=torch.Generator().manual_seed(31),
            **kwargs,
        )
        assert torch.equal(reference, actual.images)
        assert info["sampling_inverse_calls"] == 1


def test_sampling_failure_restores_methods_processors_and_allows_retry():
    pipe, plan, kwargs = fixture()
    kwargs["output_type"] = "latent"
    prepare, unpack = pipe.prepare_latents, pipe._unpack_latents
    pipe.prepare_latents = prepare  # Existing instance overrides must be preserved.
    originals = [
        b.attn.processor
        for b in list(pipe.transformer.transformer_blocks)
        + list(pipe.transformer.single_transformer_blocks)
    ]

    def fail(*args):
        raise RuntimeError("injected sampling failure")

    handle = pipe.transformer.transformer_blocks[0].register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="injected sampling"):
        sample_flux_persistent(pipe, plan, backend="sdpa_reference", sdpa_backend="math", **kwargs)
    handle.remove()
    assert pipe.prepare_latents is prepare and pipe._unpack_latents == unpack
    assert not hasattr(pipe, "_flux_sampling_layout_active")
    assert not hasattr(pipe.transformer, "_flux_kernel_persistent_layout")
    assert originals == [
        b.attn.processor
        for b in list(pipe.transformer.transformer_blocks)
        + list(pipe.transformer.single_transformer_blocks)
    ]
    sample_flux_persistent(pipe, plan, backend="sdpa_reference", sdpa_backend="math", **kwargs)


def test_rejects_ambiguous_callbacks_and_stochastic_noise():
    pipe, plan, kwargs = fixture()
    with pytest.raises(ValueError, match="packed_callback"):
        sample_flux_persistent(pipe, plan, callback_on_step_end=lambda *args: {}, **kwargs)
    pipe.scheduler.register_to_config(stochastic_sampling=True)
    with pytest.raises(ValueError, match="Stochastic"):
        sample_flux_persistent(pipe, plan, **kwargs)
    assert not hasattr(pipe, "_flux_sampling_layout_active")


def test_callback_failure_and_nested_call_restore_state():
    pipe, plan, kwargs = fixture()
    kwargs["output_type"] = "latent"

    def callback(*unused):
        with pytest.raises(RuntimeError, match="Nested sampling"):
            sample_flux_persistent(pipe, plan, **kwargs)
        raise RuntimeError("injected callback failure")

    with pytest.raises(RuntimeError, match="injected callback"):
        sample_flux_persistent(
            pipe,
            plan,
            backend="sdpa_reference",
            sdpa_backend="math",
            packed_callback_on_step_end=callback,
            **kwargs,
        )
    assert "prepare_latents" not in pipe.__dict__ and "_unpack_latents" not in pipe.__dict__
    assert not hasattr(pipe, "_flux_sampling_layout_active")
