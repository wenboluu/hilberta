"""CPU tests of native Flux semantics and once-per-Transformer layout changes."""

from types import SimpleNamespace
import sys

import pytest
import torch
import torch.nn.functional as F
from diffusers.models.transformers.transformer_flux import FluxTransformer2DModel

from hilberta.flux_adapter import persistent_flux_attention
from tests.mask_reference import flux_mask_attention, flux_attention_backend


def toy_plan(full=False):
    order = torch.tensor([0, 5, 10, 15, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
    inverse = torch.argsort(order)
    phases = []
    for phase in range(4):
        rows = []
        for query in range(3):
            blocks = (
                set(range(5)) if full or phase == 3 else {0, 1, 2 + query, 2 + (query + phase) % 3}
            )
            rows.append(sorted(blocks))
        width = max(map(len, rows))
        ids = torch.full((3, width), -1, dtype=torch.int32)
        for i, row in enumerate(rows):
            ids[i, : len(row)] = torch.tensor(row, dtype=torch.int32)
        phases.append(
            SimpleNamespace(
                kv_indices=ids, kv_counts=torch.tensor(list(map(len, rows)), dtype=torch.int32)
            )
        )
    return SimpleNamespace(
        image_order=order,
        inverse_image_order=inverse,
        image_tokens=16,
        text_tokens=4,
        shared_tokens=4,
        global_tokens=8,
        total_tokens=20,
        block_size=4,
        phases=tuple(phases),
    )


def packed_mask(plan, phase):
    mask = torch.zeros(plan.total_tokens, plan.total_tokens, dtype=torch.bool)
    mask[: plan.global_tokens] = True
    table = plan.phases[phase]
    for i, count in enumerate(table.kv_counts.tolist()):
        queries = range(
            plan.global_tokens + i * plan.block_size, plan.global_tokens + (i + 1) * plan.block_size
        )
        for block in table.kv_indices[i, :count].tolist():
            keys = range(block * plan.block_size, (block + 1) * plan.block_size)
            for q in queries:
                for k in keys:
                    mask[q, k] = True
    return mask


def raster_image_masks(plan):
    # Scatter by token identities, independent of adapter inverse gathers.
    identity = torch.cat((torch.arange(plan.text_tokens), plan.text_tokens + plan.image_order))
    outputs = []
    for phase in range(4):
        dense = torch.zeros(plan.total_tokens, plan.total_tokens, dtype=torch.bool)
        dense[identity[:, None], identity[None, :]] = packed_mask(plan, phase)
        assert dense[: plan.text_tokens].all() and dense[:, : plan.text_tokens].all()
        outputs.append(dense[plan.text_tokens :, plan.text_tokens :])
    return torch.stack(outputs)


def model_and_args():
    torch.manual_seed(2707)
    model = FluxTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        num_single_layers=3,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        pooled_projection_dim=8,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    args = dict(
        hidden_states=torch.randn(2, 16, 4),
        encoder_hidden_states=torch.randn(2, 4, 12),
        pooled_projections=torch.randn(2, 8),
        timestep=torch.tensor([0.8, 0.8]),
        img_ids=torch.stack((torch.zeros(16), torch.arange(16) // 4, torch.arange(16) % 4), 1),
        txt_ids=torch.zeros(4, 3),
        return_dict=False,
    )
    return model, args


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("return_dict", [False, True])
def test_whole_transformer_matches_identity_mask_all_native_operations(full, return_dict):
    model, args = model_and_args()
    args["return_dict"] = return_dict
    plan = toy_plan(full)
    image_masks = raster_image_masks(plan)
    originals = [
        b.attn.processor
        for b in list(model.transformer_blocks) + list(model.single_transformer_blocks)
    ]
    with torch.no_grad(), flux_mask_attention(model, image_masks, "math"):
        reference = model(**args)
    snapshots = []
    # This hook predates the adapter and must still observe raster predictions.
    handle = model.register_forward_hook(
        lambda m, a, o: snapshots.append(o.sample if return_dict else o[0])
    )
    with torch.no_grad(), persistent_flux_attention(model, plan) as info:
        actual = model(**args)
        assert (
            info["transformer_calls"]
            == info["image_reorder_calls"]
            == info["image_inverse_calls"]
            == 1
        )
        assert info["image_id_reorder_calls"] == 1 and info["per_layer_image_permutations"] == 0
        assert info["layer_calls"] == [1] * 5 and info["layer_phases"] == [0, 1, 2, 3, 0]
        assert info["reference_masks_prepared"] == 4 and info["metadata_device_copies"] == 1
    handle.remove()
    reference = reference.sample if return_dict else reference[0]
    actual = actual.sample if return_dict else actual[0]
    torch.testing.assert_close(actual, reference, rtol=3e-5, atol=4e-6)
    torch.testing.assert_close(snapshots[0], actual, rtol=0, atol=0)
    assert originals == [
        b.attn.processor
        for b in list(model.transformer_blocks) + list(model.single_transformer_blocks)
    ]
    assert not hasattr(model, "_flux_kernel_persistent_layout")
    if full:
        with torch.no_grad(), flux_attention_backend("math"):
            native = model(**args)
        torch.testing.assert_close(
            actual, native.sample if return_dict else native[0], rtol=3e-5, atol=4e-6
        )


def test_entry_ids_positional_signature_input_immutability_and_two_calls():
    model, args = model_and_args()
    plan = toy_plan()
    input_copy = {
        key: value.clone() for key, value in args.items() if isinstance(value, torch.Tensor)
    }
    patches, positions = [], []
    h1 = model.x_embedder.register_forward_pre_hook(lambda m, a: patches.append(a[0].clone()))
    h2 = model.pos_embed.register_forward_pre_hook(lambda m, a: positions.append(a[0].clone()))
    with torch.no_grad(), persistent_flux_attention(model, plan) as info:
        keyword = model(**args)[0]
        # Native signature supports positional calls as well as pipeline kwargs.
        positional = model(
            args["hidden_states"],
            args["encoder_hidden_states"],
            args["pooled_projections"],
            args["timestep"],
            args["img_ids"].unsqueeze(0),
            args["txt_ids"].unsqueeze(0),
            return_dict=False,
        )[0]
        assert info["image_reorder_calls"] == info["image_inverse_calls"] == 2
        assert info["layer_calls"] == [2] * 5 and info["metadata_device_copies"] == 1
        expected_bytes = (
            2 * args["hidden_states"].numel() * args["hidden_states"].element_size() * 4
        )
        assert info["image_layout_gather_read_write_bytes"] == expected_bytes
    h1.remove()
    h2.remove()
    for patch, position in zip(patches, positions):
        torch.testing.assert_close(
            patch, args["hidden_states"][:, plan.image_order], rtol=0, atol=0
        )
        torch.testing.assert_close(
            position,
            torch.cat((args["txt_ids"], args["img_ids"][plan.image_order])),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(keyword, positional, rtol=0, atol=0)
    for key, value in input_copy.items():
        torch.testing.assert_close(args[key], value, rtol=0, atol=0)


def test_plan_metadata_are_immutable_copies_after_context_entry():
    model, args = model_and_args()
    plan = toy_plan()
    with torch.no_grad(), persistent_flux_attention(model, plan):
        first = model(**args)[0]
        plan.image_order.zero_()
        plan.inverse_image_order.zero_()
        for phase in plan.phases:
            phase.kv_indices.fill_(-1)
            phase.kv_counts.zero_()
        second = model(**args)[0]
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def test_restoration_and_context_reuse_after_native_forward_failure():
    model, args = model_and_args()
    plan = toy_plan()
    originals = [
        b.attn.processor
        for b in list(model.transformer_blocks) + list(model.single_transformer_blocks)
    ]

    def fail(*unused):
        raise RuntimeError("injected block failure")

    with torch.no_grad(), persistent_flux_attention(model, plan) as info:
        handle = model.transformer_blocks[0].register_forward_pre_hook(fail)
        with pytest.raises(RuntimeError, match="injected"):
            model(**args)
        handle.remove()
        model(**args)
        assert info["transformer_calls"] == info["image_reorder_calls"] == 2
        assert info["image_inverse_calls"] == 1
        with pytest.raises(RuntimeError, match="Nested"):
            with persistent_flux_attention(model, plan):
                pass
    assert originals == [
        b.attn.processor
        for b in list(model.transformer_blocks) + list(model.single_transformer_blocks)
    ]
    assert not hasattr(model, "_flux_kernel_persistent_layout")


@pytest.mark.parametrize(
    "bad_input,match",
    [
        ("controlnet_block_samples", "ControlNet"),
        ("controlnet_single_block_samples", "ControlNet"),
        ("joint_attention_kwargs", "IP-adapter"),
        ("hidden_states", "image-token"),
        ("encoder_hidden_states", "text-token"),
        ("img_ids", "img_ids"),
    ],
)
def test_unsupported_or_inconsistent_inputs_fail_before_attention(bad_input, match):
    model, args = model_and_args()
    plan = toy_plan()
    bad = dict(args)
    if bad_input.startswith("controlnet"):
        bad[bad_input] = [torch.zeros(2, 16, 16)]
    elif bad_input == "joint_attention_kwargs":
        bad[bad_input] = {"ip_adapter_image_embeds": torch.zeros(1)}
    else:
        bad[bad_input] = (
            args[bad_input][..., :-1, :] if bad_input == "img_ids" else args[bad_input][:, :-1]
        )
    with torch.no_grad(), persistent_flux_attention(model, plan) as info:
        with pytest.raises(ValueError, match=match):
            model(**bad)
        assert info["layer_calls"] == [0] * 5
        model(**args)
        assert info["layer_calls"] == [1] * 5


def test_sparse_dispatch_contract_uses_native_transpose_views(monkeypatch):
    model, args = model_and_args()
    plan = toy_plan()
    with torch.no_grad(), persistent_flux_attention(model, plan):
        expected = model(**args)[0]
    calls = []

    def fake_kernel(q, k, v, ids, counts, global_tokens, block_size, **options):
        # CPU dispatch-contract oracle only; the real CUDA kernel has separate
        # correctness gates. This checks adapter layout/dtype/metadata plumbing.
        phase = len(calls) % 4
        assert q.stride(-1) == k.stride(-1) == v.stride(-1) == 1
        assert not q.is_contiguous() and q.stride(1) == q.shape[-1]
        assert ids.dtype == counts.dtype == torch.int32
        assert torch.equal(ids, plan.phases[phase].kv_indices)
        assert torch.equal(counts, plan.phases[phase].kv_counts)
        assert global_tokens == 8 and block_size == 4 and options == {"global_strategy": "flash"}
        calls.append(tuple(q.stride()))
        return F.scaled_dot_product_attention(q, k, v, attn_mask=packed_mask(plan, phase))

    monkeypatch.setitem(
        sys.modules, "hilberta.sparse_kernel", SimpleNamespace(sparse_attention=fake_kernel)
    )
    with (
        torch.no_grad(),
        persistent_flux_attention(
            model, plan, backend="sparse", kernel_options={"global_strategy": "flash"}
        ) as info,
    ):
        actual = model(**args)[0]
        assert info["reference_masks_prepared"] == 0
    assert len(calls) == 5
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_invalid_duplicate_metadata_and_grad_mode_rejected():
    model, args = model_and_args()
    plan = toy_plan()
    plan.phases[0].kv_indices[0, 1] = plan.phases[0].kv_indices[0, 0]
    with pytest.raises(ValueError, match="unique"):
        with persistent_flux_attention(model, plan):
            pass
    with persistent_flux_attention(model, toy_plan()):
        with pytest.raises(RuntimeError, match="inference-only"):
            model(**args)


@pytest.mark.parametrize("tiles", [4, 16])
def test_real_builder_shared_bridge_plan_matches_native_raster_reference(tiles):
    from hilberta.plan import build_plan, full_attention_plan

    plan = build_plan(tiles, grid_size=16, text_tokens=16, shared=True, bridge=True, block_size=4)
    torch.manual_seed(2707)
    model = FluxTransformer2DModel(
        in_channels=4,
        out_channels=4,
        num_layers=2,
        num_single_layers=3,
        attention_head_dim=8,
        num_attention_heads=2,
        joint_attention_dim=12,
        pooled_projection_dim=8,
        axes_dims_rope=(2, 2, 4),
    ).eval()
    args = dict(
        hidden_states=torch.randn(1, 256, 4),
        encoder_hidden_states=torch.randn(1, 16, 12),
        pooled_projections=torch.randn(1, 8),
        timestep=torch.tensor([0.8]),
        img_ids=torch.stack((torch.zeros(256), torch.arange(256) // 16, torch.arange(256) % 16), 1),
        txt_ids=torch.zeros(16, 3),
        return_dict=False,
    )
    identity = torch.cat((torch.arange(16), 16 + plan.image_order))
    masks = []
    for phase in range(4):
        dense = torch.zeros(272, 272, dtype=torch.bool)
        dense[identity[:, None], identity[None, :]] = plan.dense_mask(phase)
        masks.append(dense[16:, 16:])
    with torch.no_grad(), flux_mask_attention(model, torch.stack(masks), "math"):
        reference = model(**args)[0]
    with torch.no_grad(), persistent_flux_attention(model, plan) as info:
        actual = model(**args)[0]
    torch.testing.assert_close(actual, reference, rtol=4e-5, atol=5e-6)
    assert info["layer_calls"] == [1] * 5
    with torch.no_grad(), flux_attention_backend("math"):
        native_full = model(**args)[0]
    with torch.no_grad(), persistent_flux_attention(model, full_attention_plan(plan)):
        reordered_full = model(**args)[0]
    torch.testing.assert_close(reordered_full, native_full, rtol=4e-5, atol=5e-6)
