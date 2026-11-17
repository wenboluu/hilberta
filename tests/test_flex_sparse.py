"""CPU checks: block adjacency equals the training mask authority; BlockMask is exact; packing round-trips."""

import unittest, torch
from hilberta.bridge_k import bridge_k_plan
from tests.helpers import raster_image_masks
from hilberta.flex_sparse import block_adjacency, build_block_mask, FlexPlanBank


class FlexSparseTopology(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plans = {
            (16, 3, 64): bridge_k_plan(16, 3, False, grid_size=64),
            (4, 1, 64): bridge_k_plan(4, 1, False),
        }

    def test_adjacency_matches_raster_authority(self):
        for (tiles, k, grid), plan in self.plans.items():
            with self.subTest(tiles=tiles, k=k, grid=grid):
                raster = raster_image_masks(plan)
                t, b = plan.text_tokens, plan.block_size
                inverse = plan.inverse_image_order
                for phase in range(4):
                    adj = block_adjacency(plan, phase)
                    dense = adj.repeat_interleave(b, 0).repeat_interleave(b, 1)
                    self.assertTrue(bool(dense[: t // b * b].all()))
                    image = dense[t:, t:][inverse][:, inverse]
                    self.assertTrue(torch.equal(image, raster[phase]))
                    self.assertTrue(torch.equal(adj[t // b :, t // b :], adj[t // b :, t // b :].T))

    def test_block_mask_exact_on_cpu(self):
        plan = self.plans[(4, 1, 64)]
        for phase in range(4):
            adj = block_adjacency(plan, phase)
            for mask_block in (64, 128):
                bm = build_block_mask(adj, "cpu", 64, mask_block)
                dense = bm.to_dense()[0, 0].bool()
                coarse = (
                    adj.reshape(
                        adj.shape[0] * 64 // mask_block, mask_block // 64, -1, mask_block // 64
                    )
                    .any(3)
                    .any(1)
                )
                self.assertTrue(torch.equal(dense, coarse))

    def test_packing_round_trip(self):
        plan = self.plans[(4, 1, 64)]
        bank = FlexPlanBank(plan)
        x = torch.randn(1, plan.total_tokens, 2, 4)
        packed = x.index_select(1, bank.index)
        self.assertTrue(torch.equal(packed.index_select(1, bank.inverse), x))
        self.assertTrue(torch.equal(packed[:, : plan.text_tokens], x[:, : plan.text_tokens]))
        self.assertTrue(
            torch.equal(bank.index[plan.text_tokens :] - plan.text_tokens, plan.image_order)
        )


if __name__ == "__main__":
    unittest.main()


class PersistentHooksCPU(unittest.TestCase):
    """Entry reorder / exit restore with a stub Transformer; no CUDA, no FLUX weights."""

    def test_round_trip_and_gradient(self):
        from hilberta.flex_sparse import FlexPlanBank, _PersistentHooks
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        plan = bridge_k_plan(4, 1, False)
        bank = FlexPlanBank(plan)
        seen = {}

        class Stub(torch.nn.Module):
            def forward(
                self, hidden_states, encoder_hidden_states, img_ids, txt_ids=None, return_dict=True
            ):
                seen["hidden"] = hidden_states
                seen["ids"] = img_ids
                out = hidden_states * 2 + img_ids[..., 1:2].to(hidden_states.dtype)
                return Transformer2DModelOutput(sample=out) if return_dict else (out,)

        m = Stub()
        hooks = _PersistentHooks(m, bank)
        h1 = m.register_forward_pre_hook(hooks.before, with_kwargs=True)
        h2 = m.register_forward_hook(hooks.after, prepend=True, always_call=True)
        x = torch.randn(1, plan.image_tokens, 3, requires_grad=True)
        text = torch.randn(1, plan.text_tokens, 3)
        ids = torch.stack(
            [
                torch.zeros(plan.image_tokens),
                torch.arange(plan.image_tokens) // 64,
                torch.arange(plan.image_tokens) % 64,
            ],
            -1,
        )
        for return_dict in (True, False):
            out = m(
                hidden_states=x, encoder_hidden_states=text, img_ids=ids, return_dict=return_dict
            )
            out = out.sample if return_dict else out[0]
            self.assertTrue(
                torch.equal(seen["hidden"], x.detach().index_select(1, plan.image_order))
            )
            self.assertTrue(torch.equal(seen["ids"], ids.index_select(0, plan.image_order)))
            self.assertTrue(torch.allclose(out, x * 2 + ids[:, 1:2]))  # restored to raster order
            self.assertFalse(bank.in_forward)
        out.sum().backward()
        self.assertTrue(torch.equal(x.grad, torch.full_like(x, 2.0)))
        self.assertEqual((bank.entry_reorders, bank.exit_restores), (2, 2))
        h1.remove()
        h2.remove()
        self.assertTrue(
            torch.equal(
                m(hidden_states=x, encoder_hidden_states=text, img_ids=ids).sample,
                x * 2 + ids[:, 1:2],
            )
        )
