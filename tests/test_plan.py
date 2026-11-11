"""Independent dense raster oracle, including phase-dependent bridge ties."""

import unittest
import torch
from hilberta.plan import build_plan, full_attention_plan
from tests.topology_oracle import base_masks, augmentation_masks


class PlanTests(unittest.TestCase):
    def test_exact_existing_raster_topology(self):
        torch.set_num_threads(4)
        for grid, block, text in [(16, 4, 16), (64, 64, 512)]:
            for tiles in [4, 16]:
                for shared in [False, True]:
                    for bridge in [False, True]:
                        with self.subTest(grid=grid, tiles=tiles, shared=shared, bridge=bridge):
                            p = build_plan(tiles, grid, text, shared, bridge, block)
                            oracle = base_masks(p.original_order, tiles)
                            if bridge:
                                oracle |= augmentation_masks(
                                    p.original_order,
                                    grid,
                                    tiles,
                                    "block_bridge",
                                    block_size=block,
                                    max_extra_blocks=1,
                                )
                            if shared:
                                ids = (
                                    torch.arange(grid**2)
                                    .reshape(grid, grid)[
                                        3 * grid // 8 : 5 * grid // 8, 3 * grid // 8 : 5 * grid // 8
                                    ]
                                    .flatten()
                                )
                                oracle[:, ids, :] = True
                                oracle[:, :, ids] = True
                            for phase in range(4):
                                mask = p.dense_mask(phase)
                                self.assertTrue(mask[:text].all())
                                self.assertTrue(mask[:, :text].all())
                                self.assertTrue(
                                    torch.equal(
                                        mask[text:, text:],
                                        oracle[phase][p.image_order][:, p.image_order],
                                    )
                                )
                            self.assertTrue(
                                torch.equal(
                                    p.image_order[p.inverse_image_order], torch.arange(grid**2)
                                )
                            )

    def test_known_bridge_budget(self):
        for tiles, expected in [(4, [24, 34, 32, 34]), (16, [56, 57, 58, 55])]:
            a, b = build_plan(tiles, bridge=False), build_plan(tiles)
            delta = [int((b.dense_mask(i) & ~a.dense_mask(i)).sum()) // 4096 for i in range(4)]
            self.assertEqual(delta, expected)
            self.assertEqual(a.physical_original_blocks[:4].tolist(), [15, 16, 47, 48])

    def test_full_layout_gate(self):
        p = full_attention_plan(build_plan(16, 16, 16, True, True, 4))
        self.assertTrue(all(p.dense_mask(i).all() for i in range(4)))

    def test_unaligned_rejected(self):
        with self.assertRaises(ValueError):
            build_plan(16, block_size=128)
        with self.assertRaises(ValueError):
            build_plan(text_tokens=17)


if __name__ == "__main__":
    unittest.main()
