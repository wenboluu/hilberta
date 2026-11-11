import unittest, torch
from hilberta.bridge_k import bridge_k_plan
from hilberta.plan import build_plan
from tests.helpers import raster_image_masks
from tests.topology_oracle import base_masks, augmentation_masks


class BridgeKTests(unittest.TestCase):
    def test_k1_exact_and_k2_oracle(self):
        torch.set_num_threads(2)
        for tiles in (4, 16):
            for shared in (False, True):
                old = raster_image_masks(build_plan(tiles, shared=shared))
                old = old | old.transpose(-1, -2)
                self.assertTrue(
                    torch.equal(raster_image_masks(bridge_k_plan(tiles, 1, shared)), old)
                )
                plan = bridge_k_plan(tiles, 2, shared)
                actual = raster_image_masks(plan)
                oracle = base_masks(plan.original_order, tiles) | augmentation_masks(
                    plan.original_order,
                    64,
                    tiles,
                    "block_bridge",
                    block_size=64,
                    max_extra_blocks=2,
                )
                if shared:
                    ids = plan.image_order[: plan.shared_tokens]
                    oracle[:, ids, :] = True
                    oracle[:, :, ids] = True
                oracle = oracle | oracle.transpose(-1, -2)
                self.assertTrue(torch.equal(actual, oracle))
                self.assertFalse((old & ~actual).any())
                self.assertTrue(torch.equal(actual, actual.transpose(-1, -2)))
                for phase in range(4):
                    mask = plan.dense_mask(phase)
                    self.assertTrue(mask[: plan.global_tokens].all())
                    self.assertTrue(mask[:, : plan.global_tokens].all())

    def test_k3_k4_no_shared_oracle(self):
        torch.set_num_threads(2)
        for tiles in (4, 16):
            previous = raster_image_masks(bridge_k_plan(tiles, 2, False))
            for k in (3, 4):
                plan = bridge_k_plan(tiles, k, False)
                actual = raster_image_masks(plan)
                oracle = base_masks(plan.original_order, tiles) | augmentation_masks(
                    plan.original_order,
                    64,
                    tiles,
                    "block_bridge",
                    block_size=64,
                    max_extra_blocks=k,
                )
                oracle = oracle | oracle.transpose(-1, -2)
                self.assertTrue(torch.equal(actual, oracle))
                self.assertFalse((previous & ~actual).any())
                self.assertFalse(actual.all(-1).any())
                self.assertEqual(plan.shared_tokens, 0)
                for phase in range(4):
                    mask = plan.dense_mask(phase)
                    self.assertTrue(mask[:512].all())
                    self.assertTrue(mask[:, :512].all())
                previous = actual

    def test_bad_k(self):
        with self.assertRaises(ValueError):
            bridge_k_plan(16, 0)


if __name__ == "__main__":
    unittest.main()
