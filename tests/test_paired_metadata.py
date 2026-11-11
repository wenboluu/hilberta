"""Paired-query metadata must preserve every original64x64 adjacency bit."""

import unittest
import torch
from hilberta.plan import build_plan
from hilberta.paired_metadata import pair_metadata, unpair_metadata


class PairedMetadataTests(unittest.TestCase):
    def test_all_plans_round_trip(self):
        torch.set_num_threads(2)
        for tiles in (4, 16):
            for shared in (False, True):
                for bridge in (False, True):
                    plan = build_plan(tiles, shared=shared, bridge=bridge)
                    for phase in plan.phases:
                        expected = [
                            r[: int(n)].tolist() for r, n in zip(phase.kv_indices, phase.kv_counts)
                        ]
                        for size in (64, 128):
                            with self.subTest(
                                tiles=tiles,
                                shared=shared,
                                bridge=bridge,
                                offset=phase.offset,
                                key_block_size=size,
                            ):
                                meta = pair_metadata(
                                    phase.kv_indices,
                                    phase.kv_counts,
                                    tokens=plan.total_tokens,
                                    global_tokens=plan.global_tokens,
                                    key_block_size=size,
                                )
                                self.assertEqual(unpair_metadata(meta), expected)
                                directed = (
                                    sum(
                                        int(f).bit_count()
                                        for r, n in zip(meta.kv_flags, meta.kv_counts)
                                        for f in r[: int(n)]
                                    )
                                    * 4096
                                )
                                self.assertEqual(directed, int(phase.kv_counts.sum()) * 4096)

    def test_different_query_and_key_halves(self):
        # Direct six-block example: first two blocks global, four ordinary rows.
        ids = torch.tensor(
            [[0, 1, 2, 4], [0, 1, 3, 5], [0, 1, 4, 5], [0, 1, 4, 5]], dtype=torch.int32
        )
        counts = torch.full((4,), 4, dtype=torch.int32)
        meta = pair_metadata(ids, counts, tokens=384, global_tokens=128, key_block_size=128)
        self.assertEqual(meta.kv_indices[0, :3].tolist(), [0, 1, 2])
        self.assertEqual(meta.kv_flags[0, :3].tolist(), [15, 9, 9])
        self.assertEqual(unpair_metadata(meta), ids.tolist())


if __name__ == "__main__":
    unittest.main()
