"""Exact top-K extension of original geometric block bridge; static CPU setup."""

from collections import Counter
from dataclasses import replace
import torch
from .plan import build_plan
from .geometry import neighbor_pairs


def bridge_k_plan(tiles=4, k=1, shared=False, *, grid_size=64):
    if isinstance(k, bool) or k not in (1, 2, 3, 4):
        raise ValueError("This bounded comparison supports integer K=1..4")
    plan = build_plan(tiles, grid_size=grid_size, shared=shared, bridge=False)
    n = plan.image_tokens
    b = plan.block_size
    nb = n // b
    g = plan.global_tokens // b
    inverse = torch.argsort(plan.original_order)
    physical = torch.argsort(plan.physical_original_blocks) + plan.text_tokens // b
    pairs = neighbor_pairs(plan.grid_size, 4)
    phases = []
    for phase, p in enumerate(plan.phases):
        labels = ((inverse - p.offset) % n) // (n // tiles)
        missing = pairs[labels[pairs[:, 0]] != labels[pairs[:, 1]]]
        counts = Counter()
        for a, c in (inverse[missing] // b).tolist():
            counts[a, c] += 1
            counts[c, a] += 1
        chosen = torch.full((nb, k), -1, dtype=torch.long)
        # Build block adjacency directly: never allocate an N-by-N token mask.
        total_blocks = plan.total_tokens // b
        adj = torch.zeros(total_blocks, total_blocks, dtype=torch.bool)
        adj[:g, :] = True
        for row, count in enumerate(p.kv_counts.tolist()):
            adj[g + row, p.kv_indices[row, :count].long()] = True
        for q in range(nb):
            options = sorted(
                [(key, score) for (query, key), score in counts.items() if query == q],
                key=lambda x: (-x[1], (x[0] - p.offset // b) % nb),
            )[:k]
            for j, (key, _) in enumerate(options):
                chosen[q, j] = key
                adj[physical[q], physical[key]] = True
        adj = adj | adj.T
        rows = [adj[q].nonzero().flatten().to(torch.int32) for q in range(g, len(adj))]
        sizes = torch.tensor([len(row) for row in rows], dtype=torch.int32)
        ids = torch.zeros(len(rows), int(sizes.max()), dtype=torch.int32)
        for q, row in enumerate(rows):
            ids[q, : len(row)] = row
        phases.append(replace(p, kv_indices=ids, kv_counts=sizes, bridge_original_blocks=chosen))
    return replace(plan, phases=tuple(phases), bridge=True)
