"""Ordering-independent connectivity analysis and explicit augmentation masks.

Token identity is raster y * grid_size + x; order[position] is a token ID.
Every returned attention mask is CPU bool [phase, query ID, key ID]. True
means accessible. Augmentation masks contain EXTRA entries only, disjoint
from base, with no duplicate softmax contributions. Offsets use exactly the
ordering-search convention: ((inverse_order - offset) % N) // tile_size.

No seam coordinates are built in. full_strip infers cuts from permanent
four-neighbor defects, or accepts explicit cuts for legacy-reference adapters.
"""

from __future__ import annotations

from collections import Counter
import math
import torch


def _order(order):
    raw = torch.as_tensor(order, device="cpu")
    if raw.ndim != 1 or raw.dtype == torch.bool or raw.is_complex():
        raise ValueError("order must be a one-dimensional integer permutation")
    result = raw.to(torch.long)
    if not torch.equal(raw, result.to(raw.dtype)):
        raise ValueError("order must contain integer token IDs")
    n = result.numel()
    if n == 0 or not torch.equal(result.sort().values, torch.arange(n)):
        raise ValueError("order must contain every token ID exactly once")
    return result


def _setup(order, tile_count, offsets=None, grid_size=None):
    order = _order(order)
    n = len(order)
    if grid_size is not None and (int(grid_size) != grid_size or int(grid_size) ** 2 != n):
        raise ValueError("grid_size squared must equal permutation length")
    if int(tile_count) != tile_count or tile_count <= 0 or n % tile_count:
        raise ValueError("tile_count must be a positive divisor of token count")
    width = n // int(tile_count)
    if offsets is None:
        if width % 4:
            raise ValueError("default four-phase schedule requires tile size divisible by four")
        offsets = [0, width // 4, width // 2, 3 * width // 4]
    offsets = list(offsets)
    if not offsets or any(int(x) != x for x in offsets):
        raise ValueError("offsets must be a nonempty integer sequence")
    offsets = [int(x) % n for x in offsets]
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(n)
    labels = torch.stack([((inverse - offset) % n) // width for offset in offsets])
    return order, inverse, labels, width, offsets


def base_masks(order, tile_count, offsets=None):
    """Build all explicit base masks in token-identity space."""
    _, _, labels, _, _ = _setup(order, tile_count, offsets)
    return labels[:, :, None] == labels[:, None, :]


def neighbor_pairs(grid_size, connectivity=4):
    """Unique undirected non-periodic 2D neighbors; each pair is ascending."""
    if int(grid_size) != grid_size or grid_size < 1 or connectivity not in (4, 8):
        raise ValueError("positive integer grid_size and connectivity 4 or 8 required")
    g = int(grid_size)
    directions = [(0, 1), (1, 0)]
    if connectivity == 8:
        directions += [(1, 1), (1, -1)]
    pairs = []
    for y in range(g):
        for x in range(g):
            for dy, dx in directions:
                yy, xx = y + dy, x + dx
                if 0 <= yy < g and 0 <= xx < g:
                    pairs.append((y * g + x, yy * g + xx))
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2)


def _pair_analysis(pairs, labels, inverse, grid_size):
    connected = labels[:, pairs[:, 0]] == labels[:, pairs[:, 1]]
    count = connected.sum(0)
    distance = (inverse[pairs[:, 0]] - inverse[pairs[:, 1]]).abs()
    distance = torch.minimum(distance, len(inverse) - distance)
    never = count == 0
    heat = torch.zeros(len(inverse), dtype=torch.long)
    heat.scatter_add_(0, pairs[never].flatten(), torch.ones(int(never.sum()) * 2, dtype=torch.long))
    missed_heat = torch.zeros(len(inverse), dtype=torch.float64)
    missed_heat.scatter_add_(
        0, pairs.flatten(), (1 - count.double() / len(labels)).repeat_interleave(2)
    )
    return {
        "pairs": pairs.tolist(),
        "connected_by_phase": connected.tolist(),
        "direct_connect_count": count.tolist(),
        "direct_connect_frequency": (count.double() / len(labels)).tolist(),
        "never_connected_pairs": pairs[never].tolist(),
        "never_connected_cyclic_distance": distance[never].tolist(),
        "cyclic_distance": distance.tolist(),
        "defect_endpoint_heatmap": heat.reshape(grid_size, grid_size).tolist(),
        "omission_frequency_endpoint_heatmap": missed_heat.reshape(grid_size, grid_size).tolist(),
        "summary": {
            "pair_count": len(pairs),
            "never_connected_pair_count": int(never.sum()),
            "connected_pair_count_by_phase": connected.sum(1).tolist(),
            "mean_direct_connect_frequency": float(count.double().mean() / len(labels))
            if len(pairs)
            else 1.0,
        },
    }


def analyze(order, grid_size, tile_count, offsets=None, teacher_weights=None):
    """JSON-ready geometry and optional weighted directed graph analysis.

    teacher_weights is finite, nonnegative [N,N] or [P,N,N]; it need not
    be symmetric or row-normalized. Mass numbers are sums of supplied
    weights, not assumed attention probabilities. Missing-frequency mass
    averages over phases; permanent mass sums only pairs missing in all
    phases (averaging weights when phase-specific). Self entries are allowed.
    """
    order, inverse, labels, width, offsets = _setup(order, tile_count, offsets, grid_size)
    grid_size = int(grid_size)
    out = {
        "grid_size": grid_size,
        "token_count": len(order),
        "tile_count": int(tile_count),
        "tile_size": width,
        "offsets": offsets,
        "four_neighbor": _pair_analysis(neighbor_pairs(grid_size, 4), labels, inverse, grid_size),
        "eight_neighbor": _pair_analysis(neighbor_pairs(grid_size, 8), labels, inverse, grid_size),
    }
    if teacher_weights is not None:
        w = torch.as_tensor(teacher_weights, device="cpu")
        n, phases = len(order), len(offsets)
        if w.shape not in ((n, n), (phases, n, n)):
            raise ValueError("teacher_weights must have shape [N,N] or [phase,N,N]")
        if w.is_complex() or not torch.isfinite(w).all() or (w < 0).any():
            raise ValueError("teacher weights must be finite and nonnegative")
        permanent = torch.ones((n, n), dtype=torch.bool)
        omitted, totals, omitted_queries = [], [], []
        mean_weights = w.double() if w.ndim == 2 else w.double().mean(0)
        for phase in range(phases):
            missing = labels[phase, :, None] != labels[phase, None, :]
            weights = w if w.ndim == 2 else w[phase]
            weights = weights.double()
            permanent &= missing
            totals.append(float(weights.sum()))
            om = weights * missing
            omitted.append(float(om.sum()))
            omitted_queries.append(om.sum(-1))
        perm_queries = (mean_weights * permanent).sum(-1)
        out["teacher"] = {
            "weight_semantics": "directed raw mass; query is row; key is column",
            "total_mass_by_phase": totals,
            "omitted_mass_by_phase": omitted,
            "omitted_mass_fraction_by_phase": [
                a / b if b else 0.0 for a, b in zip(omitted, totals)
            ],
            "mean_phase_omitted_mass": sum(omitted) / phases,
            "permanently_omitted_mean_mass": float(perm_queries.sum()),
            "permanently_omitted_mass_fraction": float(perm_queries.sum() / mean_weights.sum())
            if mean_weights.sum()
            else 0.0,
            "omitted_mass_query_heatmap_by_phase": torch.stack(omitted_queries)
            .reshape(phases, grid_size, grid_size)
            .tolist(),
            "permanent_omitted_mass_query_heatmap": perm_queries.reshape(
                grid_size, grid_size
            ).tolist(),
        }
    return out


def _defects(labels, grid_size):
    pairs = neighbor_pairs(grid_size, 4)
    connected = labels[:, pairs[:, 0]] == labels[:, pairs[:, 1]]
    return pairs[~connected.any(0)]


def _positive_int(value, name, allow_zero=False):
    if int(value) != value or value < (0 if allow_zero else 1):
        raise ValueError(f"{name} must be an integer >= {0 if allow_zero else 1}")
    return int(value)


def _ranked_cuts(defects, grid_size):
    counts = Counter()
    for first, second in defects.tolist():
        y, x = divmod(first, grid_size)
        yy, xx = divmod(second, grid_size)
        counts[("vertical", max(x, xx)) if y == yy else ("horizontal", max(y, yy))] += 1
    return sorted(counts, key=lambda cut: (-counts[cut], cut))


def _cross(mask, a, b):
    mask[a[:, None], b[None, :]] = True
    mask[b[:, None], a[None, :]] = True


def _patch_groups(grid_size, patch_size):
    grid = torch.arange(grid_size**2).reshape(grid_size, grid_size)
    groups, membership = [], torch.empty(grid_size**2, dtype=torch.long)
    for y in range(0, grid_size, patch_size):
        for x in range(0, grid_size, patch_size):
            group = grid[y : y + patch_size, x : x + patch_size].flatten()
            membership[group] = len(groups)
            groups.append(group)
    return groups, membership


def augmentation_masks(order, grid_size, tile_count, kind, offsets=None, **params):
    """Return explicit EXTRA-only masks, with deterministic regular topologies.

    none: no extra edges.
    neighbor4: symmetric permanent four-neighbor defects; max_edges optionally
      selects a deterministic prefix of complete undirected edges.
    full_strip: inferred grid cuts; full opposite-side band cross products.
      band defaults 1; max_boundaries limits ranked cuts. Explicit
      vertical_seams/horizontal_seams reproduces a supplied legacy reference.
    patch_bridge: full symmetric cross products of aligned spatial patches
      touched by permanent defects (same-patch defects add that patch square).
      patch_size defaults 2; max_bridges limits defect-count-ranked patch pairs.
    halo: each base query tile accesses width=halo keys before/after that
      tile in the cyclic phase sequence; this is directed K/V expansion.
    block_bridge: each aligned sequence query block accesses up to
      max_extra_blocks full aligned K/V blocks ranked by currently missing
      4-neighbor connections; block_size defaults 16 and must divide tile.
      Selection is directed independently per query block.

    Every topology subtracts existing base edges. Selection never slices a
    mask's rows to hit a budget, preserving the specified topology semantics.
    """
    order, inverse, labels, tile_size, offsets = _setup(order, tile_count, offsets, grid_size)
    grid_size = int(grid_size)
    n, phases = len(order), len(offsets)
    base = labels[:, :, None] == labels[:, None, :]
    extra = torch.zeros_like(base)
    aliases = {
        "4_neighbor": "neighbor4",
        "four_neighbor": "neighbor4",
        "local_patch": "patch_bridge",
        "sequence_halo": "halo",
        "block_level": "block_bridge",
        "full_strip_reference": "full_strip",
    }
    kind = aliases.get(kind, kind)
    if kind == "none":
        return extra
    if kind in ("neighbor4", "full_strip", "patch_bridge"):
        defects = _defects(labels, grid_size)
        topology = torch.zeros((n, n), dtype=torch.bool)
        if kind == "neighbor4":
            limit = _positive_int(params.get("max_edges", len(defects)), "max_edges", True)
            chosen = defects[:limit]
            topology[chosen[:, 0], chosen[:, 1]] = True
            topology[chosen[:, 1], chosen[:, 0]] = True
        elif kind == "full_strip":
            band = _positive_int(params.get("band", 1), "band", True)
            if "vertical_seams" in params or "horizontal_seams" in params:
                cuts = [("vertical", int(x)) for x in params.get("vertical_seams", [])]
                cuts += [("horizontal", int(y)) for y in params.get("horizontal_seams", [])]
                if any(not 0 < c < grid_size for _, c in cuts):
                    raise ValueError("explicit seam cuts must be interior grid coordinates")
                cuts = list(dict.fromkeys(cuts))
            else:
                cuts = _ranked_cuts(defects, grid_size)
            limit = _positive_int(params.get("max_boundaries", len(cuts)), "max_boundaries", True)
            grid = torch.arange(n).reshape(grid_size, grid_size)
            for orientation, cut in cuts[:limit]:
                if orientation == "vertical":
                    a, b = (
                        grid[:, max(0, cut - band) : cut],
                        grid[:, cut : min(grid_size, cut + band)],
                    )
                else:
                    a, b = (
                        grid[max(0, cut - band) : cut, :],
                        grid[cut : min(grid_size, cut + band), :],
                    )
                _cross(topology, a.flatten(), b.flatten())
        else:
            patch_size = _positive_int(params.get("patch_size", 2), "patch_size")
            groups, membership = _patch_groups(grid_size, patch_size)
            counts = Counter(
                tuple(sorted((int(membership[a]), int(membership[b])))) for a, b in defects.tolist()
            )
            bridges = sorted(counts, key=lambda pair: (-counts[pair], pair))
            limit = _positive_int(params.get("max_bridges", len(bridges)), "max_bridges", True)
            for a, b in bridges[:limit]:
                _cross(topology, groups[a], groups[b])
        return topology.unsqueeze(0).expand(phases, -1, -1) & ~base
    if kind == "halo":
        halo = _positive_int(params.get("halo", params.get("width", 16)), "halo", True)
        # Saturate at N, avoiding duplicate work for overlarge halo requests.
        halo = min(halo, n)
        for phase, offset in enumerate(offsets):
            sequence = order.roll(-offset)
            for tile in range(int(tile_count)):
                begin, end = tile * tile_size, (tile + 1) * tile_size
                queries = sequence[begin:end]
                key_positions = (
                    torch.cat((torch.arange(begin - halo, begin), torch.arange(end, end + halo)))
                    % n
                )
                keys = sequence[key_positions]
                extra[phase, queries[:, None], keys[None, :]] = True
        return extra & ~base
    if kind == "block_bridge":
        block_size = _positive_int(params.get("block_size", 16), "block_size")
        if tile_size % block_size:
            raise ValueError("block_size must divide base tile_size")
        count = _positive_int(params.get("max_extra_blocks", 1), "max_extra_blocks", True)
        pairs = neighbor_pairs(grid_size, 4)
        for phase, offset in enumerate(offsets):
            blocks = order.roll(-offset).reshape(-1, block_size)
            block_ids = ((inverse - offset) % n) // block_size
            missing = pairs[labels[phase, pairs[:, 0]] != labels[phase, pairs[:, 1]]]
            scores = Counter()
            for a, b in missing.tolist():
                ba, bb = int(block_ids[a]), int(block_ids[b])
                scores[(ba, bb)] += 1
                scores[(bb, ba)] += 1
            options = {}
            for (query, key), score in scores.items():
                options.setdefault(query, []).append((key, score))
            for query, choices in options.items():
                for key, _ in sorted(choices, key=lambda item: (-item[1], item[0]))[:count]:
                    extra[phase, blocks[query, :, None], blocks[key, None, :]] = True
        return extra & ~base
    raise ValueError(f"unknown augmentation kind {kind!r}")


def topology_stats(order, tile_count, extra_masks, offsets=None, block_size=16):
    """Budget and representation counts; not latency predictions.

    Segment count is the sum of nonempty contiguous K/V runs PER QUERY in
    phase-rolled sequence layout (no cyclic joining across array ends).
    Block count is nonempty query-block x key-block rectangles. Alignment
    waste counts rectangle slots not requested by this extra-only topology;
    padding at final partial blocks is included, so it is physical capacity.
    """
    order, _, labels, width, offsets = _setup(order, tile_count, offsets)
    masks = torch.as_tensor(extra_masks, dtype=torch.bool, device="cpu")
    n, phases = len(order), len(offsets)
    if masks.shape != (phases, n, n):
        raise ValueError("extra_masks must have shape [phase,N,N]")
    if (masks & (labels[:, :, None] == labels[:, None, :])).any():
        raise ValueError("extra_masks overlaps existing base edges")
    block_size = _positive_int(block_size, "block_size")
    records = []
    for phase, offset in enumerate(offsets):
        sequence = order.roll(-offset)
        mask = masks[phase][sequence][:, sequence]
        keys = mask.sum(-1)
        starts = mask & ~torch.cat((torch.zeros(n, 1, dtype=torch.bool), mask[:, :-1]), dim=-1)
        ends = mask & ~torch.cat((mask[:, 1:], torch.zeros(n, 1, dtype=torch.bool)), dim=-1)
        start_positions = starts.nonzero()[:, 1]
        end_positions = ends.nonzero()[:, 1]
        lengths = end_positions - start_positions + 1
        block_count = math.ceil(n / block_size)
        padding = block_count * block_size - n
        padded = torch.nn.functional.pad(mask, (0, padding, 0, padding)) if padding else mask
        query_kv_visits = padded[:n].reshape(n, block_count, block_size).any(2)
        touched_blocks = (
            padded.reshape(block_count, block_size, block_count, block_size).any(3).any(1)
        )
        entries = int(keys.sum())
        records.append(
            {
                "phase": phase,
                "offset": offset,
                "extra_directed_entries": entries,
                "max_extra_keys_per_query": int(keys.max()),
                "queries_touched": int((keys > 0).sum()),
                "logical_segments_per_query_sum": int(starts.sum()),
                "segment_length_histogram": dict(sorted(Counter(lengths.tolist()).items())),
                "mean_segment_length": float(lengths.float().mean()) if len(lengths) else 0.0,
                "kv_blocks_per_query_sum": int(query_kv_visits.sum()),
                "kv_block_alignment_waste_per_query_sum": int(query_kv_visits.sum()) * block_size
                - entries,
                "query_key_blocks": int(touched_blocks.sum()),
                "max_extra_kv_blocks_per_query_block": int(touched_blocks.sum(1).max()),
                "block_alignment_waste_entries": int(touched_blocks.sum()) * block_size**2
                - entries,
            }
        )
    total = sum(r["extra_directed_entries"] for r in records)
    return {
        "token_count": n,
        "tile_count": int(tile_count),
        "tile_size": width,
        "block_size": block_size,
        "base_entries_per_phase": n * width,
        "extra_entries_cycle": total,
        "extra_entries_per_phase_mean": total / phases,
        "extra_to_base_ratio": total / (phases * n * width),
        "max_extra_keys_per_query": max(r["max_extra_keys_per_query"] for r in records),
        "phase_stats": records,
        "definitions": {
            "segments": "sum of contiguous K/V runs per query in phase-rolled physical sequence",
            "blocks": "nonempty aligned query-block x key-block rectangles",
            "block_alignment_waste": "full rectangle capacity minus requested extra directed entries",
            "cost_warning": "topology counts only; no measured latency or speedup implied",
        },
    }


def fit_budget(
    order,
    grid_size,
    tile_count,
    kind,
    target_entries_per_phase,
    offsets=None,
    policy="at_most",
    **params,
):
    """Fit a monotone family of complete structures to a mean-phase budget.

    Returns (extra_masks, report). The constrained budget counts actual new
    directed entries AFTER removing base overlap. at_most never exceeds the
    target; nearest may exceed it, reports signed mismatch. Fixed parameters
    (e.g. patch_size, block_size) are preserved. Parameter search uses integer
    bisection, not row truncation or FLOP claims.
    """
    order, _, labels, width, offsets = _setup(order, tile_count, offsets, grid_size)
    if not math.isfinite(target_entries_per_phase) or target_entries_per_phase < 0:
        raise ValueError("budget must be finite and nonnegative")
    if policy not in ("at_most", "nearest"):
        raise ValueError("policy must be at_most or nearest")
    aliases = {
        "4_neighbor": "neighbor4",
        "four_neighbor": "neighbor4",
        "local_patch": "patch_bridge",
        "sequence_halo": "halo",
        "block_level": "block_bridge",
        "full_strip_reference": "full_strip",
    }
    kind = aliases.get(kind, kind)
    defects = _defects(labels, grid_size)
    if kind == "none":
        parameter, upper = "unused", 0
    elif kind == "neighbor4":
        parameter, upper = "max_edges", len(defects)
    elif kind == "full_strip":
        parameter = "max_boundaries"
        upper = (
            len(params.get("vertical_seams", [])) + len(params.get("horizontal_seams", []))
            if "vertical_seams" in params or "horizontal_seams" in params
            else len(_ranked_cuts(defects, grid_size))
        )
    elif kind == "patch_bridge":
        _, membership = _patch_groups(
            grid_size, _positive_int(params.get("patch_size", 2), "patch_size")
        )
        parameter, upper = (
            "max_bridges",
            len(
                set(
                    tuple(sorted((int(membership[a]), int(membership[b]))))
                    for a, b in defects.tolist()
                )
            ),
        )
    elif kind == "halo":
        parameter, upper = "halo", len(order)
    elif kind == "block_bridge":
        parameter, upper = (
            "max_extra_blocks",
            len(order) // _positive_int(params.get("block_size", 16), "block_size"),
        )
    else:
        raise ValueError(f"unknown augmentation kind {kind!r}")
    observations = {}

    def cost(value):
        if value not in observations:
            mask = augmentation_masks(
                order, grid_size, tile_count, kind, offsets, **(params | {parameter: value})
            )
            observations[value] = float(mask.sum()) / len(offsets)
        return observations[value]

    lo, hi = 0, upper
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if cost(mid) <= target_entries_per_phase:
            lo = mid
        else:
            hi = mid - 1
    chosen = lo
    if (
        policy == "nearest"
        and lo < upper
        and abs(cost(lo + 1) - target_entries_per_phase) < abs(cost(lo) - target_entries_per_phase)
    ):
        chosen = lo + 1
    selected = params | {parameter: chosen}
    mask = augmentation_masks(order, grid_size, tile_count, kind, offsets, **selected)
    measured = float(mask.sum()) / len(offsets)
    return mask, {
        "kind": kind,
        "parameters": selected,
        "budget_policy": policy,
        "target_entries_per_phase": target_entries_per_phase,
        "achieved_entries_per_phase": measured,
        "signed_budget_mismatch": measured - target_entries_per_phase,
        "budget_unit": "mean new directed entries over phases",
        "complete_structures_only": True,
    }
