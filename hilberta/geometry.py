"""Exact Hilbert ordering and geometric neighbors, extracted from the research implementation."""

import torch


def hilbert_order(grid_size=64):
    """Match src.utils.get_hilbert_flat_indices: x,y Hilbert, then vertical flip."""
    if grid_size < 2 or grid_size & (grid_size - 1):
        raise ValueError("Hilbert grid side must be a power of two >= 2")
    result = []
    for distance in range(grid_size**2):
        x = y = 0
        t, s = distance, 1
        while s < grid_size:
            rx, ry = (t // 2) & 1, (t ^ (t // 2)) & 1
            if ry == 0:
                if rx:
                    x, y = s - 1 - x, s - 1 - y
                x, y = y, x
            x += s * rx
            y += s * ry
            t //= 4
            s *= 2
        result.append((grid_size - 1 - y) * grid_size + x)
    return torch.tensor(result, dtype=torch.long)


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
