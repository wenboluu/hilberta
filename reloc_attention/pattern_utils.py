from typing import Literal
import numpy as np
import torch

from utils import (
    get_inverse_hilbert_indices,
    get_hilbert_flat_indices,
)


def create_closed_hilbert_mapping(p: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create forward and inverse mappings for closed Hilbert curve.
    
    Args:
        p: Grid size exponent (grid will be 2^p x 2^p)
        device: Computing device
        
    Returns:
        Tuple of (forward_mapping, inverse_mapping) tensors
    """
    base = get_hilbert_flat_indices(p).to(device)
    N = base.numel()
    index = torch.arange(N, device=device)
    cut_off = N // 4

    fwd = torch.gather(index, 0, base)
    fwd_mid = fwd[cut_off:-cut_off]

    rev = index.flip(0)
    rev_by_h = torch.gather(rev, 0, base)
    rev_mid = rev_by_h[cut_off:-cut_off]

    closed_index = torch.cat([fwd_mid, rev_mid], dim=0)

    closed_inverse = torch.empty_like(closed_index)
    closed_inverse[closed_index] = torch.arange(N, device=device)
    return closed_index, closed_inverse

def compute_pairwise_distances(coords: torch.Tensor, metric: Literal["euclidean", "manhattan"]) -> torch.Tensor:
    """Compute pairwise distances between all coordinate pairs."""
    if metric == "euclidean":
        return torch.cdist(coords, coords, p=2)
    if metric == "manhattan":
        diffs = (coords[:, None, :] - coords[None, :, :]).abs()
        return diffs.sum(dim=-1)
    raise ValueError("metric must be 'euclidean' or 'manhattan'")


def generate_row_major_coordinates(size: int, device: torch.device) -> torch.Tensor:
    """Generate 2D coordinates for tokens in row-major order."""
    rows = torch.arange(size, device=device)
    cols = torch.arange(size, device=device)
    y, x = torch.meshgrid(rows, cols, indexing="ij")
    return torch.stack((y, x), dim=-1).reshape(-1, 2).to(torch.float32)


def generate_hilbert_coordinates(p: int, device: torch.device, closed: bool) -> torch.Tensor:
    """Generate 2D coordinates for tokens in Hilbert curve order."""
    size = 2**p
    if closed:
        _, inverse = create_closed_hilbert_mapping(p, device)
    else:
        inverse = get_inverse_hilbert_indices(p).to(device=device)
    row = (inverse // size).to(torch.float32)
    col = (inverse % size).to(torch.float32)
    return torch.stack((row, col), dim=-1)


def compute_manhattan_distance_sums_1d_exact(v: torch.Tensor, size: int) -> torch.Tensor:
    """
    Integer version of _sum_abs_to_all_on_grid_coord using int64 arithmetic for exactness.
    """
    v64 = v.to(torch.int64)
    size64 = torch.tensor(size, dtype=torch.int64, device=v64.device)
    a = v64 * (v64 + 1) // 2
    b = (size64 - 1 - v64) * (size64 - v64) // 2
    return a + b


def compute_manhattan_distance_sums_2d_exact(y: torch.Tensor, x: torch.Tensor, size: int) -> torch.Tensor:
    f_y = compute_manhattan_distance_sums_1d_exact(y, size)
    f_x = compute_manhattan_distance_sums_1d_exact(x, size)
    return torch.tensor(size, dtype=torch.int64, device=y.device) * (f_y + f_x)

def compute_manhattan_disruption_scores(
    grid_side: int, 
    p: int, 
    device: torch.device, 
    closed: bool
) -> torch.Tensor:
    # Compute distance sums for original row-major layout
    y_coords_original = torch.arange(grid_side, device=device, dtype=torch.int64).repeat_interleave(grid_side)
    x_coords_original = torch.arange(grid_side, device=device, dtype=torch.int64).repeat(grid_side)
    original_distance_sums = compute_manhattan_distance_sums_2d_exact(y_coords_original, x_coords_original, grid_side)

    # Compute coordinates after Hilbert reordering
    if closed:
        _, hilbert_inverse = create_closed_hilbert_mapping(p, device)
    else:
        hilbert_inverse = get_inverse_hilbert_indices(p).to(device=device)
    
    # Get new coordinates from inverse mapping
    y_coords_after = (hilbert_inverse // grid_side).to(torch.int64)
    x_coords_after = (hilbert_inverse % grid_side).to(torch.int64)
    hilbert_distance_sums = compute_manhattan_distance_sums_2d_exact(y_coords_after, x_coords_after, grid_side)

    # Compute disruption scores = post-reordering distance sums - original distance sums
    disruption_scores = (hilbert_distance_sums - original_distance_sums).to(torch.int64)
    return disruption_scores


def compute_euclidean_disruption_scores(
    grid_side: int, 
    p: int, 
    device: torch.device, 
    closed: bool
) -> torch.Tensor:
    # Get original and reordered coordinates
    original_coords = generate_row_major_coordinates(size=grid_side, device=device)
    hilbert_coords = generate_hilbert_coordinates(p=p, device=device, closed=closed)
    
    # Compute pairwise distance matrices
    original_distances = compute_pairwise_distances(original_coords, "euclidean")
    hilbert_distances = compute_pairwise_distances(hilbert_coords, "euclidean")
    
    # Compute disruption scores for each point (sum of distance increases)
    disruption_scores = (hilbert_distances - original_distances).sum(dim=1)
    return disruption_scores


def select_topk_indices_deterministic(scores: torch.Tensor, k: int) -> torch.Tensor:
    if scores.dtype.is_floating_point:
        _, indices = torch.topk(scores, k=k, largest=True, sorted=True)
        return indices
    
    # For integer scores, use simpler tie-breaking strategy
    # Convert to float and use torch.topk for deterministic results
    _, indices = torch.topk(scores.to(torch.float32), k=k, largest=True, sorted=True)
    return indices


def find_most_affected_tokens_by_hilbert_reordering(
    size: int,
    top_k: int,
    metric: Literal["euclidean", "manhattan"] = "manhattan",
    closed: bool = True,
    device: str | torch.device | None = None,
) -> list[int]:
    device = torch.device(device) if isinstance(device, str) else (device or torch.device("cpu"))
    grid_side = int(round(np.sqrt(size)))
    p = int(np.log2(grid_side))
    
    if metric == "manhattan":
        disruption_scores = compute_manhattan_disruption_scores(grid_side, p, device, closed)
    else:  # euclidean
        disruption_scores = compute_euclidean_disruption_scores(grid_side, p, device, closed)

    k = min(top_k, disruption_scores.numel())
    top_indices = select_topk_indices_deterministic(disruption_scores, k)
    return top_indices.tolist()


if __name__ == "__main__":
    indices_64_4 = find_most_affected_tokens_by_hilbert_reordering(64, 14)
    # indices_4096_256 = find_most_affected_tokens_by_hilbert_reordering(4096, 256)
    # torch.save(indices_4096_256, "indices_4096_256.pt")
    # indices_16384_1024 = find_most_affected_tokens_by_hilbert_reordering(16384, 1024)
    # torch.save(indices_16384_1024, "indices_16384_1024.pt")