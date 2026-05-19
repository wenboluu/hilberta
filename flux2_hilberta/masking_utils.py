"""
Hilbert tile mask creation for FLUX.2 masking-based attention.

Creates explicit attention masks based on Hilbert curve tiling,
where image tokens can only attend within their tile + shared center region.
"""

import torch
from utils import get_hilbert_flat_indices


def create_hilbert_tile_mask(x, num_of_tiles, offset=0):
    """
    Create a Hilbert-curve-based attention mask for image tokens.

    Args:
        x: tensor to get device from, with shape [..., seq_len, ...]
        num_of_tiles: number of spatial tiles
        offset: sliding window offset

    Returns:
        mask: [seq_len, seq_len] float tensor with 0.0 (attend) and -inf (block)
    """
    if x.shape[-2] - 512 == 4096 or x.shape[1] == 4096:
        hilbert_index = get_hilbert_flat_indices(6).to(x.device)
    elif x.shape[-2] - 512 == 16384 or x.shape[1] == 16384:
        hilbert_index = get_hilbert_flat_indices(7).to(x.device)
    else:
        # Try to infer from the input
        hilbert_index = get_hilbert_flat_indices(6).to(x.device)

    # Make complete hilbert index (closed curve)
    index = torch.arange(hilbert_index.numel()).to(x.device)
    cut_off = index.shape[0] // 4

    hilbert_x = torch.gather(index, dim=0, index=hilbert_index)
    hilbert_x_half = hilbert_x[cut_off:-cut_off]

    x_flip = index.flip(0)
    hilbert_x_flip = torch.gather(x_flip, dim=0, index=hilbert_index)
    hilbert_x_half_flip = hilbert_x_flip[cut_off:-cut_off]

    hilbert_index = torch.cat([hilbert_x_half, hilbert_x_half_flip])

    # Apply offset for sliding window
    hilbert_index = torch.cat([hilbert_index[offset:], hilbert_index[:offset]])
    hilbert_index = hilbert_index.reshape(num_of_tiles, -1)

    # Create all pairs within each tile
    B, N = hilbert_index.shape
    tensor_i = hilbert_index.view(B, N, 1).expand(B, N, N)
    tensor_j = hilbert_index.view(B, 1, N).expand(B, N, N)
    pairs = torch.stack([tensor_i, tensor_j], dim=-1).view(B, N * N, 2)
    pairs = pairs.reshape(-1, pairs.shape[-1])

    # Build mask: -inf everywhere, 0.0 where attention is allowed
    seq_len = hilbert_index.numel()
    mask = torch.full((seq_len, seq_len), float('-inf'), device=x.device)

    # Allow within-tile attention
    mask[pairs[:, 0], pairs[:, 1]] = 0.0

    # Allow center region (rows 24-40, cols 24-40 for 64x64) to attend globally
    H = int(seq_len ** 0.5)
    center_start = H * 3 // 8  # 24 for 64x64
    center_end = H * 5 // 8    # 40 for 64x64
    W = H
    for i in range(center_start, center_end):
        start = i * W + center_start
        end = i * W + center_end
        mask[start:end, :] = 0.0
        mask[:, start:end] = 0.0

    return mask
