import itertools
import math
import os
from typing import Tuple, Union

import torch
from hilbertcurve.hilbertcurve import HilbertCurve


def isinstance_str(x: object, cls_name: str):
    """Checks whether x has any class *named* cls_name in its ancestry."""
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to input tensors using the given frequency tensor."""
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        if cos.ndim == 3:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        elif cos.ndim == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)
        return out


def get_hilbert_flat_indices(p: int) -> torch.Tensor:
    """Generate flattened indices for a 2^p x 2^p grid in Hilbert curve order."""
    size = 2 ** p
    hc = HilbertCurve(p, 2)
    indices = []
    for d in range(size * size):
        x, y = hc.point_from_distance(d)
        row = size - 1 - y  # flip vertically for bottom-left origin
        indices.append(row * size + x)
    return torch.tensor(indices, dtype=torch.long)


def get_inverse_hilbert_indices(p: int) -> torch.Tensor:
    hilbert = get_hilbert_flat_indices(p)
    inverse = torch.empty_like(hilbert)
    inverse[hilbert] = torch.arange(hilbert.numel(), device=hilbert.device)
    return inverse


# ---------------------------------------------------------------------------
# Hilbert-curve tiled attention mask generation
# ---------------------------------------------------------------------------

def create_hilbert_tile_mask(image_size: int, num_of_tiles: int, offset: int = 0) -> torch.Tensor:
    """Create a Hilbert-curve based tiled attention mask.

    Args:
        image_size: Number of image tokens (e.g. 4096 for 1024px, 16384 for 2048px).
        num_of_tiles: Number of tiles to partition the Hilbert curve into.
        offset: Circular offset applied to the Hilbert index before tiling.

    Returns:
        Float mask of shape (image_size, image_size) with 0.0 for allowed
        attention and -inf for blocked attention.
    """
    p = int(math.log2(math.isqrt(image_size)))
    hilbert_index = get_hilbert_flat_indices(p)

    # Build composite Hilbert index: forward middle half + reversed middle half
    index = torch.arange(hilbert_index.numel())
    cut_off = index.shape[0] // 4

    hilbert_fwd = torch.gather(index, 0, hilbert_index)[cut_off:-cut_off]
    hilbert_rev = torch.gather(index.flip(0), 0, hilbert_index)[cut_off:-cut_off]
    hilbert_index = torch.cat([hilbert_fwd, hilbert_rev])

    # Apply circular offset and partition into tiles
    hilbert_index = torch.cat([hilbert_index[offset:], hilbert_index[:offset]])
    tiles = hilbert_index.reshape(num_of_tiles, -1)

    # Build mask: tokens in the same tile can attend to each other
    seq_len = hilbert_index.numel()
    mask = torch.full((seq_len, seq_len), float('-inf'))
    for tile in tiles:
        mask[tile.unsqueeze(1), tile.unsqueeze(0)] = 0.0

    # Allow center region full attention (global context)
    grid_side = int(math.isqrt(image_size))
    if image_size == 4096:
        center_start, center_end = 24, 40
    elif image_size == 16384:
        center_start, center_end = 48, 80
    else:
        return mask

    for i in range(center_start, center_end):
        start = i * grid_side + center_start
        end = i * grid_side + center_end
        mask[start:end, :] = 0.0
        mask[:, start:end] = 0.0

    if image_size == 16384:
        corner_size = 4
        for cs, ce in [
            (0, corner_size),
            (seq_len - corner_size, seq_len),
        ]:
            mask[cs:ce, :] = 0.0
            mask[:, cs:ce] = 0.0
            mask[cs:ce, seq_len - corner_size:seq_len] = 0.0
            mask[:, seq_len - corner_size:seq_len] = 0.0

    return mask


def generate_masks(output_dir: str = './masks',
                   image_sizes: list = None,
                   tile_counts: list = None,
                   sliding_cycles: list = None):
    """Generate and save all Hilbert-curve tiled attention masks.

    Args:
        output_dir: Directory to save mask .pt files.
        image_sizes: List of image token counts (default: [4096]).
        tile_counts: List of tile counts (default: [4, 16]).
        sliding_cycles: List of sliding cycle values (default: [2, 4]).
    """
    if image_sizes is None:
        image_sizes = [4096]
    if tile_counts is None:
        tile_counts = [4, 16]
    if sliding_cycles is None:
        sliding_cycles = [2, 4]

    os.makedirs(output_dir, exist_ok=True)

    for image_size, num_of_tiles, sliding_cycle in itertools.product(
            image_sizes, tile_counts, sliding_cycles):
        tile_len = image_size // num_of_tiles
        offsets = [(tile_len // sliding_cycle) * i for i in range(sliding_cycle)]

        for offset in offsets:
            mask = create_hilbert_tile_mask(image_size, num_of_tiles=num_of_tiles, offset=offset)
            fname = f'image_size_{image_size}_offset_{offset}_num_of_tiles_{num_of_tiles}.pt'
            path = os.path.join(output_dir, fname)
            torch.save(mask, path)
            print(f"  Saved {fname}")

    print(f"Masks generated in {output_dir}")
