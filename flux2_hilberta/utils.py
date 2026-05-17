import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from hilbertcurve.hilbertcurve import HilbertCurve


def isinstance_str(x: object, cls_name: str):
    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False


def init_generator(device: torch.device, fallback: torch.Generator = None):
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback


def apply_rotary_emb(xq, freqs_cis, sequence_dim=None):
    """Apply rotary embeddings in FLUX.2 real (cos, sin) format.

    Args:
        xq: [..., S, H, D] or [..., H, S, D] tensor
        freqs_cis: tuple of (cos, sin) each [S, D]
        sequence_dim: which dim is the sequence dim (1 for [B,S,H,D], 2 for [B,H,S,D])
    """
    cos, sin = freqs_cis

    if sequence_dim == 1:
        # [B, S, H, D] layout (Flux2 default)
        cos = cos[None, :, None, :]
        sin = sin[None, :, None, :]
    else:
        # [B, H, S, D] layout
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]

    cos = cos.to(xq.device)
    sin = sin.to(xq.device)

    x_real, x_imag = xq.reshape(*xq.shape[:-1], -1, 2).unbind(-1)
    x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(-2)
    out = (xq.float() * cos + x_rotated.float() * sin).to(xq.dtype)
    return out


def get_hilbert_flat_indices(p: int) -> torch.Tensor:
    n = 2
    size = 2 ** p
    hilbert_curve = HilbertCurve(p, n)

    indices = []
    for d in range(size * size):
        x, y = hilbert_curve.point_from_distance(d)
        row = size - 1 - y
        flat_index = row * size + x
        indices.append(flat_index)

    return torch.tensor(indices, dtype=torch.long)


def get_inverse_hilbert_indices(p: int) -> torch.Tensor:
    hilbert = get_hilbert_flat_indices(p)
    inverse = torch.empty_like(hilbert)
    inverse[hilbert] = torch.arange(hilbert.numel(), device=hilbert.device)
    return inverse
