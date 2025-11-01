import torch
import torch.nn.functional as F
import os
from typing import Union, Tuple
import gc

def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True

    return False


def init_generator(device: torch.device, fallback: torch.Generator = None):
    """
    Forks the current default random generator given device.
    """
    if device.type == "cpu":
        return torch.Generator(device="cpu").set_state(torch.get_rng_state())
    elif device.type == "cuda":
        return torch.Generator(device=device).set_state(torch.cuda.get_rng_state())
    else:
        if fallback is None:
            return init_generator(torch.device("cpu"))
        else:
            return fallback

def do_nothing(x: torch.Tensor, mode: str = None):
    return x


def mps_gather_workaround(input, dim, index):
    if input.shape[-1] == 1:
        return torch.gather(input.unsqueeze(-1), dim - 1 if dim < 0 else dim, index.unsqueeze(-1)).squeeze(-1)
    else:
        return torch.gather(input, dim, index)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        cos = cos[None, None]
        sin = sin[None, None]
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, S, H, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)
    


def save_tensor_every_k_steps(tensor: torch.Tensor, prefix: str, output_dir: str, step: int):
    os.makedirs(output_dir, exist_ok=True)
    
    existing_files = [f for f in os.listdir(output_dir)
                      if os.path.isfile(os.path.join(output_dir, f))]
    file_count = len(existing_files)
    new_suffix = file_count + 1

    file_name = f"{prefix}_{new_suffix}.pt"
    save_path = os.path.join(output_dir, file_name)
    torch.save(tensor, save_path)
    print(f"[Step {step}]")
    return


def fold_with_indices(x, num_tiles):
    B, N, C = x.shape
    H = W = int(N**0.5)

    num_tiles_per_side = int(num_tiles**0.5)
    tile_side_len = H // num_tiles_per_side

    idx = torch.arange(N, device=x.device, dtype=torch.long)
    idx = idx.reshape(1, H, W, 1)

    tile_idx = torch.as_strided(
        idx,
        (1, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, 1),
        (N, tile_side_len * W, tile_side_len, W, 1, 1),
    )

    flatten_tile_idx = tile_idx.reshape(1, N, 1).expand(B, -1, C)

    flatten_tile_x = torch.gather(x, 1, flatten_tile_idx)
    tile_x = flatten_tile_x.reshape(B, num_tiles, -1, C)

    return tile_x, flatten_tile_idx

def unfold_with_indices(x_prime, flatten_tile_idx):
    B, N, C = flatten_tile_idx.shape

    x_prime = x_prime.reshape(B, -1, C)

    restored_x = torch.zeros_like(flatten_tile_idx, dtype=x_prime.dtype)

    restored_x = restored_x.scatter(1, flatten_tile_idx, x_prime)

    return restored_x

def reconstruct_new_rope_emb(A, rope_emb, average_method='weighted'):
    """
    Reconstruct a new rotary embedding (rope) for destination tokens using the provided tensors.
    
    Parameters:
      - A: torch.Tensor of shape [1, k, d, n]
           Attention tensor from which top-3 indices are selected.
      - rope_emb: torch.Tensor of shape [2, k, n, c]
           Rotary embedding tensor, where channel 0 contains cosine values and 
           channel 1 contains sine values.
      - average_method: str, either 'weighted' or 'direct'
           Averaging method:
             'weighted' -> Use normalized top-3 attention weights.
             'direct'   -> Simple average.
    
    Returns:
      - new_rope_emb: torch.Tensor of shape [2, k, d, c]
           The reconstructed rope embedding for each destination token group (d).
           Channel 0 is cos(avg_angle) and channel 1 is sin(avg_angle).
    """
    # A: [1, k, d, n]; rope_emb: [2, k, n, c]
    _, k, d, n = A.shape
    _, k2, n2, c = rope_emb.shape
    assert k == k2 and n == n2, "Dimension mismatch between A and rope_emb."
    
    # Step 1: Extract the top-3 indices along the n dimension from A.
    # The resulting tensors have shape [1, k, d, 3].
    topk_values, topk_indices = torch.topk(A, k=3, dim=-1)
    
    # Remove the batch dimension -> shape becomes [k, d, 3]
    topk_values = topk_values.squeeze(0)
    topk_indices = topk_indices.squeeze(0)
    
    # Step 2: Gather the corresponding rope embeddings.
    # Split rope_emb into cosine and sine parts.
    # Each part has shape [k, n, c].
    rope_emb_cos = rope_emb[0]  # shape: [k, n, c]
    rope_emb_sin = rope_emb[1]  # shape: [k, n, c]
    
    # To gather along the n dimension using an index of shape [k, d, 3],
    # we unsqueeze each part along a new dimension so that they become [k, 1, n, c].
    rope_emb_cos_unsq = rope_emb_cos.unsqueeze(1).repeat(1, d, 1, 1)  # [k, 1, n, c]
    rope_emb_sin_unsq = rope_emb_sin.unsqueeze(1).repeat(1, d, 1, 1)  # [k, 1, n, c]
    
    # Prepare an index tensor for gathering:
    # topk_indices has shape [k, d, 3]. We add an extra dimension at the end to match the embedding dim:
    # New index shape: [k, d, 3, c]
    index_for_gather = topk_indices.unsqueeze(-1).expand(-1, -1, -1, c)
    
    # Gather along the n dimension (dim=2) using the prepared index.
    # The resulting gathered tensors will have shape [k, d, 3, c].
    gathered_cos = torch.gather(rope_emb_cos_unsq, dim=2, index=index_for_gather)
    gathered_sin = torch.gather(rope_emb_sin_unsq, dim=2, index=index_for_gather)
    
    # Combine cosine and sine into one tensor with shape [2, k, d, 3, c].
    gathered = torch.stack([gathered_cos, gathered_sin], dim=0)
    
    # Step 3: Compute angles at the gathered positions.
    # Using torch.atan2(sine, cosine) yields a tensor of angles with shape [k, d, 3, c].
    gathered_angles = torch.atan2(gathered[1], gathered[0])
    
    # Step 4: Average the angles over the top-3 dimension, preserving the d dimension.
    if average_method == 'weighted':
        # Weighted average:
        # Normalize the top-3 attention weights along the top-3 dimension.
        weights_norm = topk_values / topk_values.sum(dim=-1, keepdim=True)  # shape: [k, d, 3]
        
        # For proper angle averaging, average cosine and sine separately.
        cos_angles = torch.cos(gathered_angles)  # shape: [k, d, 3, c]
        sin_angles = torch.sin(gathered_angles)  # shape: [k, d, 3, c]
        
        # Multiply the cosine and sine with the normalized weights.
        # Note: weights_norm.unsqueeze(-1) expands weights to shape [k, d, 3, 1] for broadcasting.
        weighted_avg_cos = (weights_norm.unsqueeze(-1) * cos_angles).sum(dim=2)  # sum over the top-3 dim -> [k, d, c]
        weighted_avg_sin = (weights_norm.unsqueeze(-1) * sin_angles).sum(dim=2)  # -> [k, d, c]
        
        # Compute the averaged angle from the weighted average cosine and sine.
        avg_angle = torch.atan2(weighted_avg_sin, weighted_avg_cos)  # shape: [k, d, c]
    
    elif average_method == 'direct':
        # Direct average:
        # Simply average the cosine and sine over the top-3 dimension.
        avg_cos = torch.cos(gathered_angles).mean(dim=2)  # shape: [k, d, c]
        avg_sin = torch.sin(gathered_angles).mean(dim=2)  # shape: [k, d, c]
        
        # Compute the averaged angle.
        avg_angle = torch.atan2(avg_sin, avg_cos)  # shape: [k, d, c]
    else:
        raise ValueError("average_method must be either 'weighted' or 'direct'")
    
    new_rope_emb = torch.stack([torch.cos(avg_angle), torch.sin(avg_angle)], dim=0)
    
    return new_rope_emb

def index_shift_for_tile_sliding(x, tile_len, flag = None):
    B, N, C = x.shape
    if N ** 0.5 != int(N ** 0.5):
        H = 64
        W = N // H
    else:
        H = W = int(N ** 0.5)
    # import pdb
    # pdb.set_trace()
    slide_len = tile_len // 2
    x_reshape = x.reshape(B, H, W, C)
    if flag == 'down_right':
        x_slide = torch.roll(x_reshape, shifts=(slide_len, slide_len), dims=(1, 2))
    elif flag == 'down_left':
        x_slide = torch.roll(x_reshape, shifts=(slide_len, -slide_len), dims=(1, 2))
    elif flag == 'up_right':
        x_slide = torch.roll(x_reshape, shifts=(-slide_len, slide_len), dims=(1, 2))
    elif flag == 'up_left':
        x_slide = torch.roll(x_reshape, shifts=(-slide_len, -slide_len), dims=(1, 2))
    elif flag == 'stay':
        x_slide = x_reshape
    # import pdb
    # pdb.set_trace()
    return x_slide.reshape(B, N, C)