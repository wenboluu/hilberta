"""
Reorder utilities for FLUX.2 HilbertA.

Adapted from reloc_attention/reorder_utils_sliding_shared.py for FLUX.2's architecture:
- Modulation computed outside blocks (shared modulation modules)
- RoPE computed separately for img and txt then concatenated as (cos, sin) tuple
- Two block types: Flux2TransformerBlock (double) + Flux2SingleTransformerBlock (single)
- Text token count determined dynamically from encoder_hidden_states
"""

import torch
from typing import Any, Dict, Optional, Union
from functools import lru_cache
import yaml

from diffusers.models.modeling_outputs import Transformer2DModelOutput
from utils import get_hilbert_flat_indices, get_inverse_hilbert_indices, apply_rotary_emb

# ─── Tiling primitives ───

def tile(x, num_tiles):
    B, HW, C = x.shape
    H = W = int(HW**0.5)
    num_tiles_per_side = int(num_tiles**0.5)
    tile_side_len = H // num_tiles_per_side

    x_reshaped = torch.as_strided(
        x,
        (1, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, C),
        (HW * C, tile_side_len * H * C, tile_side_len * C, H * C, C, 1),
    )
    x_reshaped = x_reshaped.reshape(-1, tile_side_len**2, C)
    x_reshaped = x_reshaped.reshape(B, HW, C).contiguous()
    return x_reshaped


def untile(x_tiled, num_tiles):
    B, HW, C = x_tiled.shape
    H = W = int(HW**0.5)
    num_tiles_per_side = int(num_tiles**0.5)
    tile_side_len = H // num_tiles_per_side

    x_tiles = x_tiled.view(
        B, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, C
    )
    x_tiles = x_tiles.permute(0, 1, 3, 2, 4, 5).contiguous()
    x_tiles = x_tiles.view(B, H, W, C)
    return x_tiles.view(B, H * W, C)


# ─── Cached Hilbert index management ───

@lru_cache(maxsize=None)
def _base_hilbert_indices(sequence_length: int) -> torch.Tensor:
    if sequence_length == 4096:
        return get_hilbert_flat_indices(6)
    if sequence_length == 16384:
        return get_hilbert_flat_indices(7)
    raise ValueError(f"Unsupported sequence_length {sequence_length}")


@lru_cache(maxsize=None)
def _base_inverse_hilbert_indices(sequence_length: int) -> torch.Tensor:
    if sequence_length == 4096:
        return get_inverse_hilbert_indices(6)
    if sequence_length == 16384:
        return get_inverse_hilbert_indices(7)
    raise ValueError(f"Unsupported sequence_length {sequence_length}")


_DEVICE_INDEX_CACHE: Dict = {}
_DEVICE_INVERSE_CACHE: Dict = {}


def _get_index_on_device(sequence_length, device):
    key = (sequence_length, device)
    cached = _DEVICE_INDEX_CACHE.get(key)
    if cached is None or cached.device != device:
        cached = _base_hilbert_indices(sequence_length).to(device, non_blocking=True)
        _DEVICE_INDEX_CACHE[key] = cached
    return cached


def _get_inverse_on_device(sequence_length, device):
    key = (sequence_length, device)
    cached = _DEVICE_INVERSE_CACHE.get(key)
    if cached is None or cached.device != device:
        cached = _base_inverse_hilbert_indices(sequence_length).to(device, non_blocking=True)
        _DEVICE_INVERSE_CACHE[key] = cached
    return cached


# ─── Hilbert reorder for FLUX.2 ───
# RoPE is (cos, sin) tuple, concatenated as [text, image]
# We reorder only the image portion.

def apply_hilbert_reorder_simple(image_rotary_emb, hidden_states, num_tiles, offset=0):
    """
    Reorder image tokens and image-portion of RoPE using Hilbert curve.
    image_rotary_emb is (cos, sin) concatenated as [text_len + img_len, D].
    hidden_states is [B, img_len, C].
    """
    img_seq_len = hidden_states.shape[1]
    cos, sin = image_rotary_emb
    txt_len = cos.shape[0] - img_seq_len

    hilbert_index = _get_index_on_device(img_seq_len, hidden_states.device)
    if offset:
        hilbert_index = torch.remainder(hilbert_index - offset, img_seq_len)

    hilbert_sorted_positions = torch.argsort(hilbert_index)

    # Reorder hidden states
    hilbert_hidden = hidden_states[:, hilbert_sorted_positions, :]

    # Reorder image portion of RoPE
    img_cos = cos[txt_len:]
    img_sin = sin[txt_len:]
    reordered_cos = torch.cat([cos[:txt_len], img_cos[hilbert_sorted_positions]], dim=0)
    reordered_sin = torch.cat([sin[:txt_len], img_sin[hilbert_sorted_positions]], dim=0)

    return (reordered_cos, reordered_sin), hilbert_hidden


def recover_hilbert_reorder_simple(image_rotary_emb, hidden_states, num_tiles, offset=0):
    """Recover original spatial order from Hilbert-reordered tokens."""
    img_seq_len = hidden_states.shape[1]
    cos, sin = image_rotary_emb
    txt_len = cos.shape[0] - img_seq_len

    hilbert_index = _get_index_on_device(img_seq_len, hidden_states.device)
    if offset:
        hilbert_index = torch.remainder(hilbert_index - offset, img_seq_len)

    hilbert_sorted_positions = torch.argsort(hilbert_index)
    inverse_positions = torch.argsort(hilbert_sorted_positions)

    recovered_hidden = hidden_states[:, inverse_positions, :]

    img_cos = cos[txt_len:]
    img_sin = sin[txt_len:]
    recovered_cos = torch.cat([cos[:txt_len], img_cos[inverse_positions]], dim=0)
    recovered_sin = torch.cat([sin[:txt_len], img_sin[inverse_positions]], dim=0)

    return (recovered_cos, recovered_sin), recovered_hidden


def apply_hilbert_reorder(image_rotary_emb, hidden_states, num_tiles, offset=0):
    """
    Reorder with center/sparse separation.
    Center region attends globally (shared with text), sparse uses local attention.
    """
    img_seq_len = hidden_states.shape[1]
    cos, sin = image_rotary_emb
    txt_len = cos.shape[0] - img_seq_len

    if img_seq_len == 4096:
        H = W = 64
        center_start, center_end = 24, 40
        center_size = 16 * 16
    elif img_seq_len == 16384:
        H = W = 128
        center_start, center_end = 48, 80
        center_size = 32 * 32
    else:
        raise ValueError(f"Unsupported image sequence length: {img_seq_len}")

    # Center region indices
    center_indices = []
    for i in range(center_start, center_end):
        for j in range(center_start, center_end):
            center_indices.append(i * W + j)
    center_indices = torch.tensor(center_indices, device=hidden_states.device)

    # Sparse indices
    all_indices = torch.arange(img_seq_len, device=hidden_states.device)
    is_center = torch.zeros(img_seq_len, dtype=torch.bool, device=hidden_states.device)
    is_center[center_indices] = True
    sparse_indices = all_indices[~is_center]

    # Split hidden states
    center_hidden = hidden_states[:, center_indices, :]
    sparse_hidden = hidden_states[:, sparse_indices, :]

    # Split image RoPE
    img_cos, img_sin = cos[txt_len:], sin[txt_len:]
    center_cos, center_sin = img_cos[center_indices], img_sin[center_indices]
    sparse_cos, sparse_sin = img_cos[sparse_indices], img_sin[sparse_indices]

    # Hilbert reorder sparse tokens
    full_hilbert_index = _get_index_on_device(img_seq_len, hidden_states.device)
    if offset:
        full_hilbert_index = torch.remainder(full_hilbert_index - offset, img_seq_len)

    sparse_hilbert_indices = full_hilbert_index[sparse_indices]
    sorted_positions = torch.argsort(sparse_hilbert_indices)

    sparse_hidden = sparse_hidden[:, sorted_positions, :]
    sparse_cos = sparse_cos[sorted_positions]
    sparse_sin = sparse_sin[sorted_positions]

    # Concatenate: [text, center, sparse]
    reordered_hidden = torch.cat([center_hidden, sparse_hidden], dim=1)
    reordered_cos = torch.cat([cos[:txt_len], center_cos, sparse_cos], dim=0)
    reordered_sin = torch.cat([sin[:txt_len], center_sin, sparse_sin], dim=0)

    return (reordered_cos, reordered_sin), reordered_hidden


def recover_hilbert_reorder(image_rotary_emb, hidden_states, num_tiles, offset=0):
    """Recover from center/sparse Hilbert ordering."""
    img_seq_len = hidden_states.shape[1]
    cos, sin = image_rotary_emb
    txt_len = cos.shape[0] - img_seq_len

    if img_seq_len == 4096:
        H = W = 64
        center_start, center_end = 24, 40
        center_size = 16 * 16
    elif img_seq_len == 16384:
        H = W = 128
        center_start, center_end = 48, 80
        center_size = 32 * 32
    else:
        raise ValueError(f"Unsupported image sequence length: {img_seq_len}")

    sparse_size = img_seq_len - center_size

    # Split
    center_hidden = hidden_states[:, :center_size, :]
    sparse_hidden = hidden_states[:, center_size:, :]

    img_cos, img_sin = cos[txt_len:], sin[txt_len:]
    center_cos, center_sin = img_cos[:center_size], img_sin[:center_size]
    sparse_cos, sparse_sin = img_cos[center_size:], img_sin[center_size:]

    # Recreate indices
    center_indices = []
    for i in range(center_start, center_end):
        for j in range(center_start, center_end):
            center_indices.append(i * W + j)
    center_indices = torch.tensor(center_indices, device=hidden_states.device)

    all_indices = torch.arange(img_seq_len, device=hidden_states.device)
    is_center = torch.zeros(img_seq_len, dtype=torch.bool, device=hidden_states.device)
    is_center[center_indices] = True
    sparse_indices = all_indices[~is_center]

    # Undo Hilbert ordering
    full_hilbert_index = _get_index_on_device(img_seq_len, hidden_states.device)
    if offset:
        full_hilbert_index = torch.remainder(full_hilbert_index - offset, img_seq_len)

    sparse_hilbert_indices = full_hilbert_index[sparse_indices]
    sorted_positions = torch.argsort(sparse_hilbert_indices)
    inverse_positions = torch.argsort(sorted_positions)

    sparse_hidden = sparse_hidden[:, inverse_positions, :]
    sparse_cos = sparse_cos[inverse_positions]
    sparse_sin = sparse_sin[inverse_positions]

    # Reconstruct
    B, _, C = center_hidden.shape
    recovered_hidden = torch.zeros(B, img_seq_len, C, device=hidden_states.device, dtype=hidden_states.dtype)
    recovered_cos = torch.zeros(img_seq_len, center_cos.shape[-1], device=hidden_states.device, dtype=center_cos.dtype)
    recovered_sin = torch.zeros(img_seq_len, center_sin.shape[-1], device=hidden_states.device, dtype=center_sin.dtype)

    recovered_hidden[:, center_indices, :] = center_hidden
    recovered_hidden[:, sparse_indices, :] = sparse_hidden
    recovered_cos[center_indices] = center_cos
    recovered_cos[sparse_indices] = sparse_cos
    recovered_sin[center_indices] = center_sin
    recovered_sin[sparse_indices] = sparse_sin

    full_cos = torch.cat([cos[:txt_len], recovered_cos], dim=0)
    full_sin = torch.cat([sin[:txt_len], recovered_sin], dim=0)

    return (full_cos, full_sin), recovered_hidden


# ─── Customized forward for Flux2Transformer2DModel ───

def customized_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: dict = None,
    return_dict: bool = True,
    step: int = 0,
) -> Union[torch.FloatTensor, "Transformer2DModelOutput"]:
    """Customized forward for Flux2Transformer2DModel with Hilbert reordering."""
    import os as _os
    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    with open(_os.path.join(_script_dir, "config.yaml"), 'r') as f:
        config = yaml.safe_load(f)

    num_tiles = config.get('num_tiles', 4)
    method = config.get('method', 'reorder_shared')
    full_attn_step = config.get('full_attn_step', [])

    from diffusers.models.transformers.transformer_flux2 import (
        Flux2Modulation, Flux2Transformer2DModelOutput
    )

    num_txt_tokens = encoder_hidden_states.shape[1]

    # 1. Timestep embedding and modulation
    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000

    temb = self.time_guidance_embed(timestep, guidance)

    double_stream_mod_img = self.double_stream_modulation_img(temb)
    double_stream_mod_txt = self.double_stream_modulation_txt(temb)
    single_stream_mod = self.single_stream_modulation(temb)

    # 2. Input projections
    hidden_states = self.x_embedder(hidden_states)
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    # 3. Compute RoPE
    if img_ids.ndim == 3:
        img_ids = img_ids[0]
    if txt_ids.ndim == 3:
        txt_ids = txt_ids[0]

    image_rotary_emb = self.pos_embed(img_ids)
    text_rotary_emb = self.pos_embed(txt_ids)
    concat_rotary_emb = (
        torch.cat([text_rotary_emb[0], image_rotary_emb[0]], dim=0),
        torch.cat([text_rotary_emb[1], image_rotary_emb[1]], dim=0),
    )

    B, N, C = hidden_states.shape

    # 4. Apply Hilbert reordering to image tokens and image portion of RoPE
    if method == "reorder_shared" and step not in full_attn_step:
        concat_rotary_emb, hidden_states = apply_hilbert_reorder(
            concat_rotary_emb, hidden_states, num_tiles, offset=0
        )
    elif method == "reorder" and step not in full_attn_step:
        concat_rotary_emb, hidden_states = apply_hilbert_reorder_simple(
            concat_rotary_emb, hidden_states, num_tiles, offset=0
        )

    # 5. Double stream blocks
    for index_block, block in enumerate(self.transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb_mod_img=double_stream_mod_img,
            temb_mod_txt=double_stream_mod_txt,
            image_rotary_emb=concat_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    # 6. Concatenate for single stream
    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # Set num_txt_tokens on single-stream processors so Triton kernel
    # knows which tokens are shared (text) vs local (image)
    for block in self.single_transformer_blocks:
        block.attn.processor._tome_info["args"]["num_txt_tokens"] = num_txt_tokens

    # 7. Single stream blocks
    for index_block, block in enumerate(self.single_transformer_blocks):
        hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb_mod=single_stream_mod,
            image_rotary_emb=concat_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    # 8. Remove text tokens
    hidden_states = hidden_states[:, num_txt_tokens:, ...]

    # 9. Recover Hilbert reordering
    if method == "reorder_shared" and step not in full_attn_step:
        concat_rotary_emb, hidden_states = recover_hilbert_reorder(
            concat_rotary_emb, hidden_states, num_tiles, offset=0
        )
    elif method == "reorder" and step not in full_attn_step:
        concat_rotary_emb, hidden_states = recover_hilbert_reorder_simple(
            concat_rotary_emb, hidden_states, num_tiles, offset=0
        )

    hidden_states = hidden_states.reshape(B, -1, C)

    # 10. Output
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if not return_dict:
        return (output,)

    return Flux2Transformer2DModelOutput(sample=output)
