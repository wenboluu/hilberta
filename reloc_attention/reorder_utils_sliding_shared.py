import torch

from typing import Any, Dict, Optional, Tuple, Union
from functools import lru_cache
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from utils import get_hilbert_flat_indices, get_inverse_hilbert_indices
logger = logging.get_logger(__name__)



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
    """
    x_tiled: (B * num_tiles, tile_side², C)
    Returns: (B, H*W, C)
    """
    B, HW, C = x_tiled.shape
    H = W = int(HW**0.5)
    num_tiles_per_side = int(num_tiles**0.5)
    tile_side_len = H // num_tiles_per_side

    # reshape each tile to (B, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, C)
    x_tiles = x_tiled.view(
        B, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, C
    )

    # move tiles back into image
    x_tiles = x_tiles.permute(0, 1, 3, 2, 4, 5).contiguous()
    x_tiles = x_tiles.view(B, H, W, C)

    return x_tiles.view(B, H * W, C)

@lru_cache(maxsize=None)
def _base_hilbert_indices(sequence_length: int) -> torch.Tensor:
    if sequence_length == 4096:
        return get_hilbert_flat_indices(6)
    if sequence_length == 16384:
        return get_hilbert_flat_indices(7)
    raise ValueError(f"Unsupported sequence_length {sequence_length} for Hilbert ordering")


@lru_cache(maxsize=None)
def _base_inverse_hilbert_indices(sequence_length: int) -> torch.Tensor:
    if sequence_length == 4096:
        return get_inverse_hilbert_indices(6)
    if sequence_length == 16384:
        return get_inverse_hilbert_indices(7)
    raise ValueError(f"Unsupported sequence_length {sequence_length} for Hilbert ordering")

_DEVICE_INDEX_CACHE: Dict[Tuple[int, torch.device], torch.Tensor] = {}
_DEVICE_INVERSE_CACHE: Dict[Tuple[int, torch.device], torch.Tensor] = {}


def _get_index_on_device(sequence_length: int, device: torch.device) -> torch.Tensor:
    key = (sequence_length, device)
    cached = _DEVICE_INDEX_CACHE.get(key)
    if cached is None or cached.device != device:
        cached = _base_hilbert_indices(sequence_length).to(device, non_blocking=True)
        _DEVICE_INDEX_CACHE[key] = cached
    return cached

def _get_inverse_on_device(sequence_length: int, device: torch.device) -> torch.Tensor:
    key = (sequence_length, device)
    cached = _DEVICE_INVERSE_CACHE.get(key)
    if cached is None or cached.device != device:
        cached = _base_inverse_hilbert_indices(sequence_length).to(device, non_blocking=True)
        _DEVICE_INVERSE_CACHE[key] = cached
    return cached


def hilbert_tile(x, offset=0):
    sequence_length = x.shape[1]
    hilbert_index = _get_index_on_device(sequence_length, x.device)
    if offset:
        hilbert_index = torch.remainder(hilbert_index - offset, sequence_length)

    gather_index = hilbert_index.view(1, sequence_length, 1).expand(x.shape[0], sequence_length, x.shape[2])
    return torch.gather(x, 1, gather_index)


def hilbert_untile(x_hilbert, offset=0):
    sequence_length = x_hilbert.shape[1]
    inverse_index = _get_inverse_on_device(sequence_length, x_hilbert.device)

    if offset:
        x_hilbert = torch.roll(x_hilbert, shifts=offset, dims=1)

    gather_index = inverse_index.view(1, sequence_length, 1).expand(x_hilbert.shape[0], sequence_length, x_hilbert.shape[2])
    return torch.gather(x_hilbert, 1, gather_index)


def apply_hilbert_reorder_simple(image_rotary_emb, hidden_states, num_tiles, offset=0):
    """
    Simple version: Reorder ALL image tokens using Hilbert curve (no center/sparse separation).

    Args:
        image_rotary_emb (tuple): (rotary_emb_1, rotary_emb_2), shape [512 + H*W, dim]
        hidden_states (torch.Tensor): [B, H*W, C]
        num_tiles (int): Number of tiles
        offset (int): Hilbert offset for sliding window

    Returns:
        image_rotary_emb: Tuple of reordered RoPE [text(512), image(4096)]
        hidden_states: Reordered hidden states [B, 4096, C]
    """
    image_rotary_emb_1, image_rotary_emb_2 = image_rotary_emb

    # Split text and image parts
    text_emb_1, image_emb_1 = image_rotary_emb_1[:512], image_rotary_emb_1[512:]
    text_emb_2, image_emb_2 = image_rotary_emb_2[:512], image_rotary_emb_2[512:]

    sequence_length = hidden_states.shape[1]

    # Get Hilbert indices with offset
    hilbert_index = _get_index_on_device(sequence_length, hidden_states.device)
    if offset:
        hilbert_index = torch.remainder(hilbert_index - offset, sequence_length)

    # Sort all tokens by Hilbert order
    hilbert_sorted_positions = torch.argsort(hilbert_index)

    # Reorder ALL image tokens
    hilbert_hidden = hidden_states[:, hilbert_sorted_positions, :]
    hilbert_emb_1 = image_emb_1[hilbert_sorted_positions, :]
    hilbert_emb_2 = image_emb_2[hilbert_sorted_positions, :]

    # Concatenate with text
    image_rotary_emb_1 = torch.cat([text_emb_1, hilbert_emb_1], dim=0)
    image_rotary_emb_2 = torch.cat([text_emb_2, hilbert_emb_2], dim=0)

    return (image_rotary_emb_1, image_rotary_emb_2), hilbert_hidden


def recover_hilbert_reorder_simple(image_rotary_emb, hidden_states, num_tiles, offset=0):
    """
    Simple version: Recover original spatial order from Hilbert-reordered tokens.

    Args:
        image_rotary_emb (tuple): (rotary_emb_1, rotary_emb_2), shape [512 + H*W, dim]
        hidden_states (torch.Tensor): [B, H*W, C] in Hilbert order
        num_tiles (int): Number of tiles
        offset (int): Hilbert offset used during reordering

    Returns:
        image_rotary_emb: Tuple of recovered RoPE
        hidden_states: Recovered hidden states [B, H*W, C] in spatial order
    """
    image_rotary_emb_1, image_rotary_emb_2 = image_rotary_emb

    # Split text and image parts
    text_emb_1, image_emb_1 = image_rotary_emb_1[:512], image_rotary_emb_1[512:]
    text_emb_2, image_emb_2 = image_rotary_emb_2[:512], image_rotary_emb_2[512:]

    sequence_length = hidden_states.shape[1]

    # Get Hilbert indices with offset (same as forward)
    hilbert_index = _get_index_on_device(sequence_length, hidden_states.device)
    if offset:
        hilbert_index = torch.remainder(hilbert_index - offset, sequence_length)

    hilbert_sorted_positions = torch.argsort(hilbert_index)

    # Get inverse permutation
    inverse_positions = torch.argsort(hilbert_sorted_positions)

    # Recover original spatial order
    recovered_hidden = hidden_states[:, inverse_positions, :]
    recovered_emb_1 = image_emb_1[inverse_positions, :]
    recovered_emb_2 = image_emb_2[inverse_positions, :]

    # Concatenate with text
    image_rotary_emb_1 = torch.cat([text_emb_1, recovered_emb_1], dim=0)
    image_rotary_emb_2 = torch.cat([text_emb_2, recovered_emb_2], dim=0)

    return (image_rotary_emb_1, image_rotary_emb_2), recovered_hidden


def apply_hilbert_reorder(image_rotary_emb, hidden_states, num_tiles, offset = 0):
    """
    Reorders the image part of rotary embeddings and hidden states using Hilbert curve,
    separating center shared region from sparse region.

    The center region is treated as globally-attentive (shared with text tokens),
    while the remaining sparse tokens use local attention with Hilbert reordering.

    Args:
        image_rotary_emb (tuple of torch.Tensor): Tuple of (rotary_emb_1, rotary_emb_2),
            each of shape [512 + H*W, dim], where 512 is text embedding and the rest is image.
        hidden_states (torch.Tensor): Input hidden states of shape [B, N, C], where N = H * W.
        num_tiles (int): Number of spatial tiles to split image into (e.g., 16 for 4x4).
        offset (int): Hilbert curve offset for sliding window.

    Returns:
        image_rotary_emb: Tuple of reordered rotary embeddings [text, center, sparse]
        hidden_states: Reordered hidden states [B, center+sparse, C]
    """
    image_rotary_emb_1, image_rotary_emb_2 = image_rotary_emb

    # Split rotary embeddings into text and image parts
    text_emb_1, image_emb_1 = image_rotary_emb_1[:512], image_rotary_emb_1[512:]
    text_emb_2, image_emb_2 = image_rotary_emb_2[:512], image_rotary_emb_2[512:]

    # Determine center region size based on sequence length
    # This matches the center region defined in masking_utils.py
    sequence_length = hidden_states.shape[1]
    if sequence_length == 4096:
        H = W = 64
        center_start, center_end = 24, 40
        center_size = 16 * 16  # 256
    elif sequence_length == 16384:
        H = W = 128
        center_start, center_end = 48, 80
        center_size = 32 * 32  # 1024
    else:
        raise ValueError(f"Unsupported sequence length: {sequence_length}")

    # Extract center region indices (rows 24-40, cols 24-40 for 4096)
    center_indices = []
    for i in range(center_start, center_end):
        for j in range(center_start, center_end):
            center_indices.append(i * W + j)
    center_indices = torch.tensor(center_indices, device=hidden_states.device)

    # Create mask for non-center (sparse) tokens
    all_indices = torch.arange(sequence_length, device=hidden_states.device)
    is_center = torch.zeros(sequence_length, dtype=torch.bool, device=hidden_states.device)
    is_center[center_indices] = True
    sparse_indices = all_indices[~is_center]

    # Extract center and sparse tokens
    center_hidden = hidden_states[:, center_indices, :]
    sparse_hidden = hidden_states[:, sparse_indices, :]

    center_emb_1 = image_emb_1[center_indices, :]
    sparse_emb_1 = image_emb_1[sparse_indices, :]

    center_emb_2 = image_emb_2[center_indices, :]
    sparse_emb_2 = image_emb_2[sparse_indices, :]

    # Apply Hilbert reorder to sparse tokens
    # Get full Hilbert indices for the entire image
    full_hilbert_index = _get_index_on_device(sequence_length, hidden_states.device)
    if offset:
        full_hilbert_index = torch.remainder(full_hilbert_index - offset, sequence_length)

    # Extract Hilbert indices for sparse positions
    sparse_hilbert_indices = full_hilbert_index[sparse_indices]

    # Sort to get the reordering: which sparse token goes where
    sorted_positions = torch.argsort(sparse_hilbert_indices)

    # Reorder sparse tokens according to their Hilbert curve order
    sparse_hidden = sparse_hidden[:, sorted_positions, :]
    sparse_emb_1 = sparse_emb_1[sorted_positions, :]
    sparse_emb_2 = sparse_emb_2[sorted_positions, :]

    # Concatenate in order: [center, sparse]
    # This creates the sequence: [center(256), sparse(3840)] for 4096
    reordered_image_hidden = torch.cat([center_hidden, sparse_hidden], dim=1)
    reordered_image_emb_1 = torch.cat([center_emb_1, sparse_emb_1], dim=0)
    reordered_image_emb_2 = torch.cat([center_emb_2, sparse_emb_2], dim=0)

    # Concatenate with text embeddings: [text(512), center(256), sparse(3840)] = 4608 total
    image_rotary_emb_1 = torch.cat([text_emb_1, reordered_image_emb_1], dim=0)
    image_rotary_emb_2 = torch.cat([text_emb_2, reordered_image_emb_2], dim=0)

    return (image_rotary_emb_1, image_rotary_emb_2), reordered_image_hidden


def recover_hilbert_reorder(image_rotary_emb, hidden_states, num_tiles, offset=0):
    """
    Recovers original ordering from Hilbert-tiled embeddings and hidden states.
    Reverses the [center, sparse] reordering back to original spatial layout.

    Args:
        image_rotary_emb (tuple): (image_rotary_emb_1, image_rotary_emb_2),
            each of shape [512 + center + sparse, dim]
        hidden_states (torch.Tensor): Shape [B, center+sparse, C]
        num_tiles (int): Number of tiles per original batch
        offset (int): Hilbert offset used during tiling

    Returns:
        image_rotary_emb: Tuple of [512 + HW, dim] restored rotary embeddings
        hidden_states: [B, HW, C] recovered hidden states in original spatial order
    """
    image_rotary_emb_1, image_rotary_emb_2 = image_rotary_emb

    # Split text + image parts
    text_emb_1, tiled_image_emb_1 = image_rotary_emb_1[:512], image_rotary_emb_1[512:]
    text_emb_2, tiled_image_emb_2 = image_rotary_emb_2[:512], image_rotary_emb_2[512:]

    # Determine sequence length and center size
    sequence_length = hidden_states.shape[1]
    if sequence_length == 4096:
        H = W = 64
        center_start, center_end = 24, 40
        center_size = 16 * 16  # 256
    elif sequence_length == 16384:
        H = W = 128
        center_start, center_end = 48, 80
        center_size = 32 * 32  # 1024
    else:
        raise ValueError(f"Unsupported sequence length: {sequence_length}")

    sparse_size = sequence_length - center_size

    # Split center and sparse parts
    center_hidden = hidden_states[:, :center_size, :]
    sparse_hidden = hidden_states[:, center_size:, :]

    center_emb_1 = tiled_image_emb_1[:center_size, :]
    sparse_emb_1 = tiled_image_emb_1[center_size:, :]

    center_emb_2 = tiled_image_emb_2[:center_size, :]
    sparse_emb_2 = tiled_image_emb_2[center_size:, :]

    # Create center indices first (needed for sparse indices)
    center_indices = []
    for i in range(center_start, center_end):
        for j in range(center_start, center_end):
            center_indices.append(i * W + j)
    center_indices = torch.tensor(center_indices, device=hidden_states.device)

    # Create sparse indices
    all_indices = torch.arange(sequence_length, device=hidden_states.device)
    is_center = torch.zeros(sequence_length, dtype=torch.bool, device=hidden_states.device)
    is_center[center_indices] = True
    sparse_indices = all_indices[~is_center]

    # Undo Hilbert ordering on sparse tokens
    # Get full Hilbert indices for the entire image
    full_hilbert_index = _get_index_on_device(sequence_length, hidden_states.device)
    if offset:
        full_hilbert_index = torch.remainder(full_hilbert_index - offset, sequence_length)

    # Extract Hilbert indices for sparse positions
    sparse_hilbert_indices = full_hilbert_index[sparse_indices]

    # Get inverse sorting to recover original order
    sorted_positions = torch.argsort(sparse_hilbert_indices)
    inverse_positions = torch.argsort(sorted_positions)

    # Recover original order of sparse tokens
    sparse_hidden = sparse_hidden[:, inverse_positions, :]
    sparse_emb_1 = sparse_emb_1[inverse_positions, :]
    sparse_emb_2 = sparse_emb_2[inverse_positions, :]

    # Reconstruct full image tensor
    B, _, C = center_hidden.shape
    recovered_image_hidden = torch.zeros(B, sequence_length, C, device=hidden_states.device, dtype=hidden_states.dtype)
    recovered_image_emb_1 = torch.zeros(sequence_length, center_emb_1.shape[1], device=hidden_states.device, dtype=center_emb_1.dtype)
    recovered_image_emb_2 = torch.zeros(sequence_length, center_emb_2.shape[1], device=hidden_states.device, dtype=center_emb_2.dtype)

    recovered_image_hidden[:, center_indices, :] = center_hidden
    recovered_image_hidden[:, sparse_indices, :] = sparse_hidden

    recovered_image_emb_1[center_indices, :] = center_emb_1
    recovered_image_emb_1[sparse_indices, :] = sparse_emb_1

    recovered_image_emb_2[center_indices, :] = center_emb_2
    recovered_image_emb_2[sparse_indices, :] = sparse_emb_2

    # Concatenate with text embeddings
    image_rotary_emb_1 = torch.cat([text_emb_1, recovered_image_emb_1], dim=0)
    image_rotary_emb_2 = torch.cat([text_emb_2, recovered_image_emb_2], dim=0)
    image_rotary_emb = (image_rotary_emb_1, image_rotary_emb_2)

    return image_rotary_emb, recovered_image_hidden


def customized_forward(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states: torch.Tensor = None,
    pooled_projections: torch.Tensor = None,
    timestep: torch.LongTensor = None,
    img_ids: torch.Tensor = None,
    txt_ids: torch.Tensor = None,
    guidance: torch.Tensor = None,
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_block_samples=None,
    controlnet_single_block_samples=None,
    return_dict: bool = True,
    controlnet_blocks_repeat: bool = False,
    step: int = 0,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    def load_config(config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    # Load config from same directory as this script
    import os as _os
    _script_dir = _os.path.dirname(_os.path.abspath(__file__))
    config = load_config(_os.path.join(_script_dir, "config.yaml"))

    num_tiles = config['num_tiles'] if 'num_tiles' in config else 16

    if joint_attention_kwargs is not None:
        joint_attention_kwargs = joint_attention_kwargs.copy()
        lora_scale = joint_attention_kwargs.pop("scale", 1.0)
    else:
        lora_scale = 1.0

    if USE_PEFT_BACKEND:
        # weight the lora layers by setting `lora_scale` for each PEFT layer
        scale_lora_layers(self, lora_scale)
    else:
        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )
    hidden_states = self.x_embedder(hidden_states)

    timestep = timestep.to(hidden_states.dtype) * 1000
    if guidance is not None:
        guidance = guidance.to(hidden_states.dtype) * 1000
    else:
        guidance = None
    temb = (
        self.time_text_embed(timestep, pooled_projections)
        if guidance is None
        else self.time_text_embed(timestep, guidance, pooled_projections)
    )
    encoder_hidden_states = self.context_embedder(encoder_hidden_states)

    if txt_ids.ndim == 3:
        logger.warning(
            "Passing `txt_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        txt_ids = txt_ids[0]
    if img_ids.ndim == 3:
        logger.warning(
            "Passing `img_ids` 3d torch.Tensor is deprecated."
            "Please remove the batch dimension and pass it as a 2d torch Tensor"
        )
        img_ids = img_ids[0]

    ids = torch.cat((txt_ids, img_ids), dim=0)
    image_rotary_emb = self.pos_embed(ids)
    B, N, C = hidden_states.shape

    # Apply initial Hilbert reordering - SIMPLE VERSION (no center/sparse separation)
    image_rotary_emb, hidden_states = apply_hilbert_reorder_simple(
        image_rotary_emb, hidden_states, num_tiles, offset = 0
    )

    for index_block, block in enumerate(self.transformer_blocks):

        # # # Entering the block
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    for index_block, block in enumerate(self.single_transformer_blocks):
        # # # Entering the block
        hidden_states = block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, :]
    # Recover using simple version (no center/sparse separation)
    image_rotary_emb, hidden_states = recover_hilbert_reorder_simple(
        image_rotary_emb, hidden_states, num_tiles, offset = 0
    )
    hidden_states = hidden_states.reshape(B, -1, C)

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)