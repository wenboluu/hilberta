import os
from typing import Optional

import torch
import torch.nn.functional as F
import yaml

from .utils import apply_rotary_emb

# Load config
with open('./src/config.yaml', 'r') as f:
    _config = yaml.safe_load(f)

num_of_tiles = _config['tiling']['num_tiles']
full_attn_step = _config['attention']['full_attn_step']
full_attn_layer = _config['attention']['full_attn_layer']
enable_seam = _config.get('seam', {}).get('enabled', False)
seam_band = _config.get('seam', {}).get('band', 1)

# Load precomputed Hilbert-curve attention masks
_mask_dir = './masks'
_mask_files = [
    (4096, 0, 4),   (4096, 256, 4),  (4096, 512, 4),  (4096, 768, 4),
    (4096, 0, 16),  (4096, 64, 16),  (4096, 128, 16), (4096, 192, 16),
]

masks = {}
for _img_size, _offset, _ntiles in _mask_files:
    _key = f"{_img_size}_{_offset}_{_ntiles}"
    _path = os.path.join(_mask_dir, f"image_size_{_img_size}_offset_{_offset}_num_of_tiles_{_ntiles}.pt")
    _raw = torch.load(_path, map_location=torch.device('cuda:0'))
    masks[_key] = torch.isfinite(_raw)


def _compute_seam_mask(H, W, vertical_seams, horizontal_seams, band=1):
    """Build a boolean mask allowing boundary tokens to attend across tile seams."""
    N = H * W
    seam = torch.zeros(N, N, dtype=torch.bool)
    rows = torch.arange(H)
    cols = torch.arange(W)
    R, C = torch.meshgrid(rows, cols, indexing="ij")
    flat = (R * W + C).reshape(-1)

    for s in vertical_seams:
        left = flat[((C >= s - band) & (C < s)).reshape(-1)]
        right = flat[((C >= s) & (C < s + band)).reshape(-1)]
        seam[left[:, None], right[None, :]] = True
        seam[right[:, None], left[None, :]] = True

    for s in horizontal_seams:
        top = flat[((R >= s - band) & (R < s)).reshape(-1)]
        bot = flat[((R >= s) & (R < s + band)).reshape(-1)]
        seam[top[:, None], bot[None, :]] = True
        seam[bot[:, None], top[None, :]] = True

    return seam


# 64x64 latent grid (1024px images), 4 tiles -> seams at columns 16,32,48 and row 32
seam_mask_4096 = None
if enable_seam:
    seam_mask_4096 = _compute_seam_mask(
        64, 64, vertical_seams=[16, 32, 48], horizontal_seams=[32], band=seam_band
    ).cuda()


class HilbertaAttnProcessor:
    """Attention processor that applies Hilbert-curve tiled masking with optional seam attention."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("HilbertaAttnProcessor requires PyTorch 2.0+.")

    def __call__(
        self,
        attn,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        step=None,
        layer_idx=None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # Joint attention: FluxTransformerBlock has encoder_hidden_states,
        # FluxSingleTransformerBlock does not
        if encoder_hidden_states is not None:
            enc_q = attn.add_q_proj(encoder_hidden_states)
            enc_k = attn.add_k_proj(encoder_hidden_states)
            enc_v = attn.add_v_proj(encoder_hidden_states)

            enc_q = enc_q.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_k = enc_k.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            enc_v = enc_v.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_added_q is not None:
                enc_q = attn.norm_added_q(enc_q)
            if attn.norm_added_k is not None:
                enc_k = attn.norm_added_k(enc_k)

            query = torch.cat([enc_q, query], dim=2)
            key = torch.cat([enc_k, key], dim=2)
            value = torch.cat([enc_v, value], dim=2)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # Build Hilbert-curve tiled attention mask
        offset = self._hilberta_info["offset"]
        L, S = query.shape[-2], key.shape[-2]
        image_size = 4096 if L == 4608 else 16384

        attn_mask = torch.ones(L, S, dtype=torch.bool, device=query.device)
        if step not in full_attn_step and layer_idx not in full_attn_layer:
            tile_mask = masks[f'{image_size}_{offset}_{num_of_tiles}'].to(query.device)
            attn_mask[-image_size:, -image_size:] = tile_mask
            if enable_seam and image_size == 4096 and seam_mask_4096 is not None:
                attn_mask[-image_size:, -image_size:] |= seam_mask_4096

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1]:],
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
