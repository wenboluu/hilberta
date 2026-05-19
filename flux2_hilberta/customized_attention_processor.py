"""
Custom attention processor for FLUX.2 with relocated attention.

Replaces Flux2AttnProcessor with a version that uses Triton kernels
for sparse local attention with Hilbert-ordered tiles.
"""

import os
import math
from typing import Optional
import yaml
import torch
import torch.nn.functional as F

from utils import apply_rotary_emb
from masking_utils import create_hilbert_tile_mask

# Load config at module import
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'), 'r') as _f:
    _config = yaml.safe_load(_f)

_method = _config.get('method', 'reorder_shared').lower()
if _method not in ['masking', 'masking_seam', 'reorder', 'reorder_shared']:
    raise ValueError("config.method must be 'masking', 'masking_seam', 'reorder', or 'reorder_shared'")

# Pre-compute masks for masking method (cached by full seq len, offset, num_tiles)
_mask_cache = {}       # hilbert mask only: {cache_key: [seq_img, seq_img]}
_full_mask_cache = {}  # full mask with text region: {cache_key: [L, L]}

# --- Seam mask: allow boundary tokens to attend across tile edges ---
_seam_mask_cache = {}  # {seq_img: [seq_img, seq_img] bool tensor}


def _compute_seam_mask(H, W, vertical_seams, horizontal_seams, band=1):
    """Compute a boolean mask allowing cross-tile attention at spatial seam boundaries."""
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


def _get_seam_mask(seq_img, device):
    """Get or compute the seam mask for a given image sequence length."""
    if seq_img not in _seam_mask_cache:
        if seq_img == 4096:
            # 64x64 grid, 4 tiles -> seams at columns 16,32,48 and row 32
            mask = _compute_seam_mask(64, 64, vertical_seams=[16, 32, 48], horizontal_seams=[32], band=1)
        elif seq_img == 16384:
            # 128x128 grid, 4 tiles -> seams at columns 32,64,96 and row 64
            mask = _compute_seam_mask(128, 128, vertical_seams=[32, 64, 96], horizontal_seams=[64], band=1)
        else:
            raise ValueError(f"Unsupported image seq len {seq_img} for seam mask")
        _seam_mask_cache[seq_img] = mask.to(device)
    return _seam_mask_cache[seq_img]


def _get_qkv_projections(attn, hidden_states, encoder_hidden_states=None):
    """Get Q/K/V projections, handling fused and unfused cases."""
    if hasattr(attn, 'fused_projections') and attn.fused_projections:
        query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        if encoder_hidden_states is not None and hasattr(attn, 'to_added_qkv'):
            eq, ek, ev = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
        else:
            eq = ek = ev = None
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
            eq = attn.add_q_proj(encoder_hidden_states)
            ek = attn.add_k_proj(encoder_hidden_states)
            ev = attn.add_v_proj(encoder_hidden_states)
        else:
            eq = ek = ev = None
    return query, key, value, eq, ek, ev


class Flux2HilbertAttnProcessor:
    """
    Attention processor for FLUX.2 double-stream blocks with Hilbert relocated attention.
    Uses Triton sparse attention kernel for image-to-image region.
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        step: int = None,
        layer_idx: int = None,
    ) -> torch.Tensor:
        query, key, value, eq, ek, ev = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        # Reshape to [B, S, H, D]
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if eq is not None:
            eq = eq.unflatten(-1, (attn.heads, -1))
            ek = ek.unflatten(-1, (attn.heads, -1))
            ev = ev.unflatten(-1, (attn.heads, -1))

            eq = attn.norm_added_q(eq)
            ek = attn.norm_added_k(ek)

            # Concatenate: [text, image]
            query = torch.cat([eq, query], dim=1)
            key = torch.cat([ek, key], dim=1)
            value = torch.cat([ev, value], dim=1)

        # Apply RoPE in [B, S, H, D] format
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        batch_size = hidden_states.shape[0]
        head_dim = query.shape[-1]
        num_of_tiles = _config['num_tiles']
        full_attn_step = _config.get('full_attn_step', [])
        full_attn_layer = _config.get('full_attn_layer', [])

        use_full_attn = (step in full_attn_step) or (layer_idx in full_attn_layer)
        offset = self._tome_info["args"]["offset"] if hasattr(self, '_tome_info') else 0

        if use_full_attn:
            # Full attention via SDPA (no mask)
            out = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            )
            out = out.transpose(1, 2).flatten(2, 3)

        elif _method in ("masking", "masking_seam"):
            # Masking-based sparse attention via SDPA with pre-computed Hilbert mask
            seq_txt = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            L = query.shape[1]
            seq_img = L - seq_txt

            seam_suffix = "_seam" if _method == "masking_seam" else ""
            full_key = f'{L}_{seq_txt}_{offset}_{num_of_tiles}{seam_suffix}'
            if full_key not in _full_mask_cache:
                hilbert_key = f'{seq_img}_{offset}_{num_of_tiles}'
                if hilbert_key not in _mask_cache:
                    _mask_cache[hilbert_key] = create_hilbert_tile_mask(
                        torch.empty(1, seq_img, device=query.device), num_of_tiles, offset
                    )
                full_mask = torch.zeros(L, L, dtype=torch.bfloat16, device=query.device)
                hilbert_mask = _mask_cache[hilbert_key].to(query.device, torch.bfloat16)
                if _method == "masking_seam":
                    # OR seam mask: allow boundary tokens to attend across tile edges
                    seam = _get_seam_mask(seq_img, query.device)
                    # hilbert_mask is 0/-inf format; seam is bool. Where seam is True, set to 0 (allow).
                    hilbert_mask = hilbert_mask.clone()
                    hilbert_mask[seam] = 0.0
                full_mask[-seq_img:, -seq_img:] = hilbert_mask
                _full_mask_cache[full_key] = full_mask
            attn_mask = _full_mask_cache[full_key].to(query.dtype)

            out = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
            )
            out = out.transpose(1, 2).flatten(2, 3)

        elif _method == "reorder":
            from triton_code.reloc_triton_kernel_parallel_sliding import attention
            # Transpose to [B, H, S, D] for Triton kernel
            q_t = query.transpose(1, 2)
            k_t = key.transpose(1, 2)
            v_t = value.transpose(1, 2)

            seq_txt = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            N_CTX_shared = seq_txt
            N_CTX = q_t.shape[-2] - N_CTX_shared
            sm_scale = 1.0 / math.sqrt(head_dim)

            out = attention(q_t, k_t, v_t, N_CTX_shared, N_CTX, False, sm_scale, num_of_tiles, offset, False)
            out = out.to(query.dtype).transpose(1, 2).flatten(2, 3)

        elif _method == "reorder_shared":
            from triton_code.reloc_triton_kernel_parallel_sliding import attention
            q_t = query.transpose(1, 2)
            k_t = key.transpose(1, 2)
            v_t = value.transpose(1, 2)

            L = q_t.shape[-2]
            seq_txt = encoder_hidden_states.shape[1] if encoder_hidden_states is not None else 0
            seq_img = L - seq_txt

            if seq_img == 4096:
                center_size = 256
            elif seq_img == 16384:
                center_size = 1024
            else:
                raise ValueError(f"Unsupported image seq len {seq_img}")

            N_CTX_shared = seq_txt + center_size
            N_CTX = L - N_CTX_shared
            sm_scale = 1.0 / math.sqrt(head_dim)

            out = attention(q_t, k_t, v_t, N_CTX_shared, N_CTX, False, sm_scale, num_of_tiles, offset, False)
            out = out.to(query.dtype).transpose(1, 2).flatten(2, 3)

        out = out.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states_out, hidden_states_out = out.split_with_sizes(
                [encoder_hidden_states.shape[1], out.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_hidden_states_out = attn.to_add_out(encoder_hidden_states_out)
            hidden_states_out = attn.to_out[0](hidden_states_out)
            hidden_states_out = attn.to_out[1](hidden_states_out)
            return hidden_states_out, encoder_hidden_states_out
        else:
            hidden_states_out = attn.to_out[0](out)
            hidden_states_out = attn.to_out[1](hidden_states_out)
            return hidden_states_out


class Flux2HilbertSingleAttnProcessor:
    """
    Attention processor for FLUX.2 single-stream blocks with Hilbert relocated attention.
    Single-stream blocks fuse QKV + MLP projections (parallel transformer block).
    """

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("Requires PyTorch 2.0")

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb: torch.Tensor | None = None,
        step: int = None,
        layer_idx: int = None,
    ) -> torch.Tensor:
        # Parallel QKV + MLP projection
        hidden_states_proj = attn.to_qkv_mlp_proj(hidden_states)
        qkv, mlp_hidden_states = torch.split(
            hidden_states_proj, [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
        )

        query, key, value = qkv.chunk(3, dim=-1)

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        batch_size = hidden_states.shape[0]
        head_dim = query.shape[-1]
        num_of_tiles = _config['num_tiles']
        full_attn_step = _config.get('full_attn_step', [])
        full_attn_layer = _config.get('full_attn_layer', [])

        use_full_attn = (step in full_attn_step) or (layer_idx in full_attn_layer)
        offset = self._tome_info["args"]["offset"] if hasattr(self, '_tome_info') else 0

        if use_full_attn:
            attn_output = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                attn_mask=attention_mask, dropout_p=0.0, is_causal=False,
            )
            attn_output = attn_output.transpose(1, 2).flatten(2, 3)

        elif _method in ("masking", "masking_seam"):
            # Single-stream: hidden_states is [text, image] concatenated
            num_txt = self._tome_info["args"].get("num_txt_tokens", 0)
            L = query.shape[1]
            seq_img = L - num_txt

            seam_suffix = "_seam" if _method == "masking_seam" else ""
            full_key = f'{L}_{num_txt}_{offset}_{num_of_tiles}{seam_suffix}'
            if full_key not in _full_mask_cache:
                hilbert_key = f'{seq_img}_{offset}_{num_of_tiles}'
                if hilbert_key not in _mask_cache:
                    _mask_cache[hilbert_key] = create_hilbert_tile_mask(
                        torch.empty(1, seq_img, device=query.device), num_of_tiles, offset
                    )
                full_mask = torch.zeros(L, L, dtype=torch.bfloat16, device=query.device)
                hilbert_mask = _mask_cache[hilbert_key].to(query.device, torch.bfloat16)
                if _method == "masking_seam":
                    seam = _get_seam_mask(seq_img, query.device)
                    hilbert_mask = hilbert_mask.clone()
                    hilbert_mask[seam] = 0.0
                full_mask[-seq_img:, -seq_img:] = hilbert_mask
                _full_mask_cache[full_key] = full_mask
            attn_mask = _full_mask_cache[full_key].to(query.dtype)

            attn_output = F.scaled_dot_product_attention(
                query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2),
                attn_mask=attn_mask, dropout_p=0.0, is_causal=False,
            )
            attn_output = attn_output.transpose(1, 2).flatten(2, 3)

        elif _method in ("reorder", "reorder_shared"):
            from triton_code.reloc_triton_kernel_parallel_sliding import attention
            q_t = query.transpose(1, 2)
            k_t = key.transpose(1, 2)
            v_t = value.transpose(1, 2)

            L = q_t.shape[-2]

            if _method == "reorder":
                # For single stream, text+image are already concatenated
                # Use text token count as shared region
                N_CTX_shared = self._tome_info["args"].get("num_txt_tokens", 0)
                N_CTX = L - N_CTX_shared
            else:
                # reorder_shared: text + center as shared
                num_txt = self._tome_info["args"].get("num_txt_tokens", 0)
                img_len = L - num_txt
                if img_len == 4096:
                    center_size = 256
                elif img_len == 16384:
                    center_size = 1024
                else:
                    center_size = 0
                N_CTX_shared = num_txt + center_size
                N_CTX = L - N_CTX_shared

            sm_scale = 1.0 / math.sqrt(head_dim)
            attn_output = attention(q_t, k_t, v_t, N_CTX_shared, N_CTX, False, sm_scale, num_of_tiles, offset, False)
            attn_output = attn_output.to(query.dtype).transpose(1, 2).flatten(2, 3)

        attn_output = attn_output.to(query.dtype)

        # MLP activation
        mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)

        # Concatenate and output projection
        out = torch.cat([attn_output, mlp_hidden_states], dim=-1)
        out = attn.to_out(out)

        return out
