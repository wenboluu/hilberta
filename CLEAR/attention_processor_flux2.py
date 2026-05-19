"""
CLEAR local window attention processors for FLUX.2-klein.

Adapted from attention_processor.py (FLUX.1) with these FLUX.2-specific changes:
- Fused/unfused QKV support (to_qkv or to_q/to_k/to_v)
- Tensor layout [B, S, H, D] via unflatten (not view+transpose)
- RoPE with sequence_dim=1
- Separate single-stream processor for to_qkv_mlp_proj + MLP activation

Four processor classes:
- Flux2AttnProcessor: teacher full attention (double-stream)
- Flux2SingleAttnProcessor: teacher full attention (single-stream)
- LocalFlexFlux2AttnProcessor: student local window (double-stream)
- LocalFlexFlux2SingleAttnProcessor: student local window (single-stream)
"""

import torch
import torch.nn.functional as F
import math
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
create_block_mask = torch.compile(create_block_mask)
from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import apply_rotary_emb
from typing import Optional
from functools import partial, lru_cache


# Global state for distillation output collection
attn_outputs_teacher = []
attn_outputs = []
BLOCK_MASK = None
HEIGHT = None
WIDTH = None


@lru_cache
def init_local_mask_flex(height, width, text_length, window_size, device):
    """Initialize local attention block mask. Same logic as CLEAR original."""
    def local_mask(b, h, q_idx, kv_idx):
        q_y = (q_idx - text_length) // width
        q_x = (q_idx - text_length) % width
        kv_y = (kv_idx - text_length) // width
        kv_x = (kv_idx - text_length) % width
        return torch.logical_or(
            torch.logical_or(q_idx < text_length, kv_idx < text_length),
            (q_y - kv_y) ** 2 + (q_x - kv_x) ** 2 < window_size ** 2
        )

    global BLOCK_MASK, HEIGHT, WIDTH
    BLOCK_MASK = create_block_mask(
        local_mask, B=None, H=None, device=device,
        Q_LEN=text_length + height * width,
        KV_LEN=text_length + height * width, _compile=True
    )
    HEIGHT = height
    WIDTH = width


# ─── Helper: FLUX.2 QKV projection ───

def _get_qkv(attn, hidden_states):
    if hasattr(attn, 'fused_projections') and attn.fused_projections:
        return attn.to_qkv(hidden_states).chunk(3, dim=-1)
    return attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)


def _get_context_qkv(attn, encoder_hidden_states):
    if hasattr(attn, 'fused_projections') and attn.fused_projections and hasattr(attn, 'to_added_qkv'):
        return attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)
    return attn.add_q_proj(encoder_hidden_states), attn.add_k_proj(encoder_hidden_states), attn.add_v_proj(encoder_hidden_states)


# ─── Double-stream: shared forward logic ───

def _double_stream_forward(attn, hidden_states, encoder_hidden_states, image_rotary_emb, attn_fn):
    """Shared QKV + RoPE logic for double-stream blocks. Returns final output tensor."""
    query, key, value = _get_qkv(attn, hidden_states)

    query = query.unflatten(-1, (attn.heads, -1))
    key = key.unflatten(-1, (attn.heads, -1))
    value = value.unflatten(-1, (attn.heads, -1))
    query = attn.norm_q(query)
    key = attn.norm_k(key)

    if encoder_hidden_states is not None:
        eq, ek, ev = _get_context_qkv(attn, encoder_hidden_states)
        eq = eq.unflatten(-1, (attn.heads, -1))
        ek = ek.unflatten(-1, (attn.heads, -1))
        ev = ev.unflatten(-1, (attn.heads, -1))
        eq = attn.norm_added_q(eq)
        ek = attn.norm_added_k(ek)
        query = torch.cat([eq, query], dim=1)
        key = torch.cat([ek, key], dim=1)
        value = torch.cat([ev, value], dim=1)

    if image_rotary_emb is not None:
        query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
        key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

    # Attention: transpose to [B, H, S, D], compute, transpose back
    out = attn_fn(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
    out = out.transpose(1, 2).flatten(2, 3).to(query.dtype)

    if encoder_hidden_states is not None:
        txt_len = encoder_hidden_states.shape[1]
        encoder_out, hidden_out = out[:, :txt_len], out[:, txt_len:]
        hidden_out = attn.to_out[0](hidden_out)
        hidden_out = attn.to_out[1](hidden_out)
        encoder_out = attn.to_add_out(encoder_out)
        return hidden_out, encoder_out
    return out


# ─── Single-stream: shared forward logic ───

def _single_stream_forward(attn, hidden_states, image_rotary_emb, attn_fn):
    """Shared QKV+MLP logic for single-stream blocks. Returns final output tensor + raw attn_output."""
    hidden_states_proj = attn.to_qkv_mlp_proj(hidden_states)
    qkv, mlp_hidden_states = torch.split(
        hidden_states_proj,
        [3 * attn.inner_dim, attn.mlp_hidden_dim * attn.mlp_mult_factor], dim=-1
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

    attn_output = attn_fn(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2))
    attn_output = attn_output.transpose(1, 2).flatten(2, 3).to(query.dtype)

    mlp_hidden_states = attn.mlp_act_fn(mlp_hidden_states)
    out = torch.cat([attn_output, mlp_hidden_states], dim=-1)
    out = attn.to_out(out)
    return out, attn_output


# ─── SDPA attention function ───

def _sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)


# ─── Teacher Processors ───

class Flux2AttnProcessor:
    """Full attention for FLUX.2 double-stream blocks (teacher)."""
    def __init__(self, distill=False):
        self.distill = distill

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, **kwargs):
        result = _double_stream_forward(attn, hidden_states, encoder_hidden_states, image_rotary_emb, _sdpa)
        if encoder_hidden_states is None and self.distill:
            attn_outputs_teacher.append(result.detach())
        return result


class Flux2SingleAttnProcessor:
    """Full attention for FLUX.2 single-stream blocks (teacher)."""
    def __init__(self, distill=False):
        self.distill = distill

    def __call__(self, attn, hidden_states, attention_mask=None,
                 image_rotary_emb=None, **kwargs):
        out, attn_output = _single_stream_forward(attn, hidden_states, image_rotary_emb, _sdpa)
        if self.distill:
            attn_outputs_teacher.append(attn_output.detach())
        return out


# ─── Student Processors ───

class LocalFlexFlux2AttnProcessor:
    """Local window attention for FLUX.2 double-stream blocks (student)."""
    def __init__(self, distill=False):
        assert BLOCK_MASK is not None, "Call init_local_mask_flex() first"
        self.flex_attn = torch.compile(partial(flex_attention, block_mask=BLOCK_MASK), dynamic=False)
        self.distill = distill

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, image_rotary_emb=None, **kwargs):
        result = _double_stream_forward(attn, hidden_states, encoder_hidden_states, image_rotary_emb, self.flex_attn)
        if encoder_hidden_states is not None and self.distill:
            # result is (hidden_out, encoder_out) tuple; collect hidden_out
            attn_outputs.append(result[0].detach())
        elif encoder_hidden_states is None and self.distill:
            attn_outputs.append(result.detach())
        return result


class LocalFlexFlux2SingleAttnProcessor:
    """Local window attention for FLUX.2 single-stream blocks (student)."""
    def __init__(self, distill=False):
        assert BLOCK_MASK is not None, "Call init_local_mask_flex() first"
        self.flex_attn = torch.compile(partial(flex_attention, block_mask=BLOCK_MASK), dynamic=False)
        self.distill = distill

    def __call__(self, attn, hidden_states, attention_mask=None,
                 image_rotary_emb=None, **kwargs):
        out, attn_output = _single_stream_forward(attn, hidden_states, image_rotary_emb, self.flex_attn)
        if self.distill:
            attn_outputs.append(attn_output.detach())
        return out
