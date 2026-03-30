import os
import math
import inspect
from typing import (
    Callable, 
    List, 
    Optional, 
    Tuple, 
    Union, 
    Type, 
    Dict, 
    Any
)
import pdb
import yaml
import torch
import torch.nn.functional as F
from torch import nn
import torch._dynamo as dynamo

# Import Attention and standard processors from diffusers
from diffusers.models.attention_processor import Attention, AttnProcessor, AttnProcessor2_0

from utils import isinstance_str, init_generator, apply_rotary_emb
# Load mask files by curve type from config
with open('./config.yaml', 'r') as _f:
    _config = yaml.safe_load(_f)
_curve_type = _config.get('curve_type', 'hilbert').lower()
if _curve_type not in ['hilbert', 'morton']:
    raise ValueError("config.curve_type must be 'hilbert' or 'morton'")
mask_dir = f'./mask_{_curve_type}'

# Load 4096 masks with 4 tiles
mask_4096_0_4 = torch.load(os.path.join(mask_dir, 'image_size_4096_offset_0_num_of_tiles_4.pt'), map_location=torch.device('cuda:0'))
mask_4096_256_4 = torch.load(os.path.join(mask_dir, 'image_size_4096_offset_256_num_of_tiles_4.pt'), map_location=torch.device('cuda:0'))
mask_4096_512_4 = torch.load(os.path.join(mask_dir, 'image_size_4096_offset_512_num_of_tiles_4.pt'), map_location=torch.device('cuda:0'))
mask_4096_768_4 = torch.load(os.path.join(mask_dir, 'image_size_4096_offset_768_num_of_tiles_4.pt'), map_location=torch.device('cuda:0'))

# Load 4096 masks with 16 tiles
mask_4096_0_16 = torch.load(os.path.join(mask_dir, 'image_size_4096_offset_0_num_of_tiles_16.pt'), map_location=torch.device('cuda:0'))
mask_4096_64_16 = torch.load(os.path.join(mask_dir, 'image_size_4096_offset_64_num_of_tiles_16.pt'), map_location=torch.device('cuda:0'))
mask_4096_128_16 = torch.load(os.path.join(mask_dir, 'image_size_4096_offset_128_num_of_tiles_16.pt'), map_location=torch.device('cuda:0'))
mask_4096_192_16 = torch.load(os.path.join(mask_dir, 'image_size_4096_offset_192_num_of_tiles_16.pt'), map_location=torch.device('cuda:0'))

# Load 16384 masks with 4 tiles
mask_16384_0_4 = torch.load(os.path.join(mask_dir, 'image_size_16384_offset_0_num_of_tiles_4.pt'), map_location=torch.device('cuda:0'))
mask_16384_1024_4 = torch.load(os.path.join(mask_dir, 'image_size_16384_offset_1024_num_of_tiles_4.pt'), map_location=torch.device('cuda:0'))
mask_16384_2048_4 = torch.load(os.path.join(mask_dir, 'image_size_16384_offset_2048_num_of_tiles_4.pt'), map_location=torch.device('cuda:0'))
mask_16384_3072_4 = torch.load(os.path.join(mask_dir, 'image_size_16384_offset_3072_num_of_tiles_4.pt'), map_location=torch.device('cuda:0'))

# Load 16384 masks with 16 tiles
mask_16384_0_16 = torch.load(os.path.join(mask_dir, 'image_size_16384_offset_0_num_of_tiles_16.pt'), map_location=torch.device('cuda:0'))
mask_16384_256_16 = torch.load(os.path.join(mask_dir, 'image_size_16384_offset_256_num_of_tiles_16.pt'), map_location=torch.device('cuda:0'))
mask_16384_512_16 = torch.load(os.path.join(mask_dir, 'image_size_16384_offset_512_num_of_tiles_16.pt'), map_location=torch.device('cuda:0'))
mask_16384_768_16 = torch.load(os.path.join(mask_dir, 'image_size_16384_offset_768_num_of_tiles_16.pt'), map_location=torch.device('cuda:0'))

# Create mask dictionary
mask_list = {
    # 4096 masks with 4 tiles
    '4096_0_4': mask_4096_0_4,
    '4096_256_4': mask_4096_256_4,
    '4096_512_4': mask_4096_512_4,
    '4096_768_4': mask_4096_768_4,
    
    # 4096 masks with 16 tiles
    '4096_0_16': mask_4096_0_16,
    '4096_64_16': mask_4096_64_16,
    '4096_128_16': mask_4096_128_16,
    '4096_192_16': mask_4096_192_16,

    # 16384 masks with 4 tiles
    '16384_0_4': mask_16384_0_4,
    '16384_1024_4': mask_16384_1024_4,
    '16384_2048_4': mask_16384_2048_4,
    '16384_3072_4': mask_16384_3072_4,

    # 16384 masks with 16 tiles
    '16384_0_16': mask_16384_0_16,
    '16384_256_16': mask_16384_256_16,
    '16384_512_16': mask_16384_512_16,
    '16384_768_16': mask_16384_768_16,
}


class FluxAttnProcessor2_0_for_transformerblock_global:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("FluxAttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
        step = None,
        layer_idx = None,
    ) -> torch.FloatTensor:
        batch_size, _, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        # `sample` projections.
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

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:

            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # attention
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        if image_rotary_emb is not None:
            from utils import apply_rotary_emb

            query = apply_rotary_emb(query, image_rotary_emb)
            key = apply_rotary_emb(key, image_rotary_emb)

        # ======================Uncomment For Inference Using Pytorch======================
        num_of_tiles = _config['num_tiles']
        full_attn_step = _config['full_attn_step']
        full_attn_layer = _config['full_attn_layer']

        offset = self._tome_info["args"]["offset"]
        L, S = query.shape[-2], key.shape[-2]
        if L == 4608:
            image_size = 4096
        else:
            image_size = 16384

        attn_mask = torch.zeros(L, S, dtype=query.dtype, device=query.device)
        if step not in full_attn_step and layer_idx not in full_attn_layer:
            mask = mask_list[f'{image_size}_{offset}_{num_of_tiles}']
            attn_mask[-image_size:, -image_size:] = attn_mask[-image_size:, -image_size:] + mask

        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        # ======================Uncomment For Inference Using Pytorch======================

        # ======================Uncomment For Inference Using Triton======================
        # N_CTX_shared = 512
        # N_CTX = query.shape[-2] - N_CTX_shared
        # GROUPS = 4

        # query_shared = query[:, :, :N_CTX_shared, :]
        # from triton_code.reloc_triton_kernel import attention
        
        # image_attn_output = attention(query, key, value, N_CTX_shared, N_CTX, False, 1.0, GROUPS, False)[:, :, N_CTX_shared:, :]

        # shared_attn_output = F.scaled_dot_product_attention(query_shared, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
        # hidden_states = torch.cat([shared_attn_output, image_attn_output], dim=2)
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # hidden_states = hidden_states.to(query.dtype)
        # ======================Uncomment For Inference Using Triton======================

        # ======================Uncomment For Parallel Triton Kernel======================
        # from contextlib import contextmanager
        # @contextmanager
        # def profile_range(name: str):
        #     with torch.profiler.record_function(name):
        #         yield

        # with profile_range("Triton_Attention_Setup"):
        #     from triton_code.reloc_triton_kernel_parallel import attention
        #     N_CTX_shared = 512
        #     N_CTX = query.shape[-2] - N_CTX_shared
        #     GROUPS = 4

        # with profile_range(f"Triton_Kernel_Groups_{GROUPS}"):
        #     hidden_states = attention(query, key, value, N_CTX_shared, N_CTX, False, 1.0 / 128**0.5, GROUPS, False).to(query.dtype)

        # with profile_range("Attention_Reshape"):
        #     hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # ======================Uncomment For Parallel Triton Kernel======================


        # ======================Uncomment For Parallel Triton Kernel with sliding======================
        # from triton_code.reloc_triton_kernel_parallel_sliding import attention
        # N_CTX_shared = 512
        # N_CTX = query.shape[-2] - N_CTX_shared
        # GROUPS = 4
        # hidden_states = attention(query, key, value, N_CTX_shared, N_CTX, False, 1.0 / 128**0.5, GROUPS, self._tome_info['args']['offset'], False).to(query.dtype)
        # hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        # ======================Uncomment For Parallel Triton Kernel with sliding======================



        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            # linear proj
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
