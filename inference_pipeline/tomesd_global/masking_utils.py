import torch

from typing import Any, Dict, Optional, Tuple, Union
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os 
from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from utils import get_hilbert_flat_indices
logger = logging.get_logger(__name__) 

def create_hilbert_tile_mask(x, num_of_tiles, offset=0):
    hilbert_index = get_hilbert_flat_indices(6).to(x.device)

    ################################# Make complete hilbert index #################################
    index = torch.arange(hilbert_index.numel()).to(x.device)    
    cut_off = index.shape[0] // 4

    hilbert_x = torch.gather(index, dim=0, index=hilbert_index)
    hilbert_x_half = hilbert_x[cut_off:-cut_off]

    x_flip = index.flip(0)
    hilbert_x_flip = torch.gather(x_flip, dim=0, index=hilbert_index)
    hilbert_x_half_flip = hilbert_x_flip[cut_off:-cut_off]

    hilbert_index = torch.cat([hilbert_x_half, hilbert_x_half_flip])
    ################################# Make complete hilbert index #################################

    hilbert_index = torch.cat([hilbert_index[offset:], hilbert_index[:offset]])
    hilbert_index = hilbert_index.reshape(num_of_tiles, -1)

    def get_all_pairs_batched_parallel(tensor):
        # tensor shape: [B, N]
        B, N = tensor.shape
        
        # Reshape for broadcasting
        tensor_i = tensor.view(B, N, 1)  # Shape: [B, N, 1]
        tensor_j = tensor.view(B, 1, N)  # Shape: [B, 1, N]
        
        # Broadcast to create all pairs
        # This expands tensor_i along the last dimension and tensor_j along the middle dimension
        # Creating a grid of all combinations for each batch
        i_grid = tensor_i.expand(B, N, N)  # Shape: [B, N, N]
        j_grid = tensor_j.expand(B, N, N)  # Shape: [B, N, N]
        
        # Stack to get pairs
        pairs = torch.stack([i_grid, j_grid], dim=-1)  # Shape: [B, N, N, 2]
        
        # Reshape to [B, N*N, 2]
        pairs = pairs.view(B, N*N, 2)
        
        return pairs

    pairs = get_all_pairs_batched_parallel(hilbert_index)
    pairs = pairs.reshape(-1, pairs.shape[-1])
    
    # Get total sequence length
    seq_len = hilbert_index.numel()
    
    mask = torch.full((seq_len, seq_len), float('-inf'), device=x.device)
    mask[pairs[:, 0], pairs[:, 1]] = 0.0   
    
    for i in range(24, 40):
        start = i*64 + 24
        end = i*64 + 40 
        mask[start:end, :] = 0.0
        mask[:, start:end] = 0.0
    
    corner_size = 4
    seq_len = 4096

    # Top-left corner
    mask[0:corner_size, :] = 0.0  
    mask[:, 0:corner_size] = 0.0  

    # Top-right corner
    mask[0:corner_size, seq_len-corner_size:seq_len] = 0.0
    mask[:, seq_len-corner_size:seq_len] = 0.0

    # Bottom-left corner
    mask[seq_len-corner_size:seq_len, 0:corner_size] = 0.0
    mask[:, 0:corner_size] = 0.0

    # Bottom-right corner
    mask[seq_len-corner_size:seq_len, seq_len-corner_size:seq_len] = 0.0
    mask[:, seq_len-corner_size:seq_len] = 0.0
    return mask

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
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
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

        # load config 
        with open('/scratch/sz3684/reorder_local_attention/inference_pipeline/tomesd_global/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
        num_of_tiles = config['num_tiles']
        sliding_cycle = config['sliding_cycle']

        offset_list = [((hidden_states.shape[1]//num_of_tiles) //sliding_cycle) * i for i in range(sliding_cycle)]

        os.makedirs('./mask', exist_ok=True)

        for offset in offset_list:
            mask = create_hilbert_tile_mask(hidden_states, num_of_tiles=num_of_tiles, offset=offset)
            f = f"./mask/mask_offset_{offset}_num_of_tiles_{num_of_tiles}.pt"
            torch.save(mask, f)

        for index_block, block in enumerate(self.transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]

        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)

        if USE_PEFT_BACKEND:
            # remove `lora_scale` from each PEFT layer
            unscale_lora_layers(self, lora_scale)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)