import torch

from typing import Any, Dict, Optional, Tuple, Union
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

def hilbert_tile(x, reverse, offset=0):
    if reverse:
        x = x.flip(1)
    hilbert_index = get_hilbert_flat_indices(6).to(x.device)
    hilbert_index_offset = torch.cat([hilbert_index[offset:], hilbert_index[:offset]])
    index = hilbert_index_offset.unsqueeze(0).expand(x.shape[0], -1).unsqueeze(-1).expand(-1, -1, x.shape[-1])
    output = torch.gather(x, 1, index)
    del hilbert_index
    del index
    torch.cuda.empty_cache()
    return output

def hilbert_untile(x_hilbert, reverse, offset=0):
    inverse_index = get_inverse_hilbert_indices(6).to(x_hilbert.device)
    if offset > 0:
        x_hilbert = torch.cat([x_hilbert[:, -offset:], x_hilbert[:, :-offset]], dim=1)

    batch_size, _, dim = x_hilbert.shape
    index = inverse_index.view(1, -1, 1).expand(batch_size, -1, dim)
    output = torch.gather(x_hilbert, 1, index)
    del inverse_index
    del index
    torch.cuda.empty_cache()
    return output

def apply_hilbert_reorder(image_rotary_emb, hidden_states, encoder_hidden_states, num_tiles, reverse = False, offset = 0):
    """
    Reorders the image part of rotary embeddings and hidden states using Hilbert curve,
    tiles them for each image patch, and prepares them for model input.
 
    Args:
        image_rotary_emb (tuple of torch.Tensor): Tuple of (rotary_emb_1, rotary_emb_2),
            each of shape [512 + H*W, dim], where 512 is text embedding and the rest is image.
        hidden_states (torch.Tensor): Input hidden states of shape [B, N, C], where N = H * W.
        encoder_hidden_states (torch.Tensor): Encoder context, shape [1, seq_len, C] or [B, seq_len, C].
        num_tiles (int): Number of spatial tiles to split image into (e.g., 16 for 4x4).

    Returns:
        image_rotary_emb: Tuple of reordered and tiled rotary embeddings
        hidden_states: Reordered and reshaped hidden states [B * num_tiles, tile_len, C]
        encoder_hidden_states: Tiled encoder hidden states [B * num_tiles, ..., C]
    """
    image_rotary_emb_1, image_rotary_emb_2 = image_rotary_emb

    # Split rotary embeddings into text and image parts
    text_emb_1, image_emb_1 = image_rotary_emb_1[:512], image_rotary_emb_1[512:]
    text_emb_2, image_emb_2 = image_rotary_emb_2[:512], image_rotary_emb_2[512:]

    # Tile and reorder image embeddings with Hilbert curve
    tiled_image_emb_1 = hilbert_tile(image_emb_1.unsqueeze(0), reverse, offset).squeeze(0)
    tiled_image_emb_2 = hilbert_tile(image_emb_2.unsqueeze(0), reverse, offset).squeeze(0)

    # Repeat text embedding for each tile
    # text_emb_1 = text_emb_1.repeat(num_tiles, 1, 1)
    text_emb_1 = text_emb_1.unsqueeze(0)
    tiled_image_emb_1 = tiled_image_emb_1.view(num_tiles, -1, tiled_image_emb_1.shape[-1])
    image_rotary_emb_1 = torch.cat([text_emb_1, tiled_image_emb_1], dim=1)

    # text_emb_2 = text_emb_2.repeat(num_tiles, 1, 1)
    text_emb_2 = text_emb_2.unsqueeze(0)
    tiled_image_emb_2 = tiled_image_emb_2.view(num_tiles, -1, tiled_image_emb_2.shape[-1])
    image_rotary_emb_2 = torch.cat([text_emb_2, tiled_image_emb_2], dim=1)

    image_rotary_emb = (image_rotary_emb_1, image_rotary_emb_2)

    # Reorder hidden states
    hidden_states = hilbert_tile(hidden_states, reverse, offset)
    B, N, C = hidden_states.shape
    hidden_states = hidden_states.view(B * num_tiles, -1, C)

    # Tile encoder hidden states
    # if encoder_hidden_states.shape[0] != num_tiles:
        # encoder_hidden_states = encoder_hidden_states.repeat(num_tiles, 1, 1)
    # else:
        # encoder_hidden_states = encoder_hidden_states

    # import pdb; pdb.set_trace()

    del image_emb_1, image_emb_2
    del tiled_image_emb_1, tiled_image_emb_2
    del text_emb_1, text_emb_2
    torch.cuda.empty_cache()

    return image_rotary_emb, hidden_states, encoder_hidden_states

def recover_hilbert_reorder(image_rotary_emb, hidden_states, encoder_hidden_states, num_tiles, reverse = False, offset=0):
    """
    Recovers original ordering from Hilbert-tiled embeddings and hidden states.

    Args:
        image_rotary_emb (tuple): (image_rotary_emb_1, image_rotary_emb_2),
            each of shape [num_tiles, seq_len, dim]
        hidden_states (torch.Tensor): Shape [B * num_tiles, tile_len, C]
        encoder_hidden_states (torch.Tensor): Shape [B * num_tiles, enc_seq_len, C]
        num_tiles (int): Number of tiles per original batch
        offset (int): Hilbert offset used during tiling

    Returns:
        image_rotary_emb: Tuple of [512 + HW, dim] restored rotary embeddings
        hidden_states: [B, HW, C] recovered hidden states
        encoder_hidden_states: [B, enc_seq_len, C] recovered encoder context
    """
    image_rotary_emb_1, image_rotary_emb_2 = image_rotary_emb

    # Split text + image parts
    text_emb_1, tiled_image_emb_1 = image_rotary_emb_1[:, :512], image_rotary_emb_1[:, 512:]
    text_emb_2, tiled_image_emb_2 = image_rotary_emb_2[:, :512], image_rotary_emb_2[:, 512:]

    # Merge tiles back to [num_tiles * tile_len, dim]
    flat_image_emb_1 = tiled_image_emb_1.reshape(-1, tiled_image_emb_1.shape[-1])
    flat_image_emb_2 = tiled_image_emb_2.reshape(-1, tiled_image_emb_2.shape[-1])

    # Undo Hilbert ordering
    recovered_image_emb_1 = hilbert_untile(flat_image_emb_1.unsqueeze(0), reverse, offset).squeeze(0)
    recovered_image_emb_2 = hilbert_untile(flat_image_emb_2.unsqueeze(0), reverse, offset).squeeze(0)

    # Merge text + image
    text_emb_1 = text_emb_1[0]  # All tiles share same text
    text_emb_2 = text_emb_2[0]
    image_rotary_emb_1 = torch.cat([text_emb_1, recovered_image_emb_1], dim=0)
    image_rotary_emb_2 = torch.cat([text_emb_2, recovered_image_emb_2], dim=0)
    image_rotary_emb = (image_rotary_emb_1, image_rotary_emb_2)

    B = hidden_states.shape[0] // num_tiles
    flat_hidden = hidden_states.reshape(B, -1, hidden_states.shape[-1])
    hidden_states = hilbert_untile(flat_hidden, reverse, offset)

    del tiled_image_emb_1, tiled_image_emb_2
    del flat_image_emb_1, flat_image_emb_2
    del text_emb_1, text_emb_2
    del flat_hidden
    torch.cuda.empty_cache()


    return image_rotary_emb, hidden_states, encoder_hidden_states


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
    t: int = 0,
) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
    def load_config(config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    config = load_config("/scratch/sz3684/reorder_local_attention/tomesd_global/config.yaml")

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

    off_set_counter = 0 

    counter = 0
    off_set_counter = 0 
    cycle = 2
    reverse = False
    off_set_unit = (N//num_tiles)//cycle
    for index_block, block in enumerate(self.transformer_blocks):
        off_set_counter = ((counter//2)%cycle) * off_set_unit

        # Reverse every one step
        tile_flag = True
        if tile_flag:
            # hilbert tile before each block        
            image_rotary_emb, hidden_states, encoder_hidden_states = apply_hilbert_reorder(
                image_rotary_emb, hidden_states, encoder_hidden_states, num_tiles, reverse = reverse, offset = off_set_counter
            )
        else:
            encoder_hidden_states = encoder_hidden_states.mean(dim=0, keepdim=True)

        # Entering the block
        print(f'Memory usage before Joint Transformer block {counter}: {torch.cuda.memory_allocated() / 1024**3} GB')
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )
        print(f'Memory usage after Joint Transformer block {counter}: {torch.cuda.memory_allocated() / 1024**3} GB')

        if tile_flag:
            image_rotary_emb, hidden_states, encoder_hidden_states = recover_hilbert_reorder(
                image_rotary_emb, hidden_states, encoder_hidden_states, num_tiles, reverse=reverse, offset = off_set_counter
            )
        counter += 1

    for index_block, block in enumerate(self.single_transformer_blocks):
        off_set_counter = ((counter//2)%cycle) * off_set_unit

        # Reverse every one step
        tile_flag = True
        if tile_flag:
            image_rotary_emb, hidden_states, encoder_hidden_states = apply_hilbert_reorder(
                image_rotary_emb, hidden_states, encoder_hidden_states, num_tiles, reverse = reverse, offset = off_set_counter
            )
        else:
            encoder_hidden_states = encoder_hidden_states.mean(dim=0, keepdim=True)


        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        # Entering the block
        hidden_states = block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

        print(f'Memory usage after Single Transformer block {counter}: {torch.cuda.memory_allocated() / 1024**3} GB')

        encoder_hidden_states, hidden_states = hidden_states[:, :encoder_hidden_states.shape[1], :], hidden_states[:, encoder_hidden_states.shape[1]:, :]

        if tile_flag:
            # hilbert untile after each block
            image_rotary_emb, hidden_states, encoder_hidden_states = recover_hilbert_reorder(
                image_rotary_emb, hidden_states, encoder_hidden_states, num_tiles, reverse = reverse, offset = off_set_counter
            )
        counter += 1


    # hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    # hidden_states = hidden_states[:, encoder_hidden_states.shape[1]:, :]
    hidden_states = hidden_states.reshape(B, -1, C)
    
    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    # output = hilbert_untile(output, offset = off_set_counter)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)



