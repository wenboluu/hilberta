import torch

from typing import Any, Dict, Optional, Tuple, Union
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import USE_PEFT_BACKEND, is_torch_version, logging, scale_lora_layers, unscale_lora_layers
from diffusers.models.modeling_outputs import Transformer2DModelOutput

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

def tile_compare(x, r, num_tiles):
    B, HW, C = x.shape
    H = W = int(HW**0.5)

    num_tiles_per_side = int(num_tiles**0.5)
    tile_side_len = H // num_tiles_per_side

    indices = torch.arange(HW, device=x.device).reshape(1, H, W, 1)
    patch_indices = torch.as_strided(
        indices,
        (1, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, 1),
        (HW * 1, tile_side_len * W * 1, tile_side_len * 1, W * 1, 1, 1),
    )

    x_reshaped = torch.as_strided(
        x,
        (1, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, C),
        (HW * C, tile_side_len * H * C, tile_side_len * C, H * C, C, 1),
    )
    x_reshaped = x_reshaped.reshape(-1, tile_side_len**2, C)
    x_reshaped = x_reshaped.reshape(B, HW, C)


    return x_reshaped, patch_indices.reshape(-1, HW)


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
    """
    The [`FluxTransformer2DModel`] forward method.

    Args:
        hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
            Input `hidden_states`.
        encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
            Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
        pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
            from the embeddings of input conditions.
        timestep ( `torch.LongTensor`):
            Used to indicate denoising step.
        block_controlnet_hidden_states: (`list` of `torch.Tensor`):
            A list of tensors that if specified are added to the residuals of transformer blocks.
        joint_attention_kwargs (`dict`, *optional*):
            A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
            `self.processor` in
            [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
        return_dict (`bool`, *optional*, defaults to `True`):
            Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
            tuple.

    Returns:
        If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
        `tuple` where the first element is the sample tensor.
    """

    def load_config(config_path):
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    
    config = load_config("/home/sz3684/diffusion/reorder_local_attention/diffusion_reorder/tomesd_global/config.yaml")

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

    #########################################################################################################
    # Split rotary embeddings into text and image parts
    image_rotary_emb_1, image_rotary_emb_2 = image_rotary_emb

    text_emb_1, image_emb_1 = image_rotary_emb_1[:512], image_rotary_emb_1[512:]
    text_emb_2, image_emb_2 = image_rotary_emb_2[:512], image_rotary_emb_2[512:]

    # Tile the image embeddings across the number of tiles
    tiled_image_emb_1 = tile(image_emb_1.unsqueeze(0), num_tiles).squeeze(0)
    tiled_image_emb_2 = tile(image_emb_2.unsqueeze(0), num_tiles).squeeze(0)

    # Concatenate text and tiled image embeddings, then reshape to (num_tiles, seq_len, dim)
    text_emb_1 = text_emb_1.view(num_tiles, -1, text_emb_1.shape[-1])
    tiled_image_emb_1 = tiled_image_emb_1.view(num_tiles, -1, tiled_image_emb_1.shape[-1])
    image_rotary_emb_1 = torch.cat([text_emb_1, tiled_image_emb_1], dim=1)

    text_emb_2 = text_emb_2.view(num_tiles, -1, text_emb_2.shape[-1])
    tiled_image_emb_2 = tiled_image_emb_2.view(num_tiles, -1, tiled_image_emb_2.shape[-1])
    image_rotary_emb_2 = torch.cat([text_emb_2, tiled_image_emb_2], dim=1)

    # Pack the updated embeddings and tile the hidden states accordingly
    image_rotary_emb = (image_rotary_emb_1, image_rotary_emb_2)
    hidden_states = tile(hidden_states, num_tiles)

    B, N, C = hidden_states.shape

    encoder_hidden_states = encoder_hidden_states.view(B*num_tiles, -1, encoder_hidden_states.shape[-1])
    encoder_hidden_states = encoder_hidden_states[0].unsqueeze(0).repeat(B*num_tiles, 1, 1)
    hidden_states = hidden_states.view(B * num_tiles, -1, hidden_states.shape[-1]) 




    # last_encoder = encoder_hidden_states[-1,:,:]
    # encoder_hidden_states = last_encoder.unsqueeze(0).repeat(B*num_tiles, 1, 1)
    # last_text = hidden_states[-1,:,:]
    # hidden_states = last_text.unsqueeze(0).repeat(B*num_tiles, 1, 1)
    # rope_1, rope_2 = image_rotary_emb
    # rope_1 = rope_1[-1].unsqueeze(0).repeat(B*num_tiles, 1, 1)
    # rope_2 = rope_2[-1].unsqueeze(0).repeat(B*num_tiles, 1, 1)
    # image_rotary_emb = (rope_1, rope_2)
    #########################################################################################################


    for index_block, block in enumerate(self.transformer_blocks):
        encoder_hidden_states, hidden_states = block(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )

    hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

    encoder_hidden_states = encoder_hidden_states.view(B, -1, C)

    for index_block, block in enumerate(self.single_transformer_blocks):
        hidden_states = block(
            hidden_states=hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=joint_attention_kwargs,
        )



    hidden_states = hidden_states[:, 512//4:, :]
        
    hidden_states = hidden_states.reshape(B, -1, C)

    #########################################################################################################
    # image_hidden_states = hidden_states[:, 512:, :]
    # text_hidden_states = hidden_states[:, :512, :]
    # image_hidden_states = untile(image_hidden_states, num_tiles)
    # hidden_states = torch.cat([text_hidden_states, image_hidden_states], dim=1) 
    #########################################################################################################

    # hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

    hidden_states = self.norm_out(hidden_states, temb)
    output = self.proj_out(hidden_states)
    output = untile(output, num_tiles)

    if USE_PEFT_BACKEND:
        # remove `lora_scale` from each PEFT layer
        unscale_lora_layers(self, lora_scale)

    if not return_dict:
        return (output,)

    return Transformer2DModelOutput(sample=output)



