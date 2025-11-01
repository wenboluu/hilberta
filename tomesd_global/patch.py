import torch
import math
import os 
from typing import Type, Dict, Any, Tuple, Callable, Optional, Union, List
import json 
from merge import do_nothing
from utils import isinstance_str, init_generator
from customized_attention_processor import FluxAttnProcessor2_0_for_transformerblock_global

def make_diffusers_flux_tome_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class ToMeBlock(block_class):
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )
            joint_attention_kwargs = joint_attention_kwargs or {}

            # Attention.
            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )

            attn_output = gate_msa.unsqueeze(1) * attn_output
            hidden_states = hidden_states + attn_output

            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            ff_output = self.ff(norm_hidden_states)
            ff_output = gate_mlp.unsqueeze(1) * ff_output

            hidden_states = hidden_states + ff_output

            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

            context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

            return encoder_hidden_states, hidden_states
    return ToMeBlock


def make_flux_single_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class ToMeBlock(block_class):
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
        ):
            residual = hidden_states
            norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
            joint_attention_kwargs = joint_attention_kwargs or {}
            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **joint_attention_kwargs,
            )

            hidden_states = torch.cat([attn_output, mlp_hidden_states], dim=2)
            gate = gate.unsqueeze(1)
            hidden_states = gate * self.proj_out(hidden_states)

            hidden_states = residual + hidden_states
            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)

            return hidden_states

    return ToMeBlock

def apply_patch(
    model: torch.nn.Module,
    ratio: float = 0.5,
    max_downsample: int = 1,
    sx: int = 2,
    sy: int = 2,
    use_rand: bool = True,
    merge_attn: bool = True,
    merge_crossattn: bool = False,
    merge_mlp: bool = False,
    dst_selection: str = "original",
    num_tiles: int = 16,
    merge_method: str = "original",
    unet_scheduler=None,
    toma_variant=None,
):
    remove_patch(model)

    is_diffusers_flux = isinstance_str(model, "FluxPipeline")

    if is_diffusers_flux:
        transformer_model = model.transformer
    else:
        print("Model is not a supported model for ToMe patching.")

    transformer_model._tome_info = {
        "size": None,
        "args": {
            "ratio": ratio,
            "max_downsample": max_downsample,
            "sx": sx,
            "sy": sy,
            "use_rand": use_rand,
            "generator": None,
            "merge_attn": merge_attn,
            "merge_crossattn": merge_crossattn,
            "merge_mlp": merge_mlp,
            "dst_selection": dst_selection,
            "k":  num_tiles * 4,
            "merge_method": merge_method,
            "unet_scheduler": unet_scheduler,
            "sliding_method": "stay",
        },
    }
    
    import types
    from reorder_utils import customized_forward
    model.transformer.forward = types.MethodType(customized_forward, model.transformer)

    make_tome_block_fn = make_diffusers_flux_tome_block
    make_single_tome_block_fn = make_flux_single_block

    for _, module in transformer_model.named_modules():
        if isinstance_str(module, "FluxTransformerBlock"):
            module.__class__ = make_tome_block_fn(module.__class__)
            module._tome_info = transformer_model._tome_info
            module.attn.processor = FluxAttnProcessor2_0_for_transformerblock_global()
            module.attn.processor._tome_info = module._tome_info
        elif isinstance_str(module, "FluxSingleTransformerBlock"):
            module.__class__ = make_single_tome_block_fn(module.__class__)
            module._tome_info = transformer_model._tome_info
            # module.attn.processor = FluxAttnProcessor2_0_for_transformerblock_global()
            # module.attn.processor._tome_info = module._tome_info
    return model

def remove_patch(model: torch.nn.Module):
    """Removes a patch from a ToMe Diffusion module if it was already patched."""
    # For diffusers
    model = model.transformer

    for _, module in model.named_modules():
        if module.__class__.__name__ == "ToMeBlock":
            module.__class__ = module._parent

    return model
