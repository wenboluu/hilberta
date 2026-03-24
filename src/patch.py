from typing import Optional, Type

import torch
import yaml

from .attention_processor import HilbertaAttnProcessor
from .utils import isinstance_str


def _make_hilberta_joint_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """Wrap a FluxTransformerBlock to average encoder states across tiles after attention."""

    class HilbertaBlock(block_class):
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            encoder_hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
            layer_idx=None,
            step: Optional[int] = None,
        ):
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
            num_of_tiles = norm_hidden_states.shape[0]
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )
            joint_attention_kwargs = joint_attention_kwargs or {}

            attn_output, context_attn_output = self.attn(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                layer_idx=layer_idx,
                step=step,
                **joint_attention_kwargs,
            )

            # Average encoder hidden states across tiles then broadcast back
            encoder_hidden_states = encoder_hidden_states.mean(dim=0, keepdim=True).expand(num_of_tiles, -1, -1)

            # Image path
            hidden_states = hidden_states + gate_msa.unsqueeze(1) * attn_output
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.ff(norm_hidden_states)

            # Context path
            encoder_hidden_states = encoder_hidden_states + c_gate_msa.unsqueeze(1) * context_attn_output
            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * self.ff_context(norm_encoder_hidden_states)

            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

            return encoder_hidden_states, hidden_states

    return HilbertaBlock


def _make_hilberta_single_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """Wrap a FluxSingleTransformerBlock to pass step/layer info through attention."""

    class HilbertaBlock(block_class):
        _parent = block_class

        def forward(
            self,
            hidden_states: torch.FloatTensor,
            temb: torch.FloatTensor,
            image_rotary_emb=None,
            joint_attention_kwargs=None,
            layer_idx=None,
            step: Optional[int] = None,
        ):
            residual = hidden_states
            norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
            mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
            joint_attention_kwargs = joint_attention_kwargs or {}

            attn_output = self.attn(
                hidden_states=norm_hidden_states,
                image_rotary_emb=image_rotary_emb,
                layer_idx=layer_idx,
                step=step,
                **joint_attention_kwargs,
            )

            hidden_states = gate.unsqueeze(1) * self.proj_out(
                torch.cat([attn_output, mlp_hidden_states], dim=2)
            )
            hidden_states = residual + hidden_states

            if hidden_states.dtype == torch.float16:
                hidden_states = hidden_states.clip(-65504, 65504)

            return hidden_states

    return HilbertaBlock


def apply_hilberta_patch(model: torch.nn.Module, num_tiles: int = 16, height=None):
    """Apply Hilberta attention patching to a FluxPipeline.

    Replaces each transformer block with a Hilberta-wrapped version that uses
    Hilbert-curve tiled attention masks with sliding offsets.
    """
    remove_hilberta_patch(model)

    if not isinstance_str(model, "FluxPipeline"):
        raise ValueError("Model is not a FluxPipeline — Hilberta patching is not supported.")

    transformer = model.transformer

    with open('./src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    num_of_tiles = config['tiling']['num_tiles']
    sliding_cycle = config['tiling']['sliding_cycle']

    if height == 1024:
        image_size = 4096
    elif height == 2048:
        image_size = 16384
    else:
        raise ValueError(f"Unsupported height {height}, expected 1024 or 2048.")

    # Build sliding offset info for each cycle position
    tile_len = image_size // num_of_tiles
    info_list = [{"offset": (tile_len // sliding_cycle) * i} for i in range(sliding_cycle)]

    counter = 0
    for _, module in transformer.named_modules():
        if isinstance_str(module, "FluxTransformerBlock"):
            module.__class__ = _make_hilberta_joint_block(module.__class__)
            module.attn.processor = HilbertaAttnProcessor()
            module.attn.processor._hilberta_info = info_list[counter % sliding_cycle]
            counter += 1
        elif isinstance_str(module, "FluxSingleTransformerBlock"):
            module.__class__ = _make_hilberta_single_block(module.__class__)
            module.attn.processor = HilbertaAttnProcessor()
            module.attn.processor._hilberta_info = info_list[counter % sliding_cycle]
            counter += 1

    return model


def remove_hilberta_patch(model: torch.nn.Module):
    """Remove Hilberta patching, restoring original block classes."""
    transformer = model.transformer
    for _, module in transformer.named_modules():
        if module.__class__.__name__ == "HilbertaBlock":
            module.__class__ = module._parent
    return model
