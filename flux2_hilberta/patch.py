"""
Model patching for FLUX.2 with Hilbert relocated attention.

Replaces Flux2TransformerBlock and Flux2SingleTransformerBlock attention processors
and replaces the transformer's forward with Hilbert-reordered version.
"""

import torch
from typing import Type, Optional
from utils import isinstance_str
from customized_attention_processor import Flux2HilbertAttnProcessor, Flux2HilbertSingleAttnProcessor
import yaml


def apply_patch(
    model: torch.nn.Module,
    num_tiles: int = 4,
    height: int = 1024,
):
    """
    Apply Hilbert relocated attention patch to a Flux2KleinPipeline.

    Args:
        model: Flux2KleinPipeline instance
        num_tiles: Number of spatial tiles (4 or 16)
        height: Image height (1024 or 2048)
    """
    import os
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    num_of_tiles = config['num_tiles']
    sliding_cycle = config['sliding_cycle']

    if height == 1024:
        image_size = 4096
    elif height == 2048:
        image_size = 16384
    else:
        raise ValueError(f"height must be 1024 or 2048, got {height}")

    # Build per-layer offset info
    info_list = []
    for i in range(sliding_cycle):
        info = {
            "size": None,
            "args": {
                "offset": (image_size // num_of_tiles) // sliding_cycle * i,
                "num_tiles": num_of_tiles,
            },
        }
        info_list.append(info)

    transformer = model.transformer

    counter = 0
    for _, module in transformer.named_modules():
        if isinstance_str(module, "Flux2TransformerBlock"):
            # Double-stream block: replace attention processor
            info_counter = counter % sliding_cycle
            module.attn.processor = Flux2HilbertAttnProcessor()
            module.attn.processor._tome_info = info_list[info_counter]
            counter += 1

        elif isinstance_str(module, "Flux2SingleTransformerBlock"):
            # Single-stream block: replace parallel self-attention processor
            info_counter = counter % sliding_cycle
            info_with_txt = {
                "size": info_list[info_counter]["size"],
                "args": {
                    **info_list[info_counter]["args"],
                    "num_txt_tokens": 0,  # Will be set at runtime
                },
            }
            module.attn.processor = Flux2HilbertSingleAttnProcessor()
            module.attn.processor._tome_info = info_with_txt
            counter += 1

    # Replace transformer forward
    from types import MethodType
    from reorder_utils import customized_forward
    transformer.forward = MethodType(customized_forward, transformer)

    return model


def remove_patch(model: torch.nn.Module):
    """Remove Hilbert patch from model."""
    transformer = model.transformer
    for _, module in transformer.named_modules():
        if hasattr(module, '_parent'):
            module.__class__ = module._parent
    return model
