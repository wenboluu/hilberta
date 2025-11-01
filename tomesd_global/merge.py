import torch
from typing import Tuple, Callable
from merge_methods import compute_attn_wts, compute_attn_wts_for_ff, stripe_attention_merge, tile_attention_merge, compute_svd, compute_stripe_svd
from facility_location import (
    tile_wise_facility,
    stripe_wise_facility,
    local_tile_wise_facility,
)
from utils import do_nothing, mps_gather_workaround, save_tensor_every_k_steps, fold_with_indices, unfold_with_indices
import os

def bipartite_soft_matching_random2d_general(
    x: torch.Tensor,
    w: int,
    h: int,
    sx: int,
    sy: int,
    r: int,
    no_rand: bool = False,
    num_of_tiles: int = 64,
    generator: torch.Generator = None,
    dst_selection: str = "original",
    unet_scheduler=None,
    key_word = None,
    rope_emb = None,
    sliding_method = None
) -> Tuple[Callable, Callable]:
    if key_word == "image":
        if_recompute_attn, if_not_merge = unet_scheduler.step_for_image()
    elif key_word == "text":
        if_recompute_attn, if_not_merge = unet_scheduler.step_for_text()
        dst_selection = "global_stripe_wise_facility"
        return False, False

    if if_not_merge:
        return False, False

    B, N, C = x.shape
    x = x / x.norm(dim=-1, keepdim=True)
    if r <= 0:
        return do_nothing, do_nothing
    
    H = W = int(N ** 0.5)
    tile_len = int(H // (num_of_tiles ** 0.5))

    # import pdb
    # pdb.set_trace()

    gather = (
        mps_gather_workaround
        if x.device.type == "mps"
        else torch.gather
    )

    from utils import index_shift_for_tile_sliding
    # def index_shift_for_tile_sliding(x, tile_len, flag = None):
        # print('I am here')
        # return x

    if sliding_method == 'up_right':
        opposite_method = 'down_left'
    elif sliding_method == 'down_left':
        opposite_method = 'up_right'
    elif sliding_method == 'up_left':
        opposite_method = 'down_right'
    elif sliding_method == 'down_right':
        opposite_method = 'up_left'
    elif sliding_method == 'stay':
        opposite_method = 'stay'
        
    x = index_shift_for_tile_sliding(x, tile_len, sliding_method)
    with torch.no_grad():

        num_dst = (w // sx) * (h // sy) if dst_selection == "original" else N - r

        def select_destination():
            if dst_selection == "tile_wise_facility":
                dst_idx = tile_wise_facility(
                    x[0].unsqueeze(0), num_dst, num_of_tiles
                ).repeat(B, 1).to(x.device)
            elif dst_selection == "local_stripe_wise_facility":
                dst_idx = stripe_wise_facility(
                    x[0].unsqueeze(0), num_dst, num_of_tiles
                ).repeat(B, 1).to(x.device)
            elif dst_selection == "global_stripe_wise_facility":
                dst_idx = stripe_wise_facility(
                    x[0].unsqueeze(0), num_dst, num_of_tiles
                ).repeat(B, 1).to(x.device)
            elif dst_selection == "local_tile_wise_facility":
                dst_idx = local_tile_wise_facility(
                    x[0].unsqueeze(0), num_dst, num_of_tiles
                ).repeat(B, 1, 1).to(x.device)
                return dst_idx
            elif dst_selection == "random":
                generator = torch.Generator(
                    device=x.device
                ).manual_seed(42)
                dst_idx = (
                    torch.randperm(
                        N, generator=generator, device=x.device
                    )
                    .unsqueeze(0)
                    .repeat(B, 1)
                    .to(x.device)
                )
            else:
                raise ValueError(f"Unknown dst_selection: {dst_selection}")

            return dst_idx[:, :num_dst].unsqueeze(2)
        
        if if_recompute_attn:
            if dst_selection == "local_stripe_wise_facility":
                dst_idx = unet_scheduler.get_dst_idx_general(select_destination, key_word)
                A, A_inv = unet_scheduler.get_A_general(stripe_attention_merge, key_word, False, x, dst_idx, num_of_tiles)
                image_rotary_emb = torch.gather(
                    rope_emb,
                    dim=1,
                    index=dst_idx.expand(2, -1, rope_emb.size(-1)),
                )
                image_rotary_emb = unet_scheduler.get_rope_emb_general(image_rotary_emb, key_word)
            elif dst_selection == 'tile_wise_facility' or dst_selection == "global_stripe_wise_facility":
                dst_idx = unet_scheduler.get_dst_idx_general(select_destination, key_word)
                # import pdb; pdb.set_trace()
                dst = gather(x, dim=1, index=dst_idx.expand(B, num_dst, C))
                A, A_inv = unet_scheduler.get_A_general(compute_attn_wts_for_ff, key_word, False, x, dst)
                image_rotary_emb = torch.gather(
                    rope_emb,
                    dim=1,
                    index=dst_idx.expand(2, -1, rope_emb.size(-1)),
                )
                image_rotary_emb = unet_scheduler.get_rope_emb_general(image_rotary_emb, key_word)
            elif dst_selection == 'local_tile_wise_facility':
                dst_idx = unet_scheduler.get_dst_idx_general(select_destination, key_word).to(x.device)
                A, A_inv, flatten_idx = unet_scheduler.get_A_general(tile_attention_merge, key_word, True, x, dst_idx, num_of_tiles)
                rope_emb, _ = fold_with_indices(rope_emb, num_of_tiles)
                use_new_rope = False
                if use_new_rope:
                    from utils import reconstruct_new_rope_emb
                    image_rotary_emb = reconstruct_new_rope_emb(A, rope_emb, 'direct')
                else:
                    image_rotary_emb = torch.gather(
                        rope_emb,
                        dim=2,
                        index=dst_idx.repeat(2, 1, 1).unsqueeze(-1).expand(-1, -1, -1, rope_emb.size(-1))
                    ).to(x.device)
                image_rotary_emb = image_rotary_emb.reshape(2, -1, rope_emb.size(-1))
                image_rotary_emb = unet_scheduler.get_rope_emb_general(image_rotary_emb, key_word)
            elif dst_selection == 'SVD':
                generator = torch.Generator(device="cpu").manual_seed(42)  # Use CPU generator
                dst_idx = torch.arange(N).unsqueeze(0).repeat(B, 1).to(x.device)
                dst_idx = dst_idx[:, :num_dst].unsqueeze(2).to(x.device)
                A, A_inv = unet_scheduler.get_A_general(compute_stripe_svd, key_word, False, x, num_dst)
                image_rotary_emb = torch.gather(
                    rope_emb,
                    dim=1,
                    index=dst_idx.expand(2, -1, rope_emb.size(-1)),
                )
                image_rotary_emb = unet_scheduler.get_rope_emb_general(image_rotary_emb, key_word)
            elif dst_selection == 'random':
                generator = torch.Generator(device="cpu").manual_seed(42)  # Use CPU generator
                dst_idx = torch.randperm(N, generator=generator).unsqueeze(0).repeat(B, 1).to(x.device)
                dst_idx = dst_idx[:, :num_dst].unsqueeze(2).to(x.device)
                dst = gather(x, dim=1, index=dst_idx.expand(B, num_dst, C))
                A, A_inv = unet_scheduler.get_A_general(compute_attn_wts_for_ff, key_word, False, x, dst)
                image_rotary_emb = torch.gather(
                    rope_emb,
                    dim=1,
                    index=dst_idx.expand(2, -1, rope_emb.size(-1)),
                )
                image_rotary_emb = unet_scheduler.get_rope_emb_general(image_rotary_emb, key_word)
        else:
            dst_idx = unet_scheduler.get_dst_idx_general(do_nothing, key_word, x)
            if dst_selection == 'local_tile_wise_facility':
                A, A_inv, flatten_idx = unet_scheduler.get_A_general(do_nothing, key_word,True, x)
            else:
                A, A_inv = unet_scheduler.get_A_general(do_nothing, key_word, False, x)
            image_rotary_emb = unet_scheduler.get_rope_emb_general(None, key_word)

    def merge(x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(A, x), dst_idx, image_rotary_emb

    def unmerge(x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(A_inv, x)

    def stripe_wise_merge(x: torch.Tensor) -> torch.Tensor:
        x_stacked = x.reshape(A.shape[0], -1, C)
        return torch.bmm(A, x_stacked).reshape(B, -1, C), dst_idx, image_rotary_emb

    def stripe_wise_unmerge(x: torch.Tensor) -> torch.Tensor:
        x_stacked = x.reshape(A_inv.shape[0], -1, C)
        return torch.bmm(A_inv, x_stacked).reshape(B, -1, C)
    
    def stripe_merge_with_svd(x: torch.Tensor) -> torch.Tensor:
        x_stacked = x.reshape(A.shape[0], -1, C)
        return torch.bmm(A, x_stacked).reshape(B, -1, C), dst_idx, image_rotary_emb
    
    def stripe_unmerge_with_svd(x: torch.Tensor) -> torch.Tensor: 
        x_stacked = x.reshape(A_inv.shape[0], -1, C)
        return torch.bmm(A_inv, x_stacked).reshape(B, -1, C)
    
    def tile_wise_merge(x: torch.Tensor) -> torch.Tensor:
        x_slided = index_shift_for_tile_sliding(x, tile_len, sliding_method)
        x_reshaped, _ = fold_with_indices(x_slided, num_of_tiles)

        x_merged = A @ x_reshaped
        x_merged = x_merged.reshape(B, -1, C)

        # Notice that the dst_idx is temporarily not used here so there are no corresponding indices shift operations
        x_merged = index_shift_for_tile_sliding(x_merged, tile_len, opposite_method)
        return x_merged.reshape(B, -1, C), dst_idx, image_rotary_emb

    def tile_wise_unmerge(x: torch.Tensor) -> torch.Tensor:
        num_tiles = A_inv.shape[1]
        x = index_shift_for_tile_sliding(x, tile_len, sliding_method)
        x = x.reshape(B, num_tiles, -1, C)
        res = A_inv @ x

        unfold_x = unfold_with_indices(res, flatten_idx)
        unfold_x = index_shift_for_tile_sliding(unfold_x, tile_len, opposite_method)
        return unfold_x
    
    if dst_selection == "local_stripe_wise_facility":
        return stripe_wise_merge, stripe_wise_unmerge
    elif dst_selection == 'SVD':
        return stripe_merge_with_svd, stripe_unmerge_with_svd
    elif dst_selection == "local_tile_wise_facility":
        return tile_wise_merge, tile_wise_unmerge
    return merge, unmerge