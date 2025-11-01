import torch
import torch.nn.functional as F
import os
from utils import qr_based_pseudoinverse, conjugate_transpose_inverse
from utils import fold_with_indices


def compute_attn_wts(x: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Computes the attention weights for the input tensor x given the destination tensor dst.
    """
    Q = x
    K = dst

    A = K @ Q.transpose(-1, -2)  # attn_weights: [B, num_dst, N]
    A = torch.softmax(A, dim=-2)

    # _, max_indices = torch.max(A, dim=1)
    # A.zero_()
    # A.scatter_(dim=1, index=max_indices.unsqueeze(1), value=1)

    count_per_dst = A.sum(dim=-1, keepdim=True)
    avg_A = A / (count_per_dst)  
    return avg_A, A.transpose(-1, -2)

def compute_attn_wts_for_ff(x: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Computes the attention weights for the input tensor x given the destination tensor dst.
    """
    x = x / x.norm(dim=-1, keepdim=True)
    dst = dst / dst.norm(dim=-1, keepdim=True)

    Q = x
    K = dst

    A = K @ Q.transpose(-1, -2)  # attn_weights: [B, num_dst, N]
    A = torch.softmax(A * 50, dim=-2)

    # _, max_indices = torch.max(A, dim=1)
    # A.zero_()
    # A.scatter_(dim=1, index=max_indices.unsqueeze(1), value=1)

    count_per_dst = A.sum(dim=-1, keepdim=True)  
    avg_A = A / (count_per_dst)  

    # check for nan value in avg_A
    if torch.isnan(avg_A).any():
        raise ValueError("NaN value in avg_A")

    return avg_A, A.transpose(-1, -2)

def compute_svd(x: torch.Tensor, num_dst: int)-> torch.Tensor:
    U, S, Vh = torch.linalg.svd(x.float())
    
    # Select the top `num_dst` singular vectors from U
    down_projection_matrix = U[:, :, :num_dst].bfloat16()

    return down_projection_matrix.transpose(-1, -2), down_projection_matrix 

def stripe_attention_merge(x: torch.Tensor, stripe_idx_stacked: torch.Tensor, num_of_tiles: int) -> torch.Tensor:
    """
    Computes the attention weights for the input tensor x given the destination tensor dst.
    """
    # Normalize query and key tensors
    B, N, C = x.shape
    # x = x / x.norm(dim=-1, keepdim=True)
    if torch.isnan(x).any():
        raise ValueError("NaN value in x")
    
    k = num_of_tiles

    x_stacked = x.reshape(B*k, -1, C)
    # stripe_idx_stacked = stripe_idx_stacked.reshape(B*k, -1, 1)

    # dst_stacked = torch.gather(x_stacked, 1, stripe_idx_stacked.expand(-1, -1, C))

    dst_stacked = torch.gather(x, 1, stripe_idx_stacked.expand(-1, -1, C)).reshape(B*k, -1, C)
    A = dst_stacked @ x_stacked.transpose(-1, -2)  # attn_weights: [B, num_dst, N]
    A = torch.softmax(A, dim=-2)
    count_per_dst = A.sum(dim=-1, keepdim=True)  # count_per_dst: [B, num_dst, 1]
    avg_A = A / count_per_dst  # Normalize attention scores: [B, num_dst, N]
    if torch.isnan(avg_A).any():
        raise ValueError("NaN value in avg_A")

    return avg_A.bfloat16(), A.transpose(-1, -2).bfloat16()


def compute_stripe_svd(x, num_dst, num_of_tiles = 256):
    B, N, C = x.shape

    x_stacked = x.reshape(B*num_of_tiles, -1, C)
    U, S, Vh = torch.linalg.svd(x_stacked.float())

    num_dst_per_stripe = num_dst // num_of_tiles
    
    # Select the top `num_dst` singular vectors from U
    down_projection_matrix = U[:, :, :num_dst_per_stripe].bfloat16()

    return down_projection_matrix.transpose(-1, -2), down_projection_matrix

def tile_attention_merge(x: torch.Tensor, dst_idx: torch.Tensor, num_tiles: int) -> torch.Tensor:
    """
    Computes the attention weights for the input tensor x given the destination tensor dst.
    """
    B, N, C = x.shape

    x_reshaped, flatten_idx = fold_with_indices(x, num_tiles)
    x_reshaped = x_reshaped
    dst = torch.gather(x_reshaped, 2, dst_idx.unsqueeze(-1).repeat(1, 1, 1, C))

    Q = x_reshaped
    K = dst


    # Compute attention scores and normalize
    A = K @ Q.transpose(-1, -2)  # attn_weights: [B, num_dst, N]
    A = torch.softmax(A * 40, dim=-2)
    # max_indices = torch.max(A, dim=-2)[1]
    # import pdb; pdb.set_trace()
    # A.zero_()
    # A.scatter_(dim=-2, index=max_indices.unsqueeze(-2), value=1)

    count_per_dst = A.sum(dim=-1, keepdim=True)  # count_per_dst: [B, num_dst, 1]
    avg_A = A / count_per_dst  # Normalize attention scores: [B, num_dst, N]]
    A_inv= A.transpose(-1, -2)
    # A_inv = torch.linalg.pinv(avg_A.float()).bfloat16()
    return avg_A, A_inv, flatten_idx

