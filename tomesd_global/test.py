import torch
from utils import get_hilbert_flat_indices

def create_hilbert_tile_mask(x, num_of_tiles, offset=0):
    hilbert_index = get_hilbert_flat_indices(2).to(x.device)
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
    mask[pairs[:, 0], pairs[:, 1]] = 1.0    
    print(mask.shape)
    return mask

if __name__ == "__main__":
    a = torch.arange(16).cuda()
    a = a.unsqueeze(0).unsqueeze(-1)
    num_of_tiles = 4
    # print(a.reshape(a.shape[0], num_of_tiles, -1, a.shape[2]))
    
    # Test the attention mask
    attn_mask = create_hilbert_tile_mask(a, num_of_tiles, offset=0)
    print("Attention mask shape:", attn_mask.shape)
    print("Sample of attention mask:")
    # print(attn_mask[:16, :16])  # Show the full mask for the 16 tokens
