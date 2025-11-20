import torch
from utils import get_hilbert_flat_indices, get_inverse_hilbert_indices

# Hilbert tile with offset
def hilbert_tile(x, offset=0):
    hilbert_index = get_hilbert_flat_indices(6).to(x.device)
    hilbert_index_offset = torch.cat([hilbert_index[offset:], hilbert_index[:offset]])
    return torch.gather(x, 1, hilbert_index_offset.unsqueeze(0).unsqueeze(-1).repeat(x.shape[0], 1, x.shape[-1]))

# Hilbert untile that inverts the offset + mapping
def hilbert_untile(x_hilbert, offset=0):
    inverse_index = get_inverse_hilbert_indices(6).to(x_hilbert.device)
    if offset > 0:
        x_hilbert = torch.cat([x_hilbert[:, -offset:], x_hilbert[:, :-offset]], dim=1)
    return torch.gather(x_hilbert, 1, inverse_index.unsqueeze(0).unsqueeze(-1).repeat(x_hilbert.shape[0], 1, x_hilbert.shape[-1]))

# === Verification Test ===
def test_hilbert_tile_and_untile():
    B, N, D = 2, 4096, 8
    x = torch.randn(B, N, D)
    
    for offset in [0, 1, 73, 512, 1024, 2048, 4095]:
        x_hilbert = hilbert_tile(x, offset=offset)
        x_recovered = hilbert_untile(x_hilbert, offset=offset)
        is_equal = torch.allclose(x, x_recovered, atol=1e-6)
        print(f"Offset {offset:4d} | Match: {is_equal}")

        if not is_equal:
            diff = (x - x_recovered).abs().max()
            print(f"⚠️  Max diff at offset {offset}: {diff.item()}")

# Run test
if __name__ == "__main__":
    test_hilbert_tile_and_untile()
