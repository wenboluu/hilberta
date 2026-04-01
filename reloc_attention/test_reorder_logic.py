import torch
from reorder_utils_sliding_shared import apply_hilbert_reorder, recover_hilbert_reorder

# Test with small example
B = 1
sequence_length = 4096
C = 128

# Create test hidden states with unique values so we can track them
hidden_states = torch.arange(B * sequence_length * C, dtype=torch.float32).reshape(B, sequence_length, C)

# Create dummy rotary emb
emb_dim = 64
rotary_emb_1 = torch.randn(512 + sequence_length, emb_dim)
rotary_emb_2 = torch.randn(512 + sequence_length, emb_dim)
image_rotary_emb = (rotary_emb_1, rotary_emb_2)

# Apply reorder
print("Original hidden_states shape:", hidden_states.shape)
print("Sample values from original (first 5 tokens, first 3 dims):")
print(hidden_states[0, :5, :3])

reordered_emb, reordered_hidden = apply_hilbert_reorder(image_rotary_emb, hidden_states, num_tiles=4, offset=0)

print("\nReordered hidden_states shape:", reordered_hidden.shape)
print("Sample values from reordered (first 5 tokens, first 3 dims):")
print(reordered_hidden[0, :5, :3])

# Recover
recovered_emb, recovered_hidden = recover_hilbert_reorder(reordered_emb, reordered_hidden, num_tiles=4, offset=0)

print("\nRecovered hidden_states shape:", recovered_hidden.shape)
print("Sample values from recovered (first 5 tokens, first 3 dims):")
print(recovered_hidden[0, :5, :3])

# Check if recovery is perfect
diff = torch.abs(hidden_states - recovered_hidden).max()
print(f"\nMax absolute difference between original and recovered: {diff}")

if diff < 1e-5:
    print("✓ Recovery is perfect!")
else:
    print("✗ Recovery has errors!")
    # Find where the differences are
    error_mask = torch.abs(hidden_states - recovered_hidden) > 1e-5
    error_positions = torch.where(error_mask)
    print(f"Number of error positions: {len(error_positions[0])}")
    if len(error_positions[0]) > 0:
        print(f"First error at batch={error_positions[0][0]}, pos={error_positions[1][0]}, dim={error_positions[2][0]}")
        print(f"Original value: {hidden_states[error_positions[0][0], error_positions[1][0], error_positions[2][0]]}")
        print(f"Recovered value: {recovered_hidden[error_positions[0][0], error_positions[1][0], error_positions[2][0]]}")
