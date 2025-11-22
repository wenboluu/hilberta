import torch
from utils import get_hilbert_flat_indices, get_inverse_hilbert_indices

# 假设你已定义了以下函数：
# - apply_hilbert_reorder(...)
# - recover_hilbert_reorder(...)
# - hilbert_tile(...)
# - hilbert_untile(...)

from reorder_utils import apply_hilbert_reorder, recover_hilbert_reorder, hilbert_tile, hilbert_untile

def test_apply_and_recover():
    B, N, C = 1, 4096, 128  # batch, sequence, hidden_dim
    num_tiles = 16
    offset = 123

    # === Fake inputs ===
    image_dim = C
    image_tokens = N
    text_tokens = 512
    seq_len = text_tokens + image_tokens

    # Fake image rotary embeddings: (2 tensors of shape [512 + 4096, C])
    image_rotary_emb_1 = torch.randn(seq_len, image_dim)
    image_rotary_emb_2 = torch.randn(seq_len, image_dim)
    image_rotary_emb = (image_rotary_emb_1.clone(), image_rotary_emb_2.clone())

    # Hidden states: [B, 4096, C]
    hidden_states = torch.randn(B, N, C)

    # Encoder hidden states: [B, 77, C] (e.g., CLIP text embeddings)
    encoder_hidden_states = torch.randn(B, 77, C)

    # === Apply hilbert reordering ===
    reordered = apply_hilbert_reorder(
        image_rotary_emb,
        hidden_states.clone(),
        encoder_hidden_states.clone(),
        num_tiles=num_tiles,
        offset=offset,
    )

    image_rotary_emb_reordered, hidden_states_reordered, encoder_hidden_states_reordered = reordered

    # === Recover original ===
    recovered = recover_hilbert_reorder(
        image_rotary_emb_reordered,
        hidden_states_reordered,
        encoder_hidden_states_reordered,
        num_tiles=num_tiles,
        offset=offset,
    )

    image_rotary_emb_restored, hidden_states_restored, encoder_hidden_states_restored = recovered

    # === Compare ===
    image_emb_match = torch.allclose(image_rotary_emb[0][512:], image_rotary_emb_restored[0][512:], atol=1e-6) and \
                      torch.allclose(image_rotary_emb[1][512:], image_rotary_emb_restored[1][512:], atol=1e-6)
    hidden_match = torch.allclose(hidden_states, hidden_states_restored, atol=1e-6)
    encoder_match = torch.allclose(encoder_hidden_states, encoder_hidden_states_restored, atol=1e-6)

    print(f"✅ Image rotary embedding match: {image_emb_match}")
    print(f"✅ Hidden states match:         {hidden_match}")
    print(f"✅ Encoder hidden states match: {encoder_match}")

    if not (image_emb_match and hidden_match and encoder_match):
        print("⚠️  Recovery failed! Check components for differences.")

if __name__ == "__main__":
    test_apply_and_recover()
