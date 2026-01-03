import torch
import torch.nn.functional as F
import math
import yaml

def masked_scaled_dot_product_attention(
    query,
    key,
    value,
    attn_mask=None,
    dropout_p=0.0,
    is_causal=False,
    scale=None,
    enable_gqa=False,
    mask=None
) -> torch.Tensor:
    """
    Custom implementation of scaled dot-product attention with optional masking.
    """
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(~attn_mask, float("-inf"))
        else:
            attn_bias = attn_bias + attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor

    if mask is not None:
        attn_weight[:, :, -4096:, -4096:] += mask

    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


def test_attention_equivalence():
    torch.manual_seed(42)

    # Simulate inputs
    B, H, N, D = 1, 24, 4608, 128
    query = torch.randn(B, H, N, D, dtype=torch.float32).to('cuda')
    key = torch.randn(B, H, N, D, dtype=torch.float32).to('cuda')
    value = torch.randn(B, H, N, D, dtype=torch.float32).to('cuda')

    mask = torch.load(f'/scratch/sz3684/reorder_local_attention/mask/mask_offset_0_num_of_tiles_16.pt')
    # Prepare attn_mask for PyTorch official SDPA
    attn_mask = torch.zeros(N, N, dtype=query.dtype, device=query.device)
    attn_mask[-4096:, -4096:] += mask
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1).contiguous()

    # Run both attention implementations
    out_custom = masked_scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False, mask=mask)
    out_official = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)

    print(torch.allclose(out_custom, out_official, atol=1e-4))

    assert out_custom.shape == out_official.shape == (B, H, N, D)
    assert torch.allclose(out_custom, out_official, atol=1e-4), "Mismatch detected between custom and official SDPA!"


if __name__ == "__main__":
    test_attention_equivalence()
