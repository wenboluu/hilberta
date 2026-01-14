import torch
from reloc_triton_kernel import attention  

def test_flash_attention_simple():
    # Toy input
    B, H, N, D = 1, 1, 1024, 32  # small size for fast run
    device = "cuda:2"
    dtype = torch.float16

    q = torch.randn(B, H, N, D, device=device, dtype=dtype)
    k = torch.randn(B, H, N, D, device=device, dtype=dtype)
    v = torch.randn(B, H, N, D, device=device, dtype=dtype)

    # Run kernel
    out = attention(q, k, v, True, 1.0)

    print("out shape:", out.shape)

if __name__ == "__main__":
    test_flash_attention_simple()
