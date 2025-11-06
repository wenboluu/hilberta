import torch

def tile(x, num_tiles):
    B, HW, C = x.shape
    H = W = int(HW**0.5)

    num_tiles_per_side = int(num_tiles**0.5)
    tile_side_len = H // num_tiles_per_side

    x_reshaped = torch.as_strided(
        x,
        (1, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, C),
        (HW * C, tile_side_len * H * C, tile_side_len * C, H * C, C, 1),
    )
    x_reshaped = x_reshaped.reshape(-1, tile_side_len**2, C)
    x_reshaped = x_reshaped.reshape(B, HW, C)

    return x_reshaped 

def untile(x_tiled, num_tiles):
    """
    x_tiled: (B * num_tiles, tile_side², C)
    Returns: (B, H*W, C)
    """
    B, HW, C = x_tiled.shape
    H = W = int(HW**0.5)
    num_tiles_per_side = int(num_tiles**0.5)
    tile_side_len = H // num_tiles_per_side

    # reshape each tile to (B, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, C)
    x_tiles = x_tiled.view(
        B, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, C
    )

    # move tiles back into image
    x_tiles = x_tiles.permute(0, 1, 3, 2, 4, 5).contiguous()
    x_tiles = x_tiles.view(B, H, W, C)

    return x_tiles.view(B, H * W, C)


def tile_compare(x, r, num_tiles):
    B, HW, C = x.shape
    H = W = int(HW**0.5)

    num_tiles_per_side = int(num_tiles**0.5)
    tile_side_len = H // num_tiles_per_side

    indices = torch.arange(HW, device=x.device).reshape(1, H, W, 1)
    patch_indices = torch.as_strided(
        indices,
        (1, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, 1),
        (HW * 1, tile_side_len * W * 1, tile_side_len * 1, W * 1, 1, 1),
    )

    x_reshaped = torch.as_strided(
        x,
        (1, num_tiles_per_side, num_tiles_per_side, tile_side_len, tile_side_len, C),
        (HW * C, tile_side_len * H * C, tile_side_len * C, H * C, C, 1),
    )
    x_reshaped = x_reshaped.reshape(-1, tile_side_len**2, C)
    x_reshaped = x_reshaped.reshape(B, HW, C)


    return x_reshaped, patch_indices



B, H, W, C = 1, 8, 8, 1 
x = torch.arange(H * W).reshape(1, H * W, C).float()

num_tiles = 16
r = 8 

print(x.reshape(-1, H, W))

tiles = tile(x, num_tiles)
print(tiles.reshape(num_tiles, -1))

untile = untile(
    tiles,
    num_tiles=num_tiles
)
print(untile.reshape(-1, H, W))  

print(torch.allclose(
    x,
    untile,
    atol=1e-5
)) 