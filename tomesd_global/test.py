import torch
import unittest


def index_shift_for_tile_sliding(x, tile_len, flag = None):
    B, N, C = x.shape
    H = W = int(N ** 0.5)
    slide_len = tile_len // 2
    x_reshape = x.reshape(B, H, W, C)
    if flag == 'down_right':
        x_reshape = torch.roll(x_reshape, shifts=(slide_len, slide_len), dims=(1, 2))
    elif flag == 'down_left':
        x_reshape = torch.roll(x_reshape, shifts=(slide_len, -slide_len), dims=(1, 2))
    elif flag == 'up_right':
        x_reshape = torch.roll(x_reshape, shifts=(-slide_len, slide_len), dims=(1, 2))
    elif flag == 'up_left':
        x_reshape = torch.roll(x_reshape, shifts=(-slide_len, -slide_len), dims=(1, 2))
    return x_reshape.reshape(B, N, C)

class TestFoldWithIndices(unittest.TestCase):
    def test_fold_with_indices_outputs(self):
        # Parameters for the test:
        B = 1
        H = W = 4          # Image height and width
        N = H * W          # Total number of tokens
        C = 3              # Number of channels
        num_tiles = 4     # e.g. 4x4 grid

        # Create a random input tensor of shape (B, N, C)
        x = torch.arange(N).unsqueeze(0).unsqueeze(-1).expand(-1, -1, C).float()
        print(x)
        x_reshape = index_shift_for_tile_sliding(x, 2, 'up_left')
        print(x_reshape)
        print(torch.allclose(x_reshape, x))

if __name__ == '__main__':
    unittest.main()
