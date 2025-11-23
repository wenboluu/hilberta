from utils import get_hilbert_flat_indices, get_inverse_hilbert_indices
import torch


def hilbert_tile(x, reverse, offset=0):
    if reverse:
        x = x.flip(1)
    hilbert_index = get_hilbert_flat_indices(6).to(x.device)
    hilbert_index_offset = torch.cat([hilbert_index[offset:], hilbert_index[:offset]])
    return torch.gather(x, 1, hilbert_index_offset.unsqueeze(0).unsqueeze(-1).repeat(x.shape[0], 1, x.shape[-1]))

def hilbert_untile(x_hilbert, reverse, offset=0):
    # if reverse:
    #     x_hilbert = x_hilbert.flip(1)
    inverse_index = get_inverse_hilbert_indices(6).to(x_hilbert.device)
    if reverse:
        inverse_index = inverse_index.flip(0)
    if offset > 0:
        x_hilbert = torch.cat([x_hilbert[:, -offset:], x_hilbert[:, :-offset]], dim=1)

    return torch.gather(x_hilbert, 1, inverse_index.unsqueeze(0).unsqueeze(-1).repeat(x_hilbert.shape[0], 1, x_hilbert.shape[-1]))


a = torch.randn(1, 4096, 3072).cuda()
a_tile = hilbert_tile(a, True, 0)
a_recover = hilbert_untile(a_tile, True, 0)
print(torch.allclose(a, a_recover, atol=1e-5))