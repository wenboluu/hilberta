import torch
from utils import get_hilbert_flat_indices
import matplotlib.pyplot as plt
import numpy as np

hilbert_index = get_hilbert_flat_indices(3)
x = torch.arange(64)
cut_off = x.shape[0] // 4

hilbert_x = torch.gather(x, dim=0, index=hilbert_index)

hilbert_x_half = hilbert_x[cut_off:-cut_off]
print('hilbder x half', hilbert_x_half)
x_flip = x.flip(0)
hilbert_x_flip = torch.gather(x_flip, dim=0, index=hilbert_index)
hilbert_x_half_flip = hilbert_x_flip[cut_off:-cut_off]

hilbert_x_final = torch.cat([hilbert_x_half, hilbert_x_half_flip])
print('x flip half flip:', hilbert_x_final)

