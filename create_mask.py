# load config

import itertools
import os

import torch
import yaml

from src.masking_utils import create_hilbert_tile_mask

if __name__ == "__main__":
    os.makedirs('./masks', exist_ok=True)

    with open('./src/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    image_size_list = [4096, 16384]
    num_of_tiles_list = [4, 16]
    sliding_cycle_list = [2, 4]

    for image_size, num_of_tiles, sliding_cycle in itertools.product(
            image_size_list, num_of_tiles_list, sliding_cycle_list):
        hidden_states = torch.randn(1, image_size, 1)
        offset_list = [((hidden_states.shape[1] // num_of_tiles) // sliding_cycle) * i for i in range(sliding_cycle)]

        for offset in offset_list:
            mask = create_hilbert_tile_mask(hidden_states, num_of_tiles=num_of_tiles, offset=offset)
            torch.save(mask, f'./masks/image_size_{image_size}_offset_{offset}_num_of_tiles_{num_of_tiles}.pt')
