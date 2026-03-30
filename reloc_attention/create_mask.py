# load config 

import yaml
import torch
import os
from masking_utils import create_hilbert_tile_mask, create_morton_tile_mask
import itertools

if __name__ == "__main__":
    with open('./config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    num_of_tiles = config['num_tiles']
    sliding_cycle = config['sliding_cycle']
    curve_type = config.get('curve_type', 'hilbert').lower()
    if curve_type not in ['hilbert', 'morton']:
        raise ValueError("config.curve_type must be 'hilbert' or 'morton'")

    # output directory depends on curve type
    mask_dir = f'./mask_{curve_type}'
    os.makedirs(mask_dir, exist_ok=True)

    image_size_list = [4096, 16384]
    num_of_tiles_list = [4, 16]
    sliding_cycle_list = [2, 4]

    for image_size, num_of_tiles, sliding_cycle in itertools.product(image_size_list, num_of_tiles_list, sliding_cycle_list):
        hidden_states = torch.randn(1, image_size, 1)
        offset_list = [((hidden_states.shape[1]//num_of_tiles) //sliding_cycle) * i for i in range(sliding_cycle)]

        for offset in offset_list:
            if curve_type == 'hilbert':
                mask = create_hilbert_tile_mask(hidden_states, num_of_tiles=num_of_tiles, offset=offset)
            else:
                mask = create_morton_tile_mask(hidden_states, num_of_tiles=num_of_tiles, offset=offset)
            with open(f'{mask_dir}/image_size_{image_size}_offset_{offset}_num_of_tiles_{num_of_tiles}.pt', 'wb') as f:
                torch.save(mask, f)