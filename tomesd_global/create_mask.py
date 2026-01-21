# load config 

import yaml
import torch
import os
from masking_utils import create_hilbert_tile_mask
import itertools

if __name__ == "__main__":
    os.makedirs('./mask_list', exist_ok=True)

    with open('/home/sz3684/diffusion/reorder_local_attention/diffusion_reorder/tomesd_global/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    num_of_tiles = config['num_tiles']
    sliding_cycle = config['sliding_cycle']

    image_size_list = [4096, 16384]
    num_of_tiles_list = [4, 16]
    sliding_cycle_list = [2, 4]

    for image_size, num_of_tiles, sliding_cycle in itertools.product(image_size_list, num_of_tiles_list, sliding_cycle_list):
        hidden_states = torch.randn(1, image_size, 1)
        offset_list = [((hidden_states.shape[1]//num_of_tiles) //sliding_cycle) * i for i in range(sliding_cycle)]

        for offset in offset_list:
            mask = create_hilbert_tile_mask(hidden_states, num_of_tiles=num_of_tiles, offset=offset)
            with open(f'./mask_list/image_size_{image_size}_offset_{offset}_num_of_tiles_{num_of_tiles}.pt', 'wb') as f:
                torch.save(mask, f)