from pathlib import Path
import torch

mask1_dir = "/home/sz3684/diffusion/reorder_local_attention/diffusion_reorder/mask"
mask2_dir = "/home/sz3684/diffusion/reorder_local_attention/diffusion_reorder/mask_list"

mask1 = Path(mask1_dir)
mask2 = Path(mask2_dir)

name1 = "mask_offset_{offset}_num_of_tiles_{num_of_tiles}.pt"
name2 = "image_size_{image_size}_offset_{offset}_num_of_tiles_{num_of_tiles}.pt"

image_size = 4096

for offset in [0, 64, 128, 192, 256, 512, 768]:
    for num_of_tiles in [4 ,16]:
        try:
            mask1_file = mask1 / name1.format(offset=offset, num_of_tiles=num_of_tiles)
            mask2_file = mask2 / name2.format(image_size=image_size, offset=offset, num_of_tiles=num_of_tiles)
            tensor1 = torch.load(mask1_file).cpu()
            tensor2 = torch.load(mask2_file).cpu()
            if torch.allclose(tensor1, tensor2):
                print(f"mask1_file and mask2_file are the same for offset {offset} and num_of_tiles {num_of_tiles}")
            else:
                print(f"mask1_file and mask2_file are different for offset {offset} and num_of_tiles {num_of_tiles}")
        except Exception as e:
            continue