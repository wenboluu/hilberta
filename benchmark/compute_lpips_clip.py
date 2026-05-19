import torch
import torchvision.transforms as T
from torchvision.io import read_image
import lpips
import os
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm

from transformers import CLIPProcessor, CLIPModel

import re

def get_stem_to_file_map(directory):
    """
    Returns a dict mapping filename stem (without extension) to the full filename.
    Only includes files that look like images.
    """
    image_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    files = os.listdir(directory)
    stem_map = {}
    for f in files:
        ext = os.path.splitext(f)[1].lower()
        if ext in image_exts:
            stem = os.path.splitext(f)[0]
            stem_map[stem] = f
    return stem_map

def compute_lpips_clip(dir1, dir2):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lpips_fn = lpips.LPIPS(net='vgg').to(device)
    
    # Load HuggingFace CLIP model and processor
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # For LPIPS: normalize to [-1, 1]
    transform_lpips = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Build stem-to-file maps for both directories
    stem_map1 = get_stem_to_file_map(dir1)
    stem_map2 = get_stem_to_file_map(dir2)
    common_stems = sorted(list(set(stem_map1.keys()) & set(stem_map2.keys())))
    if not common_stems:
        print("No matching images found in both directories (by stem).")
        return

    lpips_scores = []
    clip_scores = []

    for stem in tqdm(common_stems, desc="Computing metrics"):
        file1 = stem_map1[stem]
        file2 = stem_map2[stem]
        img1 = read_image(os.path.join(dir1, file1)).to(device)
        img2 = read_image(os.path.join(dir2, file2)).to(device)

        # LPIPS
        img1_lpips = transform_lpips(img1)
        img2_lpips = transform_lpips(img2)
        score_lpips = lpips_fn(img1_lpips.unsqueeze(0), img2_lpips.unsqueeze(0)).item()
        lpips_scores.append(score_lpips)

        # CLIP (HuggingFace)
        img1_pil = to_pil_image(img1.cpu())
        img2_pil = to_pil_image(img2.cpu())
        # Use processor to preprocess images
        inputs = clip_processor(images=[img1_pil, img2_pil], return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        with torch.no_grad():
            image_features = clip_model.get_image_features(pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            score_clip = (image_features[0] @ image_features[1]).item()
        clip_scores.append(score_clip)

    lpips_mean = sum(lpips_scores) / len(lpips_scores)
    clip_mean = sum(clip_scores) / len(clip_scores)
    print(f"Average LPIPS: {lpips_mean:.6f} (lower is better)")
    print(f"Average CLIP similarity: {clip_mean:.6f} (higher is better)")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compute average LPIPS and CLIP similarity between two image folders.")
    parser.add_argument("dir1", type=str, help="Path to first image directory (e.g., originals)")
    parser.add_argument("dir2", type=str, help="Path to second image directory (e.g., generated)")
    args = parser.parse_args()
    compute_lpips_clip(args.dir1, args.dir2) 
