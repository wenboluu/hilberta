import os
import glob
import random
import argparse
from PIL import Image

# === Argument parsing ===
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='./outputs/lora_1024_4', help='Directory containing PNG files')
parser.add_argument('--dim', type=str, required=True, help='mxn, e.g., 3x3, Grid dimension')
args = parser.parse_args()

grid_h, grid_w = map(int, args.dim.split('x'))
png_dir = args.dir
total_required = grid_w * grid_h

# === Collect PNG files ===
image_files = []
for ext in ("*.png", "*.jpg", "*.jpeg"):
    image_files.extend(glob.glob(os.path.join(png_dir, ext)))

if len(image_files) < total_required:
    raise ValueError(f"Not enough image files in {png_dir}. Required: {total_required}, Found: {len(image_files)}")

selected_files = random.sample(image_files, total_required)

# === Load and resize images ===
images = []
for file_path in selected_files:
    img = Image.open(file_path)
    img_resized = img.resize((img.width // 2, img.height // 2))
    images.append(img_resized)

# === Get dimensions ===
img_width, img_height = images[0].size
grid_width = img_width * grid_w
grid_height = img_height * grid_h

# === Create output image ===
grid_img = Image.new('RGB', (grid_width, grid_height))

for idx, img in enumerate(images):
    row = idx // grid_w
    col = idx % grid_w
    x = col * img_width
    y = row * img_height
    grid_img.paste(img, (x, y))

# === Save ===
output_path = os.path.join(os.getcwd(), f"random_grid_{grid_h}x{grid_w}.jpg")
grid_img.save(output_path)
print(f"Created {grid_h}x{grid_w} grid of randomly selected images at {output_path}")
