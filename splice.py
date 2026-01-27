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

grid_w, grid_h = map(int, args.dim.split('x'))
png_dir = args.dir
total_required = grid_w * grid_h

# === Collect PNG files ===
png_files = glob.glob(os.path.join(png_dir, "*.png"))
if len(png_files) < total_required:
    raise ValueError(f"Not enough PNG files in {png_dir}. Required: {total_required}, Found: {len(png_files)}")

selected_files = random.sample(png_files, total_required)

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
output_path = os.path.join(os.getcwd(), f"random_grid_{grid_w}x{grid_h}.jpg")
grid_img.save(output_path)
print(f"Created {grid_w}x{grid_h} grid of randomly selected images at {output_path}")
