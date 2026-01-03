import os
import argparse
from PIL import Image, ImageChops

def generate_diff_image(image_path1, image_path2, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the two images
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)

    # Compute the difference
    diff = ImageChops.difference(img1, img2)

    # Generate the output filename based on the input filenames
    base_name1 = os.path.splitext(os.path.basename(image_path1))[0]
    base_name2 = os.path.splitext(os.path.basename(image_path2))[0]
    output_filename = f"{base_name1}_diff_{base_name2}.png"

    # Save the diff image
    output_path = os.path.join(output_dir, output_filename)
    diff.save(output_path)

    print(f"Diff image saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a diff image from two input images.")
    parser.add_argument("image_path1", type=str, help="Path to the first image.")
    parser.add_argument("image_path2", type=str, help="Path to the second image.")

    args = parser.parse_args()

    generate_diff_image(args.image_path1, args.image_path2, "diff_output")
