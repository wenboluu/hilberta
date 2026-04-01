from modelscope.hub.snapshot_download import snapshot_download
from diffusers import FluxKontextPipeline
import torch
from diffusers.utils import load_image

# NOTE: Update cache_dir and model_dir paths for your server
model_dir = snapshot_download(
    model_id="black-forest-labs/FLUX.1-Kontext-dev",
    cache_dir="/data2/sz3684/.cache/modelscope",  # TODO: Update this path
    revision=None
)

model_dir = "/data2/sz3684/.cache/modelscope/black-forest-labs/FLUX.1-Kontext-dev"  # TODO: Update this path

pipeline = FluxKontextPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

pipeline.to("cuda:7")

input_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

image = pipeline(
  image=input_image,
  prompt="Add a hat to the cat",
  guidance_scale=2.5
).images[0]
image.save(f"flux-edit-dev.png")
