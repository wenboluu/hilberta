from modelscope.hub.snapshot_download import snapshot_download
from diffusers import FluxFillPipeline
import torch
from diffusers.utils import load_image
from masking_utils import apply_patch

# NOTE: Update model_dir path for your server
model_dir = "/data2/sz3684/.cache/modelscope/black-forest-labs/FLUX.1-Fill-dev"  # TODO: Update this path

image = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup.png")
mask = load_image("https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/cup_mask.png")

pipeline = FluxFillPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

import types
from masking_utils import customized_forward, customized_call
pipeline.transformer.forward = types.MethodType(customized_forward, pipeline.transformer)


apply_patch(
    pipeline,
    num_tiles=4,
    height=height,
)

pipeline.to("cuda:7")

image = pipeline(
    prompt="a white paper cup",
    image=image,
    mask_image=mask,
    height=1632,
    width=1232,
    guidance_scale=30,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]


image.save(f"flux-fill-dev.png")