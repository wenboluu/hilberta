import json
import torch
from diffusers import Flux2KleinPipeline

BATCH_SIZE = 16

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-9B", torch_dtype=dtype).to(device)

with open("coco_prompts.json", "r") as f:
    prompts = json.load(f)[:BATCH_SIZE]

generators = [torch.Generator(device=device).manual_seed(0) for _ in prompts]

images = pipe(
    prompt=prompts,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=10,
    generator=generators,
).images

for i, image in enumerate(images):
    image.save(f"flux-klein-{i}.png")
