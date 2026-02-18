import torch
from diffusers import FluxPipeline
import time 

import os 
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", 
                                    torch_dtype=torch.bfloat16, 
                                    local_files_only=True,
                                    cache_dir="/data1/wl2707/.cache/huggingface/hub").to("cuda")
# pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=28,
    max_sequence_length=512,
    generator=torch.Generator(device="cuda").manual_seed(0)
).images[0]