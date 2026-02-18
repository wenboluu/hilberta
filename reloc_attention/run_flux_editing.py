from modelscope.hub.snapshot_download import snapshot_download
from diffusers import FluxKontextPipeline  # 改成你实际模块
import torch

model_dir = snapshot_download(
    model_id="black-forest-labs/FLUX.1-Kontext-dev",
    cache_dir="/data2/sz3684/.cache/modelscope",
    revision=None  
)

model_dir = "/data2/sz3684/.cache/modelscope/black-forest-labs/FLUX.1-Kontext-dev"

pipeline = FluxKontextPipeline.from_pretrained(
    model_dir,
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)

pipeline.to("cuda")