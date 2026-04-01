import os
from pathlib import Path
import re

import torch
from diffusers import FluxPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

MODEL_REPO = "black-forest-labs/FLUX.1-dev"
# NOTE: Set cache_dir to your HuggingFace cache directory or None to use default
DEFAULT_CACHE_DIR = None  # Path("/your/cache/dir/")
DEFAULT_OUTPUT_DIR = Path(__file__).parent
DEFAULT_HEIGHT = 1024
DEFAULT_WIDTH = 1024
DEFAULT_GUIDANCE_SCALE = 3.5
DEFAULT_NUM_STEPS = 5
DEFAULT_SEED = 0

PROMPT_LIST = [
    "A group of men and woman playing a game of frisbee."
]


def prepare_pipeline(device: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
        cache_dir=str(DEFAULT_CACHE_DIR),
    ).to(device)
    return pipe


def sanitize_prompt(prompt: str) -> str:
    safe_stub = re.sub(r"[^0-9a-zA-Z]+", "_", prompt.strip())[:32]
    return safe_stub or "image"


def generate_image(pipe: FluxPipeline, prompt: str, device: str) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(DEFAULT_SEED)
    result = pipe(
        prompt,
        height=DEFAULT_HEIGHT,
        width=DEFAULT_WIDTH,
        guidance_scale=DEFAULT_GUIDANCE_SCALE,
        num_inference_steps=DEFAULT_NUM_STEPS,
        max_sequence_length=512,
        generator=generator,
    )
    return result.images[0]


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = prepare_pipeline(device)
    output_dir = DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    enable_profiler = os.environ.get("ORIGINAL_PIPE_ENABLE_PROFILER", "1") != "0"
    print(f"Profiler enabled: {enable_profiler}")

    if enable_profiler:
        activities = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        logs_dir = Path(__file__).with_name("profiler_logs")
        logs_dir.mkdir(parents=True, exist_ok=True)
        schedule = torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=1)
        handler = torch.profiler.tensorboard_trace_handler(str(logs_dir))
        print(f"Starting profiler with TensorBoard logging to {logs_dir}")
        with torch.profiler.profile(
            activities=activities,
            schedule=schedule,
            on_trace_ready=handler,
            record_shapes=False,
            with_stack=True,
            profile_memory=False,
        ) as prof:
            for prompt in PROMPT_LIST:
                image = generate_image(pipe=pipe, prompt=prompt, device=device)
                filename = f"original_pipe_{sanitize_prompt(prompt)}.png"
                image.save(output_dir / filename)
                prof.step()
            try:
                prof.step()
            except Exception:
                pass
            try:
                prof.export_chrome_trace(str(logs_dir / "trace.json"))
            except Exception:
                pass
    else:
        for prompt in PROMPT_LIST:
            image = generate_image(pipe=pipe, prompt=prompt, device=device)
            filename = f"original_pipe_{sanitize_prompt(prompt)}.png"
            image.save(output_dir / filename)


if __name__ == "__main__":
    main()
