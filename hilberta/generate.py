"""Generate matched-seed images using full or HilbertA attention."""

import argparse, json
from pathlib import Path
import torch
from diffusers import FluxPipeline
from .runtime import MODEL, REVISION, load_adapter, prompts
from .bridge_k import bridge_k_plan
from .sampling_layout import sample_flux_persistent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--adapter")
    p.add_argument("--method", choices=["full", "hilberta"], default="hilberta")
    p.add_argument("--resolution", type=int, choices=[1024, 2048, 4096], default=1024)
    p.add_argument("--tiles", type=int, choices=[4, 16], default=4)
    p.add_argument("--k", type=int, choices=[1, 3], default=1)
    p.add_argument("--seed", type=int, default=2707)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--allow-untrained", action="store_true")
    a = p.parse_args()
    if a.method == "hilberta" and not a.adapter and not a.allow_untrained:
        p.error(
            "Checkpoints are pending release. Supply --adapter or explicitly use --allow-untrained (not reported quality)."
        )
    if a.method == "full" and a.adapter:
        p.error("Full reference must not load a sparse adapter")
    rows = prompts(a.prompts)
    a.output.mkdir(parents=True, exist_ok=False)
    pipe = FluxPipeline.from_pretrained(
        a.model, revision=a.revision, torch_dtype=torch.bfloat16
    ).to(a.device)
    pipe.transformer.requires_grad_(False)
    if a.adapter:
        ck = load_adapter(pipe.transformer, a.adapter)
        cfg = ck.get("config", {})
        for key, value in [("tiles", a.tiles), ("bridge_k", a.k), ("resolution", a.resolution)]:
            if key in cfg and cfg[key] != value:
                raise ValueError(f"Adapter {key} mismatch")
        pipe.transformer.fuse_lora()
        pipe.transformer.unload_lora()
        pipe.transformer.to(dtype=torch.bfloat16)
    plan = (
        bridge_k_plan(a.tiles, a.k, False, grid_size=a.resolution // 16)
        if a.method == "hilberta"
        else None
    )
    with torch.inference_mode():
        for row in rows:
            kwargs = dict(
                prompt=row["prompt"],
                height=a.resolution,
                width=a.resolution,
                num_inference_steps=a.steps,
                guidance_scale=7.5,
                max_sequence_length=512,
                generator=torch.Generator(device=a.device).manual_seed(a.seed),
            )
            if plan:
                result, _ = sample_flux_persistent(
                    pipe,
                    plan,
                    backend="paired",
                    kernel_options=dict(global_strategy="triton", num_warps=8, num_stages=2),
                    **kwargs,
                )
            else:
                result = pipe(**kwargs)
            result.images[0].save(a.output / (str(row["id"]) + ".png"))
    (a.output / "protocol.json").write_text(json.dumps(vars(a), default=str, indent=2))


if __name__ == "__main__":
    main()
