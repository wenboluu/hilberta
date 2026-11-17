"""Cache full FLUX teacher trajectories from user-supplied caption manifests."""

import argparse, hashlib, json
from pathlib import Path
import torch
from diffusers import FluxPipeline
from .runtime import MODEL, REVISION, prompts, atomic_save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--resolution", type=int, choices=[1024, 2048], default=1024)
    p.add_argument("--seeds", type=int, nargs="+", default=[2707, 2708])
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--device", default="cuda:0")
    a = p.parse_args()
    assert 0 <= a.shard < a.shards and a.steps > 0
    rows = prompts(a.prompts)[a.shard :: a.shards]
    dest = a.output / f"shard-{a.shard:03d}"
    dest.mkdir(parents=True, exist_ok=False)
    pipe = FluxPipeline.from_pretrained(
        a.model, revision=a.revision, torch_dtype=torch.bfloat16
    ).to(a.device)
    entries = []
    with torch.inference_mode():
        for row in rows:
            pe, pooled, ids = pipe.encode_prompt(
                row["prompt"], row["prompt"], device=a.device, max_sequence_length=512
            )
            textfile = f"text-{row['id']}.pt"
            atomic_save(
                dict(prompt_embeds=pe.cpu(), pooled_prompt_embeds=pooled.cpu(), text_ids=ids.cpu()),
                dest / textfile,
            )
            for seed in a.seeds:
                pending = {}
                records = []

                def before(m, args, kw):
                    pending["x"] = kw["hidden_states"].detach().clone()
                    pending["t"] = kw["timestep"].detach().clone()

                def after(m, args, out):
                    pending["v"] = out[0].detach()

                def callback(pl, i, t, kw):
                    x, v = pending["x"], pending["v"]
                    sig = pl.scheduler.sigmas
                    expected = (x.float() + (sig[i + 1] - sig[i]) * v).to(v.dtype)
                    if not torch.equal(expected, kw["latents"]):
                        raise RuntimeError("Teacher Euler replay mismatch")
                    if not torch.isfinite(v).all():
                        raise RuntimeError("Nonfinite teacher prediction")
                    records.append(
                        dict(x_t=x.cpu(), v=v.cpu(), model_timestep=pending["t"].cpu(), step=i)
                    )
                    return kw

                h1 = pipe.transformer.register_forward_pre_hook(before, with_kwargs=True)
                h2 = pipe.transformer.register_forward_hook(after)
                try:
                    pipe(
                        prompt_embeds=pe,
                        pooled_prompt_embeds=pooled,
                        height=a.resolution,
                        width=a.resolution,
                        num_inference_steps=a.steps,
                        guidance_scale=7.5,
                        max_sequence_length=512,
                        generator=torch.Generator(device=a.device).manual_seed(seed),
                        output_type="latent",
                        callback_on_step_end=callback,
                    )
                finally:
                    h1.remove()
                    h2.remove()
                assert len(records) == a.steps
                file = f"trajectory-{row['id']}-{seed}.pt"
                atomic_save(
                    dict(
                        records=records,
                        text_file=textfile,
                        resolution=a.resolution,
                        seed=seed,
                        prompt=row,
                    ),
                    dest / file,
                )
                entries.extend(dict(file=file, step=i) for i in range(a.steps))
    (dest / "index.json").write_text(json.dumps(entries))
    (dest / "protocol.json").write_text(
        json.dumps(
            dict(
                model=a.model,
                revision=a.revision,
                resolution=a.resolution,
                seeds=a.seeds,
                steps=a.steps,
                guidance=7.5,
                manifest_sha256=hashlib.sha256(Path(a.prompts).read_bytes()).hexdigest(),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
