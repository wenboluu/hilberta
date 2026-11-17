"""Portable DDP LoRA distillation over cached teacher records.

New release runner; reported paper results were produced by the historical
training system. Numerical/GPU production validation of this CLI is pending.
"""

import argparse, contextlib, hashlib, json, os, random
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from diffusers import FluxTransformer2DModel, FluxPipeline
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from .runtime import MODEL, REVISION, add_lora, load_adapter, atomic_save
from .bridge_k import bridge_k_plan
from .flex_sparse import flex_plan_attention


class Records:
    def __init__(self, root):
        self.entries = []
        self.path = None
        self.obj = None
        self.text = None
        protocols = []
        for idx in sorted(Path(root).glob("shard-*/index.json")):
            protocol = json.loads((idx.parent / "protocol.json").read_text())
            protocols.append(protocol)
            self.entries.extend(
                (idx.parent / r["file"], r["step"]) for r in json.loads(idx.read_text())
            )
        if not self.entries:
            raise ValueError("No completed shards found")
        if any(x != protocols[0] for x in protocols):
            raise ValueError("Cache shards use different protocols")
        self.protocol = protocols[0]
        self.signature = hashlib.sha256(
            json.dumps(
                {
                    "protocol": self.protocol,
                    "records": [(p.parent.name, p.name, s) for p, s in self.entries],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def __len__(self):
        return len(self.entries)

    def get(self, i):
        path, step = self.entries[i]
        if path != self.path:
            self.obj = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            self.text = torch.load(
                path.parent / self.obj["text_file"], map_location="cpu", weights_only=True
            )
            self.path = path
        return self.obj["records"][step], self.text


def epoch_order(n, seed, epoch):
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default=MODEL)
    p.add_argument("--revision", default=REVISION)
    p.add_argument("--microbatch", type=int, default=1)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--resume")
    p.add_argument("--init-adapter")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=2707)
    a = p.parse_args()
    if a.resume and a.init_adapter:
        p.error("Choose resume OR init-adapter")
    if a.save_every < 1:
        p.error("--save-every must be positive")
    local = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    torch.cuda.set_device(local)
    device = f"cuda:{local}"
    if world > 1:
        dist.init_process_group("nccl")
    cfg = json.loads(Path(a.config).read_text())
    data = Records(a.cache)
    if cfg["shared"] or cfg["rank"] != 32 or cfg["alpha"] != 32:
        raise ValueError("Release uses no shared, rank/alpha32")
    if (
        data.protocol["steps"] != cfg["steps"]
        or data.protocol["guidance"] != cfg["guidance"]
        or data.protocol["resolution"] != cfg["resolution"]
        or data.protocol["model"] != a.model
        or data.protocol["revision"] != a.revision
    ):
        raise ValueError("Teacher/model/config mismatch")
    gb = a.microbatch * world * a.accum
    if a.microbatch < 1 or a.accum < 1 or gb != cfg["global_batch"] or len(data) % gb:
        raise ValueError(
            "microbatch*world*accum must equal configured global batch and divide record count"
        )
    if a.output.exists() and any(a.output.iterdir()) and not a.resume:
        raise ValueError("Use a new output directory")
    a.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    model = FluxTransformer2DModel.from_pretrained(
        a.model, revision=a.revision, subfolder="transformer", torch_dtype=torch.bfloat16
    ).to(device)
    model.requires_grad_(False)
    if a.init_adapter:
        load_adapter(model, a.init_adapter)
    else:
        add_lora(model)
    params = [x for x in model.parameters() if x.requires_grad]
    opt = torch.optim.AdamW(
        params,
        lr=cfg["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        foreach=False,
    )
    updates_per_epoch = len(data) // gb
    total = updates_per_epoch * cfg["epochs"]
    start = 0
    signature = dict(
        config=cfg,
        cache_signature=data.signature,
        world=world,
        microbatch=a.microbatch,
        accum=a.accum,
        seed=a.seed,
        model=a.model,
        revision=a.revision,
    )
    if a.resume:
        ck = torch.load(a.resume, map_location="cpu", weights_only=True)
        if ck["signature"] != signature:
            raise ValueError("Resume configuration differs")
        set_peft_model_state_dict(model, ck["adapter"])
        opt.load_state_dict(ck["optimizer"])
        start = ck["update"]
        torch.set_rng_state(ck["rng"][rank]["cpu"])
        torch.cuda.set_rng_state(ck["rng"][rank]["cuda"])
    model.enable_gradient_checkpointing()
    model.train()
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    plan = bridge_k_plan(cfg["tiles"], cfg["bridge_k"], False, grid_size=cfg["resolution"] // 16)
    img_ids = FluxPipeline._prepare_latent_image_ids(
        1, cfg["resolution"] // 16, cfg["resolution"] // 16, device, torch.bfloat16
    )
    if rank == 0:
        (a.output / "config.json").write_text(json.dumps(signature, indent=2))

    def save(update):
        state = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
        rng = [None] * world
        if world > 1:
            dist.all_gather_object(rng, state)
        else:
            rng = [state]
        if rank == 0:
            adapter = {k: v.detach().cpu() for k, v in get_peft_model_state_dict(model).items()}
            atomic_save(
                dict(
                    adapter=adapter,
                    config=cfg,
                    update=update,
                    signature=signature,
                    optimizer=opt.state_dict(),
                    rng=rng,
                ),
                a.output / "last.pt",
            )
            atomic_save(
                dict(adapter=adapter, config=cfg, update=update),
                a.output / f"adapter_step{update:06d}.pt",
            )

    # Keep the attention context alive during checkpoint recomputation/backward.
    with flex_plan_attention(model, plan, 64):
        for epoch in range(start // updates_per_epoch, cfg["epochs"]):
            order = epoch_order(len(data), a.seed, epoch)
            begin = start % updates_per_epoch if epoch == start // updates_per_epoch else 0
            for u in range(begin, updates_per_epoch):
                update = epoch * updates_per_epoch + u
                opt.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=device)
                lr = cfg["learning_rate"] * min(1.0, (update + 1) / max(1, cfg["warmup_updates"]))
                for g in opt.param_groups:
                    g["lr"] = lr
                for accum in range(a.accum):
                    offset = u * gb + (accum * world + rank) * a.microbatch
                    batch = [data.get(i) for i in order[offset : offset + a.microbatch]]
                    cat = lambda key: torch.cat([t[key] for r, t in batch], dim=0).to(device)
                    inputs = dict(
                        hidden_states=torch.cat([r["x_t"] for r, t in batch]).to(device),
                        timestep=torch.cat([r["model_timestep"].reshape(-1) for r, t in batch]).to(
                            device
                        ),
                        encoder_hidden_states=cat("prompt_embeds"),
                        pooled_projections=cat("pooled_prompt_embeds"),
                        txt_ids=batch[0][1]["text_ids"].to(device),
                        img_ids=img_ids,
                        guidance=torch.full(
                            (a.microbatch,), cfg["guidance"], device=device, dtype=torch.bfloat16
                        ),
                        return_dict=False,
                    )
                    target = torch.cat([r["v"] for r, t in batch]).to(device)
                    sync = (
                        wrapped.no_sync()
                        if world > 1 and accum < a.accum - 1
                        else contextlib.nullcontext()
                    )
                    with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                        pred = wrapped(**inputs)[0]
                        loss = (pred.float() - target.float()).square().mean()
                        if not torch.isfinite(loss):
                            raise RuntimeError("Nonfinite loss")
                        (loss / a.accum).backward()
                    loss_sum += loss.detach() / a.accum
                norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
                opt.step()
                if world > 1:
                    dist.all_reduce(loss_sum)
                    loss_sum /= world
                if rank == 0:
                    row = dict(
                        update=update + 1, loss=loss_sum.item(), lr=lr, grad_norm=float(norm)
                    )
                    with (a.output / "metrics.jsonl").open("a") as f:
                        f.write(json.dumps(row) + "\n")
                    print(row, flush=True)
                if (update + 1) % a.save_every == 0 or update + 1 == total:
                    save(update + 1)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
