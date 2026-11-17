"""Portable model/adapter loading; no server-specific paths."""

from pathlib import Path
import json
import torch
from peft import LoraConfig
from peft.utils import set_peft_model_state_dict

MODEL = "black-forest-labs/FLUX.1-dev"
REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"


def add_lora(model):
    targets = json.loads(Path(__file__).with_name("lora_targets.json").read_text())["modules"]
    model.add_adapter(
        LoraConfig(
            r=32,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=[x["name"] for x in targets],
            init_lora_weights=True,
        )
    )
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()


def load_adapter(model, path):
    obj = torch.load(path, map_location="cpu", weights_only=True)
    add_lora(model)
    state = obj.get("adapter", obj)
    result = set_peft_model_state_dict(model, state)
    if result.unexpected_keys or any("lora_" in k for k in result.missing_keys):
        raise ValueError(f"Incompatible adapter: {result}")
    return obj


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def prompts(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    ids = [str(r["id"]) for r in rows]
    if len(set(ids)) != len(ids) or any(
        not i or not all(c.isalnum() or c in "-_" for c in i) for i in ids
    ):
        raise ValueError("Unique filename-safe IDs are required")
    if any(not isinstance(r.get("prompt"), str) or not r["prompt"].strip() for r in rows):
        raise ValueError("Each row needs a nonempty prompt")
    return rows
