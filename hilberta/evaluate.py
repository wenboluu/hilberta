"""Matched-image FID/KID, LPIPS, CLIP, PSNR and SSIM evaluation.

Directories must contain identical PNG filenames from the same frozen manifest.
"""

import argparse, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from .runtime import prompts


def image(path, device):
    return (
        torch.from_numpy(np.asarray(Image.open(path).convert("RGB")).copy())
        .permute(2, 0, 1)[None]
        .to(device)
        .float()
        / 255
    )


def ssim(x, y):
    coord = torch.arange(11, device=x.device, dtype=x.dtype) - 5
    g = torch.exp(-coord.square() / (2 * 1.5**2))
    g /= g.sum()
    w = (g[:, None] * g[None, :]).expand(3, 1, 11, 11)
    conv = lambda z: F.conv2d(z, w, groups=3)
    ux, uy = conv(x), conv(y)
    vx = conv(x * x) - ux * ux
    vy = conv(y * y) - uy * uy
    vxy = conv(x * y) - ux * uy
    return (
        ((2 * ux * uy + 0.01**2) * (2 * vxy + 0.03**2))
        / ((ux * ux + uy * uy + 0.01**2) * (vx + vy + 0.03**2))
    ).mean()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ["images", "teacher", "real", "prompts", "output"]:
        p.add_argument("--" + k, type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--kid-subset", type=int, default=1000)
    a = p.parse_args()
    rows = prompts(a.prompts)
    expected = {str(r["id"]) + ".png" for r in rows}
    for folder in [a.images, a.teacher, a.real]:
        if {p.name for p in folder.glob("*.png")} != expected:
            raise ValueError(f"Manifest/PNG coverage mismatch: {folder}")
    if len(rows) < a.kid_subset:
        raise ValueError("KID subset exceeds evaluation set")
    import lpips, torch_fidelity
    from transformers import CLIPModel, CLIPProcessor

    model = (
        CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14", revision="32bd64288804d66eefd0ccbe215aa642df71cc41"
        )
        .to(a.device)
        .eval()
    )
    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14", revision="32bd64288804d66eefd0ccbe215aa642df71cc41"
    )
    perceptual = lpips.LPIPS(net="alex").to(a.device).eval()
    values = []
    with torch.inference_mode():
        for r in rows:
            name = str(r["id"]) + ".png"
            x, y = image(a.images / name, a.device), image(a.teacher / name, a.device)
            if x.shape != y.shape:
                raise ValueError("Teacher and student resolution mismatch")
            lp = float(
                perceptual(
                    F.interpolate(
                        x * 2 - 1,
                        size=(512, 512),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    ),
                    F.interpolate(
                        y * 2 - 1,
                        size=(512, 512),
                        mode="bilinear",
                        align_corners=False,
                        antialias=True,
                    ),
                ).item()
            )
            mse = float((x - y).square().mean())
            psnr = -10 * math.log10(mse) if mse else None
            inp = processor(
                text=[r["prompt"]],
                images=Image.open(a.images / name).convert("RGB"),
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(a.device)
            z = model(**inp)
            clip = float(F.cosine_similarity(z.image_embeds, z.text_embeds).item())
            values.append(dict(id=r["id"], lpips=lp, psnr=psnr, ssim=float(ssim(x, y)), clip=clip))
    del model, perceptual
    torch.cuda.empty_cache()
    if str(a.device).startswith("cuda"):
        torch.cuda.set_device(a.device)
    scores = torch_fidelity.calculate_metrics(
        input1=str(a.images),
        input2=str(a.real),
        cuda=str(a.device).startswith("cuda"),
        fid=True,
        kid=True,
        kid_subsets=100,
        kid_subset_size=a.kid_subset,
        rng_seed=2707,
        verbose=False,
    )
    for key in ["lpips", "psnr", "ssim", "clip"]:
        v = [r[key] for r in values if r[key] is not None]
        scores[key] = dict(
            mean=float(np.mean(v)) if v else None,
            median=float(np.median(v)) if v else None,
            p90=float(np.quantile(v, 0.9)) if v else None,
        )
    scores["psnr_infinite_count"] = sum(r["psnr"] is None for r in values)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(
        json.dumps(dict(summary=scores, per_image=values, protocol=vars(a)), default=str, indent=2)
    )


if __name__ == "__main__":
    main()
