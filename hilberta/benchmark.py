"""Synthetic resident-kernel correctness and timing; not end-to-end generation."""

import argparse, json, statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from .bridge_k import bridge_k_plan
from .paired_metadata import pair_metadata
from .paired_kernel import paired_attention


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--resolution", type=int, choices=[1024, 2048, 4096], default=1024)
    p.add_argument("--tiles", type=int, choices=[4, 16], default=4)
    p.add_argument("--k", type=int, choices=[1, 3], default=1)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeats", type=int, default=30)
    a = p.parse_args()
    torch.cuda.set_device(a.device)
    torch.manual_seed(2707)
    torch.set_num_threads(4)
    plan = bridge_k_plan(a.tiles, a.k, False, grid_size=a.resolution // 16)
    N = plan.total_tokens
    q, k, v = [torch.randn(1, 24, N, 128, device=a.device, dtype=torch.bfloat16) for _ in range(3)]

    def timed(fn):
        for _ in range(a.warmup):
            fn()
        torch.cuda.synchronize()
        times = []
        for _ in range(a.repeats):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            fn()
            e.record()
            e.synchronize()
            times.append(s.elapsed_time(e))
        return dict(
            median_ms=statistics.median(times),
            p10_ms=float(torch.tensor(times).quantile(0.1)),
            p90_ms=float(torch.tensor(times).quantile(0.9)),
        )

    rows = []
    with torch.inference_mode():
        for phase, p in enumerate(plan.phases):
            meta = pair_metadata(
                p.kv_indices, p.kv_counts, tokens=N, global_tokens=plan.global_tokens
            ).to(a.device)
            fn = lambda: paired_attention(
                q, k, v, meta, global_strategy="triton", num_warps=8, num_stages=2
            )
            out = fn()
            # Independent row-wise oracle: no N*N dense mask even at 4096.
            indices = torch.tensor([0, 511, 512, 1023, N // 2, N - 1], device=a.device)
            masks = torch.zeros(len(indices), N, device=a.device, dtype=torch.bool)
            for j, qi in enumerate(indices.tolist()):
                if qi < plan.global_tokens:
                    masks[j] = True
                else:
                    row = (qi - plan.global_tokens) // 64
                    keys = p.kv_indices[row, : int(p.kv_counts[row])].tolist()
                    for key in keys:
                        masks[j, key * 64 : (key + 1) * 64] = True
            ref = F.scaled_dot_product_attention(q[:, :, indices], k, v, attn_mask=masks)
            err = float((out[:, :, indices].float() - ref.float()).norm() / ref.float().norm())
            if not __import__("math").isfinite(err) or err > 0.02:
                raise RuntimeError(f"Correctness failed phase {phase}: {err}")
            rows.append(dict(phase=phase, relative_l2=err, **timed(fn)))
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            full = timed(lambda: F.scaled_dot_product_attention(q, k, v))
    weights = [15, 14, 14, 14]
    result = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        config=vars(a),
        full=full,
        phases=rows,
        weighted_median_ms=sum(r["median_ms"] * w for r, w in zip(rows, weights)) / 57,
        scope="resident synthetic QKV, phase-weighted single layer; setup/packing excluded",
    )
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, default=str, indent=2))


if __name__ == "__main__":
    main()
