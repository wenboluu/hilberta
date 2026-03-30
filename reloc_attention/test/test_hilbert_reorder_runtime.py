import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import sys
import time
from pathlib import Path
import types
import logging as py_logging

import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULE_ROOT = Path(__file__).resolve().parents[1]
if str(MODULE_ROOT) not in sys.path:
    sys.path.insert(0, str(MODULE_ROOT))


class _DiffusersLogging:
    @staticmethod
    def get_logger(name: str):
        return py_logging.getLogger(name)


diffusers = types.ModuleType("diffusers")
diffusers_utils = types.ModuleType("diffusers.utils")
diffusers_utils.USE_PEFT_BACKEND = False
diffusers_utils.is_torch_version = lambda *args, **kwargs: True
diffusers_utils.logging = _DiffusersLogging()
diffusers_utils.scale_lora_layers = lambda *args, **kwargs: None
diffusers_utils.unscale_lora_layers = lambda *args, **kwargs: None

diffusers_models = types.ModuleType("diffusers.models")
diffusers_modeling_outputs = types.ModuleType("diffusers.models.modeling_outputs")


class _Transformer2DModelOutput:  # minimal placeholder for import side effect
    pass


diffusers_modeling_outputs.Transformer2DModelOutput = _Transformer2DModelOutput

diffusers.utils = diffusers_utils
diffusers.models = diffusers_models
diffusers_models.modeling_outputs = diffusers_modeling_outputs

sys.modules.setdefault("diffusers", diffusers)
sys.modules.setdefault("diffusers.utils", diffusers_utils)
sys.modules.setdefault("diffusers.models", diffusers_models)
sys.modules.setdefault("diffusers.models.modeling_outputs", diffusers_modeling_outputs)

from reloc_attention import reorder_utils_sliding

def measure_and_print(device: torch.device, num_iters: int = 100) -> None:
    torch.manual_seed(0)

    batch_size = 1
    spatial_tokens = 16384  # 4096 tokens
    embed_dim = 3072
    num_tiles = 16
    offset = 0

    seq_len = 512 + spatial_tokens

    base_rotary_emb = (
        torch.randn(seq_len, embed_dim, device=device),
        torch.randn(seq_len, embed_dim, device=device),
    )
    base_hidden_states = torch.randn(batch_size, spatial_tokens, embed_dim, device=device)

    def sync() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    # warm-up run to populate caches/ensure compilation
    warm_rotary, warm_hidden = reorder_utils_sliding.apply_hilbert_reorder(
        (base_rotary_emb[0].clone(), base_rotary_emb[1].clone()),
        base_hidden_states.clone(),
        num_tiles,
        offset,
    )
    reorder_utils_sliding.recover_hilbert_reorder(
        warm_rotary,
        warm_hidden,
        num_tiles,
        offset,
    )

    apply_times = []
    recover_times = []

    for _ in range(num_iters):
        rotary_emb = (base_rotary_emb[0].clone(), base_rotary_emb[1].clone())
        hidden_states = base_hidden_states.clone()

        start = time.perf_counter()
        reordered_rotary_emb, reordered_hidden_states = reorder_utils_sliding.apply_hilbert_reorder(
            rotary_emb,
            hidden_states,
            num_tiles,
            offset,
        )
        sync()
        apply_times.append(time.perf_counter() - start)

        start = time.perf_counter()
        recovered_rotary_emb, recovered_hidden_states = reorder_utils_sliding.recover_hilbert_reorder(
            reordered_rotary_emb,
            reordered_hidden_states,
            num_tiles,
            offset,
        )
        sync()
        recover_times.append(time.perf_counter() - start)

        torch.testing.assert_close(recovered_hidden_states, hidden_states)
        torch.testing.assert_close(recovered_rotary_emb[0], rotary_emb[0])
        torch.testing.assert_close(recovered_rotary_emb[1], rotary_emb[1])

    avg_apply_ms = sum(apply_times) / len(apply_times) * 1e3
    avg_recover_ms = sum(recover_times) / len(recover_times) * 1e3

    print(f"Device: {device}")
    print(f"apply_hilbert_reorder avg over {num_iters} runs: {avg_apply_ms:.3f} ms")
    print(f"recover_hilbert_reorder avg over {num_iters} runs: {avg_recover_ms:.3f} ms")


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    measure_and_print(device)


if __name__ == "__main__":
    main()
