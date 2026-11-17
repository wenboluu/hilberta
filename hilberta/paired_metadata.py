"""CPU-only exact metadata coalescing for two adjacent 64-token query blocks."""

from dataclasses import dataclass, replace
import torch
from .sparse_kernel import validate_metadata


@dataclass(frozen=True)
class PairedMetadata:
    kv_indices: torch.Tensor
    kv_flags: torch.Tensor
    kv_counts: torch.Tensor
    tokens: int
    global_tokens: int
    key_block_size: int

    def to(self, device):
        return replace(
            self,
            kv_indices=self.kv_indices.to(device),
            kv_flags=self.kv_flags.to(device),
            kv_counts=self.kv_counts.to(device),
        )


def pair_metadata(kv_indices, kv_counts, *, tokens, global_tokens=768, key_block_size=64):
    """Merge two query lists; each flag bit keeps original 64x64 visibility.

    KV64 flags bit 0/1 correspond to query half 0/1. KV128 adds bits 2/3
    for query halves 0/1 attending its second 64-token key half. No key is
    duplicated. All work occurs once at plan setup, never inside attention.
    """
    validate_metadata(kv_indices, kv_counts, tokens=tokens, global_tokens=global_tokens)
    if key_block_size not in (64, 128) or tokens % 128 or global_tokens % 128:
        raise ValueError("Paired queries require N/global aligned to 128; KV must be 64/128")
    ratio = key_block_size // 64
    rows = []
    for q in range(0, len(kv_counts), 2):
        union = {}
        for half in range(2):
            for block in kv_indices[q + half, : int(kv_counts[q + half])].tolist():
                block_id, key_half = divmod(block, ratio)
                union[block_id] = union.get(block_id, 0) | (1 << (half + 2 * key_half))
        rows.append(sorted(union.items()))
    counts = torch.tensor([len(x) for x in rows], dtype=torch.int32)
    ids = torch.zeros((len(rows), int(counts.max())), dtype=torch.int32)
    flags = torch.zeros_like(ids)
    for q, row in enumerate(rows):
        for j, (block, flag) in enumerate(row):
            ids[q, j], flags[q, j] = block, flag
    return PairedMetadata(ids, flags, counts, tokens, global_tokens, key_block_size)


def unpair_metadata(meta):
    """Reconstruct original query block lists, independently checking flags."""
    if any(x.device.type != "cpu" for x in (meta.kv_indices, meta.kv_flags, meta.kv_counts)):
        raise ValueError("CPU metadata required for validation")
    ratio = meta.key_block_size // 64
    rows = []
    for row, count in enumerate(meta.kv_counts.tolist()):
        for half in range(2):
            ids = []
            for block, flags in zip(
                meta.kv_indices[row, :count].tolist(), meta.kv_flags[row, :count].tolist()
            ):
                for kh in range(ratio):
                    if flags & (1 << (half + 2 * kh)):
                        ids.append(block * ratio + kh)
            rows.append(sorted(ids))
    return rows
