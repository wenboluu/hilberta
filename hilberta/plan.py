"""Static, exact block plans for original composite Hilbert + shared + bridge.

Physical stream is [text, shared image blocks, remaining image blocks]. Tile
membership and bridge tie breaking always use the ORIGINAL phase ordering.
Building plans is CPU-only setup work; attention never re-tiles packed tokens.
"""

from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import torch
from .geometry import hilbert_order
from .geometry import neighbor_pairs


@dataclass(frozen=True)
class PhasePlan:
    offset: int
    kv_indices: torch.Tensor
    kv_counts: torch.Tensor
    bridge_original_blocks: torch.Tensor

    def to(self, device):
        return replace(
            self,
            kv_indices=self.kv_indices.to(device),
            kv_counts=self.kv_counts.to(device),
            bridge_original_blocks=self.bridge_original_blocks.to(device),
        )


@dataclass(frozen=True)
class AttentionPlan:
    grid_size: int
    tile_count: int
    text_tokens: int
    shared_tokens: int
    block_size: int
    image_order: torch.Tensor
    inverse_image_order: torch.Tensor
    original_order: torch.Tensor
    physical_original_blocks: torch.Tensor
    phases: tuple[PhasePlan, ...]
    bridge: bool
    full: bool = False

    @property
    def image_tokens(self):
        return self.grid_size**2

    @property
    def global_tokens(self):
        return self.text_tokens + self.shared_tokens

    @property
    def total_tokens(self):
        return self.text_tokens + self.image_tokens

    def to(self, device):
        return replace(
            self,
            image_order=self.image_order.to(device),
            inverse_image_order=self.inverse_image_order.to(device),
            original_order=self.original_order.to(device),
            physical_original_blocks=self.physical_original_blocks.to(device),
            phases=tuple(p.to(device) for p in self.phases),
        )

    def dense_mask(self, phase, device=None):
        """Explicit physical-identity mask for correctness only."""
        p = self.phases[phase % 4]
        device = p.kv_indices.device if device is None else device
        total = self.total_tokens // self.block_size
        glob = self.global_tokens // self.block_size
        blocks = torch.zeros(total, total, dtype=torch.bool)
        blocks[:glob] = True
        for q, (row, count) in enumerate(zip(p.kv_indices.cpu(), p.kv_counts.cpu().tolist())):
            blocks[glob + q, row[:count].long()] = True
        return (
            blocks.repeat_interleave(self.block_size, 0)
            .repeat_interleave(self.block_size, 1)
            .to(device)
        )

    def manifest(self):
        glob = self.global_tokens // self.block_size
        return dict(
            grid_size=self.grid_size,
            tile_count=self.tile_count,
            text_tokens=self.text_tokens,
            shared_tokens=self.shared_tokens,
            block_size=self.block_size,
            bridge=self.bridge,
            full=self.full,
            image_order_sha256=hashlib.sha256(self.image_order.cpu().numpy().tobytes()).hexdigest(),
            physical_original_blocks=self.physical_original_blocks.cpu().tolist(),
            phases=[
                dict(
                    offset=p.offset,
                    kv_count_min=int(p.kv_counts.min()),
                    kv_count_max=int(p.kv_counts.max()),
                    directed_entries=(
                        glob * (self.total_tokens // self.block_size) + int(p.kv_counts.sum())
                    )
                    * self.block_size**2,
                    bridge_original_blocks=p.bridge_original_blocks.cpu().tolist(),
                )
                for p in self.phases
            ],
        )


def build_plan(
    tile_count=4, grid_size=64, text_tokens=512, shared=True, bridge=True, block_size=64
):
    for name, value in [
        ("tile_count", tile_count),
        ("grid_size", grid_size),
        ("text_tokens", text_tokens),
        ("block_size", block_size),
    ]:
        if isinstance(value, bool) or int(value) != value or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    tile_count, grid_size, text_tokens, block_size = map(
        int, (tile_count, grid_size, text_tokens, block_size)
    )
    n = grid_size**2
    if grid_size < 8 or grid_size & (grid_size - 1):
        raise ValueError("grid_size must be a power of two >= 8")
    if n % tile_count or (n // tile_count) % (4 * block_size) or text_tokens % block_size:
        raise ValueError("Tile quarter offsets and text prefix must be block aligned")
    h = hilbert_order(grid_size)
    original = torch.cat((h[n // 4 : 3 * n // 4], n - 1 - h[n // 4 : 3 * n // 4]))
    inverse = torch.argsort(original)
    blocks = original.reshape(-1, block_size)
    nb = len(blocks)
    shared_ids = torch.zeros(n, dtype=torch.bool)
    if shared:
        start, end = 3 * grid_size // 8, 5 * grid_size // 8
        shared_ids.reshape(grid_size, grid_size)[start:end, start:end] = True
    share_counts = shared_ids[blocks].sum(1)
    if ((share_counts != 0) & (share_counts != block_size)).any():
        raise ValueError("Shared square must be a union of complete original blocks")
    share_blocks = (share_counts == block_size).nonzero().flatten()
    ordinary = (share_counts == 0).nonzero().flatten()
    packed_blocks = torch.cat((share_blocks, ordinary))
    order = blocks[packed_blocks].flatten()
    original_to_physical = torch.argsort(packed_blocks) + text_tokens // block_size
    global_blocks = text_tokens // block_size + len(share_blocks)
    pairs = neighbor_pairs(grid_size, 4)
    phases = []
    width = n // tile_count
    for offset in (0, width // 4, width // 2, 3 * width // 4):
        labels = ((inverse - offset) % n) // width
        block_labels = labels[blocks[:, 0]]
        if not torch.equal(labels[blocks], block_labels[:, None].expand_as(blocks)):
            raise AssertionError("A physical block crosses a tile boundary")
        chosen = torch.full((nb,), -1, dtype=torch.long)
        if bridge:
            missing = pairs[labels[pairs[:, 0]] != labels[pairs[:, 1]]]
            original_ids = inverse // block_size
            scores = Counter()
            for a, b in original_ids[missing].tolist():
                scores[a, b] += 1
                scores[b, a] += 1
            for q in range(nb):
                options = [(k, score) for (a, k), score in scores.items() if a == q]
                if options:
                    chosen[q] = min(
                        options, key=lambda item: (-item[1], (item[0] - offset // block_size) % nb)
                    )[0]
        rows = []
        for q in ordinary.tolist():
            same_tile = ordinary[block_labels[ordinary] == block_labels[q]]
            keys = set(range(global_blocks)) | set(original_to_physical[same_tile].tolist())
            if chosen[q] >= 0:
                keys.add(int(original_to_physical[chosen[q]]))
            rows.append(sorted(keys))
        counts = torch.tensor([len(row) for row in rows], dtype=torch.int32)
        indices = torch.zeros(len(rows), int(counts.max()), dtype=torch.int32)
        for q, row in enumerate(rows):
            indices[q, : len(row)] = torch.tensor(row, dtype=torch.int32)
        phases.append(PhasePlan(offset, indices, counts, chosen))
    return AttentionPlan(
        grid_size,
        tile_count,
        text_tokens,
        len(share_blocks) * block_size,
        block_size,
        order,
        torch.argsort(order),
        original,
        packed_blocks,
        tuple(phases),
        bool(bridge),
    )


def full_attention_plan(plan):
    """Same persistent layout, all connections: permutation-equivalence gate."""
    nb = plan.total_tokens // plan.block_size
    nq = (plan.total_tokens - plan.global_tokens) // plan.block_size
    ids = torch.arange(nb, dtype=torch.int32).repeat(nq, 1)
    counts = torch.full((nq,), nb, dtype=torch.int32)
    return replace(
        plan,
        phases=tuple(
            replace(p, kv_indices=ids.clone(), kv_counts=counts.clone()) for p in plan.phases
        ),
        full=True,
    )
