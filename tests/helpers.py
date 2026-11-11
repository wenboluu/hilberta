import torch


def raster_image_masks(plan):
    t = plan.text_tokens
    inv = plan.inverse_image_order
    return torch.stack([plan.dense_mask(i)[t:, t:][inv][:, inv] for i in range(4)])


def inverse_permutation(order):
    order = torch.as_tensor(order, dtype=torch.long)
    n = order.numel()
    if order.ndim != 1 or not torch.equal(
        order.sort().values, torch.arange(n, device=order.device)
    ):
        raise ValueError("order must contain every token exactly once")
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(n, device=order.device)
    return inverse


def tile_labels(order, tile_count, block_index):
    position = inverse_permutation(order)
    n = position.numel()
    if tile_count <= 0 or n % tile_count or (n // tile_count) % 4:
        raise ValueError("tile size must be integral and divisible by four")
    width = n // tile_count
    offset = (int(block_index) % 4) * (width // 4)
    return ((position - offset) % n) // width
