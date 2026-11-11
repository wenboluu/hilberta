import json
import torch
from hilberta.runtime import prompts, atomic_save
from hilberta.train import Records, epoch_order
from hilberta.evaluate import ssim


def test_records_and_epoch_coverage(tmp_path):
    shard = tmp_path / "shard-000"
    shard.mkdir()
    protocol = {
        "resolution": 1024,
        "model": "test",
        "revision": "fixed",
        "steps": 28,
        "guidance": 7.5,
    }
    (shard / "protocol.json").write_text(json.dumps(protocol))
    atomic_save({"prompt_embeds": torch.zeros(1, 2, 3)}, shard / "text.pt")
    records = [
        {"step": i, "x_t": torch.full((1, 2, 3), float(i)), "v": torch.zeros(1, 2, 3)}
        for i in range(4)
    ]
    atomic_save({"text_file": "text.pt", "records": records}, shard / "trajectory.pt")
    (shard / "index.json").write_text(
        json.dumps([{"file": "trajectory.pt", "step": i} for i in range(4)])
    )
    data = Records(tmp_path)
    assert len(data) == 4
    assert data.get(3)[0]["step"] == 3
    assert data.get(0)[0]["x_t"].sum() == 0
    order = epoch_order(32, 2707, 0)
    assert sorted(order) == list(range(32))
    assert order == epoch_order(32, 2707, 0)
    assert order != epoch_order(32, 2707, 1)


def test_manifest_rejects_duplicate_and_unsafe_ids(tmp_path):
    import pytest

    path = tmp_path / "manifest.jsonl"
    for ids in [["a", "a"], ["../x"]]:
        path.write_text("\n".join(json.dumps({"id": i, "prompt": "A scene"}) for i in ids))
        with pytest.raises(ValueError):
            prompts(path)


def test_ssim_identical():
    torch.manual_seed(1)
    x = torch.rand(1, 3, 24, 24)
    assert torch.allclose(ssim(x, x), torch.tensor(1.0))
