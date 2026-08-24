from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from tbfe.checkpoint import load_model_weights


class Network(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(2, 2)


class Wrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = Network()


def test_loads_lightning_model_prefix_into_public_wrapper(tmp_path: Path) -> None:
    source = Network()
    path = tmp_path / "legacy.ckpt"
    torch.save({"state_dict": {f"model.{key}": value for key, value in source.state_dict().items()}}, path)
    target = Wrapper()
    missing, unexpected = load_model_weights(target, path)
    assert missing == []
    assert unexpected == []
    for expected, actual in zip(source.parameters(), target.network.parameters()):
        assert torch.equal(expected, actual)
