from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tbfe.data import (
    INFERENCE_BUFFERS,
    BufferSequenceDataset,
    discover_frames,
    load_frame,
    load_npz_frame,
    preprocess_buffer,
)


CHANNELS = {
    "TbrIrradiance_1": 3,
    "WarpToCurrGbufferMask_1": 1,
    "Depth": 1,
    "Normal": 3,
    "Metallic": 1,
    "Roughness": 1,
    "Albedo": 3,
    "Irradiance": 3,
    "Color": 3,
}


def write_frame(directory: Path, frame: str, names: tuple[str, ...]) -> None:
    for name in names:
        value = np.ones((7, 9, CHANNELS[name]), dtype=np.float32)
        np.save(directory / f"{name}.{frame}.npy", value, allow_pickle=False)


def test_preprocess_buffer_uses_log_domain_and_clamps_far_depth() -> None:
    irradiance = preprocess_buffer(
        "TbrIrradiance_1", np.asarray([[[-2.0, 0.0, 3.0]]], dtype=np.float32)
    )
    expected = torch.tensor([[[0.0]], [[0.0]], [[float(np.log(4.0))]]])
    assert torch.allclose(irradiance, expected)

    depth = preprocess_buffer("Depth", np.asarray([[[49.0], [51.0]]], dtype=np.float32))
    assert depth.tolist() == [[[49.0, 0.0]]]


def test_sequence_loader_and_six_digit_preference(tmp_path: Path) -> None:
    names = INFERENCE_BUFFERS + ("Irradiance", "Color")
    write_frame(tmp_path, "0007", names)
    write_frame(tmp_path, "000007", names)
    assert discover_frames(tmp_path) == ["000007"]
    frame = load_frame(tmp_path, "000007")
    assert frame["Normal"].shape == (3, 7, 9)
    dataset = BufferSequenceDataset([tmp_path])
    assert len(dataset) == 1
    assert dataset[0]["Color"].shape == (3, 7, 9)


def test_npz_loader_rejects_missing_buffer(tmp_path: Path) -> None:
    path = tmp_path / "frame.npz"
    np.savez_compressed(path, TbrIrradiance_1=np.zeros((8, 8, 3), np.float32))
    with pytest.raises(KeyError, match="missing buffers"):
        load_npz_frame(path)
