from __future__ import annotations

from pathlib import Path

import torch

from tbfe.data import add_batch_dimension, load_npz_frame
from tbfe.models import ImageInpaintNet, TiledImageInpainter
from tbfe.render import log_irradiance_to_linear_color, pack_inputs


def test_factory_fixture_runs_without_ground_truth() -> None:
    path = Path(__file__).parents[1] / "examples" / "factory_smoke" / "frame.npz"
    buffers = add_batch_dimension(load_npz_frame(path), device="cpu")
    assert "Color" not in buffers
    assert "Irradiance" not in buffers
    assert "SkyMask" not in buffers
    assert "ShadowMask" not in buffers

    torch.manual_seed(3407)
    model = TiledImageInpainter(ImageInpaintNet()).eval()
    with torch.inference_mode():
        result = model(*pack_inputs(buffers))
        color = log_irradiance_to_linear_color(result.image, buffers["Albedo"])
    assert result.tile_count == 4
    assert color.shape == (1, 3, 128, 128)
    assert torch.isfinite(color).all()
