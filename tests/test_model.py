import torch

from tbfe.losses import ImageInpaintLoss
from tbfe.models import ImageInpaintNet, TiledImageInpainter
from tbfe.ops import extract_tiles


def _inputs(batch=2, height=13, width=17):
    torch.manual_seed(7)
    color = torch.randn(batch, 3, height, width)
    mask = torch.zeros(batch, 1, height, width)
    features = torch.randn(batch, 9, height, width)
    return color, mask, features


def test_network_accepts_batches_and_packed_deployment_input():
    network = ImageInpaintNet(base=2, feat_base=2).eval()
    color, mask, features = _inputs()
    packed = torch.cat((color, mask, features), dim=1)

    with torch.no_grad():
        residual = network(color, mask, features)
        packed_residual = network.forward_packed(packed)

    assert residual.shape == color.shape
    assert torch.isfinite(residual).all()
    torch.testing.assert_close(packed_residual, residual)


def test_tiled_inference_uses_no_ground_truth_and_handles_non_multiple_sizes():
    network = ImageInpaintNet(base=2, feat_base=2).eval()
    inpainter = TiledImageInpainter(
        network, tile_size=8, tile_expand=2, min_ratio=0.01
    )
    color, mask, features = _inputs(batch=2)
    mask[1, :, 8:, 8:] = 1

    with torch.no_grad():
        result = inpainter(color, mask, features)

    assert result.has_tiles
    assert result.tile_count >= 1
    assert result.image.shape == color.shape
    assert result.residual.shape == color.shape
    assert result.mask.shape == mask.shape
    assert result.tile_residual.shape[-2:] == (12, 12)
    assert torch.count_nonzero(result.residual[0]) == 0
    torch.testing.assert_close(result.image[0], color[0])
    torch.testing.assert_close(result.image, color + result.residual * result.mask)


def test_tiled_inference_no_selected_tiles_is_identity():
    network = ImageInpaintNet(base=2, feat_base=2).eval()
    inpainter = TiledImageInpainter(network, tile_size=8, tile_expand=2)
    color, mask, features = _inputs(batch=3, height=9, width=11)

    with torch.no_grad():
        result = inpainter(color, mask, features)

    assert not result.has_tiles
    assert result.tile_count == 0
    assert result.tile_residual.shape == (0, 3, 12, 12)
    assert torch.count_nonzero(result.residual) == 0
    assert torch.count_nonzero(result.mask) == 0
    torch.testing.assert_close(result.image, color)


def test_training_target_is_extracted_separately_from_inference_inputs():
    network = ImageInpaintNet(base=2, feat_base=2).eval()
    inpainter = TiledImageInpainter(network, tile_size=8, tile_expand=2)
    color, mask, features = _inputs(batch=1)
    mask[..., :4, :4] = 1
    target = torch.randn_like(color)

    result = inpainter(color, mask, features)
    target_tiles = extract_tiles(target, result.layout)
    losses = ImageInpaintLoss()(
        result.tile_residual,
        result.tile_color,
        target_tiles,
        result.tile_mask,
    )

    assert target_tiles.shape == result.tile_residual.shape
    assert losses["tile_cnt"] == result.tile_count
    assert losses["total_loss"].ndim == 0


def test_loss_matches_original_active_terms():
    residual = torch.ones(1, 3, 4, 4)
    input_color = torch.zeros_like(residual)
    target = torch.full_like(residual, 2)
    mask = torch.zeros(1, 1, 4, 4)
    mask[..., :2, :] = 1

    losses = ImageInpaintLoss()(residual, input_color, target, mask)

    torch.testing.assert_close(losses["loss_pred_hole"], torch.tensor(1.0))
    torch.testing.assert_close(losses["loss_known_residual"], torch.tensor(1.0))
    torch.testing.assert_close(losses["total_loss"], torch.tensor(1.1))


def test_loss_accepts_an_empty_tile_batch():
    empty_color = torch.empty(0, 3, 12, 12)
    empty_mask = torch.empty(0, 1, 12, 12)

    losses = ImageInpaintLoss()(
        empty_color,
        empty_color,
        empty_color,
        empty_mask,
    )

    assert losses["tile_cnt"] == 0
    assert losses["total_loss"].item() == 0
    losses["total_loss"].backward()
