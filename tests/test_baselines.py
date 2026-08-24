"""CPU smoke tests for the paper baselines' public tensor interfaces."""

import inspect

import pytest
import torch
import torch.nn.functional as F

from tbfe.baselines import ExtraNet, ExtraSSNet, ExtraSSTbrNet


def test_extranet_cpu_output_shape_and_explicit_inputs():
    torch.manual_seed(3)
    batch, height, width = 1, 32, 48
    model = ExtraNet().eval()
    warp = torch.rand(batch, 3, height, width)
    occlusion_warp = torch.rand(batch, 3, height, width)
    hole_mask = torch.rand(batch, 1, height, width)
    geometry = torch.rand(batch, 6, height, width)
    history = torch.rand(batch, 3, 3, height, width)
    history_holes = torch.rand(batch, 3, 1, height, width)

    with torch.no_grad():
        prediction = model(
            warp,
            occlusion_warp,
            hole_mask,
            geometry,
            history,
            history_holes,
        )

    assert prediction.shape == (batch, 3, height, width)
    assert torch.isfinite(prediction).all()
    assert "target" not in inspect.signature(model.forward).parameters


def test_extranet_rejects_incomplete_history():
    model = ExtraNet().eval()
    image = torch.zeros(1, 3, 16, 16)
    with pytest.raises(ValueError, match="3 offsets"):
        model(
            image,
            image,
            torch.zeros(1, 1, 16, 16),
            torch.zeros(1, 6, 16, 16),
            torch.zeros(1, 2, 3, 16, 16),
            torch.zeros(1, 3, 1, 16, 16),
        )


def _extrass_inputs(height: int = 32, width: int = 48):
    torch.manual_seed(5)
    color_1 = torch.rand(1, 3, height, width)
    color_3 = torch.rand(1, 3, height, width)
    fallback = 0.1 + torch.rand(1, 3, height, width)
    albedo = torch.rand(1, 3, height, width)
    normal = F.normalize(torch.rand(1, 3, height, width) + 0.1, dim=1)
    return color_1, color_3, fallback, albedo, normal


@pytest.mark.parametrize("model_type", [ExtraSSNet, ExtraSSTbrNet])
def test_extrass_variants_have_ground_truth_free_cpu_inference(model_type):
    model = model_type().eval()
    inputs = _extrass_inputs()

    with torch.no_grad():
        result = model(*inputs)

    assert result.pred_color.shape == inputs[0].shape
    assert result.pred_mask.shape == inputs[0][:, :1].shape
    assert len(result.pred_features) == 3
    assert result.target_features is None
    assert torch.isfinite(result.pred_color).all()
    assert set(torch.unique(result.pred_mask).tolist()).issubset({0.0, 1.0})


def test_extrass_optional_target_only_adds_training_features():
    model = ExtraSSNet().eval()
    inputs = _extrass_inputs()
    target = torch.rand_like(inputs[0])

    with torch.no_grad():
        inference_result = model(*inputs)
        training_result = model(*inputs, target=target)

    torch.testing.assert_close(
        training_result.pred_color, inference_result.pred_color, rtol=0, atol=0
    )
    torch.testing.assert_close(
        training_result.pred_mask, inference_result.pred_mask, rtol=0, atol=0
    )
    assert training_result.target_features is not None
    assert len(training_result.target_features) == 3


def test_extrass_and_tbr_ablation_share_checkpoint_schema():
    baseline_keys = tuple(ExtraSSNet().state_dict())
    ablation_keys = tuple(ExtraSSTbrNet().state_dict())

    assert baseline_keys == ablation_keys
    assert "encoder.pyramid1.0.0.weight" in baseline_keys
    assert "decoder1.convblock.2.weight" in baseline_keys
