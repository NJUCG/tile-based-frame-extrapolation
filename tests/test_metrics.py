from __future__ import annotations

import torch

from tbfe.metrics import psnr, ssim


def test_masked_metrics_ignore_excluded_error() -> None:
    target = torch.zeros(1, 3, 16, 16)
    prediction = target.clone()
    prediction[..., 0, 0] = 1.0
    valid = torch.ones(1, 1, 16, 16)
    valid[..., 0, 0] = 0.0
    assert psnr(prediction, target, valid) > 60
    assert torch.allclose(ssim(prediction, target, valid), torch.tensor(1.0), atol=1e-6)


def test_unmasked_psnr_detects_error() -> None:
    target = torch.zeros(1, 3, 8, 8)
    prediction = torch.ones_like(target)
    assert torch.allclose(psnr(prediction, target), torch.tensor(0.0))
