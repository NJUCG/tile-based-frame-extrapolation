"""Evaluation utilities with an explicit, shared valid-region mask."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def gamma_encode(image: torch.Tensor) -> torch.Tensor:
    return image.clamp_min(0.0).pow(1.0 / 2.2).clamp(0.0, 1.0)


def _mask_like(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(reference[:, :1])
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.shape[1] != 1:
        raise ValueError(f"valid mask must have one channel, got {mask.shape}")
    if mask.shape[-2:] != reference.shape[-2:]:
        mask = F.interpolate(mask.float(), size=reference.shape[-2:], mode="nearest")
    return mask.to(device=reference.device, dtype=reference.dtype).clamp(0.0, 1.0)


def psnr(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """PSNR over the same explicit valid region for every compared method."""

    mask = _mask_like(valid_mask, prediction)
    denominator = (mask.sum() * prediction.shape[1]).clamp_min(1.0)
    mse = (((prediction - target) ** 2) * mask).sum() / denominator
    return -10.0 * torch.log10(mse.clamp_min(torch.finfo(mse.dtype).eps))


def _gaussian_window(channels: int, device: torch.device, dtype: torch.dtype, size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - size // 2
    kernel_1d = torch.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] @ kernel_1d[None, :]
    return kernel_2d.expand(channels, 1, size, size).contiguous()


def ssim(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean SSIM, optionally restricted to a one-channel valid mask."""

    mask = _mask_like(valid_mask, prediction)
    # Neutralize excluded pixels inside the metric only so their convolution
    # neighborhoods do not leak into valid-region SSIM values.
    prediction = prediction * mask + target * (1.0 - mask)
    channels = prediction.shape[1]
    window = _gaussian_window(channels, prediction.device, prediction.dtype)
    padding = window.shape[-1] // 2
    mu_x = F.conv2d(prediction, window, padding=padding, groups=channels)
    mu_y = F.conv2d(target, window, padding=padding, groups=channels)
    var_x = F.conv2d(prediction * prediction, window, padding=padding, groups=channels) - mu_x.square()
    var_y = F.conv2d(target * target, window, padding=padding, groups=channels) - mu_y.square()
    covariance = F.conv2d(prediction * target, window, padding=padding, groups=channels) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    score = ((2.0 * mu_x * mu_y + c1) * (2.0 * covariance + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (var_x + var_y + c2)
    )
    denominator = (mask.sum() * channels).clamp_min(1.0)
    return (score * mask).sum() / denominator


def lpips_score(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    network: str = "vgg",
) -> torch.Tensor:
    """LPIPS with invalid pixels neutralized only inside the metric calculation."""

    try:
        import lpips
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("LPIPS is optional; install tbfe[metrics]") from exc
    mask = _mask_like(valid_mask, prediction)
    metric_prediction = prediction * mask + target * (1.0 - mask)
    model = lpips.LPIPS(net=network).to(prediction.device).eval()
    with torch.no_grad():
        return model(metric_prediction * 2.0 - 1.0, target * 2.0 - 1.0).mean()


def summarize(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    include_lpips: bool = False,
) -> dict[str, float]:
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    result = {
        "psnr": float(psnr(prediction, target, valid_mask)),
        "ssim": float(ssim(prediction, target, valid_mask)),
    }
    if include_lpips:
        result["lpips"] = float(lpips_score(prediction, target, valid_mask))
    if any(not math.isfinite(value) for value in result.values()):
        raise FloatingPointError(f"non-finite metric encountered: {result}")
    return result
