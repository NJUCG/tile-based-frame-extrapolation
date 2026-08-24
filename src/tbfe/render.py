"""Shared model-input packing and output composition."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from .data import INFERENCE_BUFFERS, inverse_tone_map, tone_map
from .metrics import gamma_encode


FEATURE_BUFFERS: tuple[str, ...] = ("Depth", "Normal", "Metallic", "Roughness", "Albedo")


def validate_inference_buffers(buffers: Mapping[str, torch.Tensor]) -> None:
    missing = [name for name in INFERENCE_BUFFERS if name not in buffers]
    if missing:
        raise KeyError(f"missing inference buffers: {', '.join(missing)}")
    shapes = {name: tuple(buffers[name].shape[-2:]) for name in INFERENCE_BUFFERS}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"inference buffer resolutions differ: {shapes}")


def pack_inputs(buffers: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return log irradiance, hole mask, and the 9-channel G-buffer tensor."""

    validate_inference_buffers(buffers)
    irradiance = buffers["TbrIrradiance_1"]
    hole_mask = buffers["WarpToCurrGbufferMask_1"].clamp(0.0, 1.0)
    features = torch.cat([buffers[name] for name in FEATURE_BUFFERS], dim=1)
    return irradiance, hole_mask, features


def log_irradiance_to_linear_color(log_irradiance: torch.Tensor, albedo: torch.Tensor) -> torch.Tensor:
    """Re-modulate repaired irradiance with the current-frame albedo."""

    return inverse_tone_map(log_irradiance) * albedo.clamp_min(0.0)


def linear_color_to_metric_space(linear_color: torch.Tensor) -> torch.Tensor:
    """Apply the historical log and gamma transforms used by paper metrics."""

    return gamma_encode(tone_map(linear_color))
