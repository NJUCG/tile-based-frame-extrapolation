"""Numerical preprocessing operations shared by TBR data generation.

PyTorch and OpenCV are optional at module-import time.  They are loaded only by
the operations that require them, keeping schema and NumPy I/O usable in light
weight environments.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn.functional as functional
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("This preprocessing operation requires PyTorch") from exc
    return torch, functional


def backward_warp(
    image: np.ndarray,
    motion: np.ndarray,
    *,
    mode: str = "nearest",
    padding_mode: str = "border",
) -> np.ndarray:
    """Backward-warp an HWC image using motion measured in pixel units.

    ``motion[y, x] == (dx, dy)`` samples source position ``(x-dx, y-dy)``.
    This convention matches the renderer motion vectors and the TBR shader.
    """

    if image.ndim != 3:
        raise ValueError(f"image must have shape HxWxC, got {image.shape}")
    if motion.ndim != 3 or motion.shape[2] != 2:
        raise ValueError(f"motion must have shape HxWx2, got {motion.shape}")
    if image.shape[:2] != motion.shape[:2]:
        raise ValueError("image and motion must have the same spatial shape")

    torch, functional = _require_torch()
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(image, dtype=np.float32)
    ).permute(2, 0, 1)[None]
    motion_tensor = torch.from_numpy(
        np.ascontiguousarray(motion, dtype=np.float32)
    ).permute(2, 0, 1)[None]

    _, _, height, width = image_tensor.shape
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=motion_tensor.dtype),
        torch.arange(width, dtype=motion_tensor.dtype),
        indexing="ij",
    )
    sample_x = xx - motion_tensor[0, 0]
    sample_y = yy - motion_tensor[0, 1]
    sample_x = 2 * sample_x / max(width - 1, 1) - 1
    sample_y = 2 * sample_y / max(height - 1, 1) - 1
    grid = torch.stack((sample_x, sample_y), dim=-1)[None]

    result = functional.grid_sample(
        image_tensor,
        grid.to(image_tensor.dtype),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )
    return result[0].permute(1, 2, 0).numpy()


def accumulate_backward_motion(
    current_to_previous: np.ndarray,
    previous_to_history: np.ndarray,
) -> np.ndarray:
    """Compose two backward motion-vector fields in renderer pixel units."""

    warped_previous = backward_warp(
        previous_to_history,
        current_to_previous,
        mode="nearest",
        padding_mode="border",
    )
    return warped_previous + current_to_previous


def demodulate_brdf(
    color: np.ndarray,
    base_color: np.ndarray,
    depth: np.ndarray,
    specular: np.ndarray,
    metallic: np.ndarray,
    roughness: np.ndarray,
    nov: np.ndarray,
    lut: np.ndarray,
    *,
    sky_depth: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Split HDR color into irradiance and the BRDF albedo used by TBFE.

    This intentionally retains the sampling convention of the paper code so
    existing processed data can be reproduced.
    """

    spatial = color.shape[:2]
    expected = {
        "color": (color, 3),
        "base_color": (base_color, 3),
        "depth": (depth, 1),
        "specular": (specular, 1),
        "metallic": (metallic, 1),
        "roughness": (roughness, 1),
        "nov": (nov, 1),
        "lut": (lut, 2),
    }
    for name, (array, channels) in expected.items():
        if array.ndim != 3 or array.shape[2] < channels:
            raise ValueError(f"{name} must have shape HxWx{channels} (or more), got {array.shape}")
        if name != "lut" and array.shape[:2] != spatial:
            raise ValueError(f"{name} must have spatial shape {spatial}, got {array.shape[:2]}")

    torch, functional = _require_torch()
    color = np.asarray(color[..., :3], dtype=np.float32)
    base_color = np.asarray(base_color[..., :3], dtype=np.float32)
    depth = np.asarray(depth[..., :1], dtype=np.float32)
    specular = np.asarray(specular[..., :1], dtype=np.float32)
    metallic = np.asarray(metallic[..., :1], dtype=np.float32)
    roughness = np.asarray(roughness[..., :1], dtype=np.float32)
    nov = np.asarray(nov[..., :1], dtype=np.float32)
    lut = np.asarray(lut[..., :2], dtype=np.float32)

    coordinates = np.concatenate((nov, roughness), axis=-1)
    sampled = functional.grid_sample(
        torch.from_numpy(np.ascontiguousarray(lut)).permute(2, 0, 1)[None],
        torch.from_numpy(np.ascontiguousarray(coordinates))[None],
        mode="bilinear",
        align_corners=True,
    )
    sampled_lut = sampled[0].permute(1, 2, 0).numpy()

    metallic = np.clip(metallic, 0.0, 1.0)
    specular_color = 0.08 * specular + (base_color - 0.08 * specular) * metallic
    albedo = (
        base_color * (1 - metallic)
        + sampled_lut[..., :1] * specular_color
        + sampled_lut[..., 1:2]
    )
    irradiance = color / (albedo + 1e-6)

    sky = np.repeat(depth > sky_depth, 3, axis=-1)
    albedo[sky] = 0
    irradiance[sky] = 0
    return irradiance, albedo


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    return vector / (norm + 1e-6)


def compute_gbuffer_mask(
    current: Mapping[str, np.ndarray],
    warped_previous: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Compute the TBR invalid-region mask from warped G-buffer attributes.

    Required current keys depend on the supplied warped attributes:

    - ``stencil`` detects dynamic-object boundaries;
    - ``normal`` plus ``stencil`` detects self-occlusion;
    - ``world_position`` plus ``stencil`` additionally needs current ``nov``,
      ``depth``, and ``sky_mask`` for camera disocclusion.
    """

    masks: list[np.ndarray] = []
    if "stencil" in warped_previous:
        stencil = current["stencil"][..., :1]
        warped_stencil = warped_previous["stencil"][..., :1]
        masks.append(stencil != warped_stencil)

    if "normal" in warped_previous and "stencil" in warped_previous:
        warped_stencil = warped_previous["stencil"][..., :1]
        normal_difference = np.sum(
            _normalized(current["normal"]) * _normalized(warped_previous["normal"]),
            axis=-1,
            keepdims=True,
        ) < 0.98
        masks.append(np.logical_and(warped_stencil, normal_difference))

    if "world_position" in warped_previous and "stencil" in warped_previous:
        depth = current["depth"] * (1 - current["sky_mask"])
        bias = 7.5 * np.abs(current["nov"][..., :1]) + 45 * (
            1 - np.abs(current["nov"][..., :1])
        )
        bias = bias + depth[..., :1] * 50
        distance = np.linalg.norm(
            current["world_position"] - warped_previous["world_position"],
            axis=-1,
            keepdims=True,
        )
        position_difference = distance > bias
        warped_stencil = warped_previous["stencil"][..., :1]
        masks.append(
            np.logical_and(
                np.logical_not(warped_stencil.astype(bool)),
                position_difference,
            )
        )

    if not masks:
        raise ValueError("At least one warped G-buffer attribute is required")
    combined = np.logical_or.reduce(masks)
    if combined.ndim == 2:
        combined = combined[..., None]
    return combined.astype(np.float32)


def dilate_mask(mask: np.ndarray, *, radius: int = 1) -> np.ndarray:
    """Dilate a one-channel mask by ``radius`` pixels using a square kernel."""

    if mask.ndim != 3 or mask.shape[2] != 1:
        raise ValueError(f"mask must have shape HxWx1, got {mask.shape}")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return mask.copy()
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError("Mask dilation requires OpenCV") from exc
    size = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    result = cv2.dilate(mask, kernel, iterations=1)
    if result.ndim == 2:
        result = result[..., None]
    return result.astype(np.float32, copy=False)
