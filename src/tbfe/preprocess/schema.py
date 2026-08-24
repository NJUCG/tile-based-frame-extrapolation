"""Canonical on-disk buffer schema used by the public TBFE pipeline.

All image-like arrays are stored channel-last (``H x W x C``) as NumPy
``float32`` arrays.  Keeping this contract in one module prevents the training,
inference, and preprocessing entry points from silently disagreeing about
buffer names or channel counts.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class BufferSpec:
    """Description of one processed frame buffer."""

    channels: int
    role: str
    description: str


_BUFFER_SPECS = {
    "TbrIrradiance_1": BufferSpec(
        3,
        "inference",
        "Previous-frame irradiance warped to the extrapolated frame by TBR.",
    ),
    "WarpToCurrGbufferMask_1": BufferSpec(
        1,
        "inference",
        "Disocclusion/invalid-region mask; one means the pixel needs repair.",
    ),
    "Depth": BufferSpec(1, "inference", "Target-frame linear scene depth."),
    "Normal": BufferSpec(3, "inference", "Target-frame world-space normal."),
    "Metallic": BufferSpec(1, "inference", "Target-frame metallic value."),
    "Roughness": BufferSpec(1, "inference", "Target-frame roughness value."),
    "Albedo": BufferSpec(3, "inference", "Target-frame BRDF albedo."),
    "SkyMask": BufferSpec(
        1,
        "auxiliary",
        "Optional renderer sky mask; not consumed by public model inference.",
    ),
    "Irradiance": BufferSpec(3, "training-target", "Ground-truth irradiance."),
    "Color": BufferSpec(3, "training-target", "Ground-truth HDR color."),
    "ShadowMask": BufferSpec(
        1,
        "evaluation-only",
        "Oracle evaluation mask; never a public inference input.",
    ),
}

BUFFER_SPECS: Mapping[str, BufferSpec] = MappingProxyType(_BUFFER_SPECS)

# These buffers are sufficient for model inference.  In particular, neither
# Color nor Irradiance (both ground truth) belongs in this tuple.
INFERENCE_BUFFER_NAMES: tuple[str, ...] = (
    "TbrIrradiance_1",
    "WarpToCurrGbufferMask_1",
    "Depth",
    "Normal",
    "Metallic",
    "Roughness",
    "Albedo",
)

TRAINING_TARGET_NAMES: tuple[str, ...] = ("Irradiance", "Color")
EVALUATION_ONLY_NAMES: tuple[str, ...] = ("ShadowMask",)
AUXILIARY_BUFFER_NAMES: tuple[str, ...] = ("SkyMask",)


def validate_buffer_array(name: str, array: np.ndarray) -> np.ndarray:
    """Validate one array against :data:`BUFFER_SPECS`.

    The input is returned unchanged so callers may retain memory mapping.  This
    function intentionally does not cast or normalize scientific data.
    """

    try:
        spec = BUFFER_SPECS[name]
    except KeyError as exc:
        known = ", ".join(sorted(BUFFER_SPECS))
        raise KeyError(f"Unknown buffer {name!r}; known buffers: {known}") from exc

    if not isinstance(array, np.ndarray):
        raise TypeError(f"{name} must be a numpy.ndarray, got {type(array).__name__}")
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape HxWxC, got {array.shape}")
    if array.shape[2] != spec.channels:
        raise ValueError(
            f"{name} must have {spec.channels} channel(s), got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must have a numeric dtype, got {array.dtype}")
    return array


def validate_buffer_set(
    buffers: Mapping[str, np.ndarray],
    required: Sequence[str] = INFERENCE_BUFFER_NAMES,
) -> tuple[int, int]:
    """Validate required buffers and return their common ``(height, width)``."""

    missing = [name for name in required if name not in buffers]
    if missing:
        raise KeyError(f"Missing required buffers: {', '.join(missing)}")

    spatial_shape: tuple[int, int] | None = None
    for name in required:
        array = validate_buffer_array(name, buffers[name])
        current = (array.shape[0], array.shape[1])
        if spatial_shape is None:
            spatial_shape = current
        elif current != spatial_shape:
            raise ValueError(
                f"All buffers must share one spatial shape; {name} has {current}, "
                f"expected {spatial_shape}"
            )

    if spatial_shape is None:
        raise ValueError("At least one required buffer is needed")
    return spatial_shape
