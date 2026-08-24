"""Public, ground-truth-free entry point for TBR's G-buffer-guided warp."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TBRWarpInputs:
    """Inputs to the G-buffer-guided TBR compute shader.

    All arrays are channel-last and share one spatial shape. Motion is a
    backward vector in pixel units: ``(dx, dy)`` samples ``(x-dx, y-dy)``.
    ``invalid_mask`` is one for a disoccluded/out-of-date pixel and zero for a
    directly reusable pixel.
    """

    previous_irradiance: np.ndarray
    motion: np.ndarray
    previous_base_color: np.ndarray
    current_base_color: np.ndarray
    previous_normal: np.ndarray
    current_normal: np.ndarray
    invalid_mask: np.ndarray

    def validated(self) -> "TBRWarpInputs":
        expected_channels = {
            "previous_irradiance": 3,
            "motion": 2,
            "previous_base_color": 3,
            "current_base_color": 3,
            "previous_normal": 3,
            "current_normal": 3,
            "invalid_mask": 1,
        }
        spatial: tuple[int, int] | None = None
        normalized: dict[str, np.ndarray] = {}
        for field in fields(self):
            name = field.name
            array = getattr(self, name)
            if not isinstance(array, np.ndarray):
                raise TypeError(f"{name} must be a numpy.ndarray")
            channels = expected_channels[name]
            if array.ndim != 3 or array.shape[2] != channels:
                raise ValueError(f"{name} must have shape HxWx{channels}, got {array.shape}")
            if not np.issubdtype(array.dtype, np.number):
                raise TypeError(f"{name} must have a numeric dtype, got {array.dtype}")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"{name} contains NaN or infinity")
            current_spatial = array.shape[:2]
            if spatial is None:
                spatial = current_spatial
            elif current_spatial != spatial:
                raise ValueError(
                    f"All TBR inputs must share one spatial shape; {name} has "
                    f"{current_spatial}, expected {spatial}"
                )
            normalized[name] = np.ascontiguousarray(array, dtype=np.float32)

        if spatial is None or spatial[0] == 0 or spatial[1] == 0:
            raise ValueError("TBR inputs must have non-empty spatial dimensions")
        return TBRWarpInputs(**normalized)


class TBRWarper:
    """Reusable OpenGL TBR warper.

    By default a headless EGL context is created on entry. Set
    ``create_context=False`` when the calling renderer already owns a current
    OpenGL 4.5 context.

    Example::

        with TBRWarper() as warper:
            irradiance = warper.warp(inputs)
    """

    def __init__(self, *, create_context: bool = True) -> None:
        self.create_context = create_context
        self._backend: Any | None = None

    def __enter__(self) -> "TBRWarper":
        if self._backend is not None:
            raise RuntimeError("TBRWarper is already open")
        # This import is deliberately delayed: importing tbfe.preprocess does
        # not import PyOpenGL or initialize EGL.
        from ._opengl import OpenGLTBRBackend

        backend = OpenGLTBRBackend(create_context=self.create_context)
        backend.open()
        self._backend = backend
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._backend is not None:
            self._backend.close()
            self._backend = None

    def warp(self, inputs: TBRWarpInputs) -> np.ndarray:
        """Return warped irradiance as a contiguous ``float32`` HWC3 array."""

        if self._backend is None:
            raise RuntimeError("Use TBRWarper as a context manager before calling warp")
        return self._backend.warp(inputs.validated())


def warp_tbr(
    inputs: TBRWarpInputs,
    *,
    create_context: bool = True,
) -> np.ndarray:
    """One-shot convenience wrapper around :class:`TBRWarper`.

    For sequences, reuse one ``TBRWarper`` instead of creating an EGL context
    for every frame.
    """

    with TBRWarper(create_context=create_context) as warper:
        return warper.warp(inputs)
