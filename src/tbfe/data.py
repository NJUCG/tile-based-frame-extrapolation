"""Dataset and buffer I/O for Tile-based Frame Extrapolation.

The public format stores one NumPy array per buffer and frame::

    <sequence>/<BufferName>.<frame-id>.npy

Arrays are expected in HWC layout. HDR color and irradiance buffers are
converted to the log domain with ``log1p(max(x, 0))`` before they are returned
to the model. G-buffer values remain linear.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


INFERENCE_BUFFERS: tuple[str, ...] = (
    "TbrIrradiance_1",
    "WarpToCurrGbufferMask_1",
    "Depth",
    "Normal",
    "Metallic",
    "Roughness",
    "Albedo",
)

TARGET_BUFFERS: tuple[str, ...] = ("Irradiance", "Color")

_FRAME_RE = re.compile(r"^(?P<buffer>.+)\.(?P<frame>\d+)\.npy$")


def tone_map(value: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Apply the log-domain transform used to train the released model."""

    if isinstance(value, np.ndarray):
        return np.log1p(np.clip(value, 0.0, None))
    return torch.log1p(value.clamp_min(0.0))


def inverse_tone_map(value: torch.Tensor) -> torch.Tensor:
    """Invert :func:`tone_map` for a tensor."""

    return torch.expm1(value).clamp_min(0.0)


def _as_hwc(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape HxW or HxWxC, got {array.shape}")
    return array


def preprocess_buffer(name: str, array: np.ndarray) -> torch.Tensor:
    """Convert one raw HWC buffer into a contiguous float32 CHW tensor."""

    value = _as_hwc(np.asarray(array), name).astype(np.float32, copy=True)
    if "Depth" in name:
        value[value > 50.0] = 0.0
    if "Color" in name or "Irradiance" in name:
        value = tone_map(value)
    return torch.from_numpy(np.ascontiguousarray(value.transpose(2, 0, 1)))


def _validate_spatial_shapes(buffers: Mapping[str, torch.Tensor]) -> None:
    shapes = {name: tuple(value.shape[-2:]) for name, value in buffers.items()}
    if len(set(shapes.values())) > 1:
        raise ValueError(f"all buffers must share a spatial resolution, got {shapes}")


def discover_frames(sequence: str | Path, anchor: str = "TbrIrradiance_1") -> list[str]:
    """Return sorted frame identifiers available for ``anchor``."""

    sequence = Path(sequence)
    if not sequence.is_dir():
        raise FileNotFoundError(f"sequence directory does not exist: {sequence}")
    frames: dict[int, str] = {}
    for path in sequence.glob(f"{anchor}.*.npy"):
        match = _FRAME_RE.match(path.name)
        if match and match.group("buffer") == anchor:
            rendered = match.group("frame")
            number = int(rendered)
            # Some internal datasets contain both four- and six-digit aliases.
            # Treat them as one frame and prefer the six-digit representation.
            if number not in frames or len(rendered) > len(frames[number]):
                frames[number] = rendered
    if not frames:
        raise FileNotFoundError(f"no {anchor}.<frame>.npy files found in {sequence}")
    return [frames[number] for number in sorted(frames)]


def load_frame(
    sequence: str | Path,
    frame: str | int,
    buffers: Sequence[str] = INFERENCE_BUFFERS,
) -> dict[str, torch.Tensor]:
    """Load one frame from a directory of per-buffer ``.npy`` files."""

    sequence = Path(sequence)
    if isinstance(frame, int):
        candidates = [f"{frame:06d}", f"{frame:04d}", str(frame)]
    else:
        candidates = [frame]

    result: dict[str, torch.Tensor] = {}
    for name in buffers:
        path = next((sequence / f"{name}.{item}.npy" for item in candidates if (sequence / f"{name}.{item}.npy").is_file()), None)
        if path is None:
            rendered = ", ".join(f"{name}.{item}.npy" for item in candidates)
            raise FileNotFoundError(f"missing buffer in {sequence}; tried {rendered}")
        result[name] = preprocess_buffer(name, np.load(path, allow_pickle=False))
    _validate_spatial_shapes(result)
    return result


def load_npz_frame(
    path: str | Path,
    buffers: Sequence[str] = INFERENCE_BUFFERS,
) -> dict[str, torch.Tensor]:
    """Load a compact smoke-test frame from a compressed NumPy archive."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        missing = [name for name in buffers if name not in archive]
        if missing:
            raise KeyError(f"{path} is missing buffers: {', '.join(missing)}")
        result = {name: preprocess_buffer(name, archive[name]) for name in buffers}
    _validate_spatial_shapes(result)
    return result


def add_batch_dimension(
    buffers: Mapping[str, torch.Tensor],
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    """Move CHW buffers to ``device`` and convert them to BCHW."""

    output: dict[str, torch.Tensor] = {}
    for name, value in buffers.items():
        if value.ndim != 3:
            raise ValueError(f"{name} must be CHW before batching, got {value.shape}")
        output[name] = value.unsqueeze(0).to(device=device)
    return output


class BufferSequenceDataset(Dataset[dict[str, torch.Tensor]]):
    """Read one or more processed sequences without assuming private paths."""

    def __init__(
        self,
        sequences: Iterable[str | Path],
        buffers: Sequence[str] = INFERENCE_BUFFERS + TARGET_BUFFERS,
        repeat: int = 1,
    ) -> None:
        self.buffers = tuple(buffers)
        self.repeat = int(repeat)
        if self.repeat < 1:
            raise ValueError("repeat must be at least one")

        self.samples: list[tuple[Path, str]] = []
        for sequence_value in sequences:
            sequence = Path(sequence_value)
            for frame in discover_frames(sequence, anchor=self.buffers[0]):
                self.samples.append((sequence, frame))
        if not self.samples:
            raise ValueError("at least one non-empty sequence is required")

    def __len__(self) -> int:
        return len(self.samples) * self.repeat

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sequence, frame = self.samples[index % len(self.samples)]
        return load_frame(sequence, frame, self.buffers)
