"""Portable readers and writers for raw and processed TBFE buffers.

OpenCV is imported only by EXR functions.  NumPy data, matrix metadata, and the
rest of the package therefore remain usable on machines without OpenCV or an
OpenGL stack (for example, CPU-only CI workers).
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from .resources import brdf_lut_path
from .schema import BUFFER_SPECS, validate_buffer_array


_NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")
_MATRIX_LINE_RE = re.compile(r"^\s*([^:#]+?)\s*:\s*(.*)$")


def frame_filename(
    buffer_name: str,
    frame_index: int,
    suffix: str = ".npy",
    *,
    digits: int = 4,
) -> str:
    """Return the canonical ``Buffer.0001.ext`` name for one frame."""

    if not buffer_name or any(char in buffer_name for char in "/\\"):
        raise ValueError("buffer_name must be a non-empty basename")
    if frame_index < 0:
        raise ValueError("frame_index must be non-negative")
    if digits < 1:
        raise ValueError("digits must be positive")
    if not suffix.startswith("."):
        suffix = f".{suffix}"
    return f"{buffer_name}.{frame_index:0{digits}d}{suffix}"


def frame_path(
    sequence_dir: str | os.PathLike[str],
    buffer_name: str,
    frame_index: int,
    suffix: str = ".npy",
    *,
    digits: int = 4,
) -> Path:
    """Return a processed frame path rooted at ``sequence_dir``."""

    return Path(sequence_dir) / frame_filename(
        buffer_name, frame_index, suffix, digits=digits
    )


def write_npy(
    path: str | os.PathLike[str],
    array: np.ndarray,
    *,
    buffer_name: str | None = None,
) -> Path:
    """Write an HWC NumPy buffer, creating parent directories as needed."""

    destination = Path(path)
    if destination.suffix != ".npy":
        raise ValueError(f"NumPy buffer path must end in .npy: {destination}")
    if buffer_name is not None:
        validate_buffer_array(buffer_name, array)
    elif not isinstance(array, np.ndarray) or array.ndim != 3:
        shape = getattr(array, "shape", None)
        raise ValueError(f"Processed buffers must have shape HxWxC, got {shape}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, np.asarray(array), allow_pickle=False)
    return destination


def read_npy(
    path: str | os.PathLike[str],
    *,
    buffer_name: str | None = None,
    mmap_mode: str | None = None,
) -> np.ndarray:
    """Read a processed HWC NumPy buffer with optional schema validation."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    array = np.load(source, mmap_mode=mmap_mode, allow_pickle=False)
    if buffer_name is not None:
        validate_buffer_array(buffer_name, array)
    elif array.ndim != 3:
        raise ValueError(f"Processed buffers must have shape HxWxC, got {array.shape}")
    return array


def load_frame_buffers(
    sequence_dir: str | os.PathLike[str],
    frame_index: int,
    buffer_names: Iterable[str],
    *,
    digits: int = 4,
    mmap_mode: str | None = None,
) -> dict[str, np.ndarray]:
    """Load named processed buffers for a single frame."""

    result: dict[str, np.ndarray] = {}
    for name in buffer_names:
        if name not in BUFFER_SPECS:
            raise KeyError(f"Unknown buffer {name!r}")
        result[name] = read_npy(
            frame_path(sequence_dir, name, frame_index, digits=digits),
            buffer_name=name,
            mmap_mode=mmap_mode,
        )
    return result


def discover_frame_indices(
    sequence_dir: str | os.PathLike[str],
    buffer_name: str,
    *,
    suffix: str = ".npy",
) -> list[int]:
    """Discover sorted frame indices for one buffer without mixing buffer types."""

    directory = Path(sequence_dir)
    pattern = re.compile(
        rf"^{re.escape(buffer_name)}\.(\d+){re.escape(suffix)}$"
    )
    indices = []
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    for candidate in directory.iterdir():
        match = pattern.match(candidate.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def _require_cv2():
    # The switch must be set before importing cv2 for OpenEXR support.
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional package
        raise RuntimeError(
            "EXR I/O requires OpenCV with OpenEXR support. Install the "
            "preprocessing dependencies before reading raw renderer buffers."
        ) from exc
    return cv2


def read_exr(
    path: str | os.PathLike[str],
    *,
    channels: int | None = None,
    target_size: tuple[int, int] | None = None,
    motion_vectors: bool = False,
) -> np.ndarray:
    """Read an EXR as RGB(A), optionally resizing it.

    Args:
        path: Source EXR file.
        channels: Number of leading channels to return after BGR(A)-to-RGB(A)
            conversion. ``None`` retains all channels.
        target_size: Optional ``(width, height)``. Nearest-neighbour sampling is
            used to match the original preprocessing code.
        motion_vectors: Scale x/y vector magnitudes when resizing. Set this
            explicitly instead of relying on a filename convention.
    """

    cv2 = _require_cv2()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    image = cv2.imread(os.fspath(source), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"OpenCV could not decode EXR file: {source}")

    if image.ndim == 2:
        image = image[:, :, None]
    elif image.shape[2] == 3:
        image = image[:, :, ::-1]
    elif image.shape[2] == 4:
        image = image[:, :, [2, 1, 0, 3]]

    if channels is not None:
        if channels <= 0 or channels > image.shape[2]:
            raise ValueError(
                f"Requested {channels} channels from EXR with shape {image.shape}"
            )
        image = image[:, :, :channels]

    if target_size is not None:
        target_width, target_height = target_size
        if target_width <= 0 or target_height <= 0:
            raise ValueError(f"target_size must be positive, got {target_size}")
        old_height, old_width = image.shape[:2]
        if (old_width, old_height) != target_size:
            image = cv2.resize(
                image,
                (target_width, target_height),
                interpolation=cv2.INTER_NEAREST,
            )
            if image.ndim == 2:
                image = image[:, :, None]
            if motion_vectors:
                if image.shape[2] < 2:
                    raise ValueError("Motion-vector EXRs must contain at least two channels")
                image[..., 0] *= target_width / old_width
                image[..., 1] *= target_height / old_height

    return np.ascontiguousarray(image)


def write_exr(path: str | os.PathLike[str], image: np.ndarray) -> Path:
    """Write an HWC float image as a four-channel OpenEXR file."""

    cv2 = _require_cv2()
    destination = Path(path)
    if image.ndim != 3 or not 1 <= image.shape[2] <= 4:
        raise ValueError(f"EXR image must have shape HxWx[1,4], got {image.shape}")

    height, width, channels = image.shape
    rgba = np.zeros((height, width, 4), dtype=image.dtype)
    if channels == 1:
        rgba[..., :3] = image
    else:
        rgba[..., :channels] = image
    rgba[..., 3] = image[..., 3] if channels == 4 else 1
    bgra = np.ascontiguousarray(rgba[..., [2, 1, 0, 3]])

    destination.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(os.fspath(destination), bgra):
        raise OSError(f"OpenCV failed to write EXR file: {destination}")
    return destination


def read_brdf_lut(
    path: str | os.PathLike[str] | None = None,
    *,
    target_size: tuple[int, int] | None = (1280, 720),
) -> np.ndarray:
    """Read the bundled two-channel BRDF LUT used by legacy preprocessing."""

    return read_exr(
        brdf_lut_path() if path is None else path,
        channels=2,
        target_size=target_size,
    )


def parse_matrix_metadata(text: str) -> Mapping[str, np.ndarray]:
    """Parse named 4x4 matrices from renderer metadata text.

    Any line of the form ``Name: [....] [....] [....] [....]`` containing
    exactly 16 numeric values is accepted. Scalar metadata is ignored.
    """

    matrices: dict[str, np.ndarray] = {}
    for line in text.splitlines():
        match = _MATRIX_LINE_RE.match(line)
        if match is None:
            continue
        name, payload = match.groups()
        numbers = [float(value) for value in _NUMBER_RE.findall(payload)]
        if len(numbers) == 16:
            matrices[name.strip()] = np.asarray(numbers, dtype=np.float32).reshape(4, 4)
    return matrices


def read_matrix_metadata(path: str | os.PathLike[str]) -> Mapping[str, np.ndarray]:
    """Read named renderer matrices from a UTF-8 metadata file."""

    source = Path(path)
    return parse_matrix_metadata(source.read_text(encoding="utf-8"))


def view_projection_matrix(
    matrices: Mapping[str, np.ndarray],
    *,
    view_name: str = "ViewMatrix",
    projection_name: str = "ProjectionMatrix",
) -> np.ndarray:
    """Compose the row-vector convention ``view @ projection`` matrix."""

    try:
        view = matrices[view_name]
        projection = matrices[projection_name]
    except KeyError as exc:
        raise KeyError(
            f"Matrix metadata must contain {view_name!r} and {projection_name!r}"
        ) from exc
    if view.shape != (4, 4) or projection.shape != (4, 4):
        raise ValueError("View and projection matrices must both have shape (4, 4)")
    return view @ projection
