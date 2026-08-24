#!/usr/bin/env python3
"""Create a compact, input-only smoke fixture from one processed frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


BUFFERS = (
    "TbrIrradiance_1",
    "WarpToCurrGbufferMask_1",
    "Depth",
    "Normal",
    "Metallic",
    "Roughness",
    "Albedo",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--frame", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    return parser.parse_args()


def load_raw(sequence: Path, name: str, frame: str) -> np.ndarray:
    path = sequence / f"{name}.{frame}.npy"
    if not path.is_file():
        raise FileNotFoundError(path)
    value = np.load(path, allow_pickle=False)
    if value.ndim == 2:
        value = value[..., None]
    if value.ndim != 3:
        raise ValueError(f"{path} is not HWC: {value.shape}")
    return value


def densest_crop(mask: np.ndarray, size: int) -> tuple[int, int]:
    height, width = mask.shape
    if size > height or size > width:
        raise ValueError(f"crop {size} does not fit source resolution {width}x{height}")
    binary = (mask > 0).astype(np.int64)
    integral = np.pad(binary, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    sums = (
        integral[size:, size:]
        - integral[:-size, size:]
        - integral[size:, :-size]
        + integral[:-size, :-size]
    )
    y, x = np.unravel_index(int(np.argmax(sums)), sums.shape)
    return int(y), int(x)


def main() -> None:
    args = parse_args()
    arrays = {name: load_raw(args.sequence, name, args.frame) for name in BUFFERS}
    shapes = {name: value.shape[:2] for name, value in arrays.items()}
    if len(set(shapes.values())) != 1:
        raise ValueError(f"buffer resolutions differ: {shapes}")
    mask = arrays["WarpToCurrGbufferMask_1"][..., 0]
    top, left = densest_crop(mask, args.size)
    cropped = {
        name: value[top : top + args.size, left : left + args.size]
        for name, value in arrays.items()
    }
    cropped.update(
        source_scene=np.asarray(args.sequence.parent.name),
        source_sequence=np.asarray(args.sequence.name),
        source_frame=np.asarray(args.frame),
        crop_xywh=np.asarray([left, top, args.size, args.size], dtype=np.int32),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **cropped)
    selected = int((cropped["WarpToCurrGbufferMask_1"] > 0).sum())
    print(f"saved {args.output}; crop=({left}, {top}, {args.size}, {args.size}); hole_values={selected}")


if __name__ == "__main__":
    main()
