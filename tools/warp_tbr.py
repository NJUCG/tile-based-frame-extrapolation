#!/usr/bin/env python3
"""Run the G-buffer-guided TBR warp on one input archive."""

from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path

import numpy as np

from tbfe.preprocess import TBRWarpInputs, warp_tbr


INPUT_NAMES = tuple(field.name for field in fields(TBRWarpInputs))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help=".npz containing the seven TBRWarpInputs fields")
    parser.add_argument("--output", type=Path, required=True, help="Output .npy warped irradiance")
    parser.add_argument(
        "--current-context",
        action="store_true",
        help="Use the caller's current OpenGL context instead of creating EGL",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with np.load(args.input, allow_pickle=False) as archive:
        missing = [name for name in INPUT_NAMES if name not in archive]
        if missing:
            raise KeyError(f"{args.input} is missing fields: {', '.join(missing)}")
        inputs = TBRWarpInputs(**{name: archive[name] for name in INPUT_NAMES})
    result = warp_tbr(inputs, create_context=not args.current_context)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, result, allow_pickle=False)
    print(f"saved {args.output}; shape={result.shape}; dtype={result.dtype}")


if __name__ == "__main__":
    main()
