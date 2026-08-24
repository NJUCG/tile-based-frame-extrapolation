#!/usr/bin/env python3
"""Benchmark the complete tile selection, network, and merge path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from tbfe.checkpoint import load_model_weights
from tbfe.data import add_batch_dimension, load_npz_frame
from tbfe.models import ImageInpaintNet, TiledImageInpainter
from tbfe.render import pack_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("examples/factory_smoke/frame.npz"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--tile-expand", type=int, default=8)
    parser.add_argument("--min-ratio", type=float, default=0.01)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))
    model = TiledImageInpainter(
        ImageInpaintNet(),
        tile_size=args.tile_size,
        tile_expand=args.tile_expand,
        min_ratio=args.min_ratio,
    ).to(device).eval()
    if args.checkpoint:
        load_model_weights(model, args.checkpoint, strict=True, map_location=device)
    batch = add_batch_dimension(load_npz_frame(args.input), device)
    inputs = pack_inputs(batch)

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(args.warmup):
            result = model(*inputs)
        synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            result = model(*inputs)
        synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000.0 / args.iterations
    print(json.dumps({
        "device": str(device),
        "iterations": args.iterations,
        "mean_ms": elapsed_ms,
        "selected_tiles": result.tile_count,
        "input_shape": list(inputs[0].shape),
        "checkpoint_loaded": args.checkpoint is not None,
    }, indent=2))


if __name__ == "__main__":
    main()
