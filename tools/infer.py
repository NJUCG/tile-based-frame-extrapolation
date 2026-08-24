#!/usr/bin/env python3
"""Run GT-free Tile-based Frame Extrapolation on one processed frame."""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

from tbfe.checkpoint import load_model_weights
from tbfe.data import add_batch_dimension, load_frame, load_npz_frame
from tbfe.models import ImageInpaintNet, TiledImageInpainter
from tbfe.render import log_irradiance_to_linear_color, pack_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="A smoke-test .npz or processed sequence directory")
    parser.add_argument("--frame", help="Frame id when --input is a sequence directory")
    parser.add_argument("--checkpoint", type=Path, help="Trusted .pt/.ckpt/.safetensors weights")
    parser.add_argument("--random-init", action="store_true", help="Only verify execution; output has no visual meaning")
    parser.add_argument("--output", type=Path, default=Path("outputs/prediction.npz"))
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or an explicit torch device")
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--tile-expand", type=int, default=8)
    parser.add_argument("--min-ratio", type=float, default=0.01)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    args = parse_args()
    if (args.checkpoint is None) == (not args.random_init):
        raise SystemExit("provide exactly one of --checkpoint or --random-init")
    if args.input.suffix.lower() == ".npz":
        raw = load_npz_frame(args.input)
    else:
        if args.frame is None:
            raise SystemExit("--frame is required when --input is a sequence directory")
        raw = load_frame(args.input, args.frame)

    device = resolve_device(args.device)
    torch.manual_seed(3407)
    network = ImageInpaintNet()
    model = TiledImageInpainter(
        network,
        tile_size=args.tile_size,
        tile_expand=args.tile_expand,
        min_ratio=args.min_ratio,
    ).to(device)
    if args.checkpoint is not None:
        load_model_weights(model, args.checkpoint, strict=True, map_location=device)
    else:
        warnings.warn(
            "using randomly initialized parameters: this run checks only data and model plumbing",
            stacklevel=1,
        )

    buffers = add_batch_dimension(raw, device=device)
    irradiance, hole_mask, features = pack_inputs(buffers)
    model.eval()
    with torch.inference_mode():
        result = model(irradiance, hole_mask, features)
        linear_color = log_irradiance_to_linear_color(result.image, buffers["Albedo"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        PredictedIrradiance=result.image[0].permute(1, 2, 0).cpu().numpy(),
        PredictedColor=linear_color[0].permute(1, 2, 0).cpu().numpy(),
        SelectedMask=result.mask[0].permute(1, 2, 0).cpu().numpy(),
    )
    print(
        f"saved {args.output} | device={device} | selected_tiles={result.tile_count} "
        f"| shape={tuple(linear_color.shape[-2:])}"
    )


if __name__ == "__main__":
    main()
