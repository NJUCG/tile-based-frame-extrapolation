#!/usr/bin/env python3
"""Export the per-tile repair kernel; tile scheduling stays in host code."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from tbfe.checkpoint import load_model_weights
from tbfe.models import ImageInpaintNet, TiledImageInpainter


class PackedTileKernel(torch.nn.Module):
    def __init__(self, network: ImageInpaintNet) -> None:
        super().__init__()
        self.network = network

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        return self.network.forward_packed(packed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("outputs/tbfe_tile.onnx"))
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--tile-expand", type=int, default=8)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wrapper = TiledImageInpainter(ImageInpaintNet(), tile_size=args.tile_size, tile_expand=args.tile_expand)
    load_model_weights(wrapper, args.checkpoint, strict=True)
    network = PackedTileKernel(wrapper.network.eval()).eval()
    extent = args.tile_size + 2 * args.tile_expand
    packed = torch.zeros(1, 13, extent, extent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        network,
        (packed,),
        args.output,
        input_names=["packed_input"],
        output_names=["residual"],
        dynamic_axes={"packed_input": {0: "tiles"}, "residual": {0: "tiles"}},
        opset_version=args.opset,
    )
    print(f"saved tile kernel to {args.output}")


if __name__ == "__main__":
    main()
