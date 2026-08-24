#!/usr/bin/env python3
"""Evaluate predictions without modifying them with ground-truth pixels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from tbfe.data import tone_map
from tbfe.metrics import gamma_encode, summarize


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", type=Path, required=True, help=".npz/.npy containing linear predicted color")
    parser.add_argument("--target", type=Path, required=True, help=".npz/.npy containing linear reference color")
    parser.add_argument("--prediction-key", default="PredictedColor")
    parser.add_argument("--target-key", default="Color")
    parser.add_argument("--valid-mask", type=Path, help="Optional .npy mask; one means evaluated")
    parser.add_argument("--exclude-mask", type=Path, action="append", default=[], help="Optional .npy mask; one means excluded")
    parser.add_argument("--lpips", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, help="Optional JSON output")
    return parser.parse_args()


def load_array(path: Path, key: str) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if key not in archive:
                raise KeyError(f"{path} does not contain {key}")
            return np.asarray(archive[key])
    return np.asarray(np.load(path, allow_pickle=False))


def to_bchw(value: np.ndarray, device: torch.device) -> torch.Tensor:
    if value.ndim == 2:
        value = value[..., None]
    if value.ndim != 3:
        raise ValueError(f"expected HWC array, got {value.shape}")
    return torch.from_numpy(value.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    prediction = to_bchw(load_array(args.prediction, args.prediction_key), device)
    target = to_bchw(load_array(args.target, args.target_key), device)
    prediction = gamma_encode(tone_map(prediction))
    target = gamma_encode(tone_map(target))

    valid = None
    if args.valid_mask:
        valid = to_bchw(load_array(args.valid_mask, "mask"), device)[:, :1]
    for path in args.exclude_mask:
        exclusion = to_bchw(load_array(path, "mask"), device)[:, :1].clamp(0.0, 1.0)
        valid = (torch.ones_like(exclusion) if valid is None else valid) * (1.0 - exclusion)

    metrics = summarize(prediction, target, valid_mask=valid, include_lpips=args.lpips)
    rendered = json.dumps(metrics, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
