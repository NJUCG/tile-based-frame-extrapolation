#!/usr/bin/env python3
"""Minimal, path-agnostic training entry point for the tiled repair network."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from tbfe.data import BufferSequenceDataset
from tbfe.losses import ImageInpaintLoss
from tbfe.metrics import psnr
from tbfe.models import ImageInpaintNet, TiledImageInpainter
from tbfe.ops import extract_tiles
from tbfe.render import linear_color_to_metric_space, log_irradiance_to_linear_color, pack_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/train.yaml"))
    parser.add_argument("--device", help="Override config device")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def make_loader(sequences: list[str], cfg: dict, shuffle: bool) -> DataLoader:
    if not sequences:
        raise ValueError("dataset sequence list is empty; edit configs/train.yaml")
    dataset = BufferSequenceDataset(sequences)
    workers = int(cfg.get("num_workers", 4))
    return DataLoader(
        dataset,
        batch_size=int(cfg.get("batch_size", 1)),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=bool(cfg.get("pin_memory", True)),
        persistent_workers=workers > 0,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device, non_blocking=True) for name, value in batch.items()}


@torch.no_grad()
def validate(model: TiledImageInpainter, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores: list[float] = []
    for batch in loader:
        batch = move_batch(batch, device)
        irradiance, mask, features = pack_inputs(batch)
        result = model(irradiance, mask, features)
        predicted_color = log_irradiance_to_linear_color(result.image, batch["Albedo"])
        prediction = linear_color_to_metric_space(predicted_color)
        target = torch.clamp(batch["Color"], min=0.0).pow(1.0 / 2.2).clamp(0.0, 1.0)
        scores.append(float(psnr(prediction, target)))
    return sum(scores) / max(len(scores), 1)


def main() -> None:
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = int(cfg.get("seed", 3407))
    seed_everything(seed)
    device = resolve_device(args.device or cfg.get("device", "auto"))

    model_cfg = cfg.get("model", {})
    tiling_cfg = cfg.get("tiling", {})
    network = ImageInpaintNet(**model_cfg)
    model = TiledImageInpainter(network, **tiling_cfg).to(device)
    loss_fn = ImageInpaintLoss(**cfg.get("loss", {})).to(device)

    data_cfg = cfg["data"]
    train_loader = make_loader(list(data_cfg.get("train_sequences", [])), data_cfg, shuffle=True)
    val_sequences = list(data_cfg.get("val_sequences", []))
    val_loader = make_loader(val_sequences, data_cfg, shuffle=False) if val_sequences else None

    train_cfg = cfg.get("training", {})
    optimizer = torch.optim.Adam(model.parameters(), lr=float(train_cfg.get("learning_rate", 4e-4)))
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=list(train_cfg.get("milestones", [20, 40, 60, 70, 80, 90])),
        gamma=float(train_cfg.get("gamma", 0.5)),
    )
    output = Path(train_cfg.get("output", "outputs/train"))
    output.mkdir(parents=True, exist_ok=True)
    epochs = int(train_cfg.get("epochs", 100))
    best_psnr = float("-inf")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        updates = 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            irradiance, mask, features = pack_inputs(batch)
            result = model(irradiance, mask, features)
            if not result.has_tiles:
                continue
            target_tiles = extract_tiles(batch["Irradiance"], result.layout)
            losses = loss_fn(result.tile_residual, result.tile_color, target_tiles, result.tile_mask)
            optimizer.zero_grad(set_to_none=True)
            losses["total_loss"].backward()
            optimizer.step()
            running_loss += float(losses["total_loss"].detach())
            updates += 1
        scheduler.step()

        val_psnr = validate(model, val_loader, device) if val_loader else float("nan")
        report = {
            "epoch": epoch,
            "train_loss": running_loss / max(updates, 1),
            "updates": updates,
            "val_psnr": val_psnr,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(report, sort_keys=True))
        torch.save(model.state_dict(), output / "latest.pt")
        if val_loader and val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), output / "best.pt")


if __name__ == "__main__":
    main()
