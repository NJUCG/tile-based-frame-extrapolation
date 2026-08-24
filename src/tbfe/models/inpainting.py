"""Neural repair model used by Tile-based Frame Extrapolation.

The network predicts an irradiance residual.  It never consumes reference
irradiance or ground-truth colour: those tensors belong exclusively to the
training loss or evaluator.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from tbfe.ops.tiling import (
    TileLayout,
    extract_tiles,
    merge_tiles_and_mask,
    select_tiles,
)


def _group_norm(channels: int, max_groups: int = 8) -> nn.GroupNorm:
    """Match the group-count rule used for the paper model."""

    return nn.GroupNorm(max(1, math.gcd(channels, max_groups)), channels)


class _DoubleConv(nn.Module):
    """Two reflect-padded convolutions used by both encoders."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            _group_norm(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            _group_norm(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.block(tensor)


class _DoubleConvPReLU(nn.Module):
    """Two decoder convolutions with channel-wise PReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )
        self.norm1 = _group_norm(out_channels)
        self.act1 = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            3,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )
        self.norm2 = _group_norm(out_channels)
        self.act2 = nn.PReLU(out_channels)

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = self.act1(self.norm1(self.conv1(tensor)))
        return self.act2(self.norm2(self.conv2(tensor)))


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            3,
            stride=2,
            padding=1,
            bias=False,
            padding_mode="reflect",
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.conv(tensor)


class _Up(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv = _DoubleConvPReLU(out_channels + skip_channels, out_channels)

    def forward(self, tensor: Tensor, skip: Tensor) -> Tensor:
        tensor = self.up(tensor)
        if tensor.shape[-2:] != skip.shape[-2:]:
            tensor = F.interpolate(
                tensor, size=skip.shape[-2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat((tensor, skip), dim=1))


class ImageInpaintNet(nn.Module):
    """Dual-encoder U-Net that predicts an irradiance residual.

    Args:
        base: Base channel count of the colour encoder and decoder.
        feat_base: Base channel count of the G-buffer encoder.
        color_ch: Number of input irradiance channels.
        feat_ch: Number of packed G-buffer channels.
        out_ch: Number of residual channels.

    ``features`` uses the paper ordering: depth (1), normal (3), metallic (1),
    roughness (1), and albedo (3), for nine channels in total by default.

    Module names intentionally match the training implementation so that a
    plain model ``state_dict`` can be loaded without renaming keys.
    """

    def __init__(
        self,
        base: int = 16,
        feat_base: int = 16,
        color_ch: int = 3,
        feat_ch: int = 9,
        out_ch: int = 3,
    ) -> None:
        super().__init__()
        if min(base, feat_base, color_ch, feat_ch, out_ch) <= 0:
            raise ValueError("all model channel counts must be positive")

        self.color_ch = int(color_ch)
        self.feat_ch = int(feat_ch)
        self.out_ch = int(out_ch)

        e1, e2, e3, e4 = base, base * 2, base * 4, base * 8
        f1, f2, f3, f4 = feat_base * 2, feat_base * 4, feat_base * 6, feat_base * 8
        d1, d2, d3 = base * 2, base * 4, base * 8

        self.color_enc1 = _DoubleConv(color_ch + 1, e1)
        self.color_down1 = _Down(e1, e2)
        self.color_enc2 = _DoubleConv(e2, e2)
        self.color_down2 = _Down(e2, e3)
        self.color_enc3 = _DoubleConv(e3, e3)
        self.color_down3 = _Down(e3, e4)
        self.color_bott = _DoubleConv(e4, e4)

        self.feat_enc1 = _DoubleConv(feat_ch, f1)
        self.feat_down1 = _Down(f1, f2)
        self.feat_enc2 = _DoubleConv(f2, f2)
        self.feat_down2 = _Down(f2, f3)
        self.feat_enc3 = _DoubleConv(f3, f3)
        self.feat_down3 = _Down(f3, f4)
        self.feat_bott = _DoubleConv(f4, f4)

        self.up2 = _Up(e4 + f4, e3 + f3, d3)
        self.up1 = _Up(d3, e2 + f2, d2)
        self.up0 = _Up(d2, e1 + f1, d1)
        self.head = nn.Sequential(
            nn.Conv2d(
                d1,
                d1 // 2,
                3,
                padding=1,
                bias=False,
                padding_mode="reflect",
            ),
            _group_norm(d1 // 2),
            nn.PReLU(d1 // 2),
            nn.Conv2d(d1 // 2, out_ch, 1),
        )

    @staticmethod
    def _validate_inputs(color: Tensor, hole_mask: Tensor, features: Tensor) -> None:
        names_and_tensors = (
            ("color", color),
            ("hole_mask", hole_mask),
            ("features", features),
        )
        for name, tensor in names_and_tensors:
            if not isinstance(tensor, Tensor) or tensor.ndim != 4:
                shape = getattr(tensor, "shape", None)
                raise ValueError(f"{name} must be a [B, C, H, W] tensor, got {shape}")

        reference = (color.shape[0], color.shape[-2], color.shape[-1])
        for name, tensor in names_and_tensors[1:]:
            current = (tensor.shape[0], tensor.shape[-2], tensor.shape[-1])
            if current != reference:
                raise ValueError(
                    f"{name} batch/spatial dimensions {current} do not match color {reference}"
                )
        if color.device != hole_mask.device or color.device != features.device:
            raise ValueError("color, hole_mask, and features must be on the same device")
        if color.dtype != hole_mask.dtype or color.dtype != features.dtype:
            raise ValueError("color, hole_mask, and features must have the same dtype")

    def inference(self, color_and_mask: Tensor, features: Tensor) -> Tensor:
        """Run the tensor-only network kernel used by training and deployment.

        Args:
            color_and_mask: ``[B, color_ch + 1, H, W]`` packed tensor.
            features: ``[B, feat_ch, H, W]`` packed G-buffer tensor.
        """

        if color_and_mask.ndim != 4 or color_and_mask.shape[1] != self.color_ch + 1:
            raise ValueError(
                f"color_and_mask must have {self.color_ch + 1} channels, "
                f"got shape {tuple(color_and_mask.shape)}"
            )
        if features.ndim != 4 or features.shape[1] != self.feat_ch:
            raise ValueError(
                f"features must have {self.feat_ch} channels, got shape {tuple(features.shape)}"
            )
        color = color_and_mask[:, : self.color_ch]
        hole_mask = color_and_mask[:, self.color_ch : self.color_ch + 1]
        self._validate_inputs(color, hole_mask, features)

        ce1 = self.color_enc1(color_and_mask)
        ce2 = self.color_enc2(self.color_down1(ce1))
        ce3 = self.color_enc3(self.color_down2(ce2))
        color_bottleneck = self.color_bott(self.color_down3(ce3))

        fe1 = self.feat_enc1(features)
        fe2 = self.feat_enc2(self.feat_down1(fe1))
        fe3 = self.feat_enc3(self.feat_down2(fe2))
        feature_bottleneck = self.feat_bott(self.feat_down3(fe3))

        bottleneck = torch.cat((color_bottleneck, feature_bottleneck), dim=1)
        decoder2 = self.up2(bottleneck, torch.cat((ce3, fe3), dim=1))
        decoder1 = self.up1(decoder2, torch.cat((ce2, fe2), dim=1))
        decoder0 = self.up0(decoder1, torch.cat((ce1, fe1), dim=1))
        residual = self.head(decoder0)
        return torch.nan_to_num(residual, nan=0.0, posinf=1e6, neginf=-1e6)

    def forward(self, color: Tensor, hole_mask: Tensor, features: Tensor) -> Tensor:
        """Predict a residual without accessing ground truth."""

        self._validate_inputs(color, hole_mask, features)
        if color.shape[1] != self.color_ch:
            raise ValueError(f"color must have {self.color_ch} channels, got {color.shape[1]}")
        if hole_mask.shape[1] != 1:
            raise ValueError(f"hole_mask must have one channel, got {hole_mask.shape[1]}")
        if features.shape[1] != self.feat_ch:
            raise ValueError(
                f"features must have {self.feat_ch} channels, got {features.shape[1]}"
            )
        return self.inference(torch.cat((color, hole_mask), dim=1), features)

    def forward_packed(self, packed: Tensor) -> Tensor:
        """Deployment helper for a single packed ONNX/TensorRT input."""

        expected_channels = self.color_ch + 1 + self.feat_ch
        if packed.ndim != 4 or packed.shape[1] != expected_channels:
            raise ValueError(
                f"packed input must have {expected_channels} channels, "
                f"got shape {tuple(packed.shape)}"
            )
        color_end = self.color_ch
        feature_start = color_end + 1
        return self.forward(
            packed[:, :color_end],
            packed[:, color_end:feature_start],
            packed[:, feature_start:],
        )


@dataclass
class TiledInpaintingResult:
    """Outputs and training metadata from tiled inference."""

    image: Tensor
    residual: Tensor
    mask: Tensor
    tile_residual: Tensor
    tile_color: Tensor
    tile_mask: Tensor
    layout: TileLayout

    @property
    def has_tiles(self) -> bool:
        return self.layout.has_tiles

    @property
    def tile_count(self) -> int:
        return self.layout.tile_count


class TiledImageInpainter(nn.Module):
    """Select damaged tiles, run :class:`ImageInpaintNet`, and merge results."""

    def __init__(
        self,
        network: ImageInpaintNet | None = None,
        *,
        tile_size: int = 64,
        min_ratio: float = 0.01,
        tile_expand: int = 8,
    ) -> None:
        super().__init__()
        if tile_size <= 0:
            raise ValueError(f"tile_size must be positive, got {tile_size}")
        if tile_expand < 0:
            raise ValueError(f"tile_expand must be non-negative, got {tile_expand}")
        if not 0.0 <= min_ratio <= 1.0:
            raise ValueError(f"min_ratio must be in [0, 1], got {min_ratio}")
        self.network = network if network is not None else ImageInpaintNet()
        self.tile_size = int(tile_size)
        self.min_ratio = float(min_ratio)
        self.tile_expand = int(tile_expand)

    def forward(self, color: Tensor, hole_mask: Tensor, features: Tensor) -> TiledInpaintingResult:
        """Repair ``color`` using only its mask and current-frame G-buffers."""

        self.network._validate_inputs(color, hole_mask, features)
        if color.shape[1] != self.network.color_ch:
            raise ValueError(
                f"color must have {self.network.color_ch} channels, got {color.shape[1]}"
            )
        if hole_mask.shape[1] != 1:
            raise ValueError(f"hole_mask must have one channel, got {hole_mask.shape[1]}")
        if features.shape[1] != self.network.feat_ch:
            raise ValueError(
                f"features must have {self.network.feat_ch} channels, got {features.shape[1]}"
            )
        if self.network.out_ch != color.shape[1]:
            raise ValueError(
                "tiled residual composition requires network out_ch to equal color channels"
            )

        layout = select_tiles(
            hole_mask,
            tile_size=self.tile_size,
            min_ratio=self.min_ratio,
            tile_expand=self.tile_expand,
        )
        tile_color = extract_tiles(color, layout)
        tile_mask = extract_tiles(hole_mask, layout)

        if layout.has_tiles:
            tile_features = extract_tiles(features, layout)
            tile_residual = self.network(tile_color, tile_mask, tile_features)
            residual, selected_mask = merge_tiles_and_mask(
                tile_residual, tile_mask, layout
            )
        else:
            tile_residual = color.new_empty(
                (0, self.network.out_ch, layout.tile_extract, layout.tile_extract)
            )
            residual = color.new_zeros(
                (color.shape[0], self.network.out_ch, color.shape[-2], color.shape[-1])
            )
            selected_mask = hole_mask.new_zeros(hole_mask.shape)

        image = color + residual * selected_mask
        return TiledInpaintingResult(
            image=image,
            residual=residual,
            mask=selected_mask,
            tile_residual=tile_residual,
            tile_color=tile_color,
            tile_mask=tile_mask,
            layout=layout,
        )


__all__ = ["ImageInpaintNet", "TiledImageInpainter", "TiledInpaintingResult"]
