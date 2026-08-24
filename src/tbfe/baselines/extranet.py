"""ExtraNet baseline with a tensor-only, ground-truth-free inference API."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class _DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mid_channels: int | None = None,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.double_conv(tensor)


class _LightweightGatedConv2d(nn.Module):
    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        padding: int,
        kernel_size: int,
        stride: int,
    ) -> None:
        super().__init__()
        self.conv_feature = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.conv_mask = nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=1,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            nn.Sigmoid(),
        )

    def forward(self, tensor: Tensor) -> Tensor:
        return self.conv_feature(tensor) * self.conv_mask(tensor)


class _DownLightweightGated(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.downsample = _LightweightGatedConv2d(
            in_channels, in_channels, kernel_size=3, padding=1, stride=2
        )
        self.conv1 = _LightweightGatedConv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = _LightweightGatedConv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, tensor: Tensor) -> Tensor:
        tensor = self.downsample(tensor)
        tensor = self.relu1(self.bn1(self.conv1(tensor)))
        return self.relu2(self.bn2(self.conv2(tensor)))


class _Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        # The paper implementation always used this bilinear branch.
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, tensor: Tensor, skip: Tensor) -> Tensor:
        tensor = self.up(tensor)
        diff_y = skip.size(2) - tensor.size(2)
        diff_x = skip.size(3) - tensor.size(3)
        tensor = F.pad(
            tensor,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        return self.conv(torch.cat((skip, tensor), dim=1))


class ExtraNet(nn.Module):
    """Gated U-Net baseline used in the paper.

    The explicit inputs correspond to the legacy buffer dictionary as follows:

    - ``warp_irradiance_1``: RGB irradiance from the ordinary one-frame warp.
    - ``occlusion_warp_irradiance_1``: RGB irradiance from the occlusion-aware
      one-frame warp.
    - ``hole_mask_1``: one for an invalid/hole pixel and zero for a valid pixel
      (the inverse of the legacy ``WarpToCurrGbufferMask_1`` buffer).
    - ``geometry_features``: depth (1), normal (3), metallic (1), and
      roughness (1), concatenated in that order.
    - ``history_irradiance``: occlusion-aware RGB warps at offsets 1, 3, and 5,
      shaped ``[B, 3, 3, H, W]``.
    - ``history_hole_masks``: matching invalid-pixel masks, shaped
      ``[B, 3, 1, H, W]``.

    No reference image or ground-truth buffer is consumed during inference.
    Module attribute names match the paper implementation so its model
    ``state_dict`` remains loadable after removing a leading ``model.`` prefix.
    """

    def __init__(
        self,
        n_channels: int = 18,
        n_classes: int = 3,
        *,
        skip: bool = True,
    ) -> None:
        super().__init__()
        if n_channels <= 0 or n_classes <= 0:
            raise ValueError("n_channels and n_classes must be positive")
        if skip and n_classes != 3:
            raise ValueError("the historical RGB skip requires n_classes=3")

        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.skip = bool(skip)

        self.convHis1 = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
        )
        self.convHis2 = nn.Sequential(
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.convHis3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.lowlevelGated = _LightweightGatedConv2d(
            32 * 3, 32, kernel_size=3, stride=1, padding=1
        )
        self.conv1 = _LightweightGatedConv2d(
            self.n_channels, 24, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = _LightweightGatedConv2d(
            24, 24, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(24)
        self.relu2 = nn.ReLU(inplace=True)
        self.down1 = _DownLightweightGated(24, 24)
        self.down2 = _DownLightweightGated(24, 32)
        self.down3 = _DownLightweightGated(32, 32)

        self.up1 = _Up(96, 32)
        self.up2 = _Up(56, 24)
        self.up3 = _Up(48, 24)
        self.outc = nn.Conv2d(24, self.n_classes, kernel_size=1)

    @staticmethod
    def _validate_image(name: str, tensor: Tensor, channels: int) -> None:
        if not isinstance(tensor, Tensor) or tensor.ndim != 4:
            raise ValueError(f"{name} must be a [B, C, H, W] tensor")
        if tensor.shape[1] != channels:
            raise ValueError(f"{name} must have {channels} channels, got {tensor.shape[1]}")

    def _predict(
        self,
        warp_pair: Tensor,
        geometry_features: Tensor,
        hole_mask: Tensor,
        history: Tensor,
    ) -> Tensor:
        batch, _, height, width = warp_pair.shape
        history = history.reshape(batch * 3, 4, height, width)

        history_down1 = self.convHis1(history)
        history_down2 = self.convHis2(history_down1)
        history_down3 = self.convHis3(history_down2)
        history_down3 = history_down3.reshape(
            batch, 3 * 32, history_down3.shape[-2], history_down3.shape[-1]
        )
        motion_features = self.lowlevelGated(history_down3)

        tensor1 = torch.cat((warp_pair, warp_pair * hole_mask, geometry_features), dim=1)
        tensor1 = self.relu1(self.bn1(self.conv1(tensor1)))
        tensor1 = self.relu2(self.bn2(self.conv2(tensor1)))
        tensor2 = self.down1(tensor1)
        tensor3 = self.down2(tensor2)
        tensor4 = self.down3(tensor3)

        tensor4 = torch.cat((tensor4, motion_features), dim=1)
        result = self.up1(tensor4, tensor3)
        result = self.up2(result, tensor2)
        result = self.up3(result, tensor1)
        prediction = self.outc(result)
        if self.skip:
            prediction = prediction + warp_pair[:, :3]
        return prediction

    def forward(
        self,
        warp_irradiance_1: Tensor,
        occlusion_warp_irradiance_1: Tensor,
        hole_mask_1: Tensor,
        geometry_features: Tensor,
        history_irradiance: Tensor,
        history_hole_masks: Tensor,
    ) -> Tensor:
        """Predict RGB irradiance from renderer-provided inputs only."""

        self._validate_image("warp_irradiance_1", warp_irradiance_1, 3)
        self._validate_image(
            "occlusion_warp_irradiance_1", occlusion_warp_irradiance_1, 3
        )
        self._validate_image("hole_mask_1", hole_mask_1, 1)
        self._validate_image("geometry_features", geometry_features, 6)
        if history_irradiance.ndim != 5 or history_irradiance.shape[1:3] != (3, 3):
            raise ValueError(
                "history_irradiance must have shape [B, 3 offsets, 3, H, W]"
            )
        if history_hole_masks.ndim != 5 or history_hole_masks.shape[1:3] != (3, 1):
            raise ValueError(
                "history_hole_masks must have shape [B, 3 offsets, 1, H, W]"
            )

        tensors = (
            occlusion_warp_irradiance_1,
            hole_mask_1,
            geometry_features,
            history_irradiance,
            history_hole_masks,
        )
        reference = (warp_irradiance_1.shape[0], *warp_irradiance_1.shape[-2:])
        for tensor in tensors:
            if (tensor.shape[0], *tensor.shape[-2:]) != reference:
                raise ValueError("all ExtraNet inputs must share batch and spatial dimensions")
            if tensor.device != warp_irradiance_1.device:
                raise ValueError("all ExtraNet inputs must be on the same device")
            if tensor.dtype != warp_irradiance_1.dtype:
                raise ValueError("all ExtraNet inputs must have the same dtype")

        warp_pair = torch.cat(
            (warp_irradiance_1, occlusion_warp_irradiance_1), dim=1
        )
        packed_channels = warp_pair.shape[1] * 2 + geometry_features.shape[1]
        if packed_channels != self.n_channels:
            raise ValueError(
                f"packed model input has {packed_channels} channels, expected {self.n_channels}"
            )
        history = torch.cat((history_irradiance, history_hole_masks), dim=2)
        return self._predict(warp_pair, geometry_features, hole_mask_1, history)


__all__ = ["ExtraNet"]
