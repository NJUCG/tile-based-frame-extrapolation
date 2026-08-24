"""ExtraSS baselines used by Tile-based Frame Extrapolation.

The feature pyramid and coarse-to-fine flow decoder are adapted from IFRNet
(https://github.com/ltkong218/IFRNet), which is distributed under the MIT
License.  Attribution and the retained license notice are recorded in the
repository-level ``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def _resize(tensor: Tensor, scale_factor: float) -> Tensor:
    return F.interpolate(
        tensor, scale_factor=scale_factor, mode="bilinear", align_corners=False
    )


def _conv_prelu(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias=bias,
        ),
        nn.PReLU(out_channels),
    )


def _backward_warp(
    image: Tensor,
    flow: Tensor,
    *,
    mode: str = "bilinear",
    padding_mode: str = "border",
) -> Tensor:
    """Backward-warp ``image`` with pixel-space ``flow``.

    This is the small ``grid_sample`` operation that ExtraSS used from the
    legacy utility module.  It is kept local so importing a baseline does not
    pull in OpenCV, LPIPS, Kornia, or the restricted soft-splat dependency.
    """

    height, width = image.shape[-2:]
    if flow.shape[-2:] != (height, width):
        flow = F.interpolate(
            flow, size=(height, width), mode="bilinear", align_corners=True
        )
    flow_x = 2 * flow[:, 0:1] / max(width - 1, 1)
    flow_y = 2 * flow[:, 1:2] / max(height - 1, 1)

    xx = torch.linspace(-1, 1, width, device=flow.device, dtype=flow.dtype)
    yy = torch.linspace(-1, 1, height, device=flow.device, dtype=flow.dtype)
    xx, yy = torch.meshgrid(xx, yy, indexing="xy")
    grid = torch.stack((xx, yy), dim=0).unsqueeze(0)
    grid = grid + torch.cat((flow_x, flow_y), dim=1)
    return F.grid_sample(
        image,
        grid.permute(0, 2, 3, 1),
        mode=mode,
        padding_mode=padding_mode,
        align_corners=True,
    )


class _ResBlock(nn.Module):
    def __init__(self, in_channels: int, side_channels: int, bias: bool = True) -> None:
        super().__init__()
        self.side_channels = side_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias),
            nn.PReLU(in_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, 3, 1, 1, bias=bias),
            nn.PReLU(side_channels),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias),
            nn.PReLU(in_channels),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(side_channels, side_channels, 3, 1, 1, bias=bias),
            nn.PReLU(side_channels),
        )
        self.conv5 = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=bias)
        self.prelu = nn.PReLU(in_channels)

    def forward(self, tensor: Tensor) -> Tensor:
        output = self.conv1(tensor)
        output[:, -self.side_channels :] = self.conv2(
            output[:, -self.side_channels :].clone()
        )
        output = self.conv3(output)
        output[:, -self.side_channels :] = self.conv4(
            output[:, -self.side_channels :].clone()
        )
        return self.prelu(tensor + self.conv5(output))


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.pyramid1 = nn.Sequential(
            _conv_prelu(3, 18, 3, 2, 1),
            _conv_prelu(18, 18),
        )
        self.pyramid2 = nn.Sequential(
            _conv_prelu(18, 24, 3, 2, 1),
            _conv_prelu(24, 24),
        )
        self.pyramid3 = nn.Sequential(
            _conv_prelu(24, 36, 3, 2, 1),
            _conv_prelu(36, 36),
        )
        self.pyramid4 = nn.Sequential(
            _conv_prelu(36, 48, 3, 2, 1),
            _conv_prelu(48, 48),
        )

    def forward(self, image: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        feature1 = self.pyramid1(image)
        feature2 = self.pyramid2(feature1)
        feature3 = self.pyramid3(feature2)
        feature4 = self.pyramid4(feature3)
        return feature1, feature2, feature3, feature4


class _Decoder4(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            _conv_prelu(96, 96),
            _ResBlock(96, 24),
            nn.ConvTranspose2d(96, 40, 4, 2, 1, bias=True),
        )

    def forward(self, feature0: Tensor, feature1: Tensor) -> Tensor:
        return self.convblock(torch.cat((feature0, feature1), dim=1))


def _match_decoder_size(tensor: Tensor, reference: Tensor) -> Tensor:
    if tensor.shape[-2:] == reference.shape[-2:]:
        return tensor
    return F.interpolate(
        tensor, size=reference.shape[-2:], mode="bilinear", align_corners=False
    )


class _Decoder3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            _conv_prelu(3 * 36 + 4, 108),
            _ResBlock(108, 24),
            nn.ConvTranspose2d(108, 28, 4, 2, 1, bias=True),
        )

    def forward(
        self,
        intermediate: Tensor,
        feature0: Tensor,
        feature1: Tensor,
        flow0: Tensor,
        flow1: Tensor,
    ) -> Tensor:
        feature0 = _match_decoder_size(_backward_warp(feature0, flow0), intermediate)
        feature1 = _match_decoder_size(_backward_warp(feature1, flow1), intermediate)
        flow0 = _match_decoder_size(flow0, intermediate)
        flow1 = _match_decoder_size(flow1, intermediate)
        return self.convblock(
            torch.cat((intermediate, feature0, feature1, flow0, flow1), dim=1)
        )


class _Decoder2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            _conv_prelu(76, 72),
            _ResBlock(72, 24),
            nn.ConvTranspose2d(72, 22, 4, 2, 1, bias=True),
        )

    def forward(
        self,
        intermediate: Tensor,
        feature0: Tensor,
        feature1: Tensor,
        flow0: Tensor,
        flow1: Tensor,
    ) -> Tensor:
        feature0 = _match_decoder_size(_backward_warp(feature0, flow0), intermediate)
        feature1 = _match_decoder_size(_backward_warp(feature1, flow1), intermediate)
        flow0 = _match_decoder_size(flow0, intermediate)
        flow1 = _match_decoder_size(flow1, intermediate)
        return self.convblock(
            torch.cat((intermediate, feature0, feature1, flow0, flow1), dim=1)
        )


class _Decoder1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.convblock = nn.Sequential(
            _conv_prelu(58, 54),
            _ResBlock(54, 24),
            nn.ConvTranspose2d(54, 5, 4, 2, 1, bias=True),
        )

    def forward(
        self,
        intermediate: Tensor,
        feature0: Tensor,
        feature1: Tensor,
        flow0: Tensor,
        flow1: Tensor,
    ) -> Tensor:
        feature0 = _match_decoder_size(_backward_warp(feature0, flow0), intermediate)
        feature1 = _match_decoder_size(_backward_warp(feature1, flow1), intermediate)
        flow0 = _match_decoder_size(flow0, intermediate)
        flow1 = _match_decoder_size(flow1, intermediate)
        return self.convblock(
            torch.cat((intermediate, feature0, feature1, flow0, flow1), dim=1)
        )


def _nearest_downsample(tensor: Tensor, scale_factor: int) -> Tensor:
    size = (tensor.size(2) // scale_factor, tensor.size(3) // scale_factor)
    return F.interpolate(tensor, size=size, mode="nearest")


def _nearest_upsample(tensor: Tensor, scale_factor: int) -> Tensor:
    return F.interpolate(tensor, scale_factor=scale_factor, mode="nearest")


def _compute_blend_mask(
    albedo: Tensor,
    normal: Tensor,
    half_resolution_prediction: Tensor,
    fallback_color: Tensor,
    *,
    scale_factor: int = 2,
    delta1: float = 0.1,
    delta2: float = 0.9,
) -> Tensor:
    def downsample_then_upsample(tensor: Tensor) -> Tensor:
        return _nearest_upsample(
            _nearest_downsample(tensor, scale_factor), scale_factor
        )

    upsampled_albedo = downsample_then_upsample(albedo)
    upsampled_normal = downsample_then_upsample(normal)
    upsampled_prediction = _nearest_upsample(
        half_resolution_prediction, scale_factor
    )

    target_size = fallback_color.shape[-2:]
    if upsampled_albedo.shape[-2:] != target_size:
        upsampled_albedo = F.interpolate(
            upsampled_albedo, size=target_size, mode="nearest"
        )
    if upsampled_normal.shape[-2:] != target_size:
        upsampled_normal = F.interpolate(
            upsampled_normal, size=target_size, mode="nearest"
        )
    if upsampled_prediction.shape[-2:] != target_size:
        upsampled_prediction = F.interpolate(
            upsampled_prediction, size=target_size, mode="nearest"
        )

    albedo_term = 1 - (1 / (3**0.5)) * torch.norm(
        upsampled_albedo - albedo, p=2, dim=1, keepdim=True
    )
    normal_dot = torch.sum(upsampled_normal * normal, dim=1, keepdim=True)
    mask1 = (albedo_term * normal_dot > delta1).float()

    difference = torch.norm(
        upsampled_prediction - fallback_color, p=1, dim=1, keepdim=True
    )
    prediction_norm = torch.norm(
        upsampled_prediction, p=1, dim=1, keepdim=True
    )
    mask2 = (difference / prediction_norm > delta2).float()
    return mask1 * mask2


def _blend_prediction(
    half_resolution_prediction: Tensor,
    fallback_color: Tensor,
    mask: Tensor,
) -> Tensor:
    upsampled = _nearest_upsample(half_resolution_prediction, 2)
    target_size = fallback_color.shape[-2:]
    if upsampled.shape[-2:] != target_size:
        upsampled = F.interpolate(upsampled, size=target_size, mode="nearest")
    if mask.shape[-2:] != target_size:
        mask = F.interpolate(mask, size=target_size, mode="nearest")
    return upsampled * mask + fallback_color * (1 - mask)


@dataclass
class ExtraSSResult:
    """ExtraSS prediction and optional target features used by its training loss."""

    pred_color: Tensor
    pred_mask: Tensor
    pred_features: tuple[Tensor, Tensor, Tensor]
    target_features: tuple[Tensor, Tensor, Tensor] | None = None


class ExtraSSNet(nn.Module):
    """ExtraSS baseline with an explicit, oracle-free prediction interface.

    Its feature encoder and flow decoder are adapted from IFRNet's MIT-licensed
    architecture; see ``THIRD_PARTY_NOTICES.md``.  ``target`` is optional and
    is encoded only to supply feature-pyramid supervision during training.  It
    never participates in computing ``pred_color`` or ``pred_mask``.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()
        self.decoder4 = _Decoder4()
        self.decoder3 = _Decoder3()
        self.decoder2 = _Decoder2()
        self.decoder1 = _Decoder1()

    @staticmethod
    def _validate_inputs(
        color_1: Tensor,
        color_3: Tensor,
        fallback_color: Tensor,
        albedo: Tensor,
        normal: Tensor,
        target: Tensor | None,
    ) -> None:
        named_tensors = (
            ("color_1", color_1),
            ("color_3", color_3),
            ("fallback_color", fallback_color),
            ("albedo", albedo),
            ("normal", normal),
        )
        if target is not None:
            named_tensors += (("target", target),)
        reference = None
        for name, tensor in named_tensors:
            if not isinstance(tensor, Tensor) or tensor.ndim != 4:
                raise ValueError(f"{name} must be a [B, 3, H, W] tensor")
            if tensor.shape[1] != 3:
                raise ValueError(f"{name} must have three channels, got {tensor.shape[1]}")
            current = (tensor.shape[0], *tensor.shape[-2:])
            if reference is None:
                reference = current
            elif current != reference:
                raise ValueError("all ExtraSS inputs must share batch and spatial dimensions")
            if tensor.device != color_1.device:
                raise ValueError("all ExtraSS inputs must be on the same device")
            if tensor.dtype != color_1.dtype:
                raise ValueError("all ExtraSS inputs must have the same dtype")
        if color_1.shape[-2] < 2 or color_1.shape[-1] < 2:
            raise ValueError("ExtraSS inputs must be at least 2 x 2 pixels")

    def _run(
        self,
        color_1: Tensor,
        color_3: Tensor,
        fallback_color: Tensor,
        albedo: Tensor,
        normal: Tensor,
        target: Tensor | None,
    ) -> ExtraSSResult:
        self._validate_inputs(color_1, color_3, fallback_color, albedo, normal, target)

        mean = (
            torch.cat((color_1, color_3), dim=2)
            .mean(1, keepdim=True)
            .mean(2, keepdim=True)
            .mean(3, keepdim=True)
        )
        color_1_half = F.pad(_nearest_downsample(color_1, 2), (0, 0, 4, 4)) - mean
        color_3_half = F.pad(_nearest_downsample(color_3, 2), (0, 0, 4, 4)) - mean

        target_features: tuple[Tensor, Tensor, Tensor] | None = None
        if target is not None:
            target_half = F.pad(_nearest_downsample(target, 2), (0, 0, 4, 4)) - mean
            target_feature1, target_feature2, target_feature3, _ = self.encoder(
                target_half
            )
            target_features = (
                target_feature1,
                target_feature2,
                target_feature3,
            )

        feature0_1, feature0_2, feature0_3, feature0_4 = self.encoder(color_1_half)
        feature1_1, feature1_2, feature1_3, feature1_4 = self.encoder(color_3_half)

        output4 = self.decoder4(feature0_4, feature1_4)
        flow0_4 = output4[:, 0:2]
        flow1_4 = output4[:, 2:4]
        predicted_feature3 = output4[:, 4:]

        output3 = self.decoder3(
            predicted_feature3, feature0_3, feature1_3, flow0_4, flow1_4
        )
        flow0_3 = output3[:, 0:2] + 2.0 * _resize(flow0_4, 2.0)
        flow1_3 = output3[:, 2:4] + 2.0 * _resize(flow1_4, 2.0)
        predicted_feature2 = output3[:, 4:]

        output2 = self.decoder2(
            predicted_feature2, feature0_2, feature1_2, flow0_3, flow1_3
        )
        flow0_2 = output2[:, 0:2] + 2.0 * _resize(flow0_3, 2.0)
        flow1_2 = output2[:, 2:4] + 2.0 * _resize(flow1_3, 2.0)
        predicted_feature1 = output2[:, 4:]

        output1 = self.decoder1(
            predicted_feature1, feature0_1, feature1_1, flow0_2, flow1_2
        )
        flow0_1 = output1[:, 0:2] + 2.0 * _resize(flow0_2, 2.0)
        flow1_1 = output1[:, 2:4] + 2.0 * _resize(flow1_2, 2.0)
        merge_mask = torch.sigmoid(output1[:, 4:5])

        color_1_warped = _match_decoder_size(
            _backward_warp(color_1_half, flow0_1), merge_mask
        )
        color_3_warped = _match_decoder_size(
            _backward_warp(color_3_half, flow1_1), merge_mask
        )
        if mean.shape[-2:] != merge_mask.shape[-2:]:
            mean = F.interpolate(
                mean,
                size=merge_mask.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        merged = merge_mask * color_1_warped + (1 - merge_mask) * color_3_warped + mean
        half_resolution_prediction = merged[:, :, 4:-4, :]

        blend_mask = _compute_blend_mask(
            albedo, normal, half_resolution_prediction, fallback_color
        )
        prediction = _blend_prediction(
            half_resolution_prediction, fallback_color, blend_mask
        )
        return ExtraSSResult(
            pred_color=prediction,
            pred_mask=blend_mask,
            pred_features=(
                predicted_feature1,
                predicted_feature2,
                predicted_feature3,
            ),
            target_features=target_features,
        )

    def forward(
        self,
        extra_ss_color_1: Tensor,
        extra_ss_color_3: Tensor,
        fallback_color: Tensor,
        albedo: Tensor,
        normal: Tensor,
        target: Tensor | None = None,
    ) -> ExtraSSResult:
        """Run ExtraSS; ``target`` is optional training metadata, never input."""

        return self._run(
            extra_ss_color_1,
            extra_ss_color_3,
            fallback_color,
            albedo,
            normal,
            target,
        )


class ExtraSSTbrNet(ExtraSSNet):
    """ExtraSS ablation driven by TBR rather than ExtraSS warp inputs.

    The network architecture is identical to :class:`ExtraSSNet` and retains
    its IFRNet-derived MIT attribution.  Unlike the legacy experiment loader,
    this public interface does not replace sky pixels with ground truth.
    """

    def forward(
        self,
        tbr_color_1: Tensor,
        tbr_color_3: Tensor,
        fallback_color: Tensor,
        albedo: Tensor,
        normal: Tensor,
        target: Tensor | None = None,
    ) -> ExtraSSResult:
        return self._run(
            tbr_color_1,
            tbr_color_3,
            fallback_color,
            albedo,
            normal,
            target,
        )


__all__ = ["ExtraSSNet", "ExtraSSResult", "ExtraSSTbrNet"]
