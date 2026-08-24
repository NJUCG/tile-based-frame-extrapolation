"""Training losses for the TBFE repair network."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ImageInpaintLoss(nn.Module):
    """Paper-training loss for residual irradiance repair.

    ``mask == 1`` denotes a hole.  The loss matches the training implementation:
    hole reconstruction plus a known-region residual penalty.  A boundary loss
    is returned for diagnostics but is not added to ``total_loss`` by default,
    matching the released experiment code.

    The legacy ``w_pred_hole`` and ``w_comp_hole`` arguments are accepted so old
    experiment configurations still parse; the original forward pass used
    ``w_hole`` for its sole hole-reconstruction term.
    """

    def __init__(
        self,
        pixel_loss: str = "l1",
        w_hole: float = 1.0,
        w_pred_hole: float | None = 0.3,
        w_comp_hole: float | None = 1.0,
        w_edge: float = 0.1,
        w_alpha: float = 0.1,
        boundary_ignore: int = 2,
        eps: float = 1e-3,
        *,
        include_boundary_in_total: bool = False,
    ) -> None:
        super().__init__()
        if pixel_loss not in {"l1", "charbonnier"}:
            raise ValueError("pixel_loss must be 'l1' or 'charbonnier'")
        if boundary_ignore < 0:
            raise ValueError("boundary_ignore must be non-negative")

        self.pixel_loss = pixel_loss
        self.w_hole = float(w_hole)
        self.w_pred_hole = w_pred_hole
        self.w_comp_hole = w_comp_hole
        self.w_edge = float(w_edge)
        self.w_alpha = float(w_alpha)
        self.boundary_ignore = int(boundary_ignore)
        self.eps = float(eps)
        self.include_boundary_in_total = bool(include_boundary_in_total)

        # Retained for state/config compatibility with the experiment loss.
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _pixel_error(self, difference: Tensor) -> Tensor:
        if self.pixel_loss == "l1":
            return difference.abs()
        return torch.sqrt(difference.square() + self.eps * self.eps)

    @staticmethod
    def _boundary_mask(mask: Tensor, kernel_size: int = 3) -> Tensor:
        kernel = mask.new_ones((1, 1, kernel_size, kernel_size))
        convolved = F.conv2d(mask, kernel, padding=kernel_size // 2)
        dilated = (convolved > 0).to(mask.dtype)
        eroded = (convolved >= kernel_size * kernel_size).to(mask.dtype)
        return (dilated - eroded).clamp(0, 1)

    @staticmethod
    def _suppress_tile_border(mask: Tensor, margin: int) -> Tensor:
        if margin <= 0:
            return mask
        height, width = mask.shape[-2:]
        if height <= 2 * margin or width <= 2 * margin:
            return torch.zeros_like(mask)
        valid = torch.ones_like(mask)
        valid[..., :margin, :] = 0
        valid[..., -margin:, :] = 0
        valid[..., :, :margin] = 0
        valid[..., :, -margin:] = 0
        return mask * valid

    @staticmethod
    def _validate_inputs(
        residual: Tensor, input_color: Tensor, target: Tensor, mask: Tensor
    ) -> None:
        for name, tensor in (
            ("residual", residual),
            ("input_color", input_color),
            ("target", target),
            ("mask", mask),
        ):
            if not isinstance(tensor, Tensor) or tensor.ndim != 4:
                shape = getattr(tensor, "shape", None)
                raise ValueError(f"{name} must be a [N, C, H, W] tensor, got {shape}")
        if residual.shape != input_color.shape or residual.shape != target.shape:
            raise ValueError("residual, input_color, and target must have identical shapes")
        if mask.shape[0] != residual.shape[0] or mask.shape[-2:] != residual.shape[-2:]:
            raise ValueError("mask tile dimensions must match the image tensors")
        if mask.shape[1] != 1:
            raise ValueError(f"mask must have one channel, got {mask.shape[1]}")
        if not (
            residual.device == input_color.device == target.device == mask.device
        ):
            raise ValueError("all loss inputs must be on the same device")

    def forward(
        self,
        residual: Tensor,
        input_color: Tensor,
        target: Tensor,
        mask: Tensor,
    ) -> dict[str, Tensor | int]:
        """Compute loss terms for extracted training tiles.

        Empty tile tensors are valid.  They produce differentiable scalar zero
        losses so a generic training loop can safely handle all-valid batches.
        """

        self._validate_inputs(residual, input_color, target, mask)
        if residual.shape[0] == 0:
            zero = torch.zeros(
                (),
                device=residual.device,
                dtype=residual.dtype,
                requires_grad=torch.is_grad_enabled(),
            )
            return {
                "total_loss": zero,
                "loss_pred_hole": zero,
                "loss_boundary": zero,
                "loss_known_residual": zero,
                "tile_cnt": 0,
            }

        mask = mask.to(dtype=residual.dtype).clamp(0, 1)
        prediction = input_color + residual * mask
        difference = torch.nan_to_num(
            prediction - target, nan=0.0, posinf=1e6, neginf=-1e6
        )
        per_pixel_prediction = self._pixel_error(difference).mean(dim=1, keepdim=True)

        hole_pixels = mask.sum().clamp_min(1.0)
        loss_pred_hole = (per_pixel_prediction * mask).sum() / hole_pixels

        loss_boundary = residual.new_zeros(())
        if self.w_edge > 0:
            boundary = self._boundary_mask(mask)
            boundary = self._suppress_tile_border(boundary, self.boundary_ignore)
            boundary = boundary * mask
            boundary_pixels = boundary.sum().clamp_min(1.0)
            loss_boundary = (per_pixel_prediction * boundary).sum() / boundary_pixels

        loss_known_residual = residual.new_zeros(())
        if self.w_alpha > 0:
            known_mask = 1.0 - mask
            per_pixel_residual = self._pixel_error(residual).mean(dim=1, keepdim=True)
            known_pixels = known_mask.sum().clamp_min(1.0)
            loss_known_residual = (
                per_pixel_residual * known_mask
            ).sum() / known_pixels

        total_loss = (
            self.w_hole * loss_pred_hole
            + self.w_alpha * loss_known_residual
        )
        if self.include_boundary_in_total:
            total_loss = total_loss + self.w_edge * loss_boundary

        return {
            "total_loss": torch.nan_to_num(
                total_loss, nan=0.0, posinf=1e6, neginf=-1e6
            ),
            "loss_pred_hole": loss_pred_hole,
            "loss_boundary": loss_boundary,
            "loss_known_residual": loss_known_residual,
            "tile_cnt": residual.shape[0],
        }


class ImageInpaintNetLoss(ImageInpaintLoss):
    """Backward-compatible class name used by the experiment configuration."""


__all__ = ["ImageInpaintLoss", "ImageInpaintNetLoss"]
