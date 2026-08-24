"""Tile selection, extraction, and reconstruction for TBFE.

The selector operates on a regular ``tile_size`` grid.  A tile is evaluated
using its centre region and selected when at least ``min_ratio`` of that region
is marked as a hole.  ``tile_expand`` adds context around selected tiles without
changing the selection decision.

All functions support batches with independently selected tiles.  Spatial
inputs need not be multiples of ``tile_size``; right/bottom padding is removed
again during reconstruction.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class TileLayout:
    """Metadata shared by extraction and reconstruction.

    ``batch_indices`` and ``tile_indices`` have one entry per selected tile.
    Tile indices are row-major within each image's padded tile grid.
    """

    batch_size: int
    height: int
    width: int
    padded_height: int
    padded_width: int
    tile_size: int
    tile_expand: int
    tiles_y: int
    tiles_x: int
    batch_indices: Tensor
    tile_indices: Tensor

    @property
    def tile_extract(self) -> int:
        """Spatial size passed to the network for each selected tile."""

        return self.tile_size + 2 * self.tile_expand

    @property
    def tiles_per_image(self) -> int:
        return self.tiles_y * self.tiles_x

    @property
    def tile_count(self) -> int:
        return int(self.batch_indices.numel())

    @property
    def has_tiles(self) -> bool:
        return self.tile_count > 0


def _validate_image_tensor(tensor: Tensor, name: str) -> None:
    if not isinstance(tensor, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != 4:
        raise ValueError(f"{name} must have shape [B, C, H, W], got {tuple(tensor.shape)}")
    if tensor.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one image")
    if tensor.shape[-2] < 1 or tensor.shape[-1] < 1:
        raise ValueError(f"{name} must have non-empty spatial dimensions")


def _pad_spatial(tensor: Tensor, padding: tuple[int, int, int, int]) -> Tensor:
    """Pad with reflection when legal, otherwise replicate edge pixels."""

    if padding == (0, 0, 0, 0):
        return tensor

    left, right, top, bottom = padding
    height, width = tensor.shape[-2:]
    mode = "reflect"
    if max(left, right) >= width or max(top, bottom) >= height:
        mode = "replicate"
    return F.pad(tensor, padding, mode=mode)


def select_tiles(
    hole_mask: Tensor,
    *,
    tile_size: int = 64,
    min_ratio: float = 0.01,
    tile_expand: int = 0,
) -> TileLayout:
    """Select grid tiles based on the fraction of hole pixels.

    Args:
        hole_mask: ``[B, 1, H, W]`` tensor; values greater than zero are holes.
        tile_size: Size and stride of the regular selection grid.
        min_ratio: Inclusive minimum hole fraction for selecting a tile.
        tile_expand: Context pixels extracted on every side after selection.

    Returns:
        A :class:`TileLayout`, including an empty selection when no tile meets
        the threshold.
    """

    _validate_image_tensor(hole_mask, "hole_mask")
    if hole_mask.shape[1] != 1:
        raise ValueError(f"hole_mask must have one channel, got {hole_mask.shape[1]}")
    if tile_size <= 0:
        raise ValueError(f"tile_size must be positive, got {tile_size}")
    if tile_expand < 0:
        raise ValueError(f"tile_expand must be non-negative, got {tile_expand}")
    if not 0.0 <= min_ratio <= 1.0:
        raise ValueError(f"min_ratio must be in [0, 1], got {min_ratio}")

    batch_size, _, height, width = hole_mask.shape
    pad_h = (tile_size - height % tile_size) % tile_size
    pad_w = (tile_size - width % tile_size) % tile_size
    padded = _pad_spatial(hole_mask, (0, pad_w, 0, pad_h))
    padded_height, padded_width = padded.shape[-2:]
    tiles_y = padded_height // tile_size
    tiles_x = padded_width // tile_size

    # [B, 1, tiles_y, tiles_x, tile, tile] -> [B, L, 1, tile, tile]
    mask_tiles = padded.unfold(2, tile_size, tile_size).unfold(3, tile_size, tile_size)
    mask_tiles = mask_tiles.permute(0, 2, 3, 1, 4, 5).reshape(
        batch_size, tiles_y * tiles_x, 1, tile_size, tile_size
    )
    # Keep the original implementation's float32 ratio calculation even when
    # inference tensors use another floating-point dtype.
    ratios = (mask_tiles > 0).float().mean(dim=(2, 3, 4))
    selected = ratios >= min_ratio
    batch_indices, tile_indices = torch.where(selected)

    return TileLayout(
        batch_size=batch_size,
        height=height,
        width=width,
        padded_height=padded_height,
        padded_width=padded_width,
        tile_size=int(tile_size),
        tile_expand=int(tile_expand),
        tiles_y=tiles_y,
        tiles_x=tiles_x,
        batch_indices=batch_indices,
        tile_indices=tile_indices,
    )


def extract_tiles(tensor: Tensor, layout: TileLayout) -> Tensor:
    """Extract selected tiles from ``tensor`` according to ``layout``.

    The returned shape is ``[N, C, tile_extract, tile_extract]``.  ``N`` may be
    zero.  Tensor batch/spatial dimensions must match those used for selection.
    """

    _validate_image_tensor(tensor, "tensor")
    expected = (layout.batch_size, layout.height, layout.width)
    actual = (tensor.shape[0], tensor.shape[-2], tensor.shape[-1])
    if actual != expected:
        raise ValueError(
            "tensor does not match tile layout: expected "
            f"[B, H, W]={expected}, got {actual}"
        )

    pad_h = layout.padded_height - layout.height
    pad_w = layout.padded_width - layout.width
    padded = _pad_spatial(tensor, (0, pad_w, 0, pad_h))
    if layout.tile_expand:
        expand = layout.tile_expand
        padded = _pad_spatial(padded, (expand, expand, expand, expand))

    extract_size = layout.tile_extract
    windows = padded.unfold(2, extract_size, layout.tile_size).unfold(
        3, extract_size, layout.tile_size
    )
    windows = windows.permute(0, 2, 3, 1, 4, 5)

    if not layout.has_tiles:
        return tensor.new_empty((0, tensor.shape[1], extract_size, extract_size))

    tile_y = torch.div(layout.tile_indices, layout.tiles_x, rounding_mode="floor")
    tile_x = layout.tile_indices % layout.tiles_x
    return windows[layout.batch_indices, tile_y, tile_x].contiguous()


def merge_tiles(tiles: Tensor, layout: TileLayout) -> Tensor:
    """Average selected (possibly overlapping) tiles back into full images.

    Pixels not covered by any selected tile are zero.  For an empty selection,
    a correctly shaped all-zero tensor is returned.
    """

    if not isinstance(tiles, Tensor) or tiles.ndim != 4:
        shape = getattr(tiles, "shape", None)
        raise ValueError(f"tiles must have shape [N, C, H, W], got {shape}")
    if tiles.shape[-2] < 1 or tiles.shape[-1] < 1:
        raise ValueError("tiles must have non-empty spatial dimensions")
    expected_spatial = (layout.tile_extract, layout.tile_extract)
    if tiles.shape[0] != layout.tile_count:
        raise ValueError(
            f"expected {layout.tile_count} selected tiles, got {tiles.shape[0]}"
        )
    if tiles.shape[-2:] != expected_spatial:
        raise ValueError(
            f"expected tile shape {expected_spatial}, got {tuple(tiles.shape[-2:])}"
        )

    channels = tiles.shape[1]
    if not layout.has_tiles:
        return tiles.new_zeros((layout.batch_size, channels, layout.height, layout.width))

    tile_area = layout.tile_extract * layout.tile_extract
    tile_columns = tiles.new_zeros(
        (layout.batch_size, channels * tile_area, layout.tiles_per_image)
    )
    tile_columns[layout.batch_indices, :, layout.tile_indices] = tiles.reshape(
        layout.tile_count, channels * tile_area
    )

    output_size = (
        layout.padded_height + 2 * layout.tile_expand,
        layout.padded_width + 2 * layout.tile_expand,
    )
    summed = F.fold(
        tile_columns,
        output_size=output_size,
        kernel_size=layout.tile_extract,
        stride=layout.tile_size,
    )

    weight_columns = tiles.new_zeros(
        (layout.batch_size, tile_area, layout.tiles_per_image)
    )
    weight_columns[layout.batch_indices, :, layout.tile_indices] = 1
    weights = F.fold(
        weight_columns,
        output_size=output_size,
        kernel_size=layout.tile_extract,
        stride=layout.tile_size,
    )
    merged = summed / weights.clamp_min(1)
    merged = torch.where(weights > 0, merged, torch.zeros_like(merged))

    expand = layout.tile_expand
    merged = merged[
        :,
        :,
        expand : expand + layout.padded_height,
        expand : expand + layout.padded_width,
    ]
    return merged[:, :, : layout.height, : layout.width]


def merge_tiles_and_mask(
    tile_values: Tensor, tile_mask: Tensor, layout: TileLayout
) -> tuple[Tensor, Tensor]:
    """Merge predictions and their one-channel masks in a single fold."""

    if tile_mask.ndim != 4 or tile_mask.shape[1] != 1:
        raise ValueError(
            f"tile_mask must have shape [N, 1, H, W], got {tuple(tile_mask.shape)}"
        )
    if tile_values.shape[0] != tile_mask.shape[0] or tile_values.shape[-2:] != tile_mask.shape[-2:]:
        raise ValueError("tile_values and tile_mask must have matching tile dimensions")

    channels = tile_values.shape[1]
    merged = merge_tiles(torch.cat((tile_values, tile_mask), dim=1), layout)
    return merged[:, :channels], merged[:, channels:]


__all__ = [
    "TileLayout",
    "extract_tiles",
    "merge_tiles",
    "merge_tiles_and_mask",
    "select_tiles",
]
