import pytest
import torch

from tbfe.ops import extract_tiles, merge_tiles, select_tiles


def test_selection_is_independent_for_each_batch_item():
    mask = torch.zeros(2, 1, 8, 8)
    mask[0, :, :4, :4] = 1
    mask[1, :, 4:, 4:] = 1

    layout = select_tiles(mask, tile_size=4, min_ratio=0.5, tile_expand=1)

    assert layout.batch_indices.tolist() == [0, 1]
    assert layout.tile_indices.tolist() == [0, 3]
    assert layout.tile_extract == 6
    assert layout.tile_count == 2


@pytest.mark.parametrize("tile_expand", [0, 2])
def test_extract_merge_round_trip_with_non_multiple_image(tile_expand):
    torch.manual_seed(4)
    image = torch.randn(2, 3, 5, 7)
    mask = torch.ones(2, 1, 5, 7)
    layout = select_tiles(
        mask,
        tile_size=4,
        min_ratio=0.01,
        tile_expand=tile_expand,
    )

    tiles = extract_tiles(image, layout)
    reconstructed = merge_tiles(tiles, layout)

    assert tiles.shape == (8, 3, 4 + 2 * tile_expand, 4 + 2 * tile_expand)
    torch.testing.assert_close(reconstructed, image)


def test_empty_selection_has_well_formed_empty_tiles_and_zero_merge():
    image = torch.randn(3, 2, 9, 11)
    mask = torch.zeros(3, 1, 9, 11)
    layout = select_tiles(mask, tile_size=8, min_ratio=0.01, tile_expand=2)

    tiles = extract_tiles(image, layout)
    reconstructed = merge_tiles(tiles, layout)

    assert not layout.has_tiles
    assert tiles.shape == (0, 2, 12, 12)
    assert reconstructed.shape == image.shape
    assert torch.count_nonzero(reconstructed) == 0


def test_extract_rejects_tensor_from_another_layout_shape():
    mask = torch.ones(1, 1, 8, 8)
    layout = select_tiles(mask, tile_size=4)

    with pytest.raises(ValueError, match="does not match tile layout"):
        extract_tiles(torch.ones(1, 3, 7, 8), layout)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tile_size": 0}, "tile_size"),
        ({"tile_expand": -1}, "tile_expand"),
        ({"min_ratio": 1.1}, "min_ratio"),
    ],
)
def test_selection_validates_configuration(kwargs, message):
    with pytest.raises(ValueError, match=message):
        select_tiles(torch.ones(1, 1, 4, 4), **kwargs)
