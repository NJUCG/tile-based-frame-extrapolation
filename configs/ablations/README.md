# Tile ablations

The public entry points expose the three tile parameters used by the paper
code:

- `tile_size`: regular selection-grid size and stride;
- `tile_expand`: context added on every side after tile selection;
- `min_ratio`: minimum invalid-pixel fraction for selecting a tile.

The paper default is `tile_size=64`, `tile_expand=8`, and `min_ratio=0.01`.
Override these fields in `configs/train.yaml` or pass the matching flags to
`tools/infer.py` and `tools/benchmark.py`. Per-ablation weights are not
distributed.
