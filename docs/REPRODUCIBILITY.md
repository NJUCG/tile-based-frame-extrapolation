# Reproducibility scope

This release freezes the algorithmic implementation but does not distribute
the private scene dataset or scene-specific trained weights.

- The main network architecture and tiled execution path are public.
- The extracted implementation was numerically compared with the research
  code: network, tile residual, merged residual, and merged mask had maximum
  absolute difference 0 on fixed test inputs.
- Baseline architectures and the TBR/ExtraSS ablation are public.
- Tile-ablation parameters are exposed, but per-run checkpoints are not.
- Dataset splits and paper result tables cannot be independently reproduced
  from this repository alone.

For fair local evaluation, generate every method's prediction without ground
truth and pass the same valid-region mask to `tools/evaluate.py`. Do not save
predictions with sky or shadow pixels copied from the reference frame.
