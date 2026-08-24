# Model card

## Intended use

The tiled repair network fills invalid irradiance regions produced by TBR in a
rendered-frame extrapolation pipeline. It expects the seven renderer buffers
documented in `DATA_FORMAT.md`.

## Weights

No pretrained weights are distributed. The authors found that trained models
are scene-dependent. Users should train and validate on their own rendering
distribution.

## Limitations

- Randomly initialized smoke-test output is not representative.
- Generalization across engines, content distributions, and G-buffer
  conventions has not been established.
- The model does not reconstruct sky/shadow pixels from ground truth during
  inference.
- Runtime depends on the number of selected tiles.

## Evaluation

The repository reports PSNR/SSIM and optional LPIPS using a caller-supplied
valid mask. Any exclusions must be identical for all compared methods.
