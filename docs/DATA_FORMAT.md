# Processed data format

Each sequence is a flat directory of NumPy arrays:

```text
SequenceName/
  TbrIrradiance_1.0007.npy
  WarpToCurrGbufferMask_1.0007.npy
  Depth.0007.npy
  Normal.0007.npy
  Metallic.0007.npy
  Roughness.0007.npy
  Albedo.0007.npy
  Irradiance.0007.npy       # training only
  Color.0007.npy            # training/evaluation only
```

Arrays are uncompressed `.npy`, `float32`, and HWC. Frame identifiers may be
four or six digits as long as all buffers use the same identifier.

The loader converts names containing `Color` or `Irradiance` to
`log1p(max(value, 0))`. Depth greater than 50 is mapped to zero, matching the
paper training code. Other G-buffers remain linear.

Inference requires exactly seven buffers, listed in the root README. Ground
truth and `SkyMask`/`ShadowMask` are not model inputs.
