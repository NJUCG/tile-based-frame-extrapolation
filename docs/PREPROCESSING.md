# TBR preprocessing

`tbfe.preprocess` separates portable buffer operations from the optional
OpenGL runtime.

## Warp input convention

`TBRWarpInputs` contains seven HWC float arrays of one resolution:

- previous irradiance, HWC3;
- backward motion in pixel units, HWC2;
- previous and current base color, HWC3 each;
- previous and current normal, HWC3 each;
- invalid-region mask, HWC1, where one means repair is required.

Motion `(dx, dy)` samples the previous image at `(x-dx, y-dy)`.

## API

```python
from tbfe.preprocess import TBRWarpInputs, warp_tbr

inputs = TBRWarpInputs(
    previous_irradiance=...,
    motion=...,
    previous_base_color=...,
    current_base_color=...,
    previous_normal=...,
    current_normal=...,
    invalid_mask=...,
)
warped = warp_tbr(inputs)
```

For sequences, reuse `TBRWarper` as a context manager. `warp_tbr` creates a
headless EGL/OpenGL 4.5 context by default; callers embedded in a renderer may
set `create_context=False`.

Helpers for BRDF demodulation, G-buffer-mask construction, motion accumulation,
EXR/NPY I/O, and renderer matrix metadata are documented in their docstrings.
Raw renderer export is engine-specific and is not distributed as data.
