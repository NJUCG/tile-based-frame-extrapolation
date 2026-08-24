# TBFE preprocessing API

This package contains the paper's G-buffer-guided TBR warp and the small set of
portable preprocessing operations needed to prepare its inputs. It deliberately
does not depend on the private training dataset layout.

## Processed frame format

Each buffer is a NumPy `float32` array in `H x W x C` layout and is named
`BufferName.0001.npy`. `schema.py` is the authoritative list of channels and
roles. Ground-truth `Color` and `Irradiance` are training targets,
`ShadowMask` is evaluation-only, and `SkyMask` is an optional renderer
auxiliary; none is required by public inference.

Renderer motion vectors are backward vectors in pixel units. At pixel `(x, y)`,
`(dx, dy)` samples `(x - dx, y - dy)`. The invalid-region mask uses one for a
pixel requiring repair and zero for a directly reusable pixel.

## TBR warp

The OpenGL backend is lazy so importing `tbfe.preprocess` and using its NumPy
I/O does not require OpenGL:

```python
from tbfe.preprocess import TBRWarper, TBRWarpInputs

inputs = TBRWarpInputs(
    previous_irradiance=previous_irradiance,
    motion=motion,
    previous_base_color=previous_base_color,
    current_base_color=current_base_color,
    previous_normal=previous_normal,
    current_normal=current_normal,
    invalid_mask=invalid_mask,
)

with TBRWarper() as warper:  # creates a headless EGL context
    warped_irradiance = warper.warp(inputs)
```

Reuse one `TBRWarper` for an entire sequence. A caller with an existing current
OpenGL 4.5 context can pass `create_context=False`.

The compute shader and BRDF LUT are resolved relative to the installed Python
package, never relative to the shell working directory. Packaging configuration
must include `shaders/GGWarp.comp` and `assets/Precomputed.exr`.
