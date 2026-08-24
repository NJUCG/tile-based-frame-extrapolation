"""Preprocessing and G-buffer-guided warp APIs for Tile-based Frame Extrapolation.

Importing this package is safe without OpenGL. The graphics runtime is loaded
only when :class:`TBRWarper` is opened.
"""

from .io import (
    discover_frame_indices,
    frame_filename,
    frame_path,
    load_frame_buffers,
    parse_matrix_metadata,
    read_brdf_lut,
    read_exr,
    read_matrix_metadata,
    read_npy,
    view_projection_matrix,
    write_exr,
    write_npy,
)
from .ops import (
    accumulate_backward_motion,
    backward_warp,
    compute_gbuffer_mask,
    demodulate_brdf,
    dilate_mask,
)
from .resources import brdf_lut_path, load_shader_source, shader_path
from .schema import (
    AUXILIARY_BUFFER_NAMES,
    BUFFER_SPECS,
    EVALUATION_ONLY_NAMES,
    INFERENCE_BUFFER_NAMES,
    TRAINING_TARGET_NAMES,
    BufferSpec,
    validate_buffer_array,
    validate_buffer_set,
)
from .warp import TBRWarper, TBRWarpInputs, warp_tbr

__all__ = [
    "AUXILIARY_BUFFER_NAMES",
    "BUFFER_SPECS",
    "EVALUATION_ONLY_NAMES",
    "INFERENCE_BUFFER_NAMES",
    "TRAINING_TARGET_NAMES",
    "BufferSpec",
    "TBRWarper",
    "TBRWarpInputs",
    "accumulate_backward_motion",
    "backward_warp",
    "brdf_lut_path",
    "compute_gbuffer_mask",
    "demodulate_brdf",
    "dilate_mask",
    "discover_frame_indices",
    "frame_filename",
    "frame_path",
    "load_frame_buffers",
    "load_shader_source",
    "parse_matrix_metadata",
    "read_brdf_lut",
    "read_exr",
    "read_matrix_metadata",
    "read_npy",
    "shader_path",
    "validate_buffer_array",
    "validate_buffer_set",
    "view_projection_matrix",
    "warp_tbr",
    "write_exr",
    "write_npy",
]
