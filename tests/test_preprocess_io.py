from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tbfe.preprocess import (
    INFERENCE_BUFFER_NAMES,
    TBRWarpInputs,
    brdf_lut_path,
    discover_frame_indices,
    frame_filename,
    frame_path,
    load_frame_buffers,
    load_shader_source,
    parse_matrix_metadata,
    read_npy,
    shader_path,
    validate_buffer_set,
    view_projection_matrix,
    write_npy,
)


def _inference_buffers(height: int = 3, width: int = 5) -> dict[str, np.ndarray]:
    channels = {
        "TbrIrradiance_1": 3,
        "WarpToCurrGbufferMask_1": 1,
        "Depth": 1,
        "Normal": 3,
        "Metallic": 1,
        "Roughness": 1,
        "Albedo": 3,
    }
    return {
        name: np.zeros((height, width, channel_count), dtype=np.float32)
        for name, channel_count in channels.items()
    }


def test_frame_names_are_explicit_and_stable(tmp_path: Path) -> None:
    assert frame_filename("Depth", 7) == "Depth.0007.npy"
    assert frame_filename("Depth", 7, digits=6) == "Depth.000007.npy"
    assert frame_filename("Depth", 10001, "exr") == "Depth.10001.exr"
    assert frame_path(tmp_path, "Normal", 12) == tmp_path / "Normal.0012.npy"

    with pytest.raises(ValueError, match="basename"):
        frame_filename("nested/Depth", 7)
    with pytest.raises(ValueError, match="non-negative"):
        frame_filename("Depth", -1)


def test_npy_round_trip_and_frame_discovery(tmp_path: Path) -> None:
    depth = np.arange(12, dtype=np.float32).reshape(3, 4, 1)
    path = frame_path(tmp_path, "Depth", 9)
    assert write_npy(path, depth, buffer_name="Depth") == path
    np.testing.assert_array_equal(read_npy(path, buffer_name="Depth"), depth)

    write_npy(frame_path(tmp_path, "Depth", 2), depth, buffer_name="Depth")
    write_npy(
        frame_path(tmp_path, "Metallic", 1),
        np.zeros_like(depth),
        buffer_name="Metallic",
    )
    assert discover_frame_indices(tmp_path, "Depth") == [2, 9]


def test_load_and_validate_inference_frame(tmp_path: Path) -> None:
    expected = _inference_buffers()
    for name, array in expected.items():
        write_npy(frame_path(tmp_path, name, 3), array, buffer_name=name)

    loaded = load_frame_buffers(tmp_path, 3, INFERENCE_BUFFER_NAMES)
    assert tuple(loaded) == INFERENCE_BUFFER_NAMES
    assert validate_buffer_set(loaded) == (3, 5)

    loaded["Depth"] = np.zeros((2, 5, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="spatial shape"):
        validate_buffer_set(loaded)


def test_matrix_metadata_parser_uses_named_sections() -> None:
    text = """
ClipToView: [1 0 0 0] [0 1 0 0] [0 0 1 0] [0 0 0 1]
ViewMatrix: [1 0 0 0] [0 2 0 0] [0 0 3 0] [4 5 6 1]
ProjectionMatrix: [2 0 0 0] [0 3 0 0] [0 0 4 0] [0 0 0 1]
FOV: 90
"""
    matrices = parse_matrix_metadata(text)
    assert set(matrices) == {"ClipToView", "ViewMatrix", "ProjectionMatrix"}
    np.testing.assert_allclose(
        view_projection_matrix(matrices),
        matrices["ViewMatrix"] @ matrices["ProjectionMatrix"],
    )


def test_resources_do_not_depend_on_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert shader_path().is_file()
    assert brdf_lut_path().is_file()
    source = load_shader_source()
    assert "layout(local_size_x = 16" in source
    assert "offset.x >= frameSize.x" in source


def test_tbr_input_validation_does_not_open_opengl() -> None:
    shape = (2, 4)
    inputs = TBRWarpInputs(
        previous_irradiance=np.zeros((*shape, 3), dtype=np.float64),
        motion=np.zeros((*shape, 2), dtype=np.float64),
        previous_base_color=np.zeros((*shape, 3), dtype=np.float64),
        current_base_color=np.zeros((*shape, 3), dtype=np.float64),
        previous_normal=np.zeros((*shape, 3), dtype=np.float64),
        current_normal=np.zeros((*shape, 3), dtype=np.float64),
        invalid_mask=np.zeros((*shape, 1), dtype=np.float64),
    ).validated()
    assert inputs.motion.dtype == np.float32
    assert inputs.motion.flags.c_contiguous


def test_package_import_does_not_require_pyopengl() -> None:
    # A clean interpreter demonstrates that public preprocessing imports before
    # the optional OpenGL backend is loaded.
    script = (
        "import sys; import tbfe.preprocess; "
        "assert 'OpenGL.GL' not in sys.modules; "
        "assert 'OpenGL.EGL' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", script], check=True)
