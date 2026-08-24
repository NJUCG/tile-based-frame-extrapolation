"""Paths and readers for resources shipped with the TBFE Python package."""

from __future__ import annotations

from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent


def shader_path() -> Path:
    """Return the package-relative path of the TBR compute shader."""

    path = _PACKAGE_DIR / "shaders" / "GGWarp.comp"
    if not path.is_file():
        raise FileNotFoundError(
            f"TBR shader is missing from the installation: {path}. "
            "Install the package with its package data."
        )
    return path


def brdf_lut_path() -> Path:
    """Return the package-relative path of the precomputed BRDF lookup table."""

    path = _PACKAGE_DIR / "assets" / "Precomputed.exr"
    if not path.is_file():
        raise FileNotFoundError(
            f"BRDF lookup table is missing from the installation: {path}. "
            "Install the package with its package data."
        )
    return path


def load_shader_source() -> str:
    """Load the compute shader without relying on the process working directory."""

    return shader_path().read_text(encoding="utf-8")
