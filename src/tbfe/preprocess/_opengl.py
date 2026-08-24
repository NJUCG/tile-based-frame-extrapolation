"""Lazy OpenGL backend for :mod:`tbfe.preprocess.warp`.

This is an implementation detail.  The module itself does not import PyOpenGL;
bindings are loaded only when :class:`OpenGLTBRBackend` is opened.
"""

from __future__ import annotations

import ctypes
import os
from typing import Any

import numpy as np

from .resources import load_shader_source


def _load_bindings(*, egl: bool):
    if egl:
        # PyOpenGL chooses its platform at first import. Do not overwrite a
        # caller's explicit selection, but make headless operation the default.
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    try:
        import OpenGL.GL as gl  # type: ignore
        import OpenGL.EGL as egl_module  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "TBR warp requires PyOpenGL and a driver exposing EGL/OpenGL 4.5"
        ) from exc
    return gl, egl_module


def _decode_log(log: Any) -> str:
    return log.decode("utf-8", errors="replace") if isinstance(log, bytes) else str(log)


def _pad_rgba(array: np.ndarray) -> np.ndarray:
    height, width, channels = array.shape
    if channels > 4:
        raise ValueError(f"OpenGL input may have at most four channels, got {channels}")
    if channels == 4:
        return np.ascontiguousarray(array, dtype=np.float32)
    result = np.ones((height, width, 4), dtype=np.float32)
    result[..., :channels] = array
    return result


class OpenGLTBRBackend:
    """Compile and dispatch the packaged TBR shader on a current GL context."""

    def __init__(self, *, create_context: bool = True) -> None:
        self.create_context = create_context
        self.gl: Any | None = None
        self.egl: Any | None = None
        self.display: Any | None = None
        self.surface: Any | None = None
        self.context: Any | None = None
        self.program: int | None = None

    def open(self) -> None:
        if self.gl is not None:
            raise RuntimeError("OpenGL TBR backend is already open")
        self.gl, self.egl = _load_bindings(egl=self.create_context)
        try:
            if self.create_context:
                self._create_egl_context()
            self.program = self._compile_program(load_shader_source())
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        gl, egl = self.gl, self.egl
        if gl is not None and self.program is not None:
            try:
                gl.glDeleteProgram(self.program)
            finally:
                self.program = None

        if egl is not None and self.create_context and self.display is not None:
            try:
                egl.eglMakeCurrent(
                    self.display,
                    egl.EGL_NO_SURFACE,
                    egl.EGL_NO_SURFACE,
                    egl.EGL_NO_CONTEXT,
                )
                if self.context not in (None, egl.EGL_NO_CONTEXT):
                    egl.eglDestroyContext(self.display, self.context)
                if self.surface not in (None, egl.EGL_NO_SURFACE):
                    egl.eglDestroySurface(self.display, self.surface)
                egl.eglTerminate(self.display)
            finally:
                self.display = self.surface = self.context = None

        self.gl = None
        self.egl = None

    def _create_egl_context(self) -> None:
        assert self.egl is not None
        egl = self.egl
        display = egl.eglGetDisplay(egl.EGL_DEFAULT_DISPLAY)
        if display == egl.EGL_NO_DISPLAY:
            raise RuntimeError("EGL could not obtain a display")
        self.display = display

        major, minor = egl.EGLint(), egl.EGLint()
        if not egl.eglInitialize(display, ctypes.byref(major), ctypes.byref(minor)):
            raise RuntimeError("EGL initialization failed")
        if not egl.eglBindAPI(egl.EGL_OPENGL_API):
            raise RuntimeError("EGL could not bind the desktop OpenGL API")

        config_attributes = (egl.EGLint * 15)(
            egl.EGL_SURFACE_TYPE,
            egl.EGL_PBUFFER_BIT,
            egl.EGL_RED_SIZE,
            8,
            egl.EGL_GREEN_SIZE,
            8,
            egl.EGL_BLUE_SIZE,
            8,
            egl.EGL_ALPHA_SIZE,
            8,
            egl.EGL_RENDERABLE_TYPE,
            egl.EGL_OPENGL_BIT,
            egl.EGL_NONE,
            egl.EGL_NONE,
            egl.EGL_NONE,
        )
        config = egl.EGLConfig()
        count = egl.EGLint()
        if not egl.eglChooseConfig(
            display,
            config_attributes,
            ctypes.byref(config),
            1,
            ctypes.byref(count),
        ) or count.value < 1:
            raise RuntimeError("EGL could not select an OpenGL pbuffer configuration")

        pbuffer_attributes = (egl.EGLint * 5)(
            egl.EGL_WIDTH,
            1,
            egl.EGL_HEIGHT,
            1,
            egl.EGL_NONE,
        )
        surface = egl.eglCreatePbufferSurface(display, config, pbuffer_attributes)
        if surface == egl.EGL_NO_SURFACE:
            raise RuntimeError("EGL could not create a pbuffer surface")
        self.surface = surface

        context_attributes = (egl.EGLint * 5)(
            egl.EGL_CONTEXT_MAJOR_VERSION,
            4,
            egl.EGL_CONTEXT_MINOR_VERSION,
            5,
            egl.EGL_NONE,
        )
        context = egl.eglCreateContext(
            display,
            config,
            egl.EGL_NO_CONTEXT,
            context_attributes,
        )
        if context == egl.EGL_NO_CONTEXT:
            raise RuntimeError("EGL could not create an OpenGL 4.5 context")
        self.context = context
        if not egl.eglMakeCurrent(display, surface, surface, context):
            raise RuntimeError("EGL could not make the OpenGL context current")

    def _compile_program(self, source: str) -> int:
        assert self.gl is not None
        gl = self.gl
        shader = gl.glCreateShader(gl.GL_COMPUTE_SHADER)
        try:
            gl.glShaderSource(shader, source)
            gl.glCompileShader(shader)
            if not gl.glGetShaderiv(shader, gl.GL_COMPILE_STATUS):
                raise RuntimeError(
                    "TBR compute shader compilation failed:\n"
                    + _decode_log(gl.glGetShaderInfoLog(shader))
                )

            program = gl.glCreateProgram()
            gl.glAttachShader(program, shader)
            gl.glLinkProgram(program)
            if not gl.glGetProgramiv(program, gl.GL_LINK_STATUS):
                log = _decode_log(gl.glGetProgramInfoLog(program))
                gl.glDeleteProgram(program)
                raise RuntimeError(f"TBR compute shader linking failed:\n{log}")
            return int(program)
        finally:
            gl.glDeleteShader(shader)

    def _create_texture(self, data: np.ndarray | None, width: int, height: int) -> int:
        assert self.gl is not None
        gl = self.gl
        texture = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA32F,
            width,
            height,
            0,
            gl.GL_RGBA,
            gl.GL_FLOAT,
            data,
        )
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        return texture

    def _read_texture(self, texture: int, width: int, height: int) -> np.ndarray:
        assert self.gl is not None
        gl = self.gl
        framebuffer = int(gl.glGenFramebuffers(1))
        try:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, framebuffer)
            gl.glFramebufferTexture2D(
                gl.GL_FRAMEBUFFER,
                gl.GL_COLOR_ATTACHMENT0,
                gl.GL_TEXTURE_2D,
                texture,
                0,
            )
            status = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
            if status != gl.GL_FRAMEBUFFER_COMPLETE:
                raise RuntimeError(f"OpenGL framebuffer is incomplete (status={status:#x})")
            raw = gl.glReadPixels(0, 0, width, height, gl.GL_RGBA, gl.GL_FLOAT)
            if isinstance(raw, np.ndarray):
                result = np.asarray(raw, dtype=np.float32)
            else:
                result = np.frombuffer(raw, dtype=np.float32)
            return result.reshape(height, width, 4).copy()
        finally:
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            gl.glDeleteFramebuffers(1, [framebuffer])

    def warp(self, inputs) -> np.ndarray:
        if self.gl is None or self.program is None:
            raise RuntimeError("OpenGL TBR backend is not open")
        gl = self.gl
        height, width = inputs.previous_irradiance.shape[:2]
        arrays = (
            inputs.previous_irradiance,
            inputs.motion,
            inputs.previous_base_color,
            inputs.current_base_color,
            inputs.previous_normal,
            inputs.current_normal,
            inputs.invalid_mask,
        )
        textures: list[int] = []
        output_texture: int | None = None
        try:
            textures = [
                self._create_texture(_pad_rgba(array), width, height) for array in arrays
            ]
            output_texture = self._create_texture(None, width, height)

            gl.glUseProgram(self.program)
            for binding, texture in enumerate(textures):
                gl.glActiveTexture(gl.GL_TEXTURE0 + binding)
                gl.glBindTexture(gl.GL_TEXTURE_2D, texture)
            gl.glBindImageTexture(
                0,
                output_texture,
                0,
                gl.GL_FALSE,
                0,
                gl.GL_WRITE_ONLY,
                gl.GL_RGBA32F,
            )
            gl.glDispatchCompute((width + 15) // 16, (height + 15) // 16, 1)
            gl.glMemoryBarrier(
                gl.GL_SHADER_IMAGE_ACCESS_BARRIER_BIT | gl.GL_TEXTURE_FETCH_BARRIER_BIT
            )
            gl.glFinish()
            return np.ascontiguousarray(
                self._read_texture(output_texture, width, height)[..., :3],
                dtype=np.float32,
            )
        finally:
            for binding in range(len(textures)):
                gl.glActiveTexture(gl.GL_TEXTURE0 + binding)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glBindImageTexture(
                0,
                0,
                0,
                gl.GL_FALSE,
                0,
                gl.GL_WRITE_ONLY,
                gl.GL_RGBA32F,
            )
            to_delete = textures + ([] if output_texture is None else [output_texture])
            if to_delete:
                gl.glDeleteTextures(to_delete)
