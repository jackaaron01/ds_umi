#!/usr/bin/env python3
"""
Working offscreen MuJoCo renderer — thin wrapper around mujoco-python-viewer.

The mujoco-python-viewer's offscreen mode works reliably (verified with
Mesa llvmpipe), while raw mjr_render + mjFB_OFFSCREEN produces black images.

Usage:
    renderer = MuJoCoRenderer(model, width=320, height=240)
    img = renderer.render(data)
    renderer.close()
"""
import numpy as np
import mujoco
from mujoco_viewer import MujocoViewer


class MuJoCoRenderer:
    """Offscreen MuJoCo renderer using mujoco-python-viewer internally."""

    def __init__(self, model: mujoco.MjModel, width: int = 320, height: int = 240):
        self._model = model
        self._width = width
        self._height = height
        # Create a dummy data for viewer initialization (will be updated per-frame)
        self._data = mujoco.MjData(model)
        mujoco.mj_forward(model, self._data)
        self._viewer = MujocoViewer(model, self._data, mode="offscreen",
                                    width=width, height=height,
                                    title="umi_renderer")

    def render(self, data: mujoco.MjData = None) -> np.ndarray:
        """Render current scene and return RGB image.

        Args:
            data: MuJoCo data (if None, uses internal data — update qpos first)

        Returns:
            RGB image as uint8 numpy array (height, width, 3)
        """
        if data is not None:
            # Copy joint positions to internal data
            self._data.qpos[:] = data.qpos[:]
            self._data.qvel[:] = data.qvel[:] if data.qvel is not None else 0
            mujoco.mj_forward(self._model, self._data)

        return self._viewer.read_pixels()

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def close(self):
        """Close the viewer but DON'T terminate GLFW globally.

        MujocoViewer.close() calls glfw.terminate() which kills the GLFW
        library globally, preventing any future windows. We manually clean
        up the context and window instead.
        """
        if hasattr(self, '_viewer') and self._viewer:
            v = self._viewer
            if hasattr(v, 'ctx') and v.ctx:
                v.ctx.free()
            if hasattr(v, 'window') and v.window:
                from mujoco.glfw import glfw
                glfw.destroy_window(v.window)
            v.is_alive = False  # prevent double-close

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
