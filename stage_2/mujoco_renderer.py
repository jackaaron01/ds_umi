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

    def render(self, data: mujoco.MjData = None,
               camera: str = "fixed") -> np.ndarray:
        """Render current scene and return RGB image.

        Args:
            data: MuJoCo data (if None, uses internal data — update qpos first)
            camera: Camera name to use ("fixed" or "ego")

        Returns:
            RGB image as uint8 numpy array (height, width, 3)
        """
        if data is not None:
            self._data.qpos[:] = data.qpos[:]
            self._data.qvel[:] = data.qvel[:] if data.qvel is not None else 0
            mujoco.mj_forward(self._model, self._data)

        # Find camera ID by name
        cam_id = -1  # free camera
        for i in range(self._model.ncam):
            if self._model.camera(i).name == camera:
                cam_id = i
                break

        return self._viewer.read_pixels(camid=cam_id)

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def close(self):
        """Close the viewer — keep it simple to avoid segfaults.

        The MujocoViewer internally manages GLFW. We just mark it
        as not alive. The GL context/window will be cleaned up on
        process exit.
        """
        if hasattr(self, '_viewer') and self._viewer:
            try:
                self._viewer.is_alive = False
            except Exception:
                pass

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
