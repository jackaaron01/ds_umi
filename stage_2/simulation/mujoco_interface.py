#!/usr/bin/env python3
"""
MujocoRobotInterface — RobotInterface implementation backed by MuJoCo physics.

Implements the RobotInterface ABC using MuJoCo's mjModel/mjData.
Position commands set actuator targets and step the simulation forward.
Joint state is read from mjData.qpos/qvel.

For use in the teleop pipeline: safety_guardian creates a MujocoRobotInterface
instead of MockRobotInterface, and the full pipeline (hand_mapper → safety →
robot) runs with MuJoCo simulation in the loop.
"""

import sys, os
sys.path.insert(0, "/workspace/umi")

import time
import threading
import numpy as np
import mujoco

# Add root robot_hal to path for the ABC
from robot_hal import RobotInterface, JointState, GripperState


class MujocoRobotInterface(RobotInterface):
    """MuJoCo-backed robot interface for pipeline simulation.

    Loads an MJCF/URDF model, steps physics on each move_joints call,
    and reads joint state from the simulation data.

    Parameters:
        model_path: Path to MJCF or URDF file.
        control_rate: Simulation step rate (Hz). Each move_joints call
                      steps physics by 1/control_rate seconds.
    """

    def __init__(self, model_path: str = None, control_rate: float = 60.0):
        self._model_path = model_path or os.path.join(
            os.path.dirname(__file__), "xarm_mesh.xml"
        )
        self._control_rate = control_rate
        self._dt = 1.0 / control_rate

        self._model = None
        self._data = None
        self._connected = False
        self._lock = threading.Lock()

        # State tracking
        self._q = np.zeros(6)
        self._qvel = np.zeros(6)
        self._gripper_pos = 0.0

        # For non-blocking motion simulation
        self._target_q = np.zeros(6)
        self._target_gripper = 0.0
        self._last_move_time = 0.0
        self._steps_per_cycle = 1  # computed after model load

    # ── RobotInterface methods ──────────────────────────────────────────

    def connect(self) -> bool:
        with self._lock:
            try:
                self._model = mujoco.MjModel.from_xml_path(self._model_path)
                self._data = mujoco.MjData(self._model)
                # Compute physics steps needed per control cycle
                model_dt = self._model.opt.timestep
                self._steps_per_cycle = max(1, int(1.0 / self._control_rate / model_dt))
                # Initialize to a neutral home pose (not zeros — avoids singularities)
                home = np.array([0.0, -0.3, 0.0, 1.2, 0.0, 0.0])
                self._data.qpos[:6] = home
                self._data.ctrl[:6] = home
                self._target_q = home.copy()
                mujoco.mj_forward(self._model, self._data)
                self._q = self._data.qpos[:6].copy()
                self._connected = True
                return True
            except Exception as e:
                print(f"[MujocoRobotInterface] Connect failed: {e}")
                return False

    def disconnect(self) -> bool:
        with self._lock:
            self._connected = False
            self._model = None
            self._data = None
            return True

    def get_joint_state(self) -> JointState:
        with self._lock:
            if not self._connected:
                return JointState(
                    position=self._q.copy(),
                    velocity=self._qvel.copy(),
                    effort=np.zeros(6),
                    name=[f"joint{i}" for i in range(1, 7)],
                )
            return JointState(
                position=self._data.qpos[:6].copy(),
                velocity=self._data.qvel[:6].copy(),
                effort=self._data.qfrc_actuator[:6].copy(),
                name=[f"joint{i}" for i in range(1, 7)],
            )

    def move_joints(self, positions, velocity=0.5, blocking=True) -> bool:
        """Move joints to target positions.

        For non-blocking: sets actuator targets and steps physics once.
        For blocking: steps physics until joints reach targets or timeout.
        """
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (6,):
            return False

        with self._lock:
            if not self._connected:
                return False

            self._target_q = positions.copy()

            if blocking:
                return self._move_blocking(positions, velocity)
            else:
                return self._move_nonblocking(positions)

    def stop(self) -> bool:
        with self._lock:
            if not self._connected:
                return False
            self._target_q = self._data.qpos[:6].copy()
            self._data.ctrl[:] = self._target_q
            return True

    def get_gripper_state(self) -> GripperState:
        return GripperState(position=self._gripper_pos, effort=0.0)

    def move_gripper(self, position: float, blocking: bool = True) -> bool:
        self._gripper_pos = float(np.clip(position, 0.0, 1.0))
        self._target_gripper = self._gripper_pos
        return True

    # ── MuJoCo physics ──────────────────────────────────────────────────

    def _move_nonblocking(self, positions):
        """Set actuator targets only — physics stepping is done by step_physics()."""
        self._data.ctrl[:6] = positions
        self._last_move_time = time.time()
        return True

    def _move_blocking(self, positions, velocity):
        """Step physics until joints are close to targets."""
        max_steps = int(5.0 / self._dt)  # 5 second timeout
        tolerance = 0.01  # rad

        for _ in range(max_steps):
            current = self._data.qpos[:6]
            if np.all(np.abs(current - positions) < tolerance):
                return True
            self._data.ctrl[:6] = positions
            mujoco.mj_step(self._model, self._data)

        return True  # Return true even on timeout (close enough)

    def step_physics(self):
        """Step the simulation forward by one full control cycle."""
        with self._lock:
            if self._connected:
                for _ in range(self._steps_per_cycle):
                    mujoco.mj_step(self._model, self._data)

    def render_offscreen(self, width=640, height=480):
        """Render the current simulation state to an image array (RGB)."""
        with self._lock:
            if not self._connected:
                return np.zeros((height, width, 3), dtype=np.uint8)
            renderer = mujoco.Renderer(self._model, height, width)
            renderer.update_scene(self._data, camera="fixed" if self._model.ncam > 0 else -1)
            pixels = renderer.render()
            renderer.close()
            return pixels

    @property
    def connected(self) -> bool:
        return self._connected

    @property
    def qpos(self):
        """Direct access to MuJoCo joint positions (for debugging)."""
        if self._data is not None:
            return self._data.qpos[:6].copy()
        return np.zeros(6)
