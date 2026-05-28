import threading
import time
import numpy as np
from stage_1.robot_hal import RobotInterface, JointState, GripperState
from stage_1.kinematics.dh_params import XARM6_JOINT_NAMES


class MockRobotInterface(RobotInterface):
    """In-memory robot for testing the teleop pipeline without hardware."""

    def __init__(self, num_joints: int = 6):
        self._num_joints = num_joints
        self._joint_names = [f"joint{i}" for i in range(1, num_joints + 1)]
        self._lock = threading.Lock()
        self._connected = False

        # State
        self._positions = np.zeros(num_joints)
        self._velocities = np.zeros(num_joints)
        self._gripper_position = 0.0

        # Non-blocking motion state
        self._target_positions = None
        self._motion_velocity = None
        self._motion_start_time = None
        self._motion_start_positions = None
        self._motion_duration = None
        self._motion_thread = None
        self._motion_running = False

        # Injection
        self._noise_std_pos = 0.0
        self._noise_std_vel = 0.0
        self._delay_seconds = 0.0

    # ---- Helpers for testing ----
    def enable_noise(self, std_pos: float = 0.001, std_vel: float = 0.01):
        self._noise_std_pos = std_pos
        self._noise_std_vel = std_vel

    def enable_delay(self, seconds: float = 0.05):
        self._delay_seconds = seconds

    def set_joint_state(self, positions: np.ndarray):
        with self._lock:
            self._positions = positions.copy()

    # ---- RobotInterface impl ----
    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> bool:
        self.stop()
        self._connected = False
        return True

    def get_joint_state(self) -> JointState:
        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)
        with self._lock:
            self._advance_motion()
            pos = self._positions.copy()
            vel = self._velocities.copy()
        if self._noise_std_pos > 0:
            pos += np.random.normal(0, self._noise_std_pos, self._num_joints)
        if self._noise_std_vel > 0:
            vel += np.random.normal(0, self._noise_std_vel, self._num_joints)
        return JointState(
            position=pos,
            velocity=vel,
            effort=np.zeros(self._num_joints),
            name=self._joint_names,
        )

    def move_joints(
        self, positions: np.ndarray, velocity: float = 0.5, blocking: bool = True
    ) -> bool:
        positions = np.asarray(positions, dtype=np.float64)
        if positions.shape != (self._num_joints,):
            return False
        if self._delay_seconds > 0:
            time.sleep(self._delay_seconds)
        if blocking:
            with self._lock:
                # Compute distances for simulated motion time
                dists = np.abs(positions - self._positions)
                max_dist = np.max(dists)
                duration = max_dist / max(velocity, 0.01)
                self._positions = positions.copy()
                self._velocities = np.zeros(self._num_joints)
            time.sleep(duration)
        else:
            with self._lock:
                dists = np.abs(positions - self._positions)
                max_dist = np.max(dists)
                duration = max_dist / max(velocity, 0.01)
                self._target_positions = positions.copy()
                self._motion_start_positions = self._positions.copy()
                self._motion_velocity = velocity
                self._motion_start_time = time.time()
                self._motion_duration = duration
                self._motion_running = True
        return True

    def stop(self) -> bool:
        with self._lock:
            self._motion_running = False
            self._velocities = np.zeros(self._num_joints)
        return True

    def get_gripper_state(self) -> GripperState:
        with self._lock:
            pos = self._gripper_position
        return GripperState(position=pos, effort=0.0)

    def move_gripper(self, position: float, blocking: bool = True) -> bool:
        position = float(np.clip(position, 0.0, 1.0))
        with self._lock:
            self._gripper_position = position
        return True

    def move_cartesian(
        self, pose: np.ndarray, velocity: float = 0.1, blocking: bool = True
    ) -> bool:
        # Mock: no-op, return True for testing
        return True

    def get_cartesian_pose(self) -> np.ndarray:
        return np.array([0.5, 0.0, 0.4, 0.0, 0.0, 0.0])

    # ---- Internal ----
    def _advance_motion(self):
        """Interpolate non-blocking motion to the current time."""
        if not self._motion_running or self._target_positions is None:
            return
        elapsed = time.time() - self._motion_start_time
        if elapsed >= self._motion_duration:
            self._positions = self._target_positions.copy()
            self._velocities = np.zeros(self._num_joints)
            self._motion_running = False
        else:
            frac = elapsed / max(self._motion_duration, 0.001)
            self._positions = (
                self._motion_start_positions
                + frac * (self._target_positions - self._motion_start_positions)
            )
            self._velocities = (
                (self._target_positions - self._motion_start_positions)
                / max(self._motion_duration, 0.001)
            )
