import numpy as np
from stage_1.robot_hal import RobotInterface, JointState, GripperState


class XArm6Interface(RobotInterface):
    """Real xArm6 robot via xarm-python-sdk.

    The SDK is imported lazily so this module can be imported and tested
    (e.g. for type checking) without the SDK installed.
    """

    def __init__(self, ip: str = "192.168.1.100", **kwargs):
        self._ip = ip
        self._arm = None
        self._gripper_attached = False

    # ---- RobotInterface impl ----
    def connect(self) -> bool:
        try:
            from xarm.wrapper import XArmAPI
        except ImportError:
            raise RuntimeError(
                "xarm-python-sdk not installed. "
                "Install: pip install xarm-python-sdk"
            )
        self._arm = XArmAPI(self._ip, is_radian=True)
        self._arm.motion_enable(True)
        self._arm.set_mode(1)    # position control
        self._arm.set_state(0)   # ready
        return True

    def disconnect(self) -> bool:
        if self._arm is not None:
            self._arm.disconnect()
            self._arm = None
        return True

    def get_joint_state(self) -> JointState:
        if self._arm is None:
            raise RuntimeError("Not connected")
        _, angles = self._arm.get_servo_angle(is_radian=True)
        positions = np.array(angles, dtype=np.float64)  # [1..6]
        # xArm returns angles for joints 1-6; keep only first 6
        positions = positions[:6]
        velocities = np.zeros(6, dtype=np.float64)
        return JointState(
            position=positions,
            velocity=velocities,
            effort=np.zeros(6),
            name=[f"joint{i}" for i in range(1, 7)],
        )

    def move_joints(
        self, positions: np.ndarray, velocity: float = 0.5, blocking: bool = True
    ) -> bool:
        if self._arm is None:
            return False
        positions = np.asarray(positions, dtype=np.float64)
        self._arm.set_servo_angle(
            angle=list(positions),
            speed=velocity,
            wait=blocking,
            is_radian=True,
        )
        return True

    def stop(self) -> bool:
        if self._arm is None:
            return False
        self._arm.emergency_stop()
        self._arm.set_state(0)  # re-enable after stop
        return True

    def get_gripper_state(self) -> GripperState:
        if self._arm is None:
            raise RuntimeError("Not connected")
        _, pos = self._arm.get_gripper_position()
        return GripperState(position=float(pos), effort=0.0)

    def move_gripper(self, position: float, blocking: bool = True) -> bool:
        if self._arm is None:
            return False
        pos = float(np.clip(position, 0.0, 1.0))
        self._arm.set_gripper_position(pos=pos, wait=blocking)
        return True

    def move_cartesian(
        self, pose: np.ndarray, velocity: float = 0.1, blocking: bool = True
    ) -> bool:
        if self._arm is None:
            return False
        x, y, z, roll, pitch, yaw = pose
        self._arm.set_position(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            speed=velocity,
            wait=blocking,
            is_radian=True,
        )
        return True

    def get_cartesian_pose(self) -> np.ndarray:
        if self._arm is None:
            raise RuntimeError("Not connected")
        _, pose = self._arm.get_position(is_radian=True)
        return np.array(pose, dtype=np.float64)
