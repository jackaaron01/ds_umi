import os
import numpy as np
from stage_1.kinematics.utils import quaternion_to_rotation_matrix


class HandToRobotTransform:
    """Maps Quest3 tracking-space wrist poses into the robot base frame.

    The transform applies: rotation → scaling → translation.
    p_robot = scale * R @ p_quest + offset

    The calibration parameters (R, offset, scale) can be:
      - Default hardcoded values (Quest3 → robot convention mapping)
      - Loaded from a YAML calibration file (see calibrate.py)
    """

    # Default rotation: Quest3 convention (+X right, +Y up, +Z backward)
    # → Robot convention (+X forward, +Y left, +Z up)
    DEFAULT_ROTATION = np.array([
        [ 0.0,  0.0, -1.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
    ])

    def __init__(
        self,
        scale: float = 3.0,
        offset: np.ndarray = None,
        rotation: np.ndarray = None,
    ):
        self.scale = scale
        self.offset = (
            offset
            if offset is not None
            else np.array([0.5, 0.0, 0.2])  # robot workspace center
        )
        self._R_quest_to_robot = (
            rotation if rotation is not None else self.DEFAULT_ROTATION
        )

    def transform_position(self, p_quest: np.ndarray) -> np.ndarray:
        """Map Quest3 wrist position to robot end-effector target position."""
        p_robot_frame = self.scale * self._R_quest_to_robot @ p_quest
        return p_robot_frame + self.offset

    def transform_orientation_quat(self, q_quest_xyzw: np.ndarray) -> np.ndarray:
        """Map Quest3 wrist orientation quaternion [x, y, z, w] to robot frame.

        Returns quaternion [w, x, y, z] suitable for pose_to_transform.
        """
        R_quest = quaternion_to_rotation_matrix(
            np.array([q_quest_xyzw[3], q_quest_xyzw[0], q_quest_xyzw[1], q_quest_xyzw[2]])
        )
        R_robot = self._R_quest_to_robot @ R_quest
        from stage_1.kinematics.utils import rotation_matrix_to_quaternion
        return rotation_matrix_to_quaternion(R_robot)

    @staticmethod
    def from_yaml(filepath: str) -> "HandToRobotTransform":
        """Factory: load calibration from a YAML file produced by calibrate.py."""
        import yaml

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return HandToRobotTransform(
            scale=data.get("scale", 3.0),
            offset=np.array(data.get("translation", [0.5, 0.0, 0.2])),
            rotation=np.array(data.get("rotation_matrix", HandToRobotTransform.DEFAULT_ROTATION)),
        )

    @staticmethod
    def default_transform():
        """Factory: a reasonable default calibration for mock/testing."""
        return HandToRobotTransform(scale=3.0)

    @staticmethod
    def mock_transform():
        """Factory: transform for mock pipeline testing.

        Maps the mock tracker's default wrist position (z=0.2 in Quest3 space)
        to a comfortably reachable robot workspace position [0.4, 0, 0.3].
        """
        return HandToRobotTransform(
            scale=1.0,
            rotation=HandToRobotTransform.DEFAULT_ROTATION.copy(),
            offset=np.array([0.6, 0.0, 0.3]),
        )
