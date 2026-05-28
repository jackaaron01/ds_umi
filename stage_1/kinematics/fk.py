import numpy as np
from stage_1.kinematics.dh_params import XARM6_DH_PARAMS
from stage_1.kinematics.utils import dh_transform


def link_transforms(q: np.ndarray, dh_params: np.ndarray = XARM6_DH_PARAMS) -> list:
    """Compute homogeneous transforms for all links (base to each joint frame).

    Args:
        q: joint angles [6,] in radians
        dh_params: (6, 4) DH parameter table [a, alpha, d, theta_offset]

    Returns:
        List of 7 (4,4) transforms T_0_i for i = 0..6, where T_0_0 = identity.
        T_0_6 is the end-effector frame.
    """
    T = np.eye(4)
    transforms = [T.copy()]
    for i in range(6):
        a, alpha, d, offset = dh_params[i]
        theta = q[i] + offset
        T_i = dh_transform(a, alpha, d, theta)
        T = T @ T_i
        transforms.append(T.copy())
    return transforms


def forward_kinematics(q: np.ndarray, dh_params: np.ndarray = XARM6_DH_PARAMS) -> tuple:
    """Compute positions and orientations for all joints.

    Returns:
        positions: (7, 3) array of joint frame origins
        orientations: (7, 3, 3) array of rotation matrices
    """
    transforms = link_transforms(q, dh_params)
    positions = np.array([T[:3, 3] for T in transforms])
    orientations = np.array([T[:3, :3] for T in transforms])
    return positions, orientations


def end_effector_pose(q: np.ndarray, dh_params: np.ndarray = XARM6_DH_PARAMS) -> tuple:
    """Compute end-effector pose.

    Returns:
        R: (3, 3) rotation matrix
        p: (3,) position vector
    """
    transforms = link_transforms(q, dh_params)
    T_ee = transforms[-1]
    return T_ee[:3, :3], T_ee[:3, 3]
