import numpy as np


def dh_transform(a: float, alpha: float, d: float, theta: float) -> np.ndarray:
    """Standard DH transformation matrix (4x4).
    T = Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
    """
    ct = np.cos(theta)
    st = np.sin(theta)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    return np.array([
        [ct,  -st * ca,   st * sa,  a * ct],
        [st,   ct * ca,  -ct * sa,  a * st],
        [0,    sa,         ca,       d     ],
        [0,    0,          0,        1     ],
    ])


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix from 3-vector."""
    return np.array([
        [0,      -v[2],   v[1]],
        [v[2],    0,     -v[0]],
        [-v[1],   v[0],   0   ],
    ])


def rotation_matrix_to_euler(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to roll, pitch, yaw (XYZ fixed-axis convention)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return np.array([roll, pitch, yaw])


def euler_to_rotation_matrix(rpy: np.ndarray) -> np.ndarray:
    """Convert roll, pitch, yaw (XYZ fixed-axis) to 3x3 rotation matrix."""
    r, p, y = rpy
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])

    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])

    return Rz @ Ry @ Rx


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w, x, y, z] to 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,  2*x*y - 2*w*z,      2*x*z + 2*w*y],
        [2*x*y + 2*w*z,      1 - 2*x*x - 2*z*z,  2*y*z - 2*w*x],
        [2*x*z - 2*w*y,      2*y*z + 2*w*x,      1 - 2*x*x - 2*y*y],
    ])


def so3_log(R: np.ndarray) -> np.ndarray:
    """Logarithmic map on SO(3): rotation matrix -> axis-angle vector."""
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if abs(theta) < 1e-10:
        return np.zeros(3)
    return theta / (2.0 * np.sin(theta)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])


def pose_error(T_current: np.ndarray, T_desired: np.ndarray) -> np.ndarray:
    """Compute 6-vector pose error from two 4x4 homogeneous transforms.
    Returns [pos_err_x, pos_err_y, pos_err_z, rot_err_x, rot_err_y, rot_err_z].
    Rotation error is in the axis-angle (so3 log) representation.
    """
    p_cur = T_current[:3, 3]
    R_cur = T_current[:3, :3]
    p_des = T_desired[:3, 3]
    R_des = T_desired[:3, :3]

    pos_err = p_des - p_cur
    R_err = R_cur.T @ R_des
    rot_err = so3_log(R_err)
    return np.concatenate([pos_err, rot_err])


def pose_to_transform(position: np.ndarray, orientation_xyzw: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position [x,y,z] and quaternion [x,y,z,w]."""
    q = np.array([orientation_xyzw[3], orientation_xyzw[0], orientation_xyzw[1], orientation_xyzw[2]])
    R = quaternion_to_rotation_matrix(q)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    return T
