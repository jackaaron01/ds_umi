import numpy as np
from stage_1.kinematics.dh_params import XARM6_DH_PARAMS
from stage_1.kinematics.fk import link_transforms


def compute_jacobian(q: np.ndarray, dh_params: np.ndarray = XARM6_DH_PARAMS) -> np.ndarray:
    """Compute the 6x6 geometric Jacobian at joint configuration q.

    For revolute joint i, column i is:
      J[:3, i] = z_i x (p_ee - p_i)   -- linear velocity component
      J[3:, i] = z_i                    -- angular velocity component

    Where z_i and p_i come from T_0_i (transform from base to frame i).
    """
    transforms = link_transforms(q, dh_params)
    p_ee = transforms[-1][:3, 3]

    J = np.zeros((6, 6))
    for i in range(6):
        T_i = transforms[i]
        z_i = T_i[:3, 2]       # z-axis of frame i
        p_i = T_i[:3, 3]       # origin of frame i
        J[:3, i] = np.cross(z_i, p_ee - p_i)
        J[3:, i] = z_i

    return J
