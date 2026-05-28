import numpy as np

# xArm6 standard DH parameters.
# Each row: (a, alpha, d, theta_offset)
# a: link length along x_i (m)
# alpha: link twist about x_i (rad)
# d: link offset along z_{i-1} (m)
# theta_offset: constant offset added to joint angle (rad)
XARM6_DH_PARAMS = np.array([
    [0.0,     0.0,           0.267,   0.0   ],  # Joint 1
    [0.0,     -np.pi / 2,    0.0,    -np.pi / 2],  # Joint 2
    [0.2895,   0.0,           0.0,     0.0   ],  # Joint 3
    [0.0775,  -np.pi / 2,    0.3425,  0.0   ],  # Joint 4
    [0.0,      np.pi / 2,    0.0,     0.0   ],  # Joint 5
    [0.0,     -np.pi / 2,    0.0975,  0.0   ],  # Joint 6
])

# Joint limits (rad): [min, max] per joint
XARM6_JOINT_LIMITS = np.array([
    [-2.0 * np.pi,  2.0 * np.pi],      # Joint 1 (actually +/-360 deg)
    [-2.2515,       2.2515     ],      # Joint 2 (approx +/-129 deg)
    [-2.0 * np.pi,  2.0 * np.pi],      # Joint 3
    [-2.0 * np.pi,  2.0 * np.pi],      # Joint 4
    [-2.0 * np.pi,  2.0 * np.pi],      # Joint 5
    [-2.0 * np.pi,  2.0 * np.pi],      # Joint 6
])

# Velocity limits (rad/s)
XARM6_VELOCITY_LIMITS = np.full(6, np.pi)  # 180 deg/s for all joints

# Joint names matching the URDF convention
XARM6_JOINT_NAMES = [f"joint{i}" for i in range(1, 7)]
