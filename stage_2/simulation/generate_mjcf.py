#!/usr/bin/env python3
"""
Generate an accurate MuJoCo MJCF model for the xArm6 robot arm.

Uses a two-body-per-joint kinematic structure to correctly model standard
DH parameters: rotor body at parent origin (hinge about Z), then link body
offset by DH fixed parameters [a, 0, d] with twist α.

For each joint i with DH params (a, α, d, θ_offset):
  body "j{i}_rotor" at parent origin, hinge axis=[0,0,1]
  body "link{i}" offset by R_z(θ_offset)·[a,0,d], rotated by R_z(θ_offset)·R_x(α)

At qpos=θ: T = Rot_z(θ + θ_offset) · Trans_z(d) · Trans_x(a) · Rot_x(α)  ✓
"""

import os
import sys
import numpy as np


# xArm6 standard DH parameters: (a, alpha, d, theta_offset)
XARM6_DH = np.array([
    [0.0,     0.0,           0.267,   0.0        ],  # Joint 1
    [0.0,     -np.pi / 2,    0.0,    -np.pi / 2  ],  # Joint 2
    [0.2895,   0.0,           0.0,     0.0        ],  # Joint 3
    [0.0775,  -np.pi / 2,    0.3425,  0.0        ],  # Joint 4
    [0.0,      np.pi / 2,    0.0,     0.0        ],  # Joint 5
    [0.0,     -np.pi / 2,    0.0975,  0.0        ],  # Joint 6
])

XARM6_JOINT_LIMITS = np.array([
    [-2.0 * np.pi,  2.0 * np.pi],
    [-2.2515,       2.2515     ],
    [-2.0 * np.pi,  2.0 * np.pi],
    [-2.0 * np.pi,  2.0 * np.pi],
    [-2.0 * np.pi,  2.0 * np.pi],
    [-2.0 * np.pi,  2.0 * np.pi],
])

# Link visual geometry: (geom_type, size_args, rgba)
LINK_GEOMS = [
    ("cylinder", "0.08 0.05",  "0.3 0.3 0.3 1"),   # base
    ("cylinder", "0.06 0.12",  "0.4 0.4 0.6 1"),   # link1
    ("capsule",  "0.05 0.15",  "0.4 0.4 0.6 1"),   # link2
    ("capsule",  "0.04 0.12",  "0.4 0.4 0.6 1"),   # link3
    ("capsule",  "0.035 0.10", "0.4 0.4 0.6 1"),   # link4
    ("capsule",  "0.03 0.08",  "0.4 0.4 0.6 1"),   # link5
    ("capsule",  "0.025 0.06", "0.4 0.4 0.6 1"),   # link6
]


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_x(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def mat_to_euler_xyz(R_mat: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to intrinsic X-Y-Z Euler angles."""
    sy = R_mat[0, 2]
    if abs(sy) > 0.99999:  # gimbal lock
        ey = np.pi / 2 if sy > 0 else -np.pi / 2
        ex = 0.0
        ez = np.arctan2(R_mat[1, 0], R_mat[1, 1])
    else:
        ey = np.arcsin(sy)
        cy = np.cos(ey)
        ex = np.arctan2(-R_mat[1, 2] / cy, R_mat[2, 2] / cy)
        ez = np.arctan2(-R_mat[0, 1] / cy, R_mat[0, 0] / cy)
    return np.array([ex, ey, ez])


def generate_xml():
    """Generate MJCF XML using two-body-per-joint for correct DH kinematics."""
    lines = []
    lines.append('<mujoco model="xarm6">')
    lines.append('  <compiler angle="radian"/>')
    lines.append('  <option gravity="0 0 -9.81" timestep="0.002"/>')
    lines.append('  <default>')
    lines.append('    <joint limited="true" damping="2"/>')
    lines.append('    <position ctrllimited="true" forcerange="-50 50" kp="500"/>')
    lines.append('  </default>')
    lines.append('  <worldbody>')
    # Base body at world origin
    lines.append('    <body name="base" pos="0 0 0">')
    lines.append('      <geom type="cylinder" size="0.08 0.05" pos="0 0 0.133" rgba="0.3 0.3 0.3 1"/>')

    for i, (a, alpha, d, toff) in enumerate(XARM6_DH):
        joint_name = f"joint{i + 1}"
        rotor_name = f"j{i + 1}_rotor"
        link_name = f"link{i + 1}"
        jlo, jhi = XARM6_JOINT_LIMITS[i]

        # Link pose relative to rotor at θ=0:
        p = np.array([a, 0.0, d])
        link_pos = rot_z(toff) @ p
        link_euler = mat_to_euler_xyz(rot_z(toff) @ rot_x(alpha))

        geom_type, geom_size, geom_rgba = LINK_GEOMS[i + 1]

        # Indentation: each level adds 2 spaces
        # Level 0: worldbody, Level 1: base, Level 2: j1_rotor, Level 3: link1, ...
        base_indent = "      "  # 6 spaces (inside worldbody > base)
        r_indent = base_indent + "  " * (2 * i + 1)
        l_indent = r_indent + "  "

        # Rotor body: at parent origin, hinge about Z
        lines.append(f'{r_indent}<body name="{rotor_name}" pos="0 0 0">')
        lines.append(
            f'{r_indent}  <joint name="{joint_name}" type="hinge" '
            f'axis="0 0 1" range="{jlo:.4f} {jhi:.4f}"/>'
        )
        # Link body: DH fixed offset from rotor
        lines.append(
            f'{l_indent}<body name="{link_name}" '
            f'pos="{" ".join(f"{v:.6f}" for v in link_pos)}" '
            f'euler="{" ".join(f"{v:.6f}" for v in link_euler)}">'
        )
        lines.append(
            f'{l_indent}  <geom type="{geom_type}" '
            f'size="{geom_size}" rgba="{geom_rgba}"/>'
        )

    # EE site at link6 origin
    ee_indent = "      " + "  " * 13
    lines.append(f'{ee_indent}<site name="ee" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>')

    # Close bodies: 6 links + 6 rotors + base
    for i in range(6):
        link_indent = "      " + "  " * (2 * i + 3)
        rotor_indent = "      " + "  " * (2 * i + 1)
        lines.append(f'{link_indent}</body>')   # close link
        lines.append(f'{rotor_indent}</body>')   # close rotor

    lines.append('    </body>')  # close base
    lines.append('  </worldbody>')
    lines.append('  <actuator>')
    for i in range(6):
        jlo, jhi = XARM6_JOINT_LIMITS[i]
        lines.append(
            f'    <position name="motor{i + 1}" joint="joint{i + 1}" '
            f'ctrlrange="{jlo:.4f} {jhi:.4f}" kp="500"/>'
        )
    lines.append('  </actuator>')
    lines.append('</mujoco>')
    return "\n".join(lines)


def verify_fk(xml_path: str):
    """Verify MuJoCo FK matches hand-written FK."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
    from stage_1.kinematics.fk import end_effector_pose
    from stage_1.kinematics.dh_params import XARM6_DH_PARAMS

    # Zero config
    data.qpos[:] = np.zeros(6)
    mujoco.mj_forward(model, data)
    mj_ee = data.site_xpos[model.site("ee").id].copy()
    _, p_fk = end_effector_pose(np.zeros(6), XARM6_DH_PARAMS)
    err_zero = np.linalg.norm(mj_ee - p_fk)

    # Random configs
    rng = np.random.RandomState(42)
    max_err = 0.0
    for _ in range(100):
        q = rng.uniform(XARM6_JOINT_LIMITS[:, 0], XARM6_JOINT_LIMITS[:, 1])
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        mj_ee = data.site_xpos[model.site("ee").id].copy()
        _, p_fk = end_effector_pose(q, XARM6_DH_PARAMS)
        err = np.linalg.norm(mj_ee - p_fk)
        max_err = max(max_err, err)

    return err_zero, max_err


def main():
    xml = generate_xml()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "xarm6.xml")
    with open(out_path, "w") as f:
        f.write(xml)
    print(f"Generated: {out_path}")

    import mujoco
    model = mujoco.MjModel.from_xml_path(out_path)
    print(f"Model: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")

    err_zero, max_err = verify_fk(out_path)
    print(f"FK error at q=0:      {err_zero:.6f}m")
    print(f"FK error max (100 cfg): {max_err:.6f}m")

    if max_err < 1e-3:
        print("✓ FK verification PASSED (<1mm)")
    else:
        print(f"✗ FK verification FAILED (max error {max_err * 1000:.2f}mm > 1mm)")
        sys.exit(1)


if __name__ == "__main__":
    main()
