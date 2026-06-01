#!/usr/bin/env python3
"""
Generate a minimal MuJoCo MJCF model for a 6-DOF robot arm.

Body layout is NOT matched to xArm6 kinematics — this is a "generic 6-DOF arm"
for pipeline integration testing. The MujocoRobotInterface sets qpos directly
and reads back from qpos, effectively using MuJoCo as a visualization + physics
placeholder rather than a kinematic reference.

For accurate xArm6 kinematics, use our hand-written FK/IK, which IS accurate.
"""

import os, numpy as np

lines = []
lines.append('<mujoco model="xarm6_sim">')
lines.append('  <compiler angle="radian"/>')
lines.append('  <option gravity="0 0 -9.81"/>')
lines.append('  <default>')
lines.append('    <joint limited="true" damping="2"/>')
lines.append('    <position ctrllimited="true" forcerange="-50 50" kp="500"/>')
lines.append('  </default>')
lines.append('  <worldbody>')
lines.append('    <body name="base" pos="0 0 0">')
lines.append('      <geom type="cylinder" size="0.08 0.05" pos="0 0 0.133" rgba="0.3 0.3 0.3 1"/>')

# Build a 6-DOF arm chain with roughly xArm6-like dimensions
# Each link: body with a hinge joint, plus a visual geom
# Total reach: ~0.7m

# Joint definitions: (name, pos_offset, geom_size, geom_pos, joint_range)
link_specs = [
    ("link1", "joint1", [0.0, 0.0, 0.133], [0.05, 0.15], [0, 0, 0.075], [-6.283, 6.283]),
    ("link2", "joint2", [0.0, 0.0, 0.0],   [0.04, 0.30], [0.15, 0, 0],  [-2.251, 2.251]),
    ("link3", "joint3", [0.3, 0.0, 0.0],   [0.035, 0.25], [0.125, 0, 0], [-6.283, 6.283]),
    ("link4", "joint4", [0.25, 0.0, -0.04], [0.03, 0.20], [0.10, 0, 0],  [-6.283, 6.283]),
    ("link5", "joint5", [0.12, 0.0, 0.0],  [0.025, 0.15], [0.075, 0, 0], [-6.283, 6.283]),
    ("link6", "joint6", [0.08, 0.0, 0.0],  [0.02, 0.10], [0.05, 0, 0],  [-6.283, 6.283]),
]

indent = "      "
for i, (body_name, joint_name, pos, gsize, gpos, jrange) in enumerate(link_specs):
    blk_indent = indent + "  " * (i + 1)
    lines.append(f'{blk_indent}<body name="{body_name}" pos="{" ".join(f"{p:.6f}" for p in pos)}">')
    lines.append(f'{blk_indent}  <geom type="capsule" size="{" ".join(f"{s:.6f}" for s in gsize)}" pos="{" ".join(f"{p:.6f}" for p in gpos)}" rgba="0.4 0.4 0.6 1"/>')
    lines.append(f'{blk_indent}  <joint name="{joint_name}" type="hinge" axis="0 0 1" range="{jrange[0]:.4f} {jrange[1]:.4f}"/>')

# End-effector site at tip
ee_indent = indent + "  " * 7
lines.append(f'{ee_indent}<site name="ee" pos="0.08 0 0" size="0.01" rgba="1 0 0 1"/>')

# Close all bodies
for i in range(6, 0, -1):
    lines.append(f'{indent}{"  " * i}</body>')

lines.append('    </body>')
lines.append('  </worldbody>')
lines.append('  <actuator>')
for i in range(6):
    lines.append(f'    <position name="motor{i+1}" joint="joint{i+1}" ctrlrange="-6.283 6.283" kp="500"/>')
lines.append('  </actuator>')
lines.append('</mujoco>')

xml = "\n".join(lines)
out_dir = os.path.dirname(os.path.abspath(__file__))
out_path = os.path.join(out_dir, "xarm6.xml")
with open(out_path, "w") as f:
    f.write(xml)

# Verify it loads
import mujoco
model = mujoco.MjModel.from_xml_path(out_path)
data = mujoco.MjData(model)
data.qpos[:] = np.zeros(6)
mujoco.mj_forward(model, data)
ee = data.site_xpos[model.site("ee").id]
print(f"Generated: {out_path}")
print(f"Model: {model.nbody} bodies, {model.njnt} joints, {model.nu} actuators")
print(f"EE at zero: [{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]")
print("Ready for MujocoRobotInterface integration.")
