#!/usr/bin/env python3
"""Show IK configuration and test results."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from mujoco_ik import MujocoIK
import numpy as np
import mujoco

ik = MujocoIK(os.path.join(os.path.dirname(__file__), "xarm6_gripper.xml"), "ee")
home = np.deg2rad([0, -20, -75, 0, 90, 0])
pos_home, rot_home = ik.fk(home)

print("=" * 70)
print("IK CONFIGURATION")
print("=" * 70)
print(f"Model:     xarm6_gripper.xml")
print(f"Site:      ee (joint6 + 0.13m, euler=0,-90,0)")
print(f"Joints:    {ik.n_joints} arm DOFs")
print(f"HOME deg:  {np.array2string(np.degrees(home), precision=0, separator=',')}")
print(f"HOME FK:   pos={np.array2string(pos_home, precision=3)}")
print(f"           Z-axis={np.array2string(rot_home[:,2], precision=3)}")

print()
print("SOLVER PARAMETERS:")
print("  pos_weight=45.0   rot_weight=5.0   damping=0.05")
print("  tolerance=0.001m  max_iter=50      trust_region=0.3rad")
print("  reg_weights=[5,1,1,5,1,1]  smooth_weight=0.05")

print()
print("=" * 70)
print("IK 6-DIRECTION REACH TEST (5cm from HOME)")
print("=" * 70)
tests = [
    ("fwd   (-X)", np.array([-0.05,  0.00,  0.00])),
    ("back  (+X)", np.array([ 0.05,  0.00,  0.00])),
    ("left  (+Y)", np.array([ 0.00,  0.05,  0.00])),
    ("right (-Y)", np.array([ 0.00, -0.05,  0.00])),
    ("up    (+Z)", np.array([ 0.00,  0.00,  0.05])),
    ("down  (-Z)", np.array([ 0.00,  0.00, -0.05])),
]
for name, d in tests:
    tgt = pos_home + d
    q = ik.solve(tgt, None, home.copy(), home)
    pf, _ = ik.fk(q)
    err = np.linalg.norm(tgt - pf) * 1000
    dq_deg = np.degrees(q - home)
    print(f"  {name:12s} | dq={np.array2string(dq_deg, precision=0, separator=',', max_line_width=80)}")
    print(f"              | EE={np.array2string(pf, precision=3)} err={err:.1f}mm")

print()
print("=" * 70)
print("JACOBIAN AT HOME (position part, 3×6)")
print("=" * 70)
model = ik._model
data = ik._data
nv = ik._nv
data.qpos[:nv] = home.copy()
mujoco.mj_forward(model, data)
jac_pos = np.zeros((3, ik._nv_full))
jac_rot = np.zeros((3, ik._nv_full))
mujoco.mj_jac(model, data, jac_pos, jac_rot,
              data.site_xpos[ik._site_id], ik._body_id)
J = jac_pos[:, :nv]
U, S, Vt = np.linalg.svd(J)
print(f"cond(J) = {S[0]/S[-1]:.0f}  |  σ = {np.array2string(S, precision=2)}")
for i in range(6):
    print(f"  J[:,{i}] joint{i+1}: {np.array2string(J[:,i], precision=3, max_line_width=30)}")

print()
print("=" * 70)
print("PSEUDOINVERSE: target = HOME + fwd 10cm,  one iteration")
print("=" * 70)
tgt = pos_home + np.array([-0.10, 0, 0])
pos_err = tgt - data.site_xpos[ik._site_id]
weighted_err = 45.0 * pos_err
JJT = J @ J.T
J_inv = J.T @ np.linalg.solve(JJT + 0.05**2 * np.eye(3), np.eye(3))
dq = J_inv @ weighted_err
dq = np.clip(dq, -0.3, 0.3)
print(f"pos_err (raw): {np.round(pos_err, 3)}")
print(f"pos_err ×45:   {np.round(weighted_err, 1)}")
print(f"dq (deg):      {np.array2string(np.degrees(dq), precision=1, separator=',')}")
