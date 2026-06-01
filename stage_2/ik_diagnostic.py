#!/usr/bin/env python3
"""Diagnose IK convergence issues in mock pipeline."""
import sys
sys.path.insert(0, "/workspace/umi")

import numpy as np
from stage_1.kinematics.fk import end_effector_pose
from stage_1.kinematics.ik import solve_ik
from stage_1.kinematics.dh_params import XARM6_DH_PARAMS, XARM6_JOINT_LIMITS
from stage_1.kinematics.utils import pose_error
from stage_1.teleop_bridge.calibration import HandToRobotTransform

# Recreate what mock pipeline does
transform = HandToRobotTransform.mock_transform()

# Sample the tracker trajectory at several time points
print("=== IK Convergence Diagnostic ===\n")
print(f"mock_transform: scale={transform.scale}, offset={transform.offset}")
print(f"DEFAULT_ROTATION:\n{transform._R_quest_to_robot}\n")

times = np.linspace(0, 20, 41)  # 20 seconds, 41 samples
omega = 0.5
Ax, Ay, Az = 0.03, 0.02, 0.02
offset_z = 0.2

failures = 0
errors_pos = []
errors_rot = []
q_last = np.zeros(6)  # running seed

for t in times:
    # Mock tracker position (Quest3 space)
    x = Ax * np.sin(omega * t)
    y = Ay * np.sin(2.0 * omega * t)
    z = Az * np.sin(0.5 * omega * t) + offset_z
    p_quest = np.array([x, y, z])

    # Identity orientation (what mock tracker publishes)
    q_quest_xyzw = np.array([0.0, 0.0, 0.0, 1.0])  # w=1

    # Apply transform
    p_robot = transform.transform_position(p_quest)
    q_robot_wxyz = transform.transform_orientation_quat(q_quest_xyzw)

    # Build target transform
    from stage_1.kinematics.utils import quaternion_to_rotation_matrix
    R_target = quaternion_to_rotation_matrix(q_robot_wxyz)
    T_target = np.eye(4)
    T_target[:3, :3] = R_target
    T_target[:3, 3] = p_robot

    # Use running seed (simulates fixed hand_mapper behavior)
    if t == 0:
        q_seed = np.zeros(6)
    else:
        q_seed = q_last  # use previous best-effort as seed

    q_sol, success, iters, error = solve_ik(T_target, q_init=q_seed)

    # Always update seed (the fix)
    q_last = q_sol.copy()

    if not success:
        failures += 1
        # Check: is the pose reachable at all?
        # FK at q_sol (best effort)
        R_best, p_best = end_effector_pose(q_sol, XARM6_DH_PARAMS)
        pe = np.linalg.norm(p_robot - p_best)
        errors_pos.append(pe)
        # Rotation error
        from stage_1.kinematics.utils import so3_log
        R_err = R_target @ R_best.T
        re = np.linalg.norm(so3_log(R_err))
        errors_rot.append(re)

print(f"Samples: {len(times)}")
print(f"IK failures (from zero seed): {failures}/{len(times)}")

if errors_pos:
    print(f"\nOn failures:")
    print(f"  Position error: mean={np.mean(errors_pos)*1000:.1f}mm, max={np.max(errors_pos)*1000:.1f}mm")
    print(f"  Rotation error: mean={np.mean(errors_rot):.3f}rad, max={np.max(errors_rot):.3f}rad")

# Check reachability of the default orientation at workspace center
print("\n--- Orientation reachability check ---")
print("Can xArm6 reach DEFAULT_ROTATION at position [0.4, 0, 0.3]?")

# Try with a grid of seed configurations
T_test = np.eye(4)
T_test[:3, :3] = transform._R_quest_to_robot  # DEFAULT_ROTATION
T_test[:3, 3] = [0.4, 0.0, 0.3]

# Try some plausible seeds
seeds = [
    np.zeros(6),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    np.array([0.0, -0.5, 0.0, 1.5, 0.0, 0.0]),  # typical elbow-out
    np.array([0.0, -0.8, 0.0, 1.2, 0.0, 0.0]),  # typical elbow-down
    np.array([0.0, -0.3, 0.3, 2.0, 0.0, 0.0]),
    np.array([0.5, -0.2, 0.1, 1.8, 0.5, -0.3]),
]

for seed in seeds:
    q_sol, success, iters, error = solve_ik(T_test, q_init=seed, max_iterations=200)
    status = "CONVERGED" if success else f"FAILED (error={error:.4f})"
    print(f"  seed={np.array2string(seed, precision=1)}: {status} in {iters} iters")

# What if we use a different rotation that's easier to reach?
print("\n--- Simpler orientation test ---")
# z-up orientation (end-effector pointing down is more natural for xArm6)
R_z_up = np.eye(3)  # identity = z-up
T_simple = np.eye(4)
T_simple[:3, :3] = R_z_up
T_simple[:3, 3] = [0.4, 0.0, 0.3]

for seed in seeds[:3]:
    q_sol, success, iters, error = solve_ik(T_simple, q_init=seed, max_iterations=200)
    status = "CONVERGED" if success else f"FAILED (error={error:.4f})"
    print(f"  Identity orientation, seed={np.array2string(seed, precision=1)}: {status} in {iters} iters")
