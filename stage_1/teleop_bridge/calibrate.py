#!/usr/bin/env python3
"""Hand-eye calibration tool: compute the transformation from Quest3 tracking frame
to robot base frame using paired 3D positions (point-cloud alignment via SVD).

This is a standalone script — no ROS2 dependency.

Input: paired positions in JSON or CSV format.
Output: YAML calibration file loadable by HandToRobotTransform.

Usage:
    # From paired JSON:
    python calibrate.py --input pairs.json --output calibration.yaml

    # Interactive mode (enter points manually):
    python calibrate.py --interactive --output calibration.yaml

JSON input format:
    [
        {"quest": [x, y, z], "robot": [x, y, z]},
        ...
    ]
"""

import argparse
import json
import os
import sys

import numpy as np
import yaml


def solve_similarity(source: np.ndarray, target: np.ndarray):
    """Solve for rotation, scale, and translation that map source to target.

    Uses the SVD-based method (Umeyama 1991) for point-cloud alignment.
    Minimizes: sum_i ||target_i - (s * R * source_i + t)||^2

    Args:
        source: (N, 3) array of points in the source frame (Quest3)
        target: (N, 3) array of points in the target frame (robot)

    Returns:
        (R, t, s, rms_error)
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scalar uniform scale
        rms_error: root-mean-square residual in meters
    """
    assert source.shape == target.shape
    assert source.shape[1] == 3
    assert source.shape[0] >= 3, "Need at least 3 point pairs"

    n = source.shape[0]
    mu_s = source.mean(axis=0)
    mu_t = target.mean(axis=0)

    sigma_s = ((source - mu_s) ** 2).sum() / n
    sigma_t = ((target - mu_t) ** 2).sum() / n

    cov = (target - mu_t).T @ (source - mu_s) / n
    U, _, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    s = np.trace(np.diag(np.sqrt(sigma_t / sigma_s))) / 3.0

    t = mu_t - s * R @ mu_s

    residual = target - (s * (source @ R.T) + t)
    rms = np.sqrt((residual ** 2).sum() / n)

    return R, t, s, rms


def solve_rotation_only(source_quats_xyzw: np.ndarray, target_R: np.ndarray):
    """Solve for the rotation matrix that maps source orientations to target.

    Given N quaternion pairs (source in Quest3 frame, target as rotation matrices
    in robot frame), compute the best-fit rotation that aligns them.

    For a single pair: R_target = R_align @ R_source  →  R_align = R_target @ R_source^T
    For multiple pairs: average over the relative rotations (using SVD on mean).

    Args:
        source_quats_xyzw: (N, 4) quaternions [x, y, z, w] in Quest3 frame
        target_R: (N, 3, 3) rotation matrices in robot frame

    Returns:
        R_align: (3, 3) rotation matrix mapping Quest3 orientations to robot frame
    """
    from stage_1.kinematics.utils import quaternion_to_rotation_matrix

    n = source_quats_xyzw.shape[0]
    M_sum = np.zeros((3, 3))
    for i in range(n):
        q = np.array([
            source_quats_xyzw[i, 3],  # w
            source_quats_xyzw[i, 0],  # x
            source_quats_xyzw[i, 1],  # y
            source_quats_xyzw[i, 2],  # z
        ])
        R_source = quaternion_to_rotation_matrix(q)
        R_rel = target_R[i] @ R_source.T
        M_sum += R_rel

    U, _, Vt = np.linalg.svd(M_sum / n)
    R_align = U @ Vt
    if np.linalg.det(R_align) < 0:
        S = np.diag([1, 1, -1])
        R_align = U @ S @ Vt

    return R_align


def load_json_pairs(filepath: str) -> tuple:
    """Load paired positions from JSON file.

    Returns:
        (quest_points, robot_points): both (N, 3) arrays
    """
    with open(filepath, "r") as f:
        data = json.load(f)

    quest_pts = []
    robot_pts = []
    for pair in data:
        quest_pts.append(pair["quest"])
        robot_pts.append(pair["robot"])

    return np.array(quest_pts), np.array(robot_pts)


def interactive_collect() -> tuple:
    """Collect paired points interactively from stdin."""
    print("Enter paired points (Quest3 x y z, Robot x y z) one pair per line.")
    print("Blank line to finish.\n")

    quest_pts = []
    robot_pts = []
    pair = 0
    while True:
        line = input(f"Pair {pair}: ").strip()
        if not line:
            break
        parts = line.split()
        if len(parts) != 6:
            print("  Expected 6 numbers: qx qy qz rx ry rz")
            continue
        vals = [float(p) for p in parts]
        quest_pts.append(vals[:3])
        robot_pts.append(vals[3:])
        pair += 1

    return np.array(quest_pts), np.array(robot_pts)


def write_calibration_yaml(filepath: str, R: np.ndarray, t: np.ndarray, s: float, rms: float):
    """Write calibration result as YAML."""
    calib = {
        "rotation_matrix": R.tolist(),
        "translation": t.tolist(),
        "scale": float(s),
        "rms_error_m": float(rms),
        "num_point_pairs": int(R.shape[0]),
        "description": "Quest3-to-robot base frame calibration",
    }
    with open(filepath, "w") as f:
        yaml.dump(calib, f, default_flow_style=False)
    print(f"Calibration written to {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="Hand-eye calibration: Quest3 tracking frame → robot base frame"
    )
    parser.add_argument("--input", "-i", help="JSON file with paired positions")
    parser.add_argument("--output", "-o", default="calibration.yaml", help="Output YAML file")
    parser.add_argument("--interactive", action="store_true", help="Collect pairs interactively")
    args = parser.parse_args()

    if args.interactive:
        quest_pts, robot_pts = interactive_collect()
    elif args.input:
        quest_pts, robot_pts = load_json_pairs(args.input)
    else:
        parser.print_help()
        sys.exit(1)

    if len(quest_pts) < 3:
        print(f"Error: need at least 3 point pairs, got {len(quest_pts)}")
        sys.exit(1)

    print(f"Computing calibration from {len(quest_pts)} point pairs...")
    R, t, s, rms = solve_similarity(quest_pts, robot_pts)

    print(f"  Rotation matrix:\n{R}")
    print(f"  Translation: {t}")
    print(f"  Scale: {s:.4f}")
    print(f"  RMS error: {rms:.4f} m")

    write_calibration_yaml(args.output, R, t, s, rms)


if __name__ == "__main__":
    main()
