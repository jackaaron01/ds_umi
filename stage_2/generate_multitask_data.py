#!/usr/bin/env python3
"""
Generate multi-task reaching data for task-conditioned policy training.

Defines N distinct reach targets in joint space. For each episode:
  - Pick a random task (target joint config)
  - Start from a random initial config
  - Generate a trajectory from start → target
  - Label with task_index

Usage:
    python3 generate_multitask_data.py -n 300 -t 5 -o data/multitask_dataset --v3
"""
import os, sys, time, argparse
import numpy as np
import h5py

sys.path.insert(0, "/workspace/umi")

import mujoco
from stage_1.kinematics.dh_params import XARM6_JOINT_LIMITS
from stage_2.generate_diverse_data import (
    generate_episode_hdf5, SyntheticFeatureGenerator, random_config,
    interpolate_trajectory, CONTROL_RATE,
)

JOINT_LIMITS = np.array(XARM6_JOINT_LIMITS)

# ── Predefined reach tasks (joint-space targets) ──
# Each task is a 6-dim joint configuration within safe limits
TASK_TARGETS = [
    np.array([ 0.0, -0.5,  0.0,  1.5,  0.0,  0.0]),  # Task 0: home
    np.array([ 0.5, -0.3,  0.3,  1.2,  0.0,  0.0]),  # Task 1: forward
    np.array([-0.5, -0.6, -0.2,  1.8,  0.3,  0.0]),  # Task 2: left-reach
    np.array([ 0.8, -0.2,  0.5,  0.8, -0.2,  0.0]),  # Task 3: right-high
    np.array([ 0.0, -0.8,  0.0,  2.0,  0.0,  0.5]),  # Task 4: extended
]


def generate_episode_multitask(output_dir: str, episode_idx: int, task_idx: int,
                                model, feature_gen=None, rng=None):
    """Generate one episode: start from random → go to task target."""
    if rng is None:
        rng = np.random.RandomState(episode_idx)
    if feature_gen is None:
        feature_gen = SyntheticFeatureGenerator()

    target = TASK_TARGETS[task_idx % len(TASK_TARGETS)].copy()
    # Add small random perturbation to target for diversity
    target += rng.randn(6) * 0.05
    target = np.clip(target, JOINT_LIMITS[:, 0] + 0.1, JOINT_LIMITS[:, 1] - 0.1)

    # Start from random config
    start = random_config(rng)
    # Ensure start is far enough from target
    while np.linalg.norm(start - target) < 0.5:
        start = random_config(rng)

    # Build trajectory: start → mid → target (with waypoints)
    n_waypoints = rng.randint(3, 8)
    # Use smooth interpolation with slight curve
    alpha = np.linspace(0, 1, n_waypoints)
    # Add curvature via sine perturbation
    curve = np.sin(alpha[:, None] * np.pi) * rng.randn(6) * 0.2
    waypoints = (1 - alpha[:, None]) * start + alpha[:, None] * target + curve
    # Snap first and last to exact positions
    waypoints[0] = start
    waypoints[-1] = target

    data = mujoco.MjData(model)
    data.qpos[:6] = start.copy()
    mujoco.mj_forward(model, data)

    n_waypoints = len(waypoints)
    max_vel = rng.uniform(0.5, 3.0)
    cmd_noise_std = rng.uniform(0.001, 0.008)
    obs_noise_std = rng.uniform(0.001, 0.005)
    pause_prob = rng.uniform(0, 0.1)

    dt = 1.0 / CONTROL_RATE
    current_cmd = start.copy()
    steps_per_target = rng.randint(10, 25)

    joint_cmd_list = []
    joint_state_pos_list = []
    joint_state_vel_list = []
    timestamps_list = []
    image_feature_list = []
    task_index_list = []

    t0 = time.time()
    for w in range(n_waypoints):
        wp_target = waypoints[w]
        n_pause = rng.randint(5, 15) if rng.random() < pause_prob else 0

        for _ in range(steps_per_target + n_pause):
            delta = wp_target - current_cmd
            max_step = max_vel * dt
            if np.linalg.norm(delta) > max_step:
                delta = delta / np.linalg.norm(delta) * max_step
            current_cmd = current_cmd + delta

            noisy_cmd = current_cmd + rng.randn(6) * cmd_noise_std
            data.ctrl[:6] = noisy_cmd
            mujoco.mj_step(model, data)

            joint_cmd_list.append(current_cmd.copy())
            joint_state_pos_list.append(data.qpos[:6] + rng.randn(6) * obs_noise_std)
            joint_state_vel_list.append(data.qvel[:6])
            timestamps_list.append(time.time() - t0)
            image_feature_list.append(feature_gen.encode(data.qpos[:6]))
            task_index_list.append(task_idx)

    # Convert to arrays
    joint_cmd = np.array(joint_cmd_list, dtype=np.float64)
    joint_state_pos = np.array(joint_state_pos_list, dtype=np.float64)
    joint_state_vel = np.array(joint_state_vel_list, dtype=np.float64)
    timestamps = np.array(timestamps_list, dtype=np.float64)
    image_features = np.array(image_feature_list, dtype=np.float32)
    task_indices = np.array(task_index_list, dtype=np.int32)
    n_steps = len(joint_cmd)

    # Gripper: random open/close
    gripper_cmd = np.ones(n_steps, dtype=np.float64)
    close_start = n_steps // 3
    close_end = 2 * n_steps // 3
    gripper_cmd[close_start:close_end] = 0.0
    gripper_state = gripper_cmd.copy()

    # Write HDF5
    h5_path = os.path.join(output_dir, f"episode_{episode_idx:06d}.h5")
    with h5py.File(h5_path, "w") as f:
        ep = f.create_group(f"episode_{episode_idx:06d}")
        ep.create_dataset("joint_command/position", data=joint_cmd, compression="gzip")
        ep.create_dataset("joint_state/position", data=joint_state_pos, compression="gzip")
        ep.create_dataset("joint_state/velocity", data=joint_state_vel, compression="gzip")
        ep.create_dataset("joint_command/position_timestamp", data=timestamps, compression="gzip")
        ep.create_dataset("joint_state/position_timestamp", data=timestamps, compression="gzip")
        ep.create_dataset("gripper/command", data=gripper_cmd, compression="gzip")
        ep.create_dataset("gripper/state", data=gripper_state, compression="gzip")
        ep.create_dataset("observation/image_features", data=image_features, compression="gzip")
        ep.create_dataset("task_index", data=task_indices, compression="gzip")
        ep.attrs["num_steps"] = n_steps
        ep.attrs["task_index"] = task_idx

    return h5_path, n_steps


def main():
    parser = argparse.ArgumentParser(description="Generate multi-task reaching data")
    parser.add_argument("-n", "--episodes", type=int, default=300)
    parser.add_argument("-t", "--tasks", type=int, default=5,
                        help="Number of tasks (1-5)")
    parser.add_argument("-o", "--output", default="/workspace/umi/data/multitask_dataset")
    parser.add_argument("--model", default=None)
    parser.add_argument("--v3", action="store_true", help="Convert to LeRobot v3.0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output, exist_ok=True)

    model_path = args.model or os.path.join(
        os.path.dirname(__file__), "simulation", "xarm6.xml"
    )
    model = mujoco.MjModel.from_xml_path(model_path)
    feature_gen = SyntheticFeatureGenerator(seed=args.seed)

    n_tasks = min(args.tasks, len(TASK_TARGETS))
    print(f"Tasks: {n_tasks}")
    for i in range(n_tasks):
        print(f"  Task {i}: {TASK_TARGETS[i]}")

    total_steps = 0
    t0 = time.time()
    task_counts = [0] * n_tasks

    for ep in range(args.episodes):
        task_idx = ep % n_tasks  # cycle through tasks
        task_counts[task_idx] += 1

        h5_path, n_steps = generate_episode_multitask(
            args.output, ep, task_idx, model, feature_gen=feature_gen, rng=rng
        )
        total_steps += n_steps

        if (ep + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {ep+1}/{args.episodes} episodes, {total_steps} steps "
                  f"({elapsed:.1f}s, {total_steps/elapsed:.0f} steps/s)")

    elapsed = time.time() - t0
    print(f"\nGenerated {args.episodes} episodes, {total_steps} steps in {elapsed:.1f}s")
    print(f"Task distribution: {dict(enumerate(task_counts))}")
    print(f"Output: {args.output}/")

    # Convert to LeRobot v3.0
    if args.v3:
        v3_dir = args.output + "_v3"
        print(f"\nConverting to LeRobot v3.0: {v3_dir}")
        from stage_2.lerobot_v3_converter import (
            convert_directory, write_info_json, write_episodes_metadata,
            compute_and_write_stats,
        )
        import pandas as pd

        # Need to update converter for task_index
        stats = convert_directory(args.output, v3_dir, task_index_in_data=True)
        write_info_json(v3_dir, stats)
        write_episodes_metadata(v3_dir, stats.episodes)

        data_dir = os.path.join(v3_dir, "data", "chunk-000")
        all_dfs = [pd.read_parquet(os.path.join(data_dir, f))
                   for f in sorted(os.listdir(data_dir)) if f.endswith(".parquet")]
        compute_and_write_stats(v3_dir, all_dfs)
        print(f"v3.0 dataset: {v3_dir}/")


if __name__ == "__main__":
    main()
