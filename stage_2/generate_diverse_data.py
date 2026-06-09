#!/usr/bin/env python3
"""
Generate diverse mock training data using MuJoCo + joint-space trajectories.

Bypasses the hand_tracker→IK chain. Directly generates valid joint
trajectories and simulates them in MuJoCo, producing realistic
joint_command + joint_state pairs.

Trajectory types:
  - point_to_point: linear interpolation between 2 random configs
  - multi_waypoint: 3-7 waypoints with smooth interpolation
  - hold: maintain a configuration (teaches stationary poses)

Usage:
    python3 generate_diverse_data.py -n 200 -o data/diverse_dataset
"""

import os, sys, time, argparse, json, glob
import numpy as np
import h5py

sys.path.insert(0, "/workspace/umi")

import mujoco
from stage_1.kinematics.dh_params import XARM6_JOINT_LIMITS

# ── Configuration ──────────────────────────────────────────────────────────
CONTROL_RATE = 30  # Hz
JOINT_LIMITS = np.array(XARM6_JOINT_LIMITS)  # (6, 2)


def random_config(rng=None):
    """Generate a random valid joint configuration."""
    if rng is None:
        rng = np.random
    limits = JOINT_LIMITS
    return rng.uniform(limits[:, 0] + 0.1, limits[:, 1] - 0.1)


def interpolate_trajectory(waypoints, steps_per_segment=60):
    """Create a smooth trajectory through multiple waypoints."""
    waypoints = np.asarray(waypoints)
    n_waypoints = len(waypoints)
    if n_waypoints < 2:
        return np.tile(waypoints, (steps_per_segment, 1))

    trajectory = []
    for i in range(n_waypoints - 1):
        t = np.linspace(0, 1, steps_per_segment)
        # Smooth interpolation using cosine easing
        alpha = (1 - np.cos(t * np.pi)) / 2
        segment = (1 - alpha[:, None]) * waypoints[i] + alpha[:, None] * waypoints[i + 1]
        trajectory.append(segment)

    return np.vstack(trajectory)


def generate_episode_hdf5(output_dir: str, episode_idx: int, trajectory: np.ndarray,
                          model, control_rate: float = CONTROL_RATE,
                          steps_per_target: int = 15):
    """Simulate a joint trajectory in MuJoCo and write to HDF5.

    Uses realistic control: joint_command stays constant (target position)
    while joint_state evolves towards it over multiple physics steps.
    This mirrors real robot behavior where commands are positions and
    the robot needs time to reach them.

    Args:
        trajectory: Waypoints to visit (N, 6)
        steps_per_target: Physics steps to hold each waypoint command constant
    """
    data = mujoco.MjData(model)
    data.qpos[:6] = trajectory[0].copy()
    mujoco.mj_forward(model, data)

    n_waypoints = len(trajectory)

    # Use lists for dynamic sizing (pauses add variable steps)
    joint_cmd_list = []
    joint_state_pos_list = []
    joint_state_vel_list = []
    timestamps_list = []

    # Use velocity-limited control to simulate realistic dynamics.
    # MuJoCo's position servo reaches target instantly, but a real robot
    # takes time. We simulate this by limiting the velocity at each step.
    # Vary max velocity per episode for diversity (0.3 - 3.0 rad/s)
    rng = np.random.RandomState(episode_idx + int(time.time() * 1000) % 10000)
    max_vel = rng.uniform(0.5, 3.0)  # rad/s

    # Noise parameters for realistic human-like jitter
    cmd_noise_std = rng.uniform(0.001, 0.008)  # rad — command noise
    obs_noise_std = rng.uniform(0.001, 0.005)  # rad — observation noise
    pause_prob = rng.uniform(0, 0.15)  # probability of pausing at a waypoint

    dt = 1.0 / control_rate
    current_cmd = trajectory[0].copy()

    t0 = time.time()
    step = 0
    for w in range(n_waypoints):
        target = trajectory[w]
        # Occasionally pause at waypoints (simulates human hesitation)
        n_pause = rng.randint(5, 20) if rng.random() < pause_prob else 0

        for _ in range(steps_per_target + n_pause):
            # Move current_cmd towards target at limited velocity
            delta = target - current_cmd
            max_step = max_vel * dt
            if np.linalg.norm(delta) > max_step:
                delta = delta / np.linalg.norm(delta) * max_step
            current_cmd = current_cmd + delta

            # Add Gaussian noise to command (simulates human hand tremor)
            noisy_cmd = current_cmd + rng.randn(6) * cmd_noise_std

            # Apply the noisy command to MuJoCo
            data.ctrl[:6] = noisy_cmd
            mujoco.mj_step(model, data)

            # Record: clean command vs noisy observation
            joint_cmd_list.append(current_cmd.copy())
            joint_state_pos_list.append(data.qpos[:6] + rng.randn(6) * obs_noise_std)
            joint_state_vel_list.append(data.qvel[:6])
            timestamps_list.append(time.time() - t0)

    # Convert lists to arrays
    joint_cmd = np.array(joint_cmd_list, dtype=np.float64)
    joint_state_pos = np.array(joint_state_pos_list, dtype=np.float64)
    joint_state_vel = np.array(joint_state_vel_list, dtype=np.float64)
    timestamps = np.array(timestamps_list, dtype=np.float64)
    n_steps = len(joint_cmd)

    # Gripper: random open/close pattern
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
        ep.attrs["num_steps"] = n_steps

    return h5_path, n_steps


def main():
    parser = argparse.ArgumentParser(description="Generate diverse mock training data")
    parser.add_argument("-n", "--episodes", type=int, default=200)
    parser.add_argument("-o", "--output", default="/workspace/umi/data/diverse_dataset")
    parser.add_argument("--model", default=None)
    parser.add_argument("--v3", action="store_true", help="Convert to LeRobot v3.0 after generation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output, exist_ok=True)

    model_path = args.model or os.path.join(
        os.path.dirname(__file__), "simulation", "xarm6.xml"
    )
    model = mujoco.MjModel.from_xml_path(model_path)
    print(f"Model: {model.nbody} bodies, {model.nq} joints")

    total_steps = 0
    t0 = time.time()

    for ep in range(args.episodes):
        # Choose trajectory type
        traj_type = rng.choice(["point_to_point", "multi_waypoint", "hold"])

        if traj_type == "point_to_point":
            start = random_config(rng)
            end = random_config(rng)
            # Use interpolation to create smooth waypoints
            n_waypoints = rng.randint(3, 8)
            alpha = np.linspace(0, 1, n_waypoints)
            waypoints = (1 - alpha[:, None]) * start + alpha[:, None] * end

        elif traj_type == "multi_waypoint":
            n_waypoints = rng.randint(3, 7)
            waypoints = np.array([random_config(rng) for _ in range(n_waypoints)])

        elif traj_type == "go_to_target":
            start = random_config(rng)
            target = random_config(rng)
            mid = start + rng.uniform(0.3, 0.7) * (target - start)
            waypoints = np.array([start, mid, target])

        else:  # hold
            waypoints = np.array([random_config(rng)])

        steps_per_target = rng.randint(10, 30)
        h5_path, n_steps = generate_episode_hdf5(args.output, ep, waypoints, model,
                                                  steps_per_target=steps_per_target)
        total_steps += n_steps

        if (ep + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  {ep+1}/{args.episodes} episodes, {total_steps} total steps "
                  f"({elapsed:.1f}s, {total_steps/elapsed:.0f} steps/s)")

    elapsed = time.time() - t0
    print(f"\nGenerated {args.episodes} episodes, {total_steps} total steps "
          f"in {elapsed:.1f}s")
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

        stats = convert_directory(args.output, v3_dir)
        write_info_json(v3_dir, stats)
        write_episodes_metadata(v3_dir, stats.episodes)

        data_dir = os.path.join(v3_dir, "data", "chunk-000")
        all_dfs = [pd.read_parquet(os.path.join(data_dir, f))
                   for f in sorted(os.listdir(data_dir)) if f.endswith(".parquet")]
        compute_and_write_stats(v3_dir, all_dfs)
        print(f"v3.0 dataset: {v3_dir}/")
        print(f"  {v3_dir}/data/chunk-000/file-000.parquet")
        print(f"  {v3_dir}/meta/info.json")
        print(f"  {v3_dir}/meta/stats.json")


if __name__ == "__main__":
    main()
