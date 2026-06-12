#!/usr/bin/env python3
"""
Generate ego-centric reaching data with VISUAL goal markers.

Each episode:
  - Place a red sphere (goal marker) at a random 3D position
  - Use IK to compute the target joint configuration
  - Generate trajectory reaching toward the marker
  - Record ego camera images (showing the marker)
  - The marker position is ONLY in images, NOT in state input

This creates a true EGO task: the policy must SEE the marker to know
where to reach. Joint state alone is insufficient.

Usage:
    python3 generate_marker_data.py -n 300 -o data/marker_dataset --v3 --render
"""
import os, sys, time, argparse
import numpy as np
import h5py

sys.path.insert(0, "/workspace/umi")
os.environ["MUJOCO_GL"] = "glx"
import mujoco

from stage_2.generate_diverse_data import SyntheticFeatureGenerator, CONTROL_RATE
from stage_1.kinematics.ik import solve_ik
from stage_1.kinematics.fk import forward_kinematics
from stage_1.kinematics.dh_params import XARM6_JOINT_LIMITS

JOINT_LIMITS = np.array(XARM6_JOINT_LIMITS)


def random_joint_config(rng):
    limits = JOINT_LIMITS
    return rng.uniform(limits[:, 0] + 0.3, limits[:, 1] - 0.3)


def generate_marker_episode(output_dir, episode_idx, model,
                            feature_gen=None, rng=None, renderer=None):
    """Generate one episode reaching toward a visible marker."""
    if rng is None:
        rng = np.random.RandomState(episode_idx)
    if feature_gen is None:
        feature_gen = SyntheticFeatureGenerator()

    data = mujoco.MjData(model)

    # ── Place marker at a random 3D position in the workspace ──
    # Workspace: roughly 0.2-0.6m radius, 0.1-0.7m height from base
    marker_pos = np.array([
        rng.uniform(-0.4, 0.4),
        rng.uniform(-0.4, 0.4),
        rng.uniform(0.15, 0.6),
    ])
    # Set mocap position
    if model.nmocap > 0:
        mocap_id = 0  # first mocap body
        data.mocap_pos[mocap_id] = marker_pos

    # ── Compute IK target: reach end-effector toward marker ──
    # Start from random valid config
    start_q = random_joint_config(rng)

    # Build target transform: position=marker_pos, orientation=pointing at marker
    T_target = np.eye(4)
    T_target[:3, 3] = marker_pos
    # Default orientation: tool pointing forward
    T_target[:3, :3] = np.array([
        [0, -1, 0],
        [0, 0, 1],
        [-1, 0, 0],
    ])

    # Solve IK for target
    goal_q, ik_success, _, _ = solve_ik(T_target, q_init=start_q, max_iterations=100)
    if not ik_success:
        # Fall back to random goal
        goal_q = random_joint_config(rng)

    # Ensure start is far enough from goal
    while np.linalg.norm(start_q - goal_q) < 0.8:
        start_q = random_joint_config(rng)

    # ── Generate trajectory from start to goal ──
    n_waypoints = rng.randint(5, 12)
    alpha = np.linspace(0, 1, n_waypoints)
    curve = np.sin(alpha[:, None] * np.pi) * rng.randn(6) * 0.2
    waypoints = (1 - alpha[:, None]) * start_q + alpha[:, None] * goal_q + curve
    waypoints[0] = start_q
    waypoints[-1] = goal_q

    data.qpos[:6] = start_q.copy()
    mujoco.mj_forward(model, data)

    max_vel = rng.uniform(0.3, 2.0)
    cmd_noise_std = rng.uniform(0.001, 0.006)
    obs_noise_std = rng.uniform(0.001, 0.004)
    dt = 1.0 / CONTROL_RATE
    current_cmd = start_q.copy()
    steps_per_target = rng.randint(12, 25)
    PHYSICS_STEPS = max(1, int(1.0 / CONTROL_RATE / model.opt.timestep))

    joint_cmd_list, joint_state_list, joint_vel_list = [], [], []
    timestamps_list, img_feat_list, marker_list, image_list = [], [], [], []

    t0 = time.time()
    for w in range(len(waypoints)):
        wp = waypoints[w]
        for _ in range(steps_per_target):
            delta = wp - current_cmd
            max_step = max_vel * dt
            dist = np.linalg.norm(delta)
            if dist > max_step:
                delta = delta / dist * max_step
            current_cmd = current_cmd + delta
            noisy_cmd = current_cmd + rng.randn(6) * cmd_noise_std
            data.ctrl[:6] = noisy_cmd
            for _ in range(PHYSICS_STEPS):
                mujoco.mj_step(model, data)

            joint_cmd_list.append(current_cmd.copy())
            joint_state_list.append(data.qpos[:6] + rng.randn(6) * obs_noise_std)
            joint_vel_list.append(data.qvel[:6])
            timestamps_list.append(time.time() - t0)
            img_feat_list.append(feature_gen.encode(data.qpos[:6]))
            marker_list.append(marker_pos.copy())
            if renderer is not None:
                image_list.append(renderer.render(data, camera="ego"))

    # Lookahead actions
    joint_cmd_raw = np.array(joint_cmd_list, dtype=np.float64)
    joint_state = np.array(joint_state_list, dtype=np.float64)
    joint_vel = np.array(joint_vel_list, dtype=np.float64)
    timestamps = np.array(timestamps_list, dtype=np.float64)
    img_feat = np.array(img_feat_list, dtype=np.float32)
    markers = np.array(marker_list, dtype=np.float32)
    n_steps = len(joint_cmd_raw)

    LOOKAHEAD = rng.randint(20, 30)
    joint_cmd = np.zeros_like(joint_cmd_raw)
    for i in range(n_steps):
        joint_cmd[i] = joint_cmd_raw[min(i + LOOKAHEAD, n_steps - 1)]

    gripper_cmd = np.ones(n_steps, dtype=np.float64)
    gripper_cmd[n_steps//3:2*n_steps//3] = 0.0

    h5_path = os.path.join(output_dir, f"episode_{episode_idx:06d}.h5")
    with h5py.File(h5_path, "w") as f:
        ep = f.create_group(f"episode_{episode_idx:06d}")
        ep.create_dataset("joint_command/position", data=joint_cmd, compression="gzip")
        ep.create_dataset("joint_state/position", data=joint_state, compression="gzip")
        ep.create_dataset("joint_state/velocity", data=joint_vel, compression="gzip")
        ep.create_dataset("joint_command/position_timestamp", data=timestamps, compression="gzip")
        ep.create_dataset("joint_state/position_timestamp", data=timestamps, compression="gzip")
        ep.create_dataset("gripper/command", data=gripper_cmd, compression="gzip")
        ep.create_dataset("gripper/state", data=gripper_cmd.copy(), compression="gzip")
        ep.create_dataset("observation/image_features", data=img_feat, compression="gzip")
        ep.create_dataset("observation/marker_position", data=markers, compression="gzip")
        if renderer is not None and len(image_list) > 0:
            images = np.stack(image_list, axis=0)
            ep.create_dataset("sensors/camera/ego", data=images, compression="gzip",
                              chunks=(1, images.shape[1], images.shape[2], 3))
        ep.attrs["num_steps"] = n_steps

    return h5_path, n_steps


def main():
    parser = argparse.ArgumentParser(description="Generate marker-based EGO data")
    parser.add_argument("-n", "--episodes", type=int, default=300)
    parser.add_argument("-o", "--output", default="/workspace/umi/data/marker_dataset")
    parser.add_argument("--v3", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output, exist_ok=True)

    model_path = os.path.join(os.path.dirname(__file__),
                              "simulation", "xarm6.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    print(f"Model: {model.nbody} bodies, mocap={model.nmocap}, "
          f"cameras={[model.camera(i).name for i in range(model.ncam)]}")

    feature_gen = SyntheticFeatureGenerator(seed=args.seed)
    renderer = None
    if args.render:
        from stage_2.mujoco_renderer import MuJoCoRenderer
        renderer = MuJoCoRenderer(model, width=args.img_size, height=args.img_size)

    total_steps = 0
    t0 = time.time()
    for ep in range(args.episodes):
        _, n_steps = generate_marker_episode(
            args.output, ep, model, feature_gen=feature_gen,
            rng=rng, renderer=renderer)
        total_steps += n_steps
        if (ep + 1) % 50 == 0:
            print(f"  {ep+1}/{args.episodes} episodes, {total_steps} steps "
                  f"({time.time()-t0:.1f}s)")

    print(f"\nGenerated {args.episodes} episodes, {total_steps} steps "
          f"in {time.time()-t0:.1f}s")

    if renderer:
        renderer.close()

    if args.v3:
        v3_dir = args.output + "_v3"
        print(f"\nConverting to v3.0: {v3_dir}")
        from stage_2.lerobot_v3_converter import (
            convert_directory, write_info_json, write_episodes_metadata,
            compute_and_write_stats,
        )
        import pandas as pd, shutil
        if os.path.exists(v3_dir):
            shutil.rmtree(v3_dir)
        stats = convert_directory(args.output, v3_dir)
        write_info_json(v3_dir, stats)
        write_episodes_metadata(v3_dir, stats.episodes)
        data_dir = os.path.join(v3_dir, "data", "chunk-000")
        all_dfs = [pd.read_parquet(os.path.join(data_dir, f))
                   for f in sorted(os.listdir(data_dir)) if f.endswith(".parquet")]
        compute_and_write_stats(v3_dir, all_dfs)
        print(f"Done: {v3_dir}/")


if __name__ == "__main__":
    main()
