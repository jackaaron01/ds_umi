#!/usr/bin/env python3
"""
Generate ego-centric (first-person) reaching data.

Adds ego camera to the model, renders from end-effector perspective,
and stores ego images alongside joint data. The policy learns to reach
goals using visual input from the robot's own perspective.

Usage:
    python3 generate_ego_data.py -n 400 -o data/ego_dataset --v3 --render
"""
import os, sys, time, argparse
import numpy as np
import h5py

sys.path.insert(0, "/workspace/umi")
os.environ["MUJOCO_GL"] = "glx"
import mujoco

from stage_1.kinematics.dh_params import XARM6_JOINT_LIMITS
from stage_2.generate_diverse_data import SyntheticFeatureGenerator, CONTROL_RATE

JOINT_LIMITS = np.array(XARM6_JOINT_LIMITS)

GOALS = [
    np.array([ 0.0, -0.5,  0.0,  1.5,  0.0,  0.0]),
    np.array([ 0.8, -0.2,  0.5,  0.8, -0.2,  0.5]),
    np.array([-0.6, -0.6, -0.4,  1.8,  0.4,  0.0]),
    np.array([ 0.5, -0.4,  0.3,  1.3,  0.0,  0.0]),
    np.array([ 0.0, -0.9,  0.2,  2.1,  0.0, -0.3]),
    np.array([-0.3, -0.3, -0.5,  1.0, -0.5,  0.2]),
    np.array([ 1.0, -0.5,  0.8,  1.5,  0.3,  0.8]),
    np.array([-1.0, -0.7, -0.8,  2.0,  0.6, -0.5]),
]


def random_config(rng):
    limits = JOINT_LIMITS
    return rng.uniform(limits[:, 0] + 0.1, limits[:, 1] - 0.1)


def generate_ego_episode(output_dir, episode_idx, goal_q, model,
                         feature_gen=None, rng=None, renderer=None):
    """Generate one episode with ego-centric camera images."""
    if rng is None:
        rng = np.random.RandomState(episode_idx)
    if feature_gen is None:
        feature_gen = SyntheticFeatureGenerator()

    goal = goal_q.copy() + rng.randn(6) * 0.03
    goal = np.clip(goal, JOINT_LIMITS[:, 0] + 0.1, JOINT_LIMITS[:, 1] - 0.1)

    start = random_config(rng)
    while np.linalg.norm(start - goal) < 1.0:
        start = random_config(rng)

    n_waypoints = rng.randint(4, 10)
    alpha = np.linspace(0, 1, n_waypoints)
    curve = np.sin(alpha[:, None] * np.pi) * rng.randn(6) * 0.3
    waypoints = (1 - alpha[:, None]) * start + alpha[:, None] * goal + curve
    waypoints[0] = start; waypoints[-1] = goal

    data = mujoco.MjData(model)
    data.qpos[:6] = start.copy()
    mujoco.mj_forward(model, data)

    max_vel = rng.uniform(0.5, 2.5)
    cmd_noise_std = rng.uniform(0.001, 0.006)
    obs_noise_std = rng.uniform(0.001, 0.004)
    pause_prob = rng.uniform(0, 0.1)
    dt = 1.0 / CONTROL_RATE
    current_cmd = start.copy()
    steps_per_target = rng.randint(15, 30)
    PHYSICS_STEPS = max(1, int(1.0 / CONTROL_RATE / model.opt.timestep))

    joint_cmd_list, joint_state_list, joint_vel_list = [], [], []
    timestamps_list, img_feat_list, goal_list, image_list = [], [], [], []

    t0 = time.time()
    for w in range(len(waypoints)):
        wp = waypoints[w]
        n_pause = rng.randint(3, 12) if rng.random() < pause_prob else 0
        for _ in range(steps_per_target + n_pause):
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
            goal_list.append(goal.copy())
            if renderer is not None:
                image_list.append(renderer.render(data, camera="ego"))

    # Lookahead actions
    joint_cmd_raw = np.array(joint_cmd_list, dtype=np.float64)
    joint_state = np.array(joint_state_list, dtype=np.float64)
    joint_vel = np.array(joint_vel_list, dtype=np.float64)
    timestamps = np.array(timestamps_list, dtype=np.float64)
    img_feat = np.array(img_feat_list, dtype=np.float32)
    goals = np.array(goal_list, dtype=np.float32)
    n_steps = len(joint_cmd_raw)

    LOOKAHEAD = rng.randint(20, 30)
    joint_cmd = np.zeros_like(joint_cmd_raw)
    for i in range(n_steps):
        fut_idx = min(i + LOOKAHEAD, n_steps - 1)
        joint_cmd[i] = joint_cmd_raw[fut_idx]

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
        ep.create_dataset("observation/goal_position", data=goals, compression="gzip")
        if renderer is not None and len(image_list) > 0:
            images = np.stack(image_list, axis=0)
            ep.create_dataset("sensors/camera/ego", data=images, compression="gzip",
                              chunks=(1, images.shape[1], images.shape[2], 3))
        ep.attrs["num_steps"] = n_steps

    return h5_path, n_steps


def main():
    parser = argparse.ArgumentParser(description="Generate ego-centric reaching data")
    parser.add_argument("-n", "--episodes", type=int, default=400)
    parser.add_argument("-o", "--output", default="/workspace/umi/data/ego_dataset")
    parser.add_argument("--v3", action="store_true")
    parser.add_argument("--render", action="store_true",
                        help="Render ego-centric camera images")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output, exist_ok=True)

    # Use the EGO-enabled model
    model_path = os.path.join(os.path.dirname(__file__),
                              "simulation", "xarm6.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    print(f"Model: {model.nbody} bodies, {model.ncam} cameras")
    for i in range(model.ncam):
        cam = model.camera(i)
        print(f"  Camera: {cam.name}")

    feature_gen = SyntheticFeatureGenerator(seed=args.seed)

    # Create ego-centric renderer
    renderer = None
    if args.render:
        from stage_2.mujoco_renderer import MuJoCoRenderer
        renderer = MuJoCoRenderer(model, width=args.img_size, height=args.img_size)
        print(f"Ego renderer: {args.img_size}x{args.img_size}")

    total_steps = 0
    t0 = time.time()
    for ep in range(args.episodes):
        goal = GOALS[ep % len(GOALS)]
        _, n_steps = generate_ego_episode(
            args.output, ep, goal, model, feature_gen=feature_gen,
            rng=rng, renderer=renderer)
        total_steps += n_steps
        if (ep + 1) % 100 == 0:
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
