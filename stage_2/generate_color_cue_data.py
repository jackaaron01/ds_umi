#!/usr/bin/env python3
"""
Generate EGO data with TWO colored markers (red/green).

CRITICAL: the target COLOR alternates per episode. Both markers are at
the SAME positions in paired episodes, but the correct action DIFFERS
based on which color is the target.

This FORCES the model to use image content — state alone cannot
determine which marker is the target.

Usage:
    python3 generate_color_cue_data.py -n 400 -o data/color_cue_dataset --render
"""
import os, sys, time, argparse
import numpy as np
import h5py

sys.path.insert(0, "/workspace/umi")
os.environ["MUJOCO_GL"] = "glx"
import mujoco

from stage_2.generate_diverse_data import SyntheticFeatureGenerator, CONTROL_RATE
from stage_1.kinematics.ik import solve_ik
from stage_1.kinematics.dh_params import XARM6_JOINT_LIMITS

JOINT_LIMITS = np.array(XARM6_JOINT_LIMITS)


def random_joint_config(rng):
    return rng.uniform(JOINT_LIMITS[:, 0] + 0.3, JOINT_LIMITS[:, 1] - 0.3)


def generate_color_episode(output_dir, episode_idx, model,
                           feature_gen=None, rng=None, renderer=None):
    """Generate episode reaching toward the correct colored marker."""
    if rng is None:
        rng = np.random.RandomState(episode_idx)
    if feature_gen is None:
        feature_gen = SyntheticFeatureGenerator()

    data = mujoco.MjData(model)
    PHYSICS_STEPS = max(1, int(1.0 / CONTROL_RATE / model.opt.timestep))

    # ── Place BOTH markers at random positions ──
    red_pos = np.array([
        rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3),
        rng.uniform(0.15, 0.5)
    ])
    green_pos = np.array([
        rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3),
        rng.uniform(0.15, 0.5)
    ])
    # Ensure markers are separated
    while np.linalg.norm(red_pos - green_pos) < 0.15:
        green_pos = np.array([
            rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3),
            rng.uniform(0.15, 0.5)
        ])

    # ── Determine target color for this episode ──
    target_is_red = (episode_idx % 2 == 0)  # even=red, odd=green
    target_pos = red_pos if target_is_red else green_pos

    # Hide the non-target marker (move far away, out of camera view)
    hidden_pos = np.array([10.0, 10.0, -5.0])  # far below floor
    if model.nmocap >= 2:
        if target_is_red:
            data.mocap_pos[0] = red_pos     # red = visible target
            data.mocap_pos[1] = hidden_pos  # green = hidden
        else:
            data.mocap_pos[0] = hidden_pos  # red = hidden
            data.mocap_pos[1] = green_pos   # green = visible target

    # ── Compute IK target ──
    T_target = np.eye(4)
    T_target[:3, 3] = target_pos
    T_target[:3, :3] = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    goal_q, ik_ok, _, _ = solve_ik(T_target, q_init=random_joint_config(rng), max_iterations=100)
    if not ik_ok:
        goal_q = random_joint_config(rng)

    # Start position
    start_q = random_joint_config(rng)
    while np.linalg.norm(start_q - goal_q) < 0.8:
        start_q = random_joint_config(rng)

    # ── Generate trajectory ──
    n_waypoints = rng.randint(5, 10)
    alpha = np.linspace(0, 1, n_waypoints)
    curve = np.sin(alpha[:, None] * np.pi) * rng.randn(6) * 0.2
    waypoints = (1 - alpha[:, None]) * start_q + alpha[:, None] * goal_q + curve
    waypoints[0] = start_q; waypoints[-1] = goal_q

    data.qpos[:6] = start_q.copy()
    mujoco.mj_forward(model, data)

    max_vel = rng.uniform(0.3, 2.0)
    cmd_noise = rng.uniform(0.001, 0.006)
    obs_noise = rng.uniform(0.001, 0.004)
    dt = 1.0 / CONTROL_RATE
    current_cmd = start_q.copy()
    steps_per_target = rng.randint(10, 22)

    joint_cmd_list, joint_state_list, joint_vel_list = [], [], []
    timestamps_list, img_feat_list, label_list, image_list = [], [], [], []
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
            noisy_cmd = current_cmd + rng.randn(6) * cmd_noise
            data.ctrl[:6] = noisy_cmd
            for _ in range(PHYSICS_STEPS):
                mujoco.mj_step(model, data)

            joint_cmd_list.append(current_cmd.copy())
            joint_state_list.append(data.qpos[:6] + rng.randn(6) * obs_noise)
            joint_vel_list.append(data.qvel[:6])
            timestamps_list.append(time.time() - t0)
            img_feat_list.append(feature_gen.encode(data.qpos[:6]))
            label_list.append(1.0 if target_is_red else 0.0)  # 1=red, 0=green
            if renderer is not None:
                image_list.append(renderer.render(data, camera="ego"))

    # Lookahead actions
    joint_cmd_raw = np.array(joint_cmd_list, dtype=np.float64)
    joint_state = np.array(joint_state_list, dtype=np.float64)
    joint_vel = np.array(joint_vel_list, dtype=np.float64)
    timestamps = np.array(timestamps_list, dtype=np.float64)
    img_feat = np.array(img_feat_list, dtype=np.float32)
    labels = np.array(label_list, dtype=np.float32)
    n_steps = len(joint_cmd_raw)

    LOOKAHEAD = rng.randint(20, 30)
    joint_cmd = np.zeros_like(joint_cmd_raw)
    for i in range(n_steps):
        joint_cmd[i] = joint_cmd_raw[min(i + LOOKAHEAD, n_steps - 1)]

    h5_path = os.path.join(output_dir, f"episode_{episode_idx:06d}.h5")
    with h5py.File(h5_path, "w") as f:
        ep = f.create_group(f"episode_{episode_idx:06d}")
        ep.create_dataset("joint_command/position", data=joint_cmd, compression="gzip")
        ep.create_dataset("joint_state/position", data=joint_state, compression="gzip")
        ep.create_dataset("joint_state/velocity", data=joint_vel, compression="gzip")
        ep.create_dataset("joint_command/position_timestamp", data=timestamps, compression="gzip")
        ep.create_dataset("joint_state/position_timestamp", data=timestamps, compression="gzip")
        ep.create_dataset("gripper/command", data=np.ones(n_steps), compression="gzip")
        ep.create_dataset("gripper/state", data=np.ones(n_steps), compression="gzip")
        ep.create_dataset("observation/image_features", data=img_feat, compression="gzip")
        ep.create_dataset("observation/target_is_red", data=labels, compression="gzip")
        if renderer is not None and len(image_list) > 0:
            images = np.stack(image_list, axis=0)
            ep.create_dataset("sensors/camera/ego", data=images, compression="gzip",
                              chunks=(1, images.shape[1], images.shape[2], 3))
        ep.attrs["num_steps"] = n_steps
        ep.attrs["target_is_red"] = 1 if target_is_red else 0

    return h5_path, n_steps


def main():
    parser = argparse.ArgumentParser(description="Generate color-cued EGO data")
    parser.add_argument("-n", "--episodes", type=int, default=400)
    parser.add_argument("-o", "--output", default="/workspace/umi/data/color_cue_dataset")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--img-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    os.makedirs(args.output, exist_ok=True)

    model_path = os.path.join(os.path.dirname(__file__),
                              "simulation", "xarm6.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    print(f"Model: {model.nbody} bodies, mocap={model.nmocap}")

    feature_gen = SyntheticFeatureGenerator(seed=args.seed)
    renderer = None
    if args.render:
        from stage_2.mujoco_renderer import MuJoCoRenderer
        renderer = MuJoCoRenderer(model, width=args.img_size, height=args.img_size)
        print(f"Renderer: {args.img_size}x{args.img_size}")

    total_steps = 0
    t0 = time.time()
    for ep in range(args.episodes):
        _, n_steps = generate_color_episode(
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


if __name__ == "__main__":
    main()
