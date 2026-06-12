#!/usr/bin/env python3
"""
Generate EGO data with VARYING marker positions within each episode.

Key innovation: the marker MOVES during the episode, so the model sees
the SAME joint state with DIFFERENT marker positions requiring DIFFERENT
actions. This FORCES the model to extract spatial information from images.

Without this, the model treats images as a binary "present/absent" switch
and ignores image content (see previous EGO experiments).

Usage:
    python3 generate_varying_marker_data.py -n 300 -o data/vary_marker_dataset
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


def generate_varying_episode(output_dir, episode_idx, model,
                             feature_gen=None, rng=None, renderer=None):
    """Generate episode where the marker moves 3-5 times."""
    if rng is None:
        rng = np.random.RandomState(episode_idx)
    if feature_gen is None:
        feature_gen = SyntheticFeatureGenerator()

    data = mujoco.MjData(model)
    n_marker_moves = rng.randint(3, 6)  # marker moves 3-5 times per episode
    PHYSICS_STEPS = max(1, int(1.0 / CONTROL_RATE / model.opt.timestep))

    joint_cmd_list, joint_state_list, joint_vel_list = [], [], []
    timestamps_list, img_feat_list, marker_list, image_list = [], [], [], []
    t0 = time.time()

    # Start from random config
    current_q = random_joint_config(rng)
    data.qpos[:6] = current_q.copy()
    mujoco.mj_forward(model, data)

    current_cmd = current_q.copy()
    max_vel = rng.uniform(0.3, 2.0)
    cmd_noise = rng.uniform(0.001, 0.006)
    obs_noise = rng.uniform(0.001, 0.004)
    dt = 1.0 / CONTROL_RATE

    for move_idx in range(n_marker_moves):
        # ── Place marker at new random position ──
        marker_pos = np.array([
            rng.uniform(-0.35, 0.35),
            rng.uniform(-0.35, 0.35),
            rng.uniform(0.12, 0.55),
        ])
        if model.nmocap > 0:
            data.mocap_pos[0] = marker_pos

        # ── Compute IK target for this marker ──
        T_target = np.eye(4)
        T_target[:3, 3] = marker_pos
        T_target[:3, :3] = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
        goal_q, ik_ok, _, _ = solve_ik(T_target, q_init=current_q, max_iterations=100)
        if not ik_ok:
            goal_q = random_joint_config(rng)

        # ── Move toward this marker for several steps ──
        # Generate waypoints from current position toward goal
        n_waypoints = rng.randint(3, 8)
        alpha = np.linspace(0, 1, n_waypoints)
        # Partial progress: only go 60-90% of the way (marker will move again)
        progress = rng.uniform(0.6, 0.9)
        waypoints = (1 - alpha[:, None] * progress) * current_q + \
                    (alpha[:, None] * progress) * goal_q

        steps_per_target = rng.randint(8, 20)
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
                marker_list.append(marker_pos.copy())
                if renderer is not None:
                    image_list.append(renderer.render(data, camera="ego"))

        # Update current position for next marker
        current_q = data.qpos[:6].copy()

    # ── Lookahead actions ──
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
        ep.create_dataset("observation/marker_position", data=markers, compression="gzip")
        if renderer is not None and len(image_list) > 0:
            images = np.stack(image_list, axis=0)
            ep.create_dataset("sensors/camera/ego", data=images, compression="gzip",
                              chunks=(1, images.shape[1], images.shape[2], 3))
        ep.attrs["num_steps"] = n_steps

    return h5_path, n_steps


def main():
    parser = argparse.ArgumentParser(description="Generate varying-marker EGO data")
    parser.add_argument("-n", "--episodes", type=int, default=300)
    parser.add_argument("-o", "--output", default="/workspace/umi/data/vary_marker_dataset")
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

    total_steps = 0
    t0 = time.time()
    for ep in range(args.episodes):
        _, n_steps = generate_varying_episode(
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
