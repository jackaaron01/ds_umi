#!/usr/bin/env python3
"""
Evaluate multi-task ACT policy in MuJoCo rollout.

For each task, runs N rollouts from random start positions and measures
the final distance to the task target. Compares:
  - Multi-task model (with correct task_index)
  - Multi-task model (with wrong task_index) — tests task conditioning
  - Single-task baseline (best single-task model)

Usage:
    python3 evaluate_multitask.py --model outputs/act_multitask/best.pt --runs 10
"""
import os, sys, argparse, time
import numpy as np
import torch
import mujoco

sys.path.insert(0, "/workspace/umi")

from lerobot.policies.act.modeling_act import ACTPolicy
from stage_1.kinematics.dh_params import XARM6_JOINT_LIMITS
from stage_2.generate_multitask_data import TASK_TARGETS

JOINT_LIMITS = np.array(XARM6_JOINT_LIMITS)


def load_multitask_model(checkpoint_path: str, device: torch.device):
    """Load task-conditioned ACT model."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = ACTPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    n_tasks = ckpt.get("n_tasks", 5)
    return model, cfg, n_tasks


def rollout_to_target(model, cfg, model_path, target_q, task_idx, n_tasks,
                      device, n_steps=100, rng=None):
    """Rollout policy from random start, measure distance to target."""
    if rng is None:
        rng = np.random.RandomState()

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    # Random start (at least 0.5 rad away from target)
    limits = JOINT_LIMITS
    start_q = rng.uniform(limits[:, 0] + 0.1, limits[:, 1] - 0.1)
    while np.linalg.norm(start_q - target_q) < 0.5:
        start_q = rng.uniform(limits[:, 0] + 0.1, limits[:, 1] - 0.1)

    d.qpos[:6] = start_q.astype(np.float64)
    mujoco.mj_forward(m, d)

    initial_dist = np.linalg.norm(d.qpos[:6] - target_q)
    min_dist = initial_dist
    final_dist = initial_dist

    # Task one-hot
    task_onehot = torch.zeros(1, n_tasks, device=device)
    task_onehot[0, task_idx] = 1.0

    has_visual = "observation.image_features" in str(cfg.input_features)

    with torch.no_grad():
        for step in range(n_steps):
            obs = torch.from_numpy(d.qpos[:6].astype(np.float32)).unsqueeze(0).to(device)
            batch = {
                "observation.environment_state": obs,
                "observation.task_index": task_onehot,
                "observation.state": obs.unsqueeze(1),
                "action": torch.zeros(1, cfg.chunk_size, 6, device=device),
                "action_is_pad": torch.zeros(1, cfg.chunk_size, dtype=torch.bool, device=device),
            }
            if has_visual:
                # Use zeros as fallback for visual features
                batch["observation.image_features"] = torch.zeros(1, 128, device=device)

            pred, _ = model.model(batch)
            action = pred[0, 0].cpu().numpy()

            # Apply action as target position
            d.ctrl[:6] = action
            mujoco.mj_step(m, d)

            dist = np.linalg.norm(d.qpos[:6] - target_q)
            min_dist = min(min_dist, dist)
            final_dist = dist

    return {
        "initial_dist": float(initial_dist),
        "final_dist": float(final_dist),
        "min_dist": float(min_dist),
        "success": final_dist < 0.2,  # within 0.2 rad = ~11°
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-task ACT policy")
    parser.add_argument("--model", default="/workspace/umi/outputs/act_multitask/best.pt")
    parser.add_argument("--baseline", default="/workspace/umi/outputs/act_state_only/best.pt")
    parser.add_argument("--runs", type=int, default=10, help="Rollouts per task")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = os.path.join(os.path.dirname(__file__), "simulation", "xarm6.xml")

    # Load multi-task model
    print(f"\nLoading multi-task model: {args.model}")
    mt_model, mt_cfg, n_tasks = load_multitask_model(args.model, device)
    print(f"  Tasks: {n_tasks}")
    print(f"  Input features: {list(mt_cfg.input_features.keys())}")

    # Load single-task baseline
    has_baseline = os.path.isfile(args.baseline)
    if has_baseline:
        print(f"\nLoading baseline: {args.baseline}")
        bl_ckpt = torch.load(args.baseline, map_location=device, weights_only=False)
        bl_cfg = bl_ckpt["cfg"]
        bl_model = ACTPolicy(bl_cfg).to(device)
        bl_model.load_state_dict(bl_ckpt["model_state_dict"])
        bl_model.eval()
        print(f"  Input features: {list(bl_cfg.input_features.keys())}")

    rng = np.random.RandomState(42)

    print(f"\n{'='*70}")
    print(f"Multi-Task Policy Evaluation ({args.runs} runs × {n_tasks} tasks)")
    print(f"{'='*70}")

    for task_idx in range(n_tasks):
        target = TASK_TARGETS[task_idx]
        print(f"\nTask {task_idx}: target={target}")

        # Multi-task with correct task index
        results = []
        for run in range(args.runs):
            r = rollout_to_target(mt_model, mt_cfg, model_path, target,
                                  task_idx, n_tasks, device,
                                  n_steps=args.steps, rng=rng)
            results.append(r)

        final_dists = [r["final_dist"] for r in results]
        successes = sum(r["success"] for r in results)
        print(f"  Multi-task (correct task): "
              f"final_dist={np.mean(final_dists):.3f}±{np.std(final_dists):.3f}, "
              f"success={successes}/{args.runs} ({100*successes/args.runs:.0f}%)")

        # Multi-task with WRONG task index (tests task conditioning)
        wrong_task = (task_idx + 1) % n_tasks
        wrong_results = []
        for run in range(args.runs):
            r = rollout_to_target(mt_model, mt_cfg, model_path, target,
                                  wrong_task, n_tasks, device,
                                  n_steps=args.steps, rng=rng)
            wrong_results.append(r)
        wrong_dists = [r["final_dist"] for r in wrong_results]
        wrong_successes = sum(r["success"] for r in wrong_results)
        print(f"  Multi-task (wrong task={wrong_task}): "
              f"final_dist={np.mean(wrong_dists):.3f}±{np.std(wrong_dists):.3f}, "
              f"success={wrong_successes}/{args.runs} ({100*wrong_successes/args.runs:.0f}%)")

        # Single-task baseline (only if available)
        if has_baseline:
            bl_results = []
            for run in range(args.runs):
                b_r = rollout_to_target(bl_model, bl_cfg, model_path, target,
                                        0, 1, device,  # baseline has no task conditioning
                                        n_steps=args.steps, rng=rng)
                bl_results.append(b_r)
            bl_dists = [r["final_dist"] for r in bl_results]
            bl_successes = sum(r["success"] for r in bl_results)
            print(f"  Single-task baseline:      "
                  f"final_dist={np.mean(bl_dists):.3f}±{np.std(bl_dists):.3f}, "
                  f"success={bl_successes}/{args.runs} ({100*bl_successes/args.runs:.0f}%)")

    print(f"\n{'='*70}")
    print("Summary: Task Conditioning Effect")
    print(f"{'='*70}")
    print("The multi-task model should perform BETTER with correct task_index")
    print("than with wrong task_index, demonstrating task conditioning works.")


if __name__ == "__main__":
    main()
