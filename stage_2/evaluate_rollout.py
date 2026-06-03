#!/usr/bin/env python3
"""
MuJoCo rollout evaluation for ACT policies.

Loads a trained ACT checkpoint, runs the policy in MuJoCo simulation
from random initial configurations, and measures:
  - Reach-home: distance to zero config after rollout
  - Reach-target: distance to target config after rollout

Usage:
    python3 evaluate_rollout.py --checkpoint outputs/act_baseline/best.pt --runs 20
"""

import os, sys, argparse, time
import numpy as np
import torch
import mujoco

sys.path.insert(0, "/workspace/umi")

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from stage_1.kinematics.dh_params import XARM6_JOINT_LIMITS

JOINT_LIMITS = np.array(XARM6_JOINT_LIMITS)


def load_model(checkpoint_path: str, device: torch.device):
    """Load ACT model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    model = ACTPolicy(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, cfg


def random_config(rng, margin=0.1):
    """Random valid joint configuration."""
    limits = JOINT_LIMITS
    return rng.uniform(limits[:, 0] + margin, limits[:, 1] - margin).astype(np.float64)


def rollout_reach_home(model, cfg, device, model_path: str,
                       n_steps: int = 100, rng=None) -> dict:
    """Rollout from random config towards home (zero) position."""
    if rng is None:
        rng = np.random.RandomState()

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    # Random start
    start_q = random_config(rng)
    d.qpos[:6] = start_q
    mujoco.mj_forward(m, d)

    target_q = np.zeros(6)
    positions = [start_q.copy()]
    errors = [np.linalg.norm(start_q - target_q)]

    for step in range(n_steps):
        # Current observation
        obs = torch.from_numpy(d.qpos[:6].copy().astype(np.float32)).unsqueeze(0).to(device)
        obs_seq = obs.unsqueeze(1)

        with torch.no_grad():
            actions, _ = model.model({
                "observation.environment_state": obs,
                "observation.state": obs_seq,
            })

        # Take first action
        action = actions[0, 0].cpu().numpy().astype(np.float64)
        d.ctrl[:6] = action
        mujoco.mj_step(m, d)

        positions.append(d.qpos[:6].copy())
        errors.append(np.linalg.norm(d.qpos[:6] - target_q))

    final_error = errors[-1]
    min_error = min(errors)
    return {
        "start_q": start_q,
        "final_q": d.qpos[:6].copy(),
        "target_q": target_q,
        "initial_error": errors[0],
        "final_error": final_error,
        "min_error": min_error,
        "error_trace": errors,
        "success": final_error < 0.3,  # within 0.3 rad ≈ 17°
    }


def rollout_reach_target(model, cfg, device, model_path: str,
                         n_steps: int = 150, rng=None) -> dict:
    """Rollout from start config to a random target config."""
    if rng is None:
        rng = np.random.RandomState()

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    start_q = random_config(rng)
    target_q = random_config(rng)
    d.qpos[:6] = start_q
    mujoco.mj_forward(m, d)

    errors = [np.linalg.norm(start_q - target_q)]

    for step in range(n_steps):
        obs = torch.from_numpy(d.qpos[:6].copy().astype(np.float32)).unsqueeze(0).to(device)
        obs_seq = obs.unsqueeze(1)

        with torch.no_grad():
            actions, _ = model.model({
                "observation.environment_state": obs,
                "observation.state": obs_seq,
            })

        action = actions[0, 0].cpu().numpy().astype(np.float64)
        d.ctrl[:6] = action
        mujoco.mj_step(m, d)
        errors.append(np.linalg.norm(d.qpos[:6] - target_q))

    return {
        "start_q": start_q,
        "target_q": target_q,
        "final_q": d.qpos[:6].copy(),
        "initial_error": errors[0],
        "final_error": errors[-1],
        "min_error": min(errors),
        "error_trace": errors,
        "success": errors[-1] < 0.5,
    }


def baseline_zero_action(model_path, n_runs=20, n_steps=100, task="home", rng=None):
    """Zero-action baseline: robot stays at start position."""
    if rng is None:
        rng = np.random.RandomState()

    results = []
    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    for _ in range(n_runs):
        start_q = random_config(rng)
        d.qpos[:6] = start_q
        mujoco.mj_forward(m, d)

        target_q = np.zeros(6) if task == "home" else random_config(rng)
        initial_error = np.linalg.norm(start_q - target_q)

        for _ in range(n_steps):
            d.ctrl[:6] = d.qpos[:6]  # hold position
            mujoco.mj_step(m, d)

        final_error = np.linalg.norm(d.qpos[:6] - target_q)
        results.append({
            "initial_error": initial_error,
            "final_error": final_error,
            "success": final_error < 0.3 if task == "home" else final_error < 0.5,
        })

    return results


def print_summary(results: list, label: str):
    """Print evaluation summary."""
    successes = [r["success"] for r in results]
    final_errors = [r["final_error"] for r in results]
    init_errors = [r["initial_error"] for r in results]
    improvement = [(1 - fe / max(ie, 1e-10)) * 100 for fe, ie in zip(final_errors, init_errors)]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Runs:           {len(results)}")
    print(f"  Success rate:   {np.mean(successes)*100:.0f}%")
    print(f"  Init error:     {np.mean(init_errors):.3f} ± {np.std(init_errors):.3f} rad")
    print(f"  Final error:    {np.mean(final_errors):.3f} ± {np.std(final_errors):.3f} rad")
    print(f"  Improvement:    {np.mean(improvement):.0f}%")
    print(f"  Min final err:  {np.min(final_errors):.3f} rad")


def main():
    parser = argparse.ArgumentParser(description="MuJoCo rollout evaluation for ACT")
    parser.add_argument("--checkpoint", default="/workspace/umi/outputs/act_baseline/best.pt")
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--model", default=None)
    parser.add_argument("--task", choices=["home", "target", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model_path = args.model or os.path.join(
        os.path.dirname(__file__), "simulation", "xarm6.xml"
    )
    rng = np.random.RandomState(args.seed)

    # Baselines
    print("\n--- Zero-action baseline ---")
    home_zero = baseline_zero_action(model_path, args.runs, args.steps, "home", rng)
    print_summary(home_zero, "Zero-action → Home")
    target_zero = baseline_zero_action(model_path, args.runs, args.steps, "target", rng)
    print_summary(target_zero, "Zero-action → Target")

    # ACT model
    if os.path.isfile(args.checkpoint):
        print(f"\nLoading checkpoint: {args.checkpoint}")
        model, cfg = load_model(args.checkpoint, device)
        step_num = torch.load(args.checkpoint, map_location="cpu", weights_only=False).get("step", "?")
        print(f"  Trained at step: {step_num}")

        if args.task in ("home", "both"):
            results = []
            t0 = time.time()
            for i in range(args.runs):
                r = rollout_reach_home(model, cfg, device, model_path, args.steps, rng)
                results.append(r)
            elapsed = time.time() - t0
            print_summary(results, f"ACT → Home ({elapsed:.1f}s)")

        if args.task in ("target", "both"):
            results = []
            t0 = time.time()
            for i in range(args.runs):
                r = rollout_reach_target(model, cfg, device, model_path, args.steps, rng)
                results.append(r)
            elapsed = time.time() - t0
            print_summary(results, f"ACT → Target ({elapsed:.1f}s)")
    else:
        print(f"\nCheckpoint not found: {args.checkpoint}")
        print("Skipping ACT evaluation. Train a model first.")


if __name__ == "__main__":
    main()
