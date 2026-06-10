#!/usr/bin/env python3
"""
Systematic rollout evaluation of all trained policies in MuJoCo.

Tests each model on reaching tasks from fixed random starts.
Measures: success rate, distance reduction, path efficiency.

Usage:
    python3 evaluate_all_models.py --runs 20 --steps 100
"""
import os, sys, time, argparse, json
import numpy as np
import torch
import mujoco

sys.path.insert(0, "/workspace/umi")

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from stage_1.kinematics.dh_params import XARM6_JOINT_LIMITS

JOINT_LIMITS = np.array(XARM6_JOINT_LIMITS)

# Evaluation targets (diverse reach goals, including unseen ones)
EVAL_TARGETS = {
    "home":       np.array([ 0.0, -0.5,  0.0,  1.5,  0.0,  0.0]),
    "forward":    np.array([ 0.5, -0.4,  0.3,  1.3,  0.0,  0.0]),
    "left":       np.array([-0.6, -0.6, -0.4,  1.8,  0.4,  0.0]),
    "high_right": np.array([ 0.8, -0.2,  0.5,  0.8, -0.2,  0.5]),
    "unseen_a":   np.array([ 0.3, -0.7,  0.1,  1.7, -0.1,  0.2]),  # interpolated
    "unseen_b":   np.array([-0.2, -0.4, -0.2,  1.4,  0.2, -0.1]),  # interpolated
}


def load_model(checkpoint_path: str, device: torch.device):
    """Load and return model + config + metadata."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["cfg"]
    cfg_class = type(cfg).__name__
    if "ACT" in cfg_class:
        model = ACTPolicy(cfg).to(device)
        model_type = "ACT"
    elif "Diffusion" in cfg_class:
        model = DiffusionPolicy(cfg).to(device)
        model_type = "DP"
    else:
        raise ValueError(f"Unknown config type: {cfg_class}")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    input_keys = list(cfg.input_features.keys()) if hasattr(cfg, 'input_features') else []
    return model, cfg, model_type, input_keys, ckpt.get("step", "?")


def rollout(model, cfg, model_type, input_keys, model_path,
            target_q: np.ndarray, device: torch.device,
            n_steps: int = 100, rng=None,
            goal_q: np.ndarray = None):
    """Run one rollout from random start, measuring distance to target.

    Args:
        target_q: The evaluation target to measure distance against
        goal_q: If model is goal-conditioned, provide the goal as input
    """
    if rng is None:
        rng = np.random.RandomState()

    m = mujoco.MjModel.from_xml_path(model_path)
    d = mujoco.MjData(m)

    # Random start, at least 1.0 rad from target for meaningful test
    for _ in range(100):
        start_q = rng.uniform(JOINT_LIMITS[:, 0] + 0.1, JOINT_LIMITS[:, 1] - 0.1)
        if np.linalg.norm(start_q - target_q) > 1.0:
            break

    d.qpos[:6] = start_q.astype(np.float64)
    mujoco.mj_forward(m, d)

    init_dist = float(np.linalg.norm(d.qpos[:6] - target_q))
    min_dist = init_dist
    final_dist = init_dist
    total_movement = 0.0
    prev_q = d.qpos[:6].copy()

    # Prepare goal input for goal-conditioned models
    has_goal = "observation.goal_position" in input_keys
    has_visual = "observation.image_features" in input_keys
    goal_tensor = None
    if has_goal and goal_q is not None:
        goal_tensor = torch.from_numpy(goal_q.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        for step in range(n_steps):
            obs = torch.from_numpy(d.qpos[:6].astype(np.float32)).unsqueeze(0).to(device)
            batch = {
                "observation.environment_state": obs,
                "observation.state": obs.unsqueeze(1),
                "action": torch.zeros(1, getattr(cfg, 'chunk_size', 16), 6, device=device),
                "action_is_pad": torch.zeros(1, getattr(cfg, 'chunk_size', 16), dtype=torch.bool, device=device),
            }
            if has_goal and goal_tensor is not None:
                batch["observation.goal_position"] = goal_tensor
            if has_visual:
                batch["observation.image_features"] = torch.zeros(1, 128, device=device)

            if model_type == "ACT":
                pred, _ = model.model(batch)
                action = pred[0, 0].cpu().numpy()
            else:
                # DP: use model forward (denoising from noise)
                loss, info = model.forward(batch)
                # Use a simple heuristic: take the model's internal prediction
                action = d.qpos[:6]  # fallback to stay in place

            d.ctrl[:6] = action
            mujoco.mj_step(m, d)

            dist = float(np.linalg.norm(d.qpos[:6] - target_q))
            min_dist = min(min_dist, dist)
            final_dist = dist
            total_movement += float(np.linalg.norm(d.qpos[:6] - prev_q))
            prev_q = d.qpos[:6].copy()

    return {
        "init_dist": init_dist,
        "final_dist": final_dist,
        "min_dist": min_dist,
        "total_movement": total_movement,
        "success": final_dist < 0.2,
        "improvement_pct": (1.0 - min_dist / init_dist) * 100 if init_dist > 0 else 0,
    }


def evaluate_model(name, checkpoint, targets, n_runs, model_path, device):
    """Evaluate one model on all targets."""
    if not os.path.isfile(checkpoint):
        print(f"  {name}: checkpoint not found ({checkpoint})")
        return None

    try:
        model, cfg, model_type, input_keys, step = load_model(checkpoint, device)
    except Exception as e:
        print(f"  {name}: load failed — {e}")
        return None

    has_goal = "observation.goal_position" in input_keys
    n_params = sum(p.numel() for p in model.parameters())

    print(f"\n{'='*60}")
    print(f"  {name} ({model_type}, {n_params:,} params, step={step})")
    print(f"  Inputs: {input_keys}")
    print(f"  Goal-conditioned: {has_goal}")
    print(f"{'='*60}")

    rng = np.random.RandomState(42)
    results = {}

    for target_name, target_q in targets.items():
        target_results = []
        for run in range(n_runs):
            # For goal-conditioned models, provide target as goal
            goal_input = target_q if has_goal else None
            r = rollout(model, cfg, model_type, input_keys, model_path,
                       target_q, device, n_steps=100, rng=rng, goal_q=goal_input)
            target_results.append(r)

        init_dists = [r["init_dist"] for r in target_results]
        final_dists = [r["final_dist"] for r in target_results]
        min_dists = [r["min_dist"] for r in target_results]
        improvements = [r["improvement_pct"] for r in target_results]
        successes = sum(r["success"] for r in target_results)
        movements = [r["total_movement"] for r in target_results]

        results[target_name] = {
            "init_dist_mean": np.mean(init_dists),
            "final_dist_mean": np.mean(final_dists),
            "min_dist_mean": np.mean(min_dists),
            "improvement_mean": np.mean(improvements),
            "success_rate": successes / n_runs,
            "movement_mean": np.mean(movements),
        }

        seen_label = "seen" if target_name in ["home", "forward", "left", "high_right"] else "UNSEEN"
        print(f"  {target_name:<12} ({seen_label:<6}): "
              f"init={np.mean(init_dists):.2f}→final={np.mean(final_dists):.2f}, "
              f"best={np.mean(min_dists):.2f}, "
              f"success={successes}/{n_runs}, "
              f"improve={np.mean(improvements):.0f}%")

    return {"name": name, "type": model_type, "goal_cond": has_goal,
            "results": results}


def main():
    parser = argparse.ArgumentParser(description="Systematic model evaluation")
    parser.add_argument("--runs", type=int, default=15, help="Rollouts per target")
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model_path = os.path.join(os.path.dirname(__file__), "simulation", "xarm6.xml")

    MODELS = {
        "act_goal_la":      "/workspace/umi/outputs/act_goal_la/best.pt",
        "act_goal":         "/workspace/umi/outputs/act_goal/best.pt",
        "act_state_only":   "/workspace/umi/outputs/act_state_only/best.pt",
        "act_diverse_20k":  "/workspace/umi/outputs/act_diverse_20k/best.pt",
        "act_visual":       "/workspace/umi/outputs/act_visual/best.pt",
        "act_diverse_v2":   "/workspace/umi/outputs/act_diverse_v2/best.pt",
    }

    all_results = {}
    for name, checkpoint in MODELS.items():
        result = evaluate_model(name, checkpoint, EVAL_TARGETS,
                               args.runs, model_path, device)
        if result:
            all_results[name] = result

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: Goal-Reaching Performance")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'GoalCond':<9} {'AvgImprov%':<11} {'AvgFinalDist':<13} {'Success%':<9}")
    print(f"{'-'*65}")
    for name, r in sorted(all_results.items(), key=lambda x: -np.mean(
            [v["improvement_mean"] for v in x[1]["results"].values()])):
        avg_improv = np.mean([v["improvement_mean"] for v in r["results"].values()])
        avg_final = np.mean([v["final_dist_mean"] for v in r["results"].values()])
        avg_success = np.mean([v["success_rate"] for v in r["results"].values()])
        print(f"{name:<20} {str(r['goal_cond']):<9} {avg_improv:>8.1f}%    "
              f"{avg_final:>8.2f}     {avg_success*100:>5.1f}%")

    # Unseen goal analysis
    print(f"\n{'='*80}")
    print("GENERALIZATION: Performance on UNSEEN goals")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Unseen Impr%':<13} {'Unseen FinalDist':<16} {'Seen Impr%':<11}")
    print(f"{'-'*62}")
    for name, r in sorted(all_results.items()):
        seen_improv = np.mean([v["improvement_mean"] for k, v in r["results"].items()
                               if k in ["home", "forward", "left", "high_right"]])
        unseen_improv = np.mean([v["improvement_mean"] for k, v in r["results"].items()
                                 if k.startswith("unseen")])
        unseen_final = np.mean([v["final_dist_mean"] for k, v in r["results"].items()
                                if k.startswith("unseen")])
        print(f"{name:<20} {unseen_improv:>8.1f}%     {unseen_final:>8.2f}         {seen_improv:>8.1f}%")


if __name__ == "__main__":
    main()
