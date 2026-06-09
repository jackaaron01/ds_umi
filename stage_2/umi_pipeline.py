#!/usr/bin/env python3
"""
UMI Pipeline — unified entry point for the full workflow.

Usage:
    # Quick validation (small test run)
    python3 umi_pipeline.py validate

    # Full training run
    python3 umi_pipeline.py train --eps 300 --steps 10000

    # Train on human teleop data
    python3 umi_pipeline.py train --data /workspace/umi/data/teleop_dataset_v3

    # Compare all trained models
    python3 umi_pipeline.py compare

    # List available models
    python3 umi_pipeline.py models
"""

import os, sys, time, json, argparse, subprocess

sys.path.insert(0, "/workspace/umi")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def run(cmd, **kwargs):
    """Run a subprocess, print output, return success."""
    print(f"  → {' '.join(cmd[:4])}...")
    result = subprocess.run(cmd, cwd=SCRIPT_DIR, **kwargs)
    return result.returncode == 0


def cmd_validate(args):
    """Quick validation: generate 10 eps, train ACT+DP briefly."""
    print("=" * 55)
    print("  UMI Pipeline — Validate")
    print("=" * 55)
    success = run([sys.executable, "validate_pipeline.py"])
    print(f"\n  {'✓ Pipeline OK' if success else '✗ Validation failed'}")
    return 0 if success else 1


def cmd_train(args):
    """Generate data and/or train models."""
    data_dir = args.data

    # Generate data if needed
    if not args.data:
        print("=" * 55)
        print(f"  Step 1/3: Generate {args.eps} episodes")
        print("=" * 55)
        data_dir = f"/workspace/umi/data/pipeline_{args.eps}"
        gen_cmd = [
            sys.executable, "generate_diverse_data.py",
            "-n", str(args.eps), "-o", data_dir, "--v3",
            "--seed", str(args.seed),
        ]
        if not run(gen_cmd):
            return 1
        data_dir = data_dir + "_v3"

    # Train ACT
    print(f"\n{'=' * 55}")
    print(f"  Step 2/3: Train ACT ({args.steps} steps)")
    print(f"{'=' * 55}")
    act_out = args.output + "/act" if args.output else "/workspace/umi/outputs/pipeline_act"
    act_cmd = [
        sys.executable, "train_act.py",
        "--data", data_dir, "--output", act_out,
        "--steps", str(args.steps), "--batch-size", str(args.batch_size),
    ]
    run(act_cmd)

    # Train DP
    if not args.skip_dp:
        print(f"\n{'=' * 55}")
        print(f"  Step 3/3: Train Diffusion Policy (5000 steps)")
        print(f"{'=' * 55}")
        dp_out = args.output + "/dp" if args.output else "/workspace/umi/outputs/pipeline_dp"
        dp_cmd = [
            sys.executable, "train_dp.py",
            "--data", data_dir, "--output", dp_out,
            "--steps", "5000", "--batch-size", str(args.batch_size),
        ]
        run(dp_cmd)

    print(f"\n{'=' * 55}")
    print(f"  Training complete")
    print(f"  ACT: {act_out}/best.pt")
    if not args.skip_dp:
        print(f"  DP:  {dp_out}/best.pt")
    print(f"{'=' * 55}")
    return 0


def cmd_compare(args):
    """Compare all available models."""
    print("=" * 55)
    print("  UMI Pipeline — Model Comparison")
    print("=" * 55)

    # Check which models exist
    models = {
        "ACT (diverse 20K)": "/workspace/umi/outputs/act_diverse_20k/best.pt",
        "ACT (diverse v2)": "/workspace/umi/outputs/act_diverse_v2/best.pt",
        "ACT (teleop)": "/workspace/umi/outputs/act_teleop/best.pt",
        "DP (diverse 10K)": "/workspace/umi/outputs/dp_diverse_10k/best.pt",
    }

    import pandas as pd, numpy as np, torch
    from lerobot.policies.act.modeling_act import ACTPolicy

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Test on teleop data
    tel_dir = "/workspace/umi/data/teleop_dataset_v3"
    tel_parquet = os.path.join(tel_dir, "data/chunk-000/file-000.parquet")
    test_frames = 0
    if os.path.isfile(tel_parquet):
        df = pd.read_parquet(tel_parquet)
        test_frames = len(df)
    else:
        print("  No teleop data found for evaluation")
        return 1

    print(f"\n  Test data: {df.episode_index.nunique()} episodes, {test_frames} frames")
    print(f"\n  {'Model':<25} {'Status':<10} {'Params':<12} {'Test MSE'}")
    print(f"  {'-'*55}")

    for name, path in models.items():
        if not os.path.isfile(path):
            print(f"  {name:<25} {'not found':<10}")
            continue

        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            cfg_class = type(ckpt["cfg"]).__name__
            if "Diffusion" in cfg_class:
                print(f"  {name:<25} {'DP (skip)':<10} {'—':<12} {'—'}")
                continue

            m = ACTPolicy(ckpt["cfg"]).to(device)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            n_params = sum(p.numel() for p in m.parameters())

            se, n = 0.0, 0
            for ep_idx, group in df.groupby("episode_index"):
                obs = np.vstack(group["observation.joint_position"].values).astype(np.float32)
                act = np.vstack(group["action.joint_position"].values).astype(np.float32)
                for i in range(min(len(obs), len(act))):
                    obs_t = torch.from_numpy(obs[i:i+1]).to(device)
                    with torch.no_grad():
                        pred, _ = m.model({
                            "observation.environment_state": obs_t,
                            "observation.state": obs_t.unsqueeze(1),
                        })
                    se += np.sum((pred[0, 0].cpu().numpy() - act[i])**2)
                    n += 6
            mse = se / max(n, 1)
            print(f"  {name:<25} {'✓':<10} {n_params:>10,}  {mse:.4f}")
        except Exception as e:
            print(f"  {name:<25} {'error':<10} {'—':<12} {str(e)[:30]}")

    print()
    return 0


def cmd_models(args):
    """List available models from registry."""
    reg_path = "/workspace/umi/outputs/model_registry.json"
    if os.path.isfile(reg_path):
        with open(reg_path) as f:
            registry = json.load(f)
        print("=" * 55)
        print("  Model Registry")
        print("=" * 55)
        for m in registry["models"]:
            exists = "✓" if os.path.isfile(m["path"]) else "✗"
            print(f"\n  [{exists}] {m['name']}")
            print(f"      Type: {m['type']}, Params: {m['params']:,}")
            print(f"      Data: {m['data']}")
            print(f"      Steps: {m['steps']}, Loss: {m.get('final_loss', '?')}")
            if m.get('test_mse_human'):
                print(f"      Test MSE (human): {m['test_mse_human']}")
            print(f"      {m.get('notes', '')}")
        print()
    else:
        print("No model registry found.")


def main():
    parser = argparse.ArgumentParser(description="UMI Pipeline")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("validate", help="Quick validation test")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("train", help="Generate data and train models")
    p.add_argument("--data", help="Existing v3 dataset directory")
    p.add_argument("--eps", type=int, default=300, help="Episodes to generate")
    p.add_argument("--steps", type=int, default=10000, help="ACT training steps")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--output", help="Output directory")
    p.add_argument("--skip-dp", action="store_true", help="Skip Diffusion Policy")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("compare", help="Compare all trained models")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("models", help="List available models")
    p.set_defaults(func=cmd_models)

    args = parser.parse_args()
    if args.cmd is None:
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
