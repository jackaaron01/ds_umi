#!/usr/bin/env python3
"""End-to-end pipeline validation: data → train → evaluate.

Generates a small dataset, trains ACT and DP briefly, verifies
loss decreases and no crashes. Use to validate the full pipeline
after any code changes.

Usage:
    python3 validate_pipeline.py
"""

import os, sys, time, tempfile, json
import numpy as np
import torch

sys.path.insert(0, "/workspace/umi")


def test_data_generation():
    """Generate 10 episodes, verify output."""
    from stage_2.generate_diverse_data import main as gen
    import argparse
    import shutil

    tmpdir = tempfile.mkdtemp(prefix="val_")
    # Run generation programmatically
    import subprocess
    result = subprocess.run([
        sys.executable, os.path.join(os.path.dirname(__file__), "generate_diverse_data.py"),
        "-n", "10", "-o", tmpdir, "--v3", "--seed", "0"
    ], capture_output=True, text=True, timeout=30)

    v3_dir = tmpdir + "_v3"
    parquet = os.path.join(v3_dir, "data/chunk-000/file-000.parquet")
    info = os.path.join(v3_dir, "meta/info.json")

    if os.path.isfile(parquet) and os.path.isfile(info):
        import pandas as pd
        df = pd.read_parquet(parquet)
        n_eps = df.episode_index.nunique()
        print(f"  ✓ Data generation: {n_eps} eps, {len(df)} frames")
        return tmpdir, v3_dir, True
    else:
        print(f"  ✗ Data generation failed")
        return tmpdir, v3_dir, False


def test_act_training(data_dir):
    """Train ACT for 200 steps, verify loss decreases."""
    from stage_2.train_act import main as train_act

    outdir = tempfile.mkdtemp(prefix="val_act_")
    import subprocess
    result = subprocess.run([
        sys.executable, os.path.join(os.path.dirname(__file__), "train_act.py"),
        "--data", data_dir, "--output", outdir,
        "--steps", "200", "--batch-size", "8", "--dim-model", "128"
    ], capture_output=True, text=True, timeout=120)

    # Parse loss from output
    for line in result.stdout.split("\n"):
        if "ratio=" in line:
            ratio = float(line.split("ratio=")[1].strip())
            passed = ratio < 0.95  # loss should decrease
            status = "✓" if passed else "✗"
            print(f"  {status} ACT training: loss ratio={ratio:.3f} (<0.95 = ok)")
            return outdir, passed

    print(f"  ✗ ACT training: could not parse output")
    return outdir, False


def test_dp_training(data_dir):
    """Train DP for 100 steps, verify no crash."""
    outdir = tempfile.mkdtemp(prefix="val_dp_")
    import subprocess
    result = subprocess.run([
        sys.executable, os.path.join(os.path.dirname(__file__), "train_dp.py"),
        "--data", data_dir, "--output", outdir,
        "--steps", "100", "--batch-size", "4"
    ], capture_output=True, text=True, timeout=120)

    passed = "Training done" in result.stdout or "Loss:" in result.stdout
    status = "✓" if passed else "✗"
    print(f"  {status} DP training: {'completed' if passed else 'failed'}")
    return outdir, passed


def main():
    print("=" * 55)
    print("  UMI Pipeline Validation")
    print("=" * 55)
    print()

    results = []

    # 1. Data generation
    print("[1/3] Data generation...")
    tmpdir, v3dir, ok = test_data_generation()
    results.append(("Data generation", ok))

    if not ok:
        print("\n  Aborting: data generation failed")
        return 1

    # 2. ACT training
    print("\n[2/3] ACT training (200 steps)...")
    actdir, ok = test_act_training(v3dir)
    results.append(("ACT training", ok))

    # 3. DP training
    print("\n[3/3] DP training (100 steps)...")
    dpdir, ok = test_dp_training(v3dir)
    results.append(("DP training", ok))

    # Summary
    print(f"\n{'=' * 55}")
    print(f"  Results:")
    for name, ok in results:
        print(f"    {'✓' if ok else '✗'} {name}")
    all_ok = all(r[1] for r in results)
    print(f"\n  Pipeline: {'✓ ALL PASSED' if all_ok else '✗ FAILURES'}")
    print(f"{'=' * 55}")

    # Cleanup
    import shutil
    for d in [tmpdir, v3dir, actdir, dpdir]:
        if d and os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
