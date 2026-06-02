#!/usr/bin/env python3
"""
ACT baseline research — generate mock data, convert to v3, verify compatibility.
"""

import os, sys, json, tempfile, time, glob, shutil
import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, "/workspace/umi")


def generate_recording(output_dir: str, duration: float = 5.0):
    """Generate a mock teleop recording using the ROS2 pipeline."""
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from stage_1.teleop_bridge.mock_hand_tracker import MockHandTracker
    from stage_1.teleop_bridge.hand_mapper import HandMapper
    from stage_1.safety.safety_node import SafetyGuardian
    from stage_1.recorder.recorder_node import RecorderNode
    from stage_1.teleop_bridge.calibration import HandToRobotTransform
    from std_srvs.srv import Trigger

    rclpy.init()
    os.makedirs(output_dir, exist_ok=True)

    tracker = MockHandTracker()
    tracker.set_parameters([
        rclpy.parameter.Parameter("amplitude_x", value=0.03),
        rclpy.parameter.Parameter("amplitude_y", value=0.02),
        rclpy.parameter.Parameter("amplitude_z", value=0.02),
        rclpy.parameter.Parameter("offset_z", value=0.2),
    ])
    mapper = HandMapper(transform=HandToRobotTransform.mock_transform())
    safety = SafetyGuardian()
    safety.set_parameters([rclpy.parameter.Parameter("robot_mode", value="mock")])
    recorder = RecorderNode()
    recorder.set_parameters([rclpy.parameter.Parameter("output_dir", value=output_dir)])

    executor = MultiThreadedExecutor(num_threads=4)
    for node in [tracker, mapper, safety, recorder]:
        executor.add_node(node)

    req = Trigger.Request()
    recorder._on_start(req, Trigger.Response())
    print(f"[RECORD] Started ({duration}s) → {output_dir}")

    start = time.time()
    while time.time() - start < duration:
        executor.spin_once(timeout_sec=0.1)

    recorder._on_stop(req, Trigger.Response())
    executor.shutdown()
    for node in [tracker, mapper, safety, recorder]:
        node.destroy_node()
    rclpy.shutdown()

    h5s = sorted(glob.glob(os.path.join(output_dir, "*.h5")))
    print(f"[RECORD] Done: {len(h5s)} file(s)")
    return h5s


def convert_to_v3(input_dir: str, output_dir: str, fps: int = 30):
    """Convert HDF5 recordings to LeRobot v3.0 (full pipeline: data + metadata + stats)."""
    import pandas as pd
    from stage_2.lerobot_v3_converter import (
        convert_directory,
        write_info_json,
        write_episodes_metadata,
        compute_and_write_stats,
    )

    stats = convert_directory(input_dir, output_dir, fps=fps)

    # Write metadata (mirrors lerobot_v3_converter.main())
    write_info_json(output_dir, stats, fps=fps)
    write_episodes_metadata(output_dir, stats.episodes)

    # Compute and write normalization statistics
    data_dir = os.path.join(output_dir, "data", "chunk-000")
    all_dfs = [
        pd.read_parquet(os.path.join(data_dir, f))
        for f in sorted(os.listdir(data_dir))
        if f.endswith(".parquet")
    ]
    compute_and_write_stats(output_dir, all_dfs)

    return stats


def verify_v3(output_dir: str) -> dict:
    """Verify LeRobot v3.0 output and ACT compatibility."""
    meta_dir = os.path.join(output_dir, "meta")
    data_dir = os.path.join(output_dir, "data")

    result = {"ok": [], "issues": [], "info": {}}

    # Check info.json
    info_path = os.path.join(meta_dir, "info.json")
    if os.path.isfile(info_path):
        with open(info_path) as f:
            info = json.load(f)
        result["info"] = info
        features = info.get("features", {})
        result["ok"].append(f"info.json: features={list(features.keys())}")

        # ACT required features
        for req in ["action.joint_position", "observation.joint_position"]:
            if req in features:
                result["ok"].append(f"  ✓ {req}")
            else:
                result["issues"].append(f"  ✗ Missing required: {req}")
    else:
        result["issues"].append("meta/info.json missing")

    # Check stats.json
    if os.path.isfile(os.path.join(meta_dir, "stats.json")):
        result["ok"].append("stats.json exists ✓")
    else:
        result["issues"].append("stats.json missing")

    # Check parquet data
    pdfs = sorted(glob.glob(os.path.join(data_dir, "chunk-*", "*.parquet")))
    if pdfs:
        df = pd.read_parquet(pdfs[0])
        n_frames = len(df)
        n_eps = df["episode_index"].nunique()
        result["ok"].append(f"data: {n_frames} frames, {n_eps} episodes")
        result["n_frames"] = n_frames
        result["n_episodes"] = n_eps
    else:
        result["issues"].append("No parquet files")

    return result


def generate_batch(output_dir: str, num_episodes: int = 50, duration: float = 5.0):
    """Generate multiple mock episodes with varied trajectory parameters."""
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from stage_1.teleop_bridge.mock_hand_tracker import MockHandTracker
    from stage_1.teleop_bridge.hand_mapper import HandMapper
    from stage_1.safety.safety_node import SafetyGuardian
    from stage_1.recorder.recorder_node import RecorderNode
    from stage_1.teleop_bridge.calibration import HandToRobotTransform
    from std_srvs.srv import Trigger

    rclpy.init()
    os.makedirs(output_dir, exist_ok=True)

    h5_files = []
    for ep in range(num_episodes):
        # Save previous episode's HDF5 (recorder always writes to episode_000000.h5)
        # Use 1-indexed naming so rename never collides with the default name
        prev_default = os.path.join(output_dir, "episode_000000.h5")
        if ep > 0 and os.path.isfile(prev_default):
            prev_name = os.path.join(output_dir, f"episode_{ep:06d}.h5")  # 1-indexed
            shutil.move(prev_default, prev_name)
            h5_files.append(prev_name)

        # Vary trajectory parameters per episode for diversity
        amp_x = 0.02 + np.random.uniform(0, 0.04)
        amp_y = 0.01 + np.random.uniform(0, 0.03)
        amp_z = 0.01 + np.random.uniform(0, 0.03)
        offset_z = 0.15 + np.random.uniform(0, 0.10)
        omega = 0.3 + np.random.uniform(0, 0.4)

        tracker = MockHandTracker()
        tracker.set_parameters([
            rclpy.parameter.Parameter("amplitude_x", value=float(amp_x)),
            rclpy.parameter.Parameter("amplitude_y", value=float(amp_y)),
            rclpy.parameter.Parameter("amplitude_z", value=float(amp_z)),
            rclpy.parameter.Parameter("offset_z", value=float(offset_z)),
            rclpy.parameter.Parameter("omega", value=float(omega)),
        ])
        mapper = HandMapper(transform=HandToRobotTransform.mock_transform())
        safety = SafetyGuardian()
        safety.set_parameters([rclpy.parameter.Parameter("robot_mode", value="mock")])
        recorder = RecorderNode()
        recorder.set_parameters([rclpy.parameter.Parameter("output_dir", value=output_dir)])

        executor = MultiThreadedExecutor(num_threads=4)
        for node in [tracker, mapper, safety, recorder]:
            executor.add_node(node)

        req = Trigger.Request()
        recorder._on_start(req, Trigger.Response())

        start = time.time()
        while time.time() - start < duration:
            executor.spin_once(timeout_sec=0.1)

        recorder._on_stop(req, Trigger.Response())
        executor.shutdown()
        for node in [tracker, mapper, safety, recorder]:
            node.destroy_node()

        # Count steps
        default_name = os.path.join(output_dir, "episode_000000.h5")
        try:
            import h5py
            with h5py.File(default_name, "r") as hf:
                eps = [k for k in hf.keys() if k.startswith("episode_")]
                n = hf[eps[0]]["joint_command/position"].shape[0] if eps else 0
        except Exception:
            n = "?"
        print(f"  Episode {ep+1}/{num_episodes}: ({n} steps, "
              f"amp_x={amp_x:.3f}, amp_y={amp_y:.3f}, omega={omega:.2f})")

    # Rename the last episode (1-indexed)
    last_default = os.path.join(output_dir, "episode_000000.h5")
    if os.path.isfile(last_default):
        last_name = os.path.join(output_dir, f"episode_{num_episodes:06d}.h5")
        shutil.move(last_default, last_name)
        h5_files.append(last_name)

    rclpy.shutdown()
    print(f"\n[RECORD] Done: {len(h5_files)} episodes in {output_dir}")
    return h5_files


def main():
    import argparse
    parser = argparse.ArgumentParser(description="ACT baseline research — mock data + v3 conversion")
    parser.add_argument("duration", nargs="?", type=float, default=5.0,
                        help="Duration per episode in seconds (default: 5.0)")
    parser.add_argument("--batch", "-n", type=int, default=1,
                        help="Number of episodes to generate (default: 1)")
    parser.add_argument("--output", "-o", default="/tmp/act_test_v3",
                        help="Output directory for v3 dataset")
    parser.add_argument("--keep-hdf5", action="store_true",
                        help="Keep HDF5 files after conversion")
    args = parser.parse_args()

    # Step 1: Generate recordings
    rec_dir = "/tmp/umi_recordings"
    if os.path.isdir(rec_dir):
        shutil.rmtree(rec_dir)

    if args.batch > 1:
        print(f"[BATCH] Generating {args.batch} episodes ({args.duration}s each)...")
        h5_files = generate_batch(rec_dir, args.batch, args.duration)
    else:
        h5_files = generate_recording(rec_dir, args.duration)

    if not h5_files:
        print("ERROR: No recordings generated")
        return 1

    # Step 2: Convert to LeRobot v3.0
    if os.path.isdir(args.output):
        shutil.rmtree(args.output)
    print(f"\n[CONVERT] {len(h5_files)} episode(s) → {args.output}")
    convert_to_v3(rec_dir, args.output)

    # Step 3: Verify
    print("\n[VERIFY]")
    result = verify_v3(args.output)
    for msg in result["ok"]:
        print(f"  {msg}")
    for msg in result["issues"]:
        print(f"  {msg}")

    # Show stats summary
    stats_path = os.path.join(args.output, "meta", "stats.json")
    if os.path.isfile(stats_path):
        print("\n[STATS]")
        with open(stats_path) as f:
            stats = json.load(f)
        for key in ["action.joint_position", "observation.joint_position"]:
            if key in stats:
                s = stats[key]
                print(f"  {key}:")
                print(f"    mean={[f'{x:.3f}' for x in s['mean']]}")
                print(f"    std ={[f'{x:.3f}' for x in s['std']]}")
                print(f"    min ={[f'{x:.3f}' for x in s['min']]}")
                print(f"    max ={[f'{x:.3f}' for x in s['max']]}")

    # Step 4: Summary
    print(f"\n{'='*60}")
    if not result["issues"]:
        print(f"✓ LeRobot v3.0 pipeline ACT-compatible! ({result.get('n_episodes', 0)} eps, {result.get('n_frames', 0)} frames)")
    else:
        print("✗ Issues found")

    print(f"""
ACT Training Plan:
  1. Record 50+ real teleop episodes
  2. Convert: python3 stage_2/lerobot_v3_converter.py --input <dir> --output <v3_dir>
  3. Train:  lerobot-train --policy.type=act --dataset.repo_id=<v3_dir> ...

Container dependencies already installed:
  - PyTorch 2.10.0 + CUDA 12.9
  - LeRobot 0.4.4 (includes ACT policy)
  - MuJoCo 3.9.0

Current data (mock, {result.get('n_frames', 0)} frames):
  - action.joint_position (6,)
  - observation.joint_position (6,)
  - gripper actions/observations
  - No camera images (state-only mode)
""")

    # Cleanup
    if not args.keep_hdf5:
        shutil.rmtree(rec_dir, ignore_errors=True)

    return 0 if not result["issues"] else 1


if __name__ == "__main__":
    sys.exit(main())
