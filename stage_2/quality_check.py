#!/usr/bin/env python3
"""
Record a mock pipeline episode and analyze data quality.

Usage (in container):
    PYTHONPATH=/workspace/umi python3 /workspace/umi/stage_2/quality_check.py
"""

import sys
sys.path.insert(0, "/workspace/umi")

import os
import tempfile
import threading
import time
from collections import deque

import numpy as np
import h5py


def record_mock_episode(duration: float = 10.0) -> str:
    """Run the mock pipeline for `duration` seconds and return the HDF5 path."""
    import rclpy

    rclpy.init()

    from stage_1.teleop_bridge.mock_hand_tracker import MockHandTracker
    from stage_1.teleop_bridge.hand_mapper import HandMapper
    from stage_1.teleop_bridge.calibration import HandToRobotTransform
    from stage_1.safety.safety_node import SafetyGuardian
    from stage_1.recorder.recorder_node import RecorderNode
    from std_srvs.srv import Trigger

    output_dir = tempfile.mkdtemp(prefix="umi_quality_")

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
    recorder.set_parameters([
        rclpy.parameter.Parameter("output_dir", value=output_dir),
    ])

    nodes = [tracker, mapper, safety, recorder]

    def spin():
        executor = rclpy.executors.MultiThreadedExecutor()
        for n in nodes:
            executor.add_node(n)
        executor.spin()

    spin_thread = threading.Thread(target=spin, daemon=True)
    spin_thread.start()

    time.sleep(1.0)  # let nodes initialize

    # Start recording via service call
    start_client = recorder.create_client(Trigger, "/recorder/start")
    start_client.wait_for_service(timeout_sec=2.0)
    start_client.call_async(Trigger.Request())

    time.sleep(duration)

    # Stop recording
    stop_client = recorder.create_client(Trigger, "/recorder/stop")
    stop_client.wait_for_service(timeout_sec=2.0)
    future = stop_client.call_async(Trigger.Request())
    rclpy.spin_until_future_complete(recorder, future, timeout_sec=2.0)

    time.sleep(0.5)

    for n in nodes:
        n.destroy_node()
    rclpy.shutdown()

    # Find the HDF5 file
    h5_files = [f for f in os.listdir(output_dir) if f.endswith(".h5")]
    if not h5_files:
        raise RuntimeError(f"No HDF5 file found in {output_dir}")
    return os.path.join(output_dir, h5_files[0])


def analyze_episode(h5_path: str) -> dict:
    """Analyze data quality of a recorded episode."""
    results = {}

    with h5py.File(h5_path, "r") as f:
        # Find episode group (HDF5Writer wraps data under /episode_XXXXXX/)
        episodes = [k for k in f.keys() if k.startswith("episode_")]
        if not episodes:
            print("ERROR: No episode group found in HDF5 file")
            return results
        ep = f[episodes[0]]
        print(f"\n--- Episode: {episodes[0]} ---")
        num_steps = ep.attrs.get("num_steps", "?")
        print(f"  num_steps (attr): {num_steps}")

        # Print all datasets within the episode group
        print("\n--- HDF5 Datasets ---")
        def print_keys(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(f"  {name}: shape={obj.shape}, dtype={obj.dtype}")
        ep.visititems(print_keys)

        # Check: joint_command
        if "joint_command/position" in ep:
            cmd = ep["joint_command/position"][:]
            n_steps = cmd.shape[0]
            results["num_steps"] = n_steps
            print(f"\n--- Joint Command Analysis ({n_steps} steps) ---")

            # 1. Smoothness: frame-to-frame differences
            diffs = np.diff(cmd, axis=0)
            max_diff = np.max(np.abs(diffs), axis=0)
            mean_diff = np.mean(np.abs(diffs), axis=0)
            print(f"  Frame-to-frame diff (max per joint): {np.array2string(max_diff, precision=4)}")
            print(f"  Frame-to-frame diff (mean per joint): {np.array2string(mean_diff, precision=4)}")

            # 2. Velocity spike detection
            dt = 1.0 / 30.0
            velocities = diffs / dt
            max_vel = np.max(np.abs(velocities), axis=0)
            spikes = np.sum(np.abs(velocities) > 3.14, axis=0)
            print(f"  Max velocity per joint (rad/s): {np.array2string(max_vel, precision=2)}")
            print(f"  Velocity spikes (>3.14 rad/s) per joint: {spikes}")

            # 3. Range check
            cmd_min = np.min(cmd, axis=0)
            cmd_max = np.max(cmd, axis=0)
            print(f"  Joint range min: {np.array2string(cmd_min, precision=3)}")
            print(f"  Joint range max: {np.array2string(cmd_max, precision=3)}")

            results["cmd_smoothness_max"] = max_diff
            results["cmd_velocity_spikes"] = int(np.sum(spikes))

        # Check: joint_state
        if "joint_state/position" in ep:
            state = ep["joint_state/position"][:]
            n_state = state.shape[0]
            print(f"\n--- Joint State Analysis ({n_state} steps) ---")

            if "joint_command/position" in ep:
                min_len = min(n_state, cmd.shape[0])
                mismatch = np.abs(cmd[:min_len] - state[:min_len])
                mean_mismatch = np.mean(mismatch, axis=0)
                print(f"  Mean cmd-state mismatch: {np.array2string(mean_mismatch, precision=4)}")

        # Check: gripper
        if "gripper/command" in ep:
            gcmd = ep["gripper/command"][:]
            print(f"\n--- Gripper Analysis ---")
            print(f"  Range: [{gcmd.min():.3f}, {gcmd.max():.3f}]")
            out_of_range = np.sum((gcmd < 0) | (gcmd > 1))
            print(f"  Out-of-range values: {out_of_range}")
            results["gripper_out_of_range"] = int(out_of_range)

        # Timing analysis using recorder timestamps
        if "joint_command/position_timestamp" in ep:
            ts = ep["joint_command/position_timestamp"][:]
            if len(ts) > 1:
                intervals = np.diff(ts.flatten())
                print(f"\n--- Timing Analysis ---")
                print(f"  Interval: mean={np.mean(intervals)*1000:.2f}ms "
                      f"std={np.std(intervals)*1000:.2f}ms "
                      f"min={np.min(intervals)*1000:.2f}ms "
                      f"max={np.max(intervals)*1000:.2f}ms")
                expected = 1.0 / 30.0
                drops = np.sum(intervals > 2 * expected)
                print(f"  Frame drops (interval > 2x expected {expected*2*1000:.0f}ms): {drops}")
                results["mean_interval_ms"] = float(np.mean(intervals) * 1000)
                results["frame_drops"] = int(drops)

        results["h5_path"] = h5_path

    return results


def main():
    print("Recording mock episode (10 seconds)...")
    h5_path = record_mock_episode(duration=10.0)
    print(f"\nRecorded to: {h5_path}")

    analyze_episode(h5_path)

    # Cleanup
    os.unlink(h5_path)
    os.rmdir(os.path.dirname(h5_path))
    print(f"\nCleaned up {h5_path}")


if __name__ == "__main__":
    main()
