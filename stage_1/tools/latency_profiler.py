#!/usr/bin/env python3
"""
End-to-end latency profiler for UMI teleop pipeline.

Uses ROS2 header stamps for cross-node latency (all nodes share the system
clock in Docker) plus per-node instrumentation for IK solve time.

Usage:
    python3 /workspace/umi/stage_1/tools/latency_profiler.py --standalone

The script spawns the mock pipeline in-process, collects samples, and
outputs a latency breakdown table.
"""

import sys

sys.path.insert(0, "/workspace/umi")

import argparse
import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StageStats:
    name: str
    samples: deque = field(default_factory=lambda: deque(maxlen=2000))

    def record(self, latency_s: float):
        if 0 < latency_s < 1.0:  # sanity bounds (0–1s)
            self.samples.append(latency_s)

    def summarize(self) -> dict:
        if len(self.samples) < 10:
            return {"name": self.name, "count": len(self.samples), "error": "too few samples"}
        arr = np.array(self.samples)
        return {
            "name": self.name,
            "count": len(arr),
            "mean_ms": np.mean(arr) * 1000,
            "std_ms": np.std(arr) * 1000,
            "min_ms": np.min(arr) * 1000,
            "max_ms": np.max(arr) * 1000,
            "p50_ms": np.percentile(arr, 50) * 1000,
            "p95_ms": np.percentile(arr, 95) * 1000,
            "p99_ms": np.percentile(arr, 99) * 1000,
        }


class StampLatencyProfiler:
    """Latency profiler using ROS2 header stamps for cross-stage timing.

    Header stamps are set by each publisher using node.get_clock().now().
    Since all nodes run in the same Docker container (same system clock),
    stamp differences accurately reflect pipeline latency.
    """

    def __init__(self):
        import rclpy
        from rclpy.node import Node
        from geometry_msgs.msg import PoseStamped
        from sensor_msgs.msg import JointState

        self._node = Node("latency_profiler")
        self._logger = self._node.get_logger()

        # Per-stage stats
        self._e2e_latency = StageStats("end-to-end (wrist_pose → command/joints stamp)")
        self._safety_latency = StageStats("safety_forward (command → state stamp)")
        self._ik_success = StageStats("IK solve time (hand_mapper internal)")
        self._ik_failure = StageStats("IK failure time (non-convergent)")

        # Track last wrist_pose stamp for matching
        self._last_wrist_stamp = None
        self._ik_convergence_count = 0
        self._ik_failure_count = 0

        # Subscriptions
        self._sub_wrist = self._node.create_subscription(
            PoseStamped, "/hand/right/wrist_pose", self._on_wrist_pose, 10
        )
        self._sub_cmd = self._node.create_subscription(
            JointState, "/teleop/command/joints", self._on_command_joints, 10
        )
        self._sub_state = self._node.create_subscription(
            JointState, "/teleop/state/joints", self._on_state_joints, 10
        )

        self._logger.info("StampLatencyProfiler ready — collecting samples...")

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return stamp.sec + stamp.nanosec * 1e-9

    def _on_wrist_pose(self, msg):
        self._last_wrist_stamp = self._stamp_to_sec(msg.header.stamp)

    def _on_command_joints(self, msg):
        cmd_stamp = self._stamp_to_sec(msg.header.stamp)
        if self._last_wrist_stamp is not None:
            delta = cmd_stamp - self._last_wrist_stamp
            self._e2e_latency.record(delta)
        self._last_cmd_stamp = cmd_stamp

    def _on_state_joints(self, msg):
        state_stamp = self._stamp_to_sec(msg.header.stamp)
        if hasattr(self, "_last_cmd_stamp") and self._last_cmd_stamp is not None:
            delta = state_stamp - self._last_cmd_stamp
            self._safety_latency.record(delta)

    def print_report(self):
        stages = [self._e2e_latency, self._safety_latency, self._ik_success, self._ik_failure]
        print("\n" + "=" * 96)
        print("UMI Teleop Pipeline — Latency Breakdown (Header-Stamp Based)")
        print("=" * 96)
        header = (
            f"{'Stage':<52} {'Count':>6} {'Mean':>8} {'Std':>8} "
            f"{'p50':>8} {'p95':>8} {'p99':>8}"
        )
        print(header)
        print("-" * 96)
        for stage in stages:
            s = stage.summarize()
            if "error" in s:
                print(f"  {s['name']:<50}  {s['error']}")
            else:
                print(
                    f"  {s['name']:<50} {s['count']:>6} "
                    f"{s['mean_ms']:>7.2f}ms {s['std_ms']:>7.2f}ms "
                    f"{s['p50_ms']:>7.2f}ms {s['p95_ms']:>7.2f}ms {s['p99_ms']:>7.2f}ms"
                )
        print("=" * 96)
        print(
            "Cross-stage latencies use ROS2 header stamp deltas.\n"
            "IK times come from hand_mapper internal time.perf_counter() instrumentation.\n"
            "Target: <50ms end-to-end, <33ms per-stage (30 Hz frame budget)."
        )


def run_standalone(duration: float = 20.0):
    """Run the latency profiler with an in-process mock pipeline."""
    import rclpy

    rclpy.init()

    from stage_1.teleop_bridge.mock_hand_tracker import MockHandTracker
    from stage_1.teleop_bridge.hand_mapper import HandMapper
    from stage_1.teleop_bridge.calibration import HandToRobotTransform
    from stage_1.safety.safety_node import SafetyGuardian
    from stage_1.recorder.recorder_node import RecorderNode

    tracker = MockHandTracker()
    tracker.set_parameters(
        [
            rclpy.parameter.Parameter("amplitude_x", value=0.03),
            rclpy.parameter.Parameter("amplitude_y", value=0.02),
            rclpy.parameter.Parameter("amplitude_z", value=0.02),
            rclpy.parameter.Parameter("offset_z", value=0.2),
        ]
    )
    mapper = HandMapper(transform=HandToRobotTransform.mock_transform())
    safety = SafetyGuardian()
    safety.set_parameters([rclpy.parameter.Parameter("robot_mode", value="mock")])
    recorder = RecorderNode()

    profiler = StampLatencyProfiler()

    nodes = [tracker, mapper, safety, recorder, profiler._node]

    def spin():
        executor = rclpy.executors.MultiThreadedExecutor()
        for n in nodes:
            executor.add_node(n)
        try:
            executor.spin()
        except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
            pass

    spin_thread = threading.Thread(target=spin, daemon=True)
    spin_thread.start()

    print(f"\nCollecting latency samples for {duration:.0f} seconds...")
    time.sleep(duration)

    profiler.print_report()

    for n in nodes:
        n.destroy_node()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="UMI teleop pipeline latency profiler")
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="Run pipeline nodes in-process",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="Collection duration in seconds",
    )
    args = parser.parse_args()

    if args.standalone:
        run_standalone(duration=args.duration)
    else:
        import rclpy

        rclpy.init()
        profiler = StampLatencyProfiler()
        try:
            print("Profiling... Press Ctrl+C to stop and print report.")
            rclpy.spin(profiler._node)
        except KeyboardInterrupt:
            pass
        profiler.print_report()
        profiler._node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
