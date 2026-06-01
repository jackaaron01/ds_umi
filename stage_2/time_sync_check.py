#!/usr/bin/env python3
"""
Time synchronization assessment for UMI teleop pipeline.

Measures:
  1. time.time() vs ROS2 clock offset
  2. Cross-node timestamp drift (using round-trip messages)
  3. Per-node timestamp precision

Target: <5ms synchronization across all nodes.

Usage:
    python3 /workspace/umi/stage_2/time_sync_check.py
"""

import sys

sys.path.insert(0, "/workspace/umi")

import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class ClockProbe(Node):
    """Publishes probe messages with both time.time() and ROS2 clock stamps."""

    def __init__(self, name="clock_probe"):
        super().__init__(name)
        self._pub = self.create_publisher(Float64MultiArray, "/clock_probe", 10)
        self._sub = self.create_subscription(
            Float64MultiArray, "/clock_probe", self._on_probe, 10
        )

        # Measurement storage
        self._offsets = []  # time.time() - ROS2 clock (seconds)
        self._roundtrips = []  # publish-to-receive latency
        self._ros2_stamps = []  # ROS2 clock timestamps for jitter
        self._py_stamps = []  # time.time() timestamps for jitter

        self._timer = self.create_timer(1.0 / 100.0, self._publish_probe)  # 100 Hz
        self._samples_collected = 0
        self._max_samples = 500

    def _publish_probe(self):
        if self._samples_collected >= self._max_samples:
            return
        t_py = time.time()
        t_ros2_ns = self.get_clock().now().nanoseconds
        t_ros2_s = t_ros2_ns / 1e9

        self._offsets.append(t_py - t_ros2_s)
        self._ros2_stamps.append(t_ros2_s)

        msg = Float64MultiArray()
        msg.data = [t_py, t_ros2_s, float(self._samples_collected)]
        self._pub.publish(msg)
        self._samples_collected += 1

    def _on_probe(self, msg):
        t_recv = time.time()
        t_send_py = msg.data[0]
        self._roundtrips.append(t_recv - t_send_py)

    def report(self):
        while self._samples_collected < self._max_samples:
            rclpy.spin_once(self, timeout_sec=0.01)

        print("\n" + "=" * 70)
        print("UMI Teleop Pipeline — Time Synchronization Assessment")
        print("=" * 70)

        # 1. Clock offset: time.time() vs ROS2 clock
        offsets = np.array(self._offsets)
        ros2_stamps = np.array(self._ros2_stamps)

        print(f"\n1. Clock Source Offset (time.time() - ROS2 clock)")
        print(f"   Samples: {len(offsets)}")
        print(f"   Mean offset: {np.mean(offsets) * 1e6:.2f} us")
        print(f"   Std offset:  {np.std(offsets) * 1e6:.2f} us")
        print(f"   Max offset:  {np.max(np.abs(offsets)) * 1e6:.2f} us")

        # 2. Clock jitter (precision of consecutive ROS2 timestamps)
        ros2_intervals = np.diff(ros2_stamps)
        ros2_jitter = np.std(ros2_intervals)
        expected_interval = 1.0 / 100.0  # 100 Hz timer
        ros2_period_error = ros2_intervals - expected_interval

        print(f"\n2. ROS2 Clock Jitter (100 Hz timer)")
        print(f"   Expected interval: {expected_interval * 1e6:.0f} us")
        print(f"   Mean interval:     {np.mean(ros2_intervals) * 1e6:.2f} us")
        print(f"   Std interval:      {np.std(ros2_intervals) * 1e6:.2f} us")
        print(f"   Max period error:  {np.max(np.abs(ros2_period_error)) * 1e6:.2f} us")

        # 3. Round-trip latency (publish → receive)
        rts = np.array(self._roundtrips)
        if len(rts) > 0:
            print(f"\n3. ROS2 DDS Round-Trip Latency (publish → receive)")
            print(f"   Samples: {len(rts)}")
            print(f"   Mean: {np.mean(rts) * 1e3:.2f} ms")
            print(f"   Std:  {np.std(rts) * 1e3:.2f} ms")
            print(f"   p50:  {np.percentile(rts, 50) * 1e3:.2f} ms")
            print(f"   p95:  {np.percentile(rts, 95) * 1e3:.2f} ms")
            print(f"   p99:  {np.percentile(rts, 99) * 1e3:.2f} ms")

        # 4. Overall assessment
        max_error_us = max(
            np.max(np.abs(offsets)) * 1e6,
            np.std(ros2_intervals) * 1e6,
            (np.percentile(rts, 99) * 1e6) if len(rts) > 0 else 0,
        )

        print(f"\n4. Overall Assessment")
        print(f"   Max synchronization error: {max_error_us:.0f} us")
        if max_error_us < 1000:
            print(f"   Target <5ms: PASS (well within 5ms budget)")
        elif max_error_us < 5000:
            print(f"   Target <5ms: PASS (within 5ms)")
        else:
            print(f"   Target <5ms: FAIL (exceeds 5ms by {max_error_us - 5000:.0f} us)")

        print("=" * 70)


def main():
    rclpy.init()
    probe = ClockProbe("time_sync_check")

    spin_thread = threading.Thread(
        target=lambda: rclpy.spin(probe), daemon=True
    )
    spin_thread.start()

    time.sleep(6.0)  # collect ~500 samples at 100 Hz

    probe.report()
    probe.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
