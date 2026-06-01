#!/usr/bin/env python3
"""
End-to-end test: mock_hand_tracker → hand_mapper → safety(mujoco) → recorder.

Verifies the full teleop pipeline runs with MuJoCo simulation in the loop.
"""

import sys
sys.path.insert(0, "/workspace/umi")

import os, tempfile, threading, time
import numpy as np
import rclpy

rclpy.init()

from stage_1.teleop_bridge.mock_hand_tracker import MockHandTracker
from stage_1.teleop_bridge.hand_mapper import HandMapper
from stage_1.teleop_bridge.calibration import HandToRobotTransform
from stage_1.safety.safety_node import SafetyGuardian
from stage_1.recorder.recorder_node import RecorderNode
from std_srvs.srv import Trigger

output_dir = tempfile.mkdtemp(prefix="sim_test_")
model_path = os.path.join(
    os.path.dirname(__file__), "xarm6.xml"
)
print(f"Model: {model_path}")
print(f"Output: {output_dir}")

# Create nodes
tracker = MockHandTracker()
tracker.set_parameters([
    rclpy.parameter.Parameter("amplitude_x", value=0.03),
    rclpy.parameter.Parameter("amplitude_y", value=0.02),
    rclpy.parameter.Parameter("amplitude_z", value=0.02),
    rclpy.parameter.Parameter("offset_z", value=0.2),
])

mapper = HandMapper(transform=HandToRobotTransform.mock_transform())

safety = SafetyGuardian()
safety.set_parameters([
    rclpy.parameter.Parameter("robot_mode", value="mujoco"),
    rclpy.parameter.Parameter("mjcf_path", value=model_path),
    rclpy.parameter.Parameter("control_rate", value=60.0),
])

recorder = RecorderNode()
recorder.set_parameters([rclpy.parameter.Parameter("output_dir", value=output_dir)])

nodes = [tracker, mapper, safety, recorder]
executor = rclpy.executors.MultiThreadedExecutor()
for n in nodes:
    executor.add_node(n)

spin_thread = threading.Thread(target=executor.spin, daemon=True)
spin_thread.start()
time.sleep(2.0)  # let nodes init and connect

# Start recording
start_client = recorder.create_client(Trigger, "/recorder/start")
start_client.wait_for_service(timeout_sec=3.0)
start_client.call_async(Trigger.Request())
print("Recording started...")

time.sleep(8.0)  # record 8 seconds

# Stop
stop_client = recorder.create_client(Trigger, "/recorder/stop")
stop_client.wait_for_service(timeout_sec=3.0)
future = stop_client.call_async(Trigger.Request())
rclpy.spin_until_future_complete(recorder, future, timeout_sec=3.0)
time.sleep(0.5)

# Check results
import h5py
h5_files = [f for f in os.listdir(output_dir) if f.endswith(".h5")]
if h5_files:
    h5_path = os.path.join(output_dir, h5_files[0])
    with h5py.File(h5_path, "r") as f:
        ep_name = [k for k in f.keys() if k.startswith("episode_")][0]
        ep = f[ep_name]
        n_cmd = ep["joint_command/position"].shape[0] if "joint_command/position" in ep else 0
        n_state = ep["joint_state/position"].shape[0] if "joint_state/position" in ep else 0
        print(f"\nResults:")
        print(f"  joint_command steps: {n_cmd}")
        print(f"  joint_state steps:   {n_state}")

        if n_cmd > 0 and n_state > 0:
            print(f"  Command range: [{ep['joint_command/position'][:,0].min():.3f}, {ep['joint_command/position'][:,0].max():.3f}]")
            print(f"  State range:   [{ep['joint_state/position'][:,0].min():.3f}, {ep['joint_state/position'][:,0].max():.3f}]")
            print("PASS: Pipeline ran with MuJoCo simulation!")
        else:
            print("FAIL: No data recorded")
    os.unlink(h5_path)
else:
    print("FAIL: No HDF5 file produced")

os.rmdir(output_dir)

# Cleanup
for n in nodes:
    n.destroy_node()
rclpy.shutdown()
