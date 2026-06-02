#!/usr/bin/env python3
"""
MuJoCo viewer node for simulation teleop visualization.

Subscribes to /teleop/state/joints and renders the xArm6 model
in a MuJoCo viewer window. Runs independently from the control pipeline.

Usage:
    python3 viewer_node.py [--model xarm6.xml]

Requires: mujoco-python-viewer (pip install mujoco-python-viewer)
"""

import os, sys, threading, time, argparse
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, "/workspace/umi")

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

import mujoco
from mujoco_viewer import MujocoViewer


class SimViewerNode(Node):
    """ROS2 node that renders MuJoCo simulation from /teleop/state/joints."""

    def __init__(self, model_path: str):
        super().__init__("sim_viewer")
        self._model = mujoco.MjModel.from_xml_path(model_path)
        self._data = mujoco.MjData(self._model)
        self._viewer = None
        self._latest_joints = np.zeros(6)
        self._lock = threading.Lock()
        self._running = True

        # Subscribe to joint state
        self._sub = self.create_subscription(
            JointState, "/teleop/state/joints", self._joint_callback, 10
        )
        self.get_logger().info(f"Viewer ready, model: {model_path}")

    def _joint_callback(self, msg: JointState):
        with self._lock:
            if len(msg.position) >= 6:
                self._latest_joints = np.array(msg.position[:6], dtype=np.float64)

    def run(self):
        """Main render loop (blocking)."""
        try:
            self._viewer = MujocoViewer(self._model, self._data)
            self.get_logger().info("Viewer window opened. Close window to exit.")

            while self._running and self._viewer.is_alive:
                # Update joint positions from latest ROS message
                with self._lock:
                    self._data.qpos[:6] = self._latest_joints

                # Forward kinematics
                mujoco.mj_forward(self._model, self._data)

                # Render
                self._viewer.render()

                # Process ROS callbacks
                rclpy.spin_once(self, timeout_sec=0.001)

        except Exception as e:
            self.get_logger().error(f"Viewer error: {e}")
        finally:
            if self._viewer and self._viewer.is_alive:
                self._viewer.close()
            self._running = False

    def stop(self):
        self._running = False


def main():
    parser = argparse.ArgumentParser(description="MuJoCo simulation viewer")
    parser.add_argument("--model", default=None, help="Path to MJCF model file")
    args = parser.parse_args()

    model_path = args.model or os.path.join(
        os.path.dirname(__file__), "xarm6.xml"
    )
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    rclpy.init()
    node = SimViewerNode(model_path)

    # Run viewer in main thread (needed for GLFW window events)
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()
        print("Viewer stopped.")


if __name__ == "__main__":
    main()
