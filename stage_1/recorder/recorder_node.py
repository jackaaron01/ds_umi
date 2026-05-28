#!/usr/bin/env python3
"""ROS2 node: subscribes to teleop topics and records synchronized HDF5 episodes."""

import os
import time
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64
from std_srvs.srv import Trigger
import numpy as np

from stage_1.recorder.hdf5_writer import HDF5Writer


class RecorderNode(Node):
    """Subscribes to command + state topics and writes time-aligned HDF5."""

    def __init__(self):
        super().__init__("recorder")
        self.declare_parameter("output_dir", os.path.expanduser("~/umi_recordings"))
        self.declare_parameter("auto_start", False)
        self.declare_parameter("enable_image_recording", False)
        self.declare_parameter("image_topic_rgb", "/camera/rgb/image_raw")
        self.declare_parameter("image_topic_depth", "/camera/depth/image_raw")

        output_dir = self.get_parameter("output_dir").value
        os.makedirs(output_dir, exist_ok=True)

        # Ring buffers: topic_category -> deque of (timestamp, data)
        self._buffers = {
            "joint_state/position": deque(maxlen=100),
            "joint_state/velocity": deque(maxlen=100),
            "joint_command/position": deque(maxlen=100),
            "gripper/state": deque(maxlen=100),
            "gripper/command": deque(maxlen=100),
        }

        enable_images = self.get_parameter("enable_image_recording").value
        if enable_images:
            self._buffers["sensors/camera/rgb"] = deque(maxlen=5)
            self._buffers["sensors/camera/depth"] = deque(maxlen=5)
        self._buffer_lock = threading.Lock()

        # Subscriptions
        self._sub_joint_cmd = self.create_subscription(
            JointState, "/teleop/command/joints", self._cb_joint_cmd, 10
        )
        self._sub_joint_state = self.create_subscription(
            JointState, "/teleop/state/joints", self._cb_joint_state, 10
        )
        self._sub_gripper_cmd = self.create_subscription(
            Float64, "/teleop/command/gripper", self._cb_gripper_cmd, 10
        )
        self._sub_gripper_state = self.create_subscription(
            Float64, "/teleop/state/gripper", self._cb_gripper_state, 10
        )

        # Image subscriptions (optional)
        if enable_images:
            rgb_topic = self.get_parameter("image_topic_rgb").value
            depth_topic = self.get_parameter("image_topic_depth").value
            self._sub_rgb = self.create_subscription(
                Image, rgb_topic, self._cb_rgb, 10
            )
            self._sub_depth = self.create_subscription(
                Image, depth_topic, self._cb_depth, 10
            )

        # Services
        self._srv_start = self.create_service(Trigger, "/recorder/start", self._on_start)
        self._srv_stop = self.create_service(Trigger, "/recorder/stop", self._on_stop)

        # Internal state
        self._writer = None
        self._episode_index = 0
        self._recording = False
        self._start_time = None
        self._output_dir = output_dir

        # Writer thread @ 30 Hz
        self._flush_period = 1.0 / 30.0
        self._flush_timer = self.create_timer(self._flush_period, self._flush)

        self.get_logger().info(f"Recorder ready (output_dir={output_dir})")

    # ---- ROS callbacks ----
    def _cb_joint_cmd(self, msg: JointState):
        ts = time.time()
        with self._buffer_lock:
            self._buffers["joint_command/position"].append(
                (ts, list(msg.position) if msg.position else [])
            )

    def _cb_joint_state(self, msg: JointState):
        ts = time.time()
        with self._buffer_lock:
            self._buffers["joint_state/position"].append(
                (ts, list(msg.position) if msg.position else [])
            )
            self._buffers["joint_state/velocity"].append(
                (ts, list(msg.velocity) if msg.velocity else [0.0] * 6)
            )

    def _cb_gripper_cmd(self, msg: Float64):
        ts = time.time()
        with self._buffer_lock:
            self._buffers["gripper/command"].append((ts, msg.data))

    def _cb_gripper_state(self, msg: Float64):
        ts = time.time()
        with self._buffer_lock:
            self._buffers["gripper/state"].append((ts, msg.data))

    def _cb_rgb(self, msg: Image):
        ts = time.time()
        with self._buffer_lock:
            self._ensure_buffer("sensors/camera/rgb", maxlen=5)
            dtype = np.uint8
            img_data = np.frombuffer(msg.data, dtype=dtype).reshape(
                msg.height, msg.width, -1
            )
            self._buffers["sensors/camera/rgb"].append((ts, img_data))

    def _cb_depth(self, msg: Image):
        ts = time.time()
        with self._buffer_lock:
            self._ensure_buffer("sensors/camera/depth", maxlen=5)
            dtype = np.uint16 if msg.encoding == "16UC1" else np.uint8
            raw = np.frombuffer(msg.data, dtype=dtype)
            img_data = raw.reshape(msg.height, msg.width, -1).squeeze()
            self._buffers["sensors/camera/depth"].append((ts, img_data))

    def _ensure_buffer(self, key: str, maxlen: int = 100):
        if key not in self._buffers:
            self._buffers[key] = deque(maxlen=maxlen)

    # ---- services ----
    def _on_start(self, req, resp):
        if self._recording:
            resp.success = False
            resp.message = "Already recording"
            return resp
        output_dir = self.get_parameter("output_dir").value
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, f"episode_{self._episode_index:06d}.h5")
        self._writer = HDF5Writer(filepath)
        self._writer.start_episode(
            self._episode_index,
            metadata={
                "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "start_time_unix": time.time(),
            },
        )
        self._recording = True
        self._start_time = time.time()
        self.get_logger().info(f"Recording started: {filepath}")
        resp.success = True
        resp.message = f"Episode {self._episode_index}"
        return resp

    def _on_stop(self, req, resp):
        if not self._recording:
            resp.success = False
            resp.message = "Not recording"
            return resp
        self._finalize_episode()
        resp.success = True
        resp.message = f"Episode {self._episode_index} saved"
        return resp

    # ---- flush loop ----
    def _flush(self):
        if not self._recording or self._writer is None:
            return
        with self._buffer_lock:
            step = {}
            for key, buf in self._buffers.items():
                if buf:
                    ts, data = buf[-1]  # latest
                    step[key] = data
                    step[f"{key}_timestamp"] = ts

            if not step:
                return

        # Require at least joint commands to write
        if "joint_command/position" not in step:
            return

        elapsed = time.time() - self._start_time
        step["timestamp"] = elapsed
        self._writer.write_step(step)

    def _finalize_episode(self):
        self._recording = False
        if self._writer is not None:
            self._writer.end_episode()
            self._writer.close()
            self.get_logger().info(
                f"Episode {self._episode_index} saved ({self._writer.step_count} steps)"
            )
            self._writer = None
            self._episode_index += 1

    def destroy_node(self):
        self._finalize_episode()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = RecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
