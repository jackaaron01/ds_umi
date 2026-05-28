#!/usr/bin/env python3
"""ROS2 node: generic USB camera driver publishing sensor_msgs/Image."""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraNode(Node):
    """Captures from a USB camera and publishes Image + CameraInfo topics."""

    def __init__(self):
        super().__init__("camera_node")
        self.declare_parameter("device_id", 0)
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 30.0)
        self.declare_parameter("topic_name", "/camera/rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/camera/rgb/camera_info")
        self.declare_parameter("frame_id", "camera_rgb_frame")
        self.declare_parameter("calibration_file", "")

        device_id = self.get_parameter("device_id").value
        width = self.get_parameter("width").value
        height = self.get_parameter("height").value
        fps = self.get_parameter("fps").value
        topic_name = self.get_parameter("topic_name").value
        camera_info_topic = self.get_parameter("camera_info_topic").value
        frame_id = self.get_parameter("frame_id").value
        calibration_file = self.get_parameter("calibration_file").value

        self._cap = cv2.VideoCapture(device_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        if not self._cap.isOpened():
            self.get_logger().error(f"Failed to open camera device {device_id}")
            raise RuntimeError(f"Cannot open camera {device_id}")

        actual_w = self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._bridge = CvBridge()
        self._frame_id = frame_id

        self._pub_image = self.create_publisher(Image, topic_name, 10)
        self._pub_camera_info = self.create_publisher(CameraInfo, camera_info_topic, 10)

        self._camera_info = None
        if calibration_file:
            self._load_calibration(calibration_file, int(actual_w), int(actual_h))

        period = 1.0 / fps if fps > 0 else 1.0 / 30.0
        self._timer = self.create_timer(period, self._capture_and_publish)

        self.get_logger().info(
            f"Camera node started: device={device_id}, {int(actual_w)}x{int(actual_h)} @ {fps}fps -> {topic_name}"
        )

    def _capture_and_publish(self):
        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture frame", throttle_duration_sec=2.0)
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stamp = self.get_clock().now().to_msg()
        img_msg = self._bridge.cv2_to_imgmsg(frame_rgb, encoding="rgb8")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self._frame_id
        self._pub_image.publish(img_msg)

        if self._camera_info is not None:
            self._camera_info.header.stamp = stamp
            self._pub_camera_info.publish(self._camera_info)

    def _load_calibration(self, filepath: str, width: int, height: int):
        import yaml

        try:
            with open(filepath, "r") as f:
                calib = yaml.safe_load(f)
            ci = CameraInfo()
            ci.width = width
            ci.height = height
            ci.k = calib.get("camera_matrix", {}).get("data", [0.0] * 9)
            ci.d = calib.get("distortion_coefficients", {}).get("data", [0.0] * 5)
            ci.r = calib.get("rectification_matrix", {}).get(
                "data", [1, 0, 0, 0, 1, 0, 0, 0, 1]
            )
            ci.p = calib.get("projection_matrix", {}).get("data", [0.0] * 12)
            self._camera_info = ci
            self.get_logger().info(f"Loaded calibration from {filepath}")
        except Exception as e:
            self.get_logger().error(f"Failed to load calibration: {e}")

    def destroy_node(self):
        if self._cap is not None:
            self._cap.release()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
