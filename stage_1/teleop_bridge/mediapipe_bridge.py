#!/usr/bin/env python3
"""
ROS2 node: receives MediaPipe hand keypoints via UDP from host,
publishes to standard ROS2 hand tracking topics.

The host runs mediapipe_ego.py which sends hand data via UDP.
This node (in Docker) receives it and publishes to:
  - /hand/right/wrist_pose   (PoseStamped)
  - /hand/right/keypoints    (Float32MultiArray, 63 floats)

The existing hand_mapper node picks these up and runs IK → joint commands.

UDP format (JSON):
  {"wrist": [x,y,z,qx,qy,qz,qw], "keypoints": [63 floats]}

Usage:
    ros2 run teleop_bridge mediapipe_bridge --ros-args -p port:=9999
"""
import json
import socket
import struct
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray


class MediaPipeBridge(Node):
    """Receives MediaPipe hand data via UDP and publishes ROS2 topics."""

    def __init__(self):
        super().__init__("mediapipe_bridge")
        self.declare_parameter("port", 9999)
        self.declare_parameter("hand", "right")

        port = self.get_parameter("port").value
        hand = self.get_parameter("hand").value

        # ROS2 publishers (same topics as mock_hand_tracker)
        self._pub_wrist = self.create_publisher(
            PoseStamped, f"/hand/{hand}/wrist_pose", 10)
        self._pub_keypoints = self.create_publisher(
            Float32MultiArray, f"/hand/{hand}/keypoints", 10)

        # UDP socket
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("0.0.0.0", port))
        self._sock.settimeout(0.01)  # non-blocking for ROS2 spin

        self._timer = self.create_timer(1/60.0, self._receive)  # 60Hz poll
        self.get_logger().info(f"MediaPipe bridge listening on UDP:{port} → /hand/{hand}/*")

    def _receive(self):
        """Poll UDP socket, parse JSON, publish ROS2."""
        try:
            data, addr = self._sock.recvfrom(65536)
            msg = json.loads(data.decode("utf-8"))

            # ── Publish wrist pose ──
            wrist = msg.get("wrist", [0, 0, 0, 0, 0, 0, 1])
            pose = PoseStamped()
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.header.frame_id = "mediapipe_tracking"
            pose.pose.position.x = float(wrist[0]) if len(wrist) > 0 else 0.0
            pose.pose.position.y = float(wrist[1]) if len(wrist) > 1 else 0.0
            pose.pose.position.z = float(wrist[2]) if len(wrist) > 2 else 0.0
            pose.pose.orientation.x = float(wrist[3]) if len(wrist) > 3 else 0.0
            pose.pose.orientation.y = float(wrist[4]) if len(wrist) > 4 else 0.0
            pose.pose.orientation.z = float(wrist[5]) if len(wrist) > 5 else 0.0
            pose.pose.orientation.w = float(wrist[6]) if len(wrist) > 6 else 1.0
            self._pub_wrist.publish(pose)

            # ── Publish keypoints ──
            kp = msg.get("keypoints", [0.0] * 63)
            kp_msg = Float32MultiArray()
            kp_msg.data = [float(v) for v in kp[:63]]
            self._pub_keypoints.publish(kp_msg)

        except socket.timeout:
            pass  # no data, normal
        except json.JSONDecodeError:
            pass  # malformed packet
        except Exception as e:
            self.get_logger().warn(f"UDP error: {e}", throttle_duration_sec=5.0)

    def destroy_node(self):
        self._sock.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MediaPipeBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
