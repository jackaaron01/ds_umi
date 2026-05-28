#!/usr/bin/env python3
"""ROS2 node: receives Quest3 hand tracking data via hand-tracking-sdk and publishes ROS2 topics."""

import threading
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray

try:
    from hand_tracking_sdk import (
        HTSClient, HTSClientConfig, StreamOutput,
        TransportMode, HandFilter, ErrorPolicy,
    )
    HAS_SDK = True
except ImportError:
    HAS_SDK = False
    TransportMode = None
    HandFilter = None
    ErrorPolicy = None


_TRANSPORT_MAP = {
    "udp": TransportMode.UDP if TransportMode else None,
    "tcp_server": TransportMode.TCP_SERVER if TransportMode else None,
    "tcp_client": TransportMode.TCP_CLIENT if TransportMode else None,
}
_HAND_MAP = {
    "left": HandFilter.LEFT if HandFilter else None,
    "right": HandFilter.RIGHT if HandFilter else None,
    "both": HandFilter.BOTH if HandFilter else None,
}


class HandTrackingNode(Node):
    """Bridges hand-tracking-sdk (UDP/TCP from Quest3) to ROS2 topics."""

    def __init__(self):
        super().__init__("hand_tracking_node")
        self.declare_parameter("transport", "udp")
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 12345)
        self.declare_parameter("hand", "right")
        self.declare_parameter("publish_interval", 1.0 / 60.0)  # 60 Hz

        transport = self.get_parameter("transport").value
        host = self.get_parameter("host").value
        port = self.get_parameter("port").value
        hand = self.get_parameter("hand").value
        publish_interval = self.get_parameter("publish_interval").value

        if not HAS_SDK:
            self.get_logger().fatal(
                "hand-tracking-sdk not installed. Install: pip install hand-tracking-sdk"
            )
            raise RuntimeError("hand-tracking-sdk not available")

        # Publishers
        self._pub_wrist = self.create_publisher(
            PoseStamped, f"/hand/{hand}/wrist_pose", 10
        )
        self._pub_keypoints = self.create_publisher(
            Float32MultiArray, f"/hand/{hand}/keypoints", 10
        )

        # State
        self._latest_wrist = None    # (x, y, z, qx, qy, qz, qw)
        self._latest_kpts = None     # 63 floats
        self._lock = threading.Lock()
        self._running = False
        self._recv_thread = None

        # Stats
        self._frame_count = 0
        self._error_count = 0

        # Start receiver thread
        self._running = True
        self._recv_thread = threading.Thread(
            target=self._receive_loop,
            args=(transport, host, port, hand),
            daemon=True,
        )
        self._recv_thread.start()

        # Publish timer
        self._pub_timer = self.create_timer(publish_interval, self._publish)

        self.get_logger().info(
            f"Hand tracking node started (transport={transport}, "
            f"host={host}, port={port}, hand={hand})"
        )

    # ---- Receive loop (background thread) ----
    def _receive_loop(self, transport: str, host: str, port: int, hand: str):
        try:
            transport_mode = _TRANSPORT_MAP.get(transport, TransportMode.UDP)
            hand_filter = _HAND_MAP.get(hand, HandFilter.RIGHT)

            config = HTSClientConfig(
                transport_mode=transport_mode,
                host=host,
                port=port,
                output=StreamOutput.FRAMES,
                hand_filter=hand_filter,
                error_policy=ErrorPolicy.TOLERANT,
            )
            client = HTSClient(config)

            for frame in client.iter_events():
                if not self._running:
                    break
                try:
                    side = frame.side.value
                    if hand != "both" and side != hand:
                        continue

                    wrist = frame.wrist
                    with self._lock:
                        self._latest_wrist = (
                            wrist.x, wrist.y, wrist.z,
                            wrist.qx, wrist.qy, wrist.qz, wrist.qw,
                        )
                        # 21 landmarks x 3 coords = 63 floats
                        kpts = []
                        for pt in frame.landmarks.points:
                            kpts.extend([pt.x, pt.y, pt.z])
                        self._latest_kpts = kpts
                        self._frame_count += 1
                except Exception:
                    self._error_count += 1
        except Exception as e:
            self.get_logger().error(f"Receiver thread error: {e}")
            self._error_count += 1

    # ---- Publish (ROS2 timer callback) ----
    def _publish(self):
        with self._lock:
            wrist = self._latest_wrist
            kpts = self._latest_kpts

        if wrist is None:
            return

        stamp = self.get_clock().now().to_msg()

        # Wrist pose
        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "quest3_tracking"
        pose.pose.position.x = wrist[0]
        pose.pose.position.y = wrist[1]
        pose.pose.position.z = wrist[2]
        pose.pose.orientation.x = wrist[3]
        pose.pose.orientation.y = wrist[4]
        pose.pose.orientation.z = wrist[5]
        pose.pose.orientation.w = wrist[6]
        self._pub_wrist.publish(pose)

        # Keypoints
        if kpts is not None:
            msg = Float32MultiArray()
            msg.data = kpts
            self._pub_keypoints.publish(msg)

    def destroy_node(self):
        self._running = False
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
        self.get_logger().info(
            f"Hand tracking node stopped (frames={self._frame_count}, "
            f"errors={self._error_count})"
        )
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = HandTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
