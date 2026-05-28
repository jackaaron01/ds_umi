#!/usr/bin/env python3
"""ROS2 node that publishes synthetic wrist poses for testing without a Quest3."""

import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray


class MockHandTracker(Node):
    """Publishes a Lissajous-figure wrist trajectory at 60 Hz."""

    def __init__(self):
        super().__init__("mock_hand_tracker")
        self.declare_parameter("frequency", 60.0)
        self.declare_parameter("amplitude_x", 0.15)
        self.declare_parameter("amplitude_y", 0.10)
        self.declare_parameter("amplitude_z", 0.10)
        self.declare_parameter("omega", 0.5)
        self.declare_parameter("offset_z", 0.3)

        freq = self.get_parameter("frequency").value
        self._pub_wrist = self.create_publisher(PoseStamped, "/hand/right/wrist_pose", 10)
        self._pub_keypoints = self.create_publisher(
            Float32MultiArray, "/hand/right/keypoints", 10
        )
        self._timer = self.create_timer(1.0 / freq, self._publish)
        self._start_time = time.time()
        self.get_logger().info("Mock hand tracker started")

    def _publish(self):
        t = time.time() - self._start_time
        Ax = self.get_parameter("amplitude_x").value
        Ay = self.get_parameter("amplitude_y").value
        Az = self.get_parameter("amplitude_z").value
        omega = self.get_parameter("omega").value
        offset_z = self.get_parameter("offset_z").value

        # Lissajous figure-8 pattern
        x = Ax * math.sin(omega * t)
        y = Ay * math.sin(2.0 * omega * t)
        z = Az * math.sin(0.5 * omega * t) + offset_z

        # Wrist pose
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = "quest3_tracking"
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = z
        pose.pose.orientation.w = 1.0  # identity orientation
        self._pub_wrist.publish(pose)

        # Keypoints: 21 landmarks × 3 coords, simplified hand model
        keypoints = [0.0] * 63
        # Thumb tip (index 4 * 3 = 12) and index tip (8 * 3 = 24) for pinch
        # Open hand: thumb tip at (x-0.02, y+0.04, z+0.06), index tip at (x+0.02, y+0.02, z+0.07)
        thumb_tip_base = [x - 0.02, y + 0.04, z + 0.06]
        index_tip_base = [x + 0.02, y + 0.02, z + 0.07]
        # Modulate pinch: sinusoid between open and closed
        pinch = 0.5 + 0.5 * math.sin(1.5 * t)
        thumb_tip = [
            thumb_tip_base[0] + 0.01 * pinch,
            thumb_tip_base[1] - 0.02 * pinch,
            thumb_tip_base[2],
        ]
        index_tip = [
            index_tip_base[0] - 0.01 * pinch,
            index_tip_base[1] + 0.02 * pinch,
            index_tip_base[2],
        ]
        # Set thumb tip (index 4)
        keypoints[12] = thumb_tip[0]
        keypoints[13] = thumb_tip[1]
        keypoints[14] = thumb_tip[2]
        # Set index tip (index 8)
        keypoints[24] = index_tip[0]
        keypoints[25] = index_tip[1]
        keypoints[26] = index_tip[2]

        msg = Float32MultiArray()
        msg.data = keypoints
        self._pub_keypoints.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = MockHandTracker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
