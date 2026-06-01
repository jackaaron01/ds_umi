#!/usr/bin/env python3
"""ROS2 node: subscribes Quest3 wrist poses, runs IK, publishes joint/gripper commands."""

import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, Float32MultiArray

from stage_1.kinematics.ik import solve_ik
from stage_1.kinematics.utils import pose_to_transform
from stage_1.teleop_bridge.calibration import HandToRobotTransform


class HandMapper(Node):
    """Maps Quest3 hand tracking data to robot joint + gripper commands."""

    def __init__(self, transform=None, **kwargs):
        super().__init__("hand_mapper", **kwargs)
        self.declare_parameter("hand", "right")
        self.declare_parameter("scale", 3.0)
        self.declare_parameter("lowpass_alpha", 0.3)
        self.declare_parameter("calibration_file", "")

        hand = self.get_parameter("hand").value
        scale = self.get_parameter("scale").value
        alpha = self.get_parameter("lowpass_alpha").value
        calib_file = self.get_parameter("calibration_file").value

        if transform is not None:
            self._transform = transform
            self.get_logger().info("Using injected transform")
        elif calib_file:
            self._transform = HandToRobotTransform.from_yaml(calib_file)
            self.get_logger().info(f"Loaded calibration from {calib_file}")
        else:
            self._transform = HandToRobotTransform(scale=scale)
        self._alpha = alpha
        self._filtered_position = None  # 3-vector, initialised on first message
        self._filtered_orientation = None  # quaternion [w, x, y, z], initialised on first message
        self._q_current = np.array([0.0, -0.5, 0.0, 1.5, 0.0, 0.0])
        self._ik_times = deque(maxlen=100)

        self._pub_joint_cmd = self.create_publisher(
            JointState, "/teleop/command/joints", 10
        )
        self._pub_gripper_cmd = self.create_publisher(
            Float64, "/teleop/command/gripper", 10
        )

        self._sub_wrist = self.create_subscription(
            PoseStamped,
            f"/hand/{hand}/wrist_pose",
            self._on_wrist_pose,
            10,
        )
        self._sub_keypoints = self.create_subscription(
            Float32MultiArray,
            f"/hand/{hand}/keypoints",
            self._on_keypoints,
            10,
        )

        self.get_logger().info(f"Hand mapper started (hand={hand}, scale={scale})")

    def _on_wrist_pose(self, msg: PoseStamped):
        p_quest = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z,
        ])
        q_quest_xyzw = np.array([
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        ])

        # Low-pass filter
        if self._filtered_position is None:
            self._filtered_position = p_quest
            self._filtered_orientation = np.array([
                q_quest_xyzw[3], q_quest_xyzw[0], q_quest_xyzw[1], q_quest_xyzw[2]
            ])
        else:
            a = self._alpha
            self._filtered_position = a * p_quest + (1.0 - a) * self._filtered_position
            filtered_q_xyzw = a * q_quest_xyzw + (1.0 - a) * np.array([
                self._filtered_orientation[1],
                self._filtered_orientation[2],
                self._filtered_orientation[3],
                self._filtered_orientation[0],
            ])
            self._filtered_orientation = np.array([
                filtered_q_xyzw[3], filtered_q_xyzw[0], filtered_q_xyzw[1], filtered_q_xyzw[2]
            ])

        # Transform to robot frame
        p_robot = self._transform.transform_position(self._filtered_position)
        q_robot_wxyz = self._transform.transform_orientation_quat(q_quest_xyzw)

        # Build target homogeneous transform
        from stage_1.kinematics.utils import quaternion_to_rotation_matrix
        R_target = quaternion_to_rotation_matrix(q_robot_wxyz)
        T_target = np.eye(4)
        T_target[:3, :3] = R_target
        T_target[:3, 3] = p_robot

        t0 = time.perf_counter()
        q_sol, success, iters, error = solve_ik(T_target, q_init=self._q_current, max_iterations=80)
        ik_dt = time.perf_counter() - t0
        self._ik_times.append(ik_dt)
        if len(self._ik_times) >= 100:
            arr = np.array(self._ik_times)
            self.get_logger().info(
                f"IK timing (n=100): mean={np.mean(arr)*1000:.2f}ms "
                f"std={np.std(arr)*1000:.2f}ms "
                f"p50={np.percentile(arr,50)*1000:.2f}ms "
                f"p95={np.percentile(arr,95)*1000:.2f}ms "
                f"max={np.max(arr)*1000:.2f}ms"
            )
            self._ik_times.clear()

        if success:
            self._q_current = q_sol
        else:
            self._q_current = q_sol  # use best-effort as seed for next frame

        cmd = JointState()
        cmd.header.stamp = self.get_clock().now().to_msg()
        cmd.name = [f"joint{i}" for i in range(1, 7)]
        cmd.position = q_sol.tolist()
        self._pub_joint_cmd.publish(cmd)

        if not success:
            self.get_logger().warn(
                f"IK failed after {iters} iterations (error={error:.4f})",
                throttle_duration_sec=1.0,
            )

    def _on_keypoints(self, msg: Float32MultiArray):
        if len(msg.data) < 63:
            return
        kp = msg.data
        # Index finger tip: landmark 8 (indices 24, 25, 26)
        # Thumb tip:        landmark 4 (indices 12, 13, 14)
        p_thumb = np.array([kp[12], kp[13], kp[14]])
        p_index = np.array([kp[24], kp[25], kp[26]])
        distance = np.linalg.norm(p_index - p_thumb)

        close_thresh = 0.015
        open_thresh = 0.080
        gripper_cmd = 1.0 - (distance - close_thresh) / (open_thresh - close_thresh)
        gripper_cmd = float(np.clip(gripper_cmd, 0.0, 1.0))

        cmd = Float64()
        cmd.data = gripper_cmd
        self._pub_gripper_cmd.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = HandMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
