#!/usr/bin/env python3
"""
Launch file for RealSense EGO teleop with MediaPipe hand tracking.

Starts all Docker-side ROS2 nodes:
  - mediapipe_bridge: UDP → ROS2 hand topics
  - hand_mapper: IK → joint commands
  - safety_node: safety guard (mock mode)
  - recorder_node: HDF5 recording

Usage (in Docker):
    ros2 launch launch ego_mediapipe.launch.py

Then on HOST:
    python3 stage_2/mediapipe_ego.py --udp
"""
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # UDP → ROS2 bridge (receives from host MediaPipe)
        Node(
            package="teleop_bridge",
            executable="mediapipe_bridge",
            name="mediapipe_bridge",
            parameters=[{"port": 9999, "hand": "right"}],
            output="screen",
        ),
        # Hand mapper: IK → joint/gripper commands
        Node(
            package="teleop_bridge",
            executable="hand_mapper",
            name="hand_mapper",
            parameters=[{"hand": "right", "scale": 0.5}],
            output="screen",
        ),
        # Safety guardian (mock mode for testing)
        Node(
            package="safety",
            executable="safety_node",
            name="safety_guardian",
            parameters=[{"robot_mode": "mujoco"}],
            output="screen",
        ),
        # Recorder (optional — use ROS2 service to start/stop)
        Node(
            package="recorder",
            executable="recorder_node",
            name="recorder",
            output="screen",
        ),
    ])
