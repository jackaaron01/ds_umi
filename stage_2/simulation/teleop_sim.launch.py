#!/usr/bin/env python3
"""
Launch file: MuJoCo simulation teleop pipeline.

Starts mock_hand_tracker → hand_mapper → safety(mujoco) → recorder.
Suitable for testing without hardware.

Usage:
    ros2 launch stage_2.simulation teleop_sim.launch.py
    ros2 launch stage_2.simulation teleop_sim.launch.py output_dir:=/tmp/recordings
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    output_dir = LaunchConfiguration("output_dir", default="/tmp/umi_sim_recordings")
    model_path = LaunchConfiguration("model_path", default="")

    return LaunchDescription([
        DeclareLaunchArgument("output_dir", default_value="/tmp/umi_sim_recordings"),
        DeclareLaunchArgument("model_path", default_value=""),

        # Mock hand tracker (60 Hz synthetic data)
        Node(
            package="teleop_bridge",
            executable="mock_hand_tracker",
            name="mock_hand_tracker",
            parameters=[{
                "frequency": 60.0,
                "amplitude_x": 0.03,
                "amplitude_y": 0.02,
                "amplitude_z": 0.02,
                "offset_z": 0.2,
            }],
        ),

        # Hand mapper (IK + low-pass filter)
        Node(
            package="teleop_bridge",
            executable="hand_mapper",
            name="hand_mapper",
            parameters=[{
                "hand": "right",
                "scale": 1.0,
            }],
        ),

        # Safety guardian with MuJoCo backend
        Node(
            package="safety",
            executable="safety_guardian",
            name="safety_guardian",
            parameters=[{
                "robot_mode": "mujoco",
                "mjcf_path": model_path,
                "control_rate": 60.0,
            }],
        ),

        # Recorder
        Node(
            package="recorder",
            executable="recorder_node",
            name="recorder",
            parameters=[{
                "output_dir": output_dir,
                "auto_start": False,
            }],
        ),
    ])
