#!/usr/bin/env python3
"""
Launch file: MuJoCo simulation teleop with keyboard control.

Starts hand_mapper → safety(mujoco) → recorder.
Keyboard teleop runs separately (python3 keyboard_teleop.py) and publishes
to /hand/right/* topics.

Usage:
    # Terminal 1:
    ros2 launch launch teleop_sim_keyboard.launch.py

    # Terminal 2:
    python3 stage_2/simulation/keyboard_teleop.py

    # Terminal 3 (optional, for visualization):
    python3 -c "
    import mujoco, time
    m = mujoco.MjModel.from_xml_path('stage_2/simulation/xarm6.xml')
    d = mujoco.MjData(m)
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            mujoco.mj_step(m, d)
            viewer.sync()
            time.sleep(0.01)
    "
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

        # NOTE: No mock_hand_tracker — use keyboard_teleop.py instead
        # Keyboard node publishes to /hand/right/wrist_pose and /hand/right/keypoints

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
