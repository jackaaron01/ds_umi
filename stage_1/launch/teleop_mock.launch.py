"""Launch the full teleop pipeline in mock mode (no hardware required)."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    output_dir = LaunchConfiguration("output_dir", default="/tmp/umi_mock_recordings")
    recording_duration = LaunchConfiguration("recording_duration", default="60")
    enable_cameras = LaunchConfiguration("enable_cameras", default="false")

    return LaunchDescription([
        DeclareLaunchArgument("output_dir", default_value="/tmp/umi_mock_recordings"),
        DeclareLaunchArgument("recording_duration", default_value="60"),
        DeclareLaunchArgument("enable_cameras", default_value="false"),

        # Mock hand tracker: publishes synthetic wrist poses at 60 Hz
        Node(
            package="teleop_bridge",
            executable="mock_hand_tracker",
            name="mock_hand_tracker",
            output="screen",
        ),

        # Hand mapper: wrist poses → IK → joint commands
        Node(
            package="teleop_bridge",
            executable="hand_mapper",
            name="hand_mapper",
            output="screen",
            parameters=[{"hand": "right", "scale": 3.0, "lowpass_alpha": 0.3}],
        ),

        # Safety guardian: validates and forwards to MockRobot
        Node(
            package="safety",
            executable="safety_guardian",
            name="safety_guardian",
            output="screen",
            parameters=[{"robot_mode": "mock"}],
        ),

        # Recorder: HDF5 recording
        Node(
            package="recorder",
            executable="recorder",
            name="recorder",
            output="screen",
            parameters=[{"output_dir": output_dir}],
        ),

        # Camera: USB camera driver (optional, controlled by enable_cameras)
        Node(
            package="perception",
            executable="camera_node",
            name="camera_node",
            output="screen",
            condition=IfCondition(enable_cameras),
            parameters=[{
                "device_id": 0,
                "width": 640,
                "height": 480,
                "fps": 30.0,
                "topic_name": "/camera/rgb/image_raw",
            }],
        ),
    ])
