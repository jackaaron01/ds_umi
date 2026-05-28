"""Launch the full teleop pipeline with real hardware (xArm6 + Quest3)."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition


def generate_launch_description():
    robot_ip = LaunchConfiguration("robot_ip", default="192.168.1.100")
    output_dir = LaunchConfiguration("output_dir", default="~/umi_recordings")
    enable_cameras = LaunchConfiguration("enable_cameras", default="false")

    return LaunchDescription([
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.100"),
        DeclareLaunchArgument("output_dir", default_value="~/umi_recordings"),
        DeclareLaunchArgument("enable_cameras", default_value="false"),

        # Hand tracking receiver: Quest3 UDP → ROS2 topics
        Node(
            package="teleop_bridge",
            executable="hand_tracking_node",
            name="hand_tracking_node",
            output="screen",
            parameters=[{
                "transport": "udp",
                "host": "0.0.0.0",
                "port": 12345,
                "hand": "right",
            }],
        ),

        # Hand mapper: Quest3 wrist poses → IK → joint commands
        Node(
            package="teleop_bridge",
            executable="hand_mapper",
            name="hand_mapper",
            output="screen",
            parameters=[{"hand": "right", "scale": 3.0, "lowpass_alpha": 0.3}],
        ),

        # Safety guardian: validates and forwards to XArm6Interface
        Node(
            package="safety",
            executable="safety_guardian",
            name="safety_guardian",
            output="screen",
            parameters=[{
                "robot_mode": "xarm6",
                "xarm6_ip": robot_ip,
                "velocity_limit": 3.14,
                "position_delta_limit": 0.3,
            }],
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
