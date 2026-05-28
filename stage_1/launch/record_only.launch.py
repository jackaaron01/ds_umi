"""Launch only the recorder node (other nodes assumed already running)."""

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    output_dir = LaunchConfiguration("output_dir", default="~/umi_recordings")

    return LaunchDescription([
        DeclareLaunchArgument("output_dir", default_value="~/umi_recordings"),

        Node(
            package="recorder",
            executable="recorder",
            name="recorder",
            output="screen",
            parameters=[{"output_dir": output_dir}],
        ),
    ])
