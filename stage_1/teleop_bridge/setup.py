from setuptools import setup

package_name = "teleop_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=["stage_1_teleop_bridge"],
    package_dir={"stage_1_teleop_bridge": "."},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev",
    maintainer_email="user@example.com",
    description="Quest3 hand-tracking to ROS2 bridge for UMI teleoperation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "mock_hand_tracker = stage_1_teleop_bridge.mock_hand_tracker:main",
            "hand_mapper = stage_1_teleop_bridge.hand_mapper:main",
            "hand_tracking_node = stage_1_teleop_bridge.hand_tracking_node:main",
            "hand_eye_calibrate = stage_1_teleop_bridge.calibrate:main",
        ],
    },
)
