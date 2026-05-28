from setuptools import setup

package_name = "perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=["stage_1_perception"],
    package_dir={"stage_1_perception": "."},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev",
    maintainer_email="user@example.com",
    description="USB camera ROS2 driver for UMI",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_node = stage_1_perception.camera_node:main",
        ],
    },
)
