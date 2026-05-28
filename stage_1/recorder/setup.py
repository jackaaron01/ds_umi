from setuptools import setup

package_name = "recorder"

setup(
    name=package_name,
    version="0.1.0",
    packages=["stage_1_recorder"],
    package_dir={"stage_1_recorder": "."},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev",
    maintainer_email="user@example.com",
    description="HDF5 recorder and LeRobot format converter for UMI",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "recorder = stage_1_recorder.recorder_node:main",
            "convert_to_lerobot = stage_1_recorder.lerobot_converter:main",
        ],
    },
)
