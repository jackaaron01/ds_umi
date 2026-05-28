from setuptools import setup

package_name = "launch"

setup(
    name=package_name,
    version="0.1.0",
    packages=["stage_1_launch"],
    package_dir={"stage_1_launch": "."},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", [
            "record_only.launch.py",
            "teleop_mock.launch.py",
            "teleop_real.launch.py",
        ]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev",
    maintainer_email="user@example.com",
    description="Launch files for UMI teleoperation pipeline",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [],
    },
)
