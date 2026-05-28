from setuptools import setup

package_name = "safety"

setup(
    name=package_name,
    version="0.1.0",
    packages=["stage_1_safety"],
    package_dir={"stage_1_safety": "."},
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="dev",
    maintainer_email="user@example.com",
    description="Safety guardian node for UMI robot operation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "safety_guardian = stage_1_safety.safety_node:main",
        ],
    },
)
