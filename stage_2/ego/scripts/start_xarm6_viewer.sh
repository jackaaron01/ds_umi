#!/bin/bash
# xArm6 Simulation Viewer Launcher
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export PYTHONPATH=/workspace/umi:${PYTHONPATH}
exec python3 /workspace/umi/stage_2/ego/scripts/xarm6_viewer.py "$@"
