#!/bin/bash
# xArm6 MuJoCo viewer — launch_passive (built-in, more robust)
set -e
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export PYTHONPATH=/workspace/umi:${PYTHONPATH}
echo "=== xArm6 Mesh Viewer (launch_passive): $(date) ==="
exec python3 -u /workspace/umi/docker/xarm6_viewer.py
