#!/bin/bash
# EGO Teleop Pipeline Launcher — run inside Docker
set -e
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export PYTHONPATH=/workspace/umi:$PYTHONPATH
echo "=== EGO Pipeline starting: $(date) ==="
ros2 launch launch ego_mediapipe.launch.py
