#!/bin/bash
set -e

export PYTHONPATH=/workspace/umi:/ros2_ws/install/lib/python3.10/site-packages:$PYTHONPATH
sudo pip install -q --upgrade "setuptools>=65,<80" 2>/dev/null
sudo pip install -q "anyio<4" 2>/dev/null

cd /ros2_ws && colcon build 2>&1 | tail -2
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

python3 /workspace/umi/stage_2/act_research.py "$@" 2>&1
