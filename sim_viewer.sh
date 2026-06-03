#!/bin/bash
# MuJoCo 3D 可视化（终端 2）
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH=/workspace/umi:/ros2_ws/install/lib/python3.10/site-packages:$PYTHONPATH
echo "Starting MuJoCo viewer..."
python3 /workspace/umi/stage_2/simulation/viewer_node.py
