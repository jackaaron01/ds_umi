#!/bin/bash
# 键盘遥操作 + 仿真管道（终端 1）
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH=/workspace/umi:/ros2_ws/install/lib/python3.10/site-packages:$PYTHONPATH
exec python3 /workspace/umi/stage_2/simulation/keyboard_teleop.py
