#!/bin/bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH=/workspace/umi:/ros2_ws/install/lib/python3.10/site-packages:$PYTHONPATH

pkill -f hand_mapper 2>/dev/null
pkill -f safety_guardian 2>/dev/null
pkill -f recorder 2>/dev/null
sleep 1

/ros2_ws/install/teleop_bridge/bin/hand_mapper --ros-args -p hand:=right -p scale:=1.0 &
/ros2_ws/install/safety/bin/safety_guardian --ros-args -p robot_mode:=mujoco -p mjcf_path:=/workspace/umi/stage_2/simulation/xarm_color/xarm_urdf_color.xml &
/ros2_ws/install/recorder/bin/recorder --ros-args -p output_dir:=/tmp/sim_teleop_recordings &

sleep 2
echo "Pipeline ready. Run:"
echo "  make exec cmd=\"bash /workspace/umi/sim_teleop.sh\""
echo "  make exec cmd=\"bash /workspace/umi/sim_viewer.sh\""
wait
