#!/bin/bash
# xArm6 MuJoCo simulation viewer — uses mujoco built-in passive viewer
# Subscribes to /teleop/state/joints and renders the robot.
set -e
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
export PYTHONPATH=/workspace/umi:${PYTHONPATH}
export MUJOCO_GL=glfw
echo "=== xArm6 Simulation Viewer: $(date) ==="
python3 -c "
import sys, os, time, threading
sys.path.insert(0, '/workspace/umi')
import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

MODEL = '/workspace/umi/stage_2/simulation/xarm6.xml'
m = mujoco.MjModel.from_xml_path(MODEL)
d = mujoco.MjData(m)

# Zero init
mujoco.mj_forward(m, d)

class JointListener(Node):
    def __init__(self):
        super().__init__('sim_viewer_mini')
        self.joints = np.zeros(6)
        self.lock = threading.Lock()
        self.sub = self.create_subscription(JointState, '/teleop/state/joints', self.cb, 10)
    def cb(self, msg):
        with self.lock:
            if len(msg.position) >= 6:
                self.joints = np.array(msg.position[:6], dtype=np.float64)

rclpy.init()
node = JointListener()
# Spin ROS in background thread
t = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
t.start()
print('[viewer] ROS listener started, opening MuJoCo window...')

with mujoco.viewer.launch_passive(m, d) as viewer:
    print('[viewer] Window opened — showing xArm6 simulation')
    while viewer.is_running():
        with node.lock:
            d.qpos[:6] = node.joints.copy()
        mujoco.mj_forward(m, d)
        # Step actuators toward current qpos (position servo)
        d.ctrl[:6] = d.qpos[:6]
        viewer.sync()
        time.sleep(0.016)  # ~60Hz
    print('[viewer] Window closed')

node.destroy_node()
rclpy.shutdown()
print('[viewer] Stopped')
"
