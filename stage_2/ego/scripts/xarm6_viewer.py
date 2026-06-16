#!/usr/bin/env python3
"""xArm6 MuJoCo simulation viewer — standalone script."""
import sys, os, time, threading
sys.path.insert(0, '/workspace/umi')

import mujoco
from mujoco.viewer import launch_passive
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

MODEL = '/workspace/umi/stage_2/ego/simulation/xarm_mesh.xml'

class JointListener(Node):
    def __init__(self):
        super().__init__('sim_viewer_standalone')
        self.joints = np.zeros(6)
        self._lock = threading.Lock()
        self.sub = self.create_subscription(JointState, '/teleop/state/joints', self._cb, 10)
        print('[viewer] Subscribed to /teleop/state/joints')

    def _cb(self, msg):
        with self._lock:
            if len(msg.position) >= 6:
                self.joints = np.array(msg.position[:6], dtype=np.float64)

    def get_joints(self):
        with self._lock:
            return self.joints.copy()

def main():
    rclpy.init()
    node = JointListener()

    # Spin ROS in daemon thread
    spin_thread = threading.Thread(target=lambda: rclpy.spin(node), daemon=True)
    spin_thread.start()

    m = mujoco.MjModel.from_xml_path(MODEL)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    print('[viewer] Opening xArm6 MuJoCo window...')
    with launch_passive(m, d) as viewer:
        print('[viewer] Window active — xArm6 simulation running')
        while viewer.is_running():
            q = node.get_joints()
            d.qpos[:6] = q
            d.ctrl[:6] = q
            mujoco.mj_forward(m, d)
            viewer.sync()
            time.sleep(0.016)
        print('[viewer] Window closed by user')

    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
