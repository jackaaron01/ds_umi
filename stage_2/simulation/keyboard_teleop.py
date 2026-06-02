#!/usr/bin/env python3
"""
Keyboard teleoperation node for MuJoCo simulation.

Publishes wrist_pose + keypoints to /hand/right/* topics, replacing
mock_hand_tracker. The rest of the pipeline (hand_mapper → safety → MuJoCo)
runs unchanged.

Usage (in container):
    python3 keyboard_teleop.py
    # Then in another terminal: ros2 launch launch teleop_mock.launch.py
    # (the keyboard node replaces mock_hand_tracker)

Controls:
    W/S      move end-effector +X/-X (forward/back)
    A/D      move end-effector +Y/-Y (left/right)
    Q/E      move end-effector +Z/-Z (up/down)
    I/K      pitch rotation
    J/L      roll rotation
    U/O      yaw rotation
    Space    close gripper (hold)
    R        reset pose
    Ctrl+C   quit
"""

import os, sys, select, termios, tty, threading, time
import numpy as np

# ROS2 imports (deferred until rclpy.init)
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32MultiArray, Header
from scipy.spatial.transform import Rotation


class KeyboardTeleop(Node):
    def __init__(self, rate: float = 30.0):
        super().__init__("keyboard_teleop")
        self._pose_pub = self.create_publisher(PoseStamped, "/hand/right/wrist_pose", 10)
        self._kp_pub = self.create_publisher(Float32MultiArray, "/hand/right/keypoints", 10)
        self._dt = 1.0 / rate
        self._timer = self.create_timer(self._dt, self._publish)

        # Current pose state
        self._pos = np.array([0.5, 0.0, 0.4], dtype=np.float64)  # x,y,z in robot frame
        self._rpy = np.array([0.0, 0.0, 0.0], dtype=np.float64)    # roll, pitch, yaw
        self._gripper = 1.0  # 1.0 = open

        # Motion parameters
        self._pos_step = 0.01  # meters per key press
        self._rot_step = 0.05  # radians per key press
        self._gripper_step = 0.05
        self._gripper_target = 1.0

        # Keyboard state
        self._keys_pressed = set()
        self._running = True

        # Start keyboard thread
        self._key_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._key_thread.start()

        self.get_logger().info("Keyboard teleop ready")
        self.get_logger().info("Controls: WASD=move, QE=up/down, IJKLUO=rotate, Space=grip, Ctrl+C=quit")

    def _keyboard_loop(self):
        """Read raw key presses in a background thread."""
        old_settings = termios.tcgetattr(sys.stdin)
        try:
            tty.setraw(sys.stdin.fileno())
            while self._running:
                if select.select([sys.stdin], [], [], 0.05)[0]:
                    c = sys.stdin.read(1)
                    if c == "\x03":  # Ctrl+C
                        self._running = False
                        break
                    elif c == "\x1b":  # Escape sequences (arrows)
                        # Try to read more
                        if select.select([sys.stdin], [], [], 0.01)[0]:
                            c2 = sys.stdin.read(1)
                            if c2 == "[" and select.select([sys.stdin], [], [], 0.01)[0]:
                                c3 = sys.stdin.read(1)
                                if c3 == "A": self._keys_pressed.add("UP")
                                elif c3 == "B": self._keys_pressed.add("DOWN")
                                elif c3 == "C": self._keys_pressed.add("RIGHT")
                                elif c3 == "D": self._keys_pressed.add("LEFT")
                            else:
                                self._keys_pressed.add(c + c2)
                        else:
                            self._keys_pressed.add(c)
                    else:
                        self._keys_pressed.add(c)
        finally:
            termios.tcsetattr(sys.stdin, old_settings[0])
            self._running = False

    def _process_keys(self):
        """Apply key presses to the pose state."""
        # Position
        if "w" in self._keys_pressed:
            self._pos[0] += self._pos_step
        if "s" in self._keys_pressed:
            self._pos[0] -= self._pos_step
        if "a" in self._keys_pressed:
            self._pos[1] += self._pos_step
        if "d" in self._keys_pressed:
            self._pos[1] -= self._pos_step
        if "e" in self._keys_pressed:
            self._pos[2] += self._pos_step
        if "q" in self._keys_pressed:
            self._pos[2] -= self._pos_step

        # Rotation
        if "i" in self._keys_pressed:
            self._rpy[0] += self._rot_step
        if "k" in self._keys_pressed:
            self._rpy[0] -= self._rot_step
        if "j" in self._keys_pressed:
            self._rpy[1] += self._rot_step
        if "l" in self._keys_pressed:
            self._rpy[1] -= self._rot_step
        if "u" in self._keys_pressed:
            self._rpy[2] += self._rot_step
        if "o" in self._keys_pressed:
            self._rpy[2] -= self._rot_step

        # Gripper
        if " " in self._keys_pressed:  # space
            self._gripper_target = 0.0  # close
        else:
            self._gripper_target = 1.0  # open

        # Reset
        if "r" in self._keys_pressed:
            self._pos = np.array([0.5, 0.0, 0.4])
            self._rpy = np.array([0.0, 0.0, 0.0])
            self._gripper = 1.0
            self._gripper_target = 1.0

        # Smooth gripper
        self._gripper += np.clip(self._gripper_target - self._gripper,
                                 -self._gripper_step, self._gripper_step)

        # Clear used keys
        self._keys_pressed.clear()

    def _publish(self):
        """Publish current pose and keypoints at the timer rate."""
        if not self._running:
            return
        self._process_keys()

        t = self.get_clock().now().to_msg()

        # PoseStamped
        quat = Rotation.from_euler("xyz", self._rpy).as_quat()  # [x,y,z,w]
        pose_msg = PoseStamped()
        pose_msg.header = Header(stamp=t, frame_id="world")
        pose_msg.pose.position.x = float(self._pos[0])
        pose_msg.pose.position.y = float(self._pos[1])
        pose_msg.pose.position.z = float(self._pos[2])
        pose_msg.pose.orientation.x = float(quat[0])
        pose_msg.pose.orientation.y = float(quat[1])
        pose_msg.pose.orientation.z = float(quat[2])
        pose_msg.pose.orientation.w = float(quat[3])
        self._pose_pub.publish(pose_msg)

        # Keypoints (21 landmarks × 3 = 63 floats)
        # Fill with placeholder values; hand_mapper uses landmarks 4+8 for gripper distance
        keypoints = np.zeros(63, dtype=np.float32)
        # Set thumb tip (landmark 4) and index tip (landmark 8) for gripper control
        # Thumb tip at fixed position, index tip moves based on gripper
        thumb_idx = 4 * 3
        index_idx = 8 * 3
        keypoints[thumb_idx:thumb_idx+3] = [0.0, 0.0, 0.0]
        # Distance: 0.015m (closed) to 0.080m (open), linear with gripper
        pinch_dist = 0.015 + self._gripper * 0.065
        keypoints[index_idx:index_idx+3] = [0.0, pinch_dist, 0.0]
        kp_msg = Float32MultiArray(data=keypoints.tolist())
        self._kp_pub.publish(kp_msg)

    @property
    def running(self) -> bool:
        return self._running


def main():
    rclpy.init()
    node = KeyboardTeleop(rate=30.0)

    print("\n" + "=" * 60)
    print("  Keyboard Teleop — MuJoCo Simulation")
    print("=" * 60)
    print()
    print("  🎮  Controls:")
    print("     W/S     move +X/-X (forward/back)")
    print("     A/D     move +Y/-Y (left/right)")
    print("     Q/E     move +Z/-Z (up/down)")
    print("     I/K     pitch  (+/-)")
    print("     J/L     roll   (+/-)")
    print("     U/O     yaw    (+/-)")
    print("     SPACE   close gripper (hold)")
    print("     R       reset pose")
    print("     Ctrl+C  quit")
    print()
    print("  Make sure the pipeline is also running:")
    print("    ros2 launch launch teleop_sim.launch.py")
    print()

    try:
        while node.running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        print("\nKeyboard teleop stopped.")


if __name__ == "__main__":
    main()
