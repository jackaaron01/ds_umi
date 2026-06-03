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

        # Subscribe to robot state for display
        from sensor_msgs.msg import JointState
        self._robot_joints = np.zeros(6)
        self._joint_sub = self.create_subscription(
            JointState, "/teleop/state/joints", self._joint_cb, 10
        )

        # Current pose state
        self._pos = np.array([0.5, 0.0, 0.4], dtype=np.float64)
        self._rpy = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        self._gripper = 1.0

        # Motion parameters
        self._pos_step = 0.01
        self._rot_step = 0.05
        self._gripper_step = 0.05
        self._gripper_target = 1.0
        self._display_counter = 0

        # Keyboard state
        self._keys_pressed = set()
        self._running = True

        # Start keyboard thread
        self._key_thread = threading.Thread(target=self._keyboard_loop, daemon=True)
        self._key_thread.start()

    def _joint_cb(self, msg):
        if len(msg.position) >= 6:
            self._robot_joints = np.array(msg.position[:6])

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

        # Speed control
        if "+" in self._keys_pressed or "=" in self._keys_pressed:
            self._pos_step = min(0.10, self._pos_step * 1.5)
            self._rot_step = min(0.30, self._rot_step * 1.5)
        if "-" in self._keys_pressed:
            self._pos_step = max(0.001, self._pos_step / 1.5)
            self._rot_step = max(0.01, self._rot_step / 1.5)

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


def _draw_display(node):
    """Draw live status display using ANSI escape codes."""
    # Clear screen and move cursor home
    sys.stdout.write("\033[2J\033[H")

    pos = node._pos
    rpy = node._rpy
    joints = node._robot_joints
    gripper = node._gripper
    step = node._pos_step
    rstep = node._rot_step

    # Top bar
    print("\033[1;37;44m  UMI Simulation Teleop  \033[0m")
    print()

    # End-effector pose
    print("  \033[1mEnd-Effector Target\033[0m")
    print(f"    Position:  X={pos[0]:6.3f}  Y={pos[1]:6.3f}  Z={pos[2]:6.3f}  m")
    print(f"    Rotation:  R={rpy[0]:6.2f}  P={rpy[1]:6.2f}  Y={rpy[2]:6.2f}  rad")
    gripper_bar = "█" * int(gripper * 10) + "░" * (10 - int(gripper * 10))
    print(f"    Gripper:   [{gripper_bar}] {gripper:.1f}  {'OPEN' if gripper > 0.5 else 'CLOSE'}")
    print()

    # Robot joint state
    print("  \033[1mRobot Joints (from MuJoCo)\033[0m")
    jstr = " ".join(f"{j:6.2f}" for j in joints)
    print(f"    J1-J6: [{jstr}] rad")
    print()

    # Controls map
    print("  \033[1mControls\033[0m                      \033[1mRotation\033[0m")
    print("    ┌───────┬───────┬───────┐    ┌───────┬───────┬───────┐")
    print("    │       │   E   │       │    │       │   I   │       │")
    print("    │       │  +Z   │       │    │       │ Pitch+│       │")
    print("    ├───────┼───────┼───────┤    ├───────┼───────┼───────┤")
    print("    │   A   │  S/W  │   D   │    │   J   │       │   L   │")
    print("    │  +Y   │ -X/+X │  -Y   │    │ Roll+ │       │ Roll- │")
    print("    ├───────┼───────┼───────┤    ├───────┼───────┼───────┤")
    print("    │       │   Q   │       │    │       │   K   │       │")
    print("    │       │  -Z   │       │    │       │ Pitch-│       │")
    print("    └───────┴───────┴───────┘    └───────┴───────┴───────┘")
    print()
    print("                                ┌───────┬───────┬───────┐")
    print("    \033[1mOther\033[0m                        │   U   │       │   O   │")
    print("    SPACE = hold to grip        │ Yaw+  │       │ Yaw-  │")
    print("    R     = reset pose          └───────┴───────┴───────┘")
    print("    +/-   = speed up/down")
    print(f"    Speed: pos={step*100:.0f}cm  rot={rstep*180/np.pi:.0f}°")
    print()
    print("  \033[90mCtrl+C to quit\033[0m")


def main():
    rclpy.init()
    node = KeyboardTeleop(rate=30.0)

    # Clear screen once
    sys.stdout.write("\033[2J\033[H")
    sys.stdout.flush()

    try:
        while node.running and rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

            # Update display at ~10 Hz
            node._display_counter += 1
            if node._display_counter % 3 == 0:
                _draw_display(node)
                sys.stdout.flush()

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        sys.stdout.write("\033[2J\033[H")  # Clear on exit
        sys.stdout.flush()
        print("Keyboard teleop stopped.")


if __name__ == "__main__":
    main()
