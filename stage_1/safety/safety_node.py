#!/usr/bin/env python3
"""ROS2 safety guardian node: validates commands and forwards to the robot."""

import sys
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64, String
from std_srvs.srv import Trigger

from stage_1.robot_hal.mock_robot import MockRobotInterface
from stage_1.kinematics.dh_params import XARM6_JOINT_LIMITS, XARM6_VELOCITY_LIMITS


STATE_NORMAL = "NORMAL"
STATE_WARNING = "WARNING"
STATE_ESTOP = "EMERGENCY_STOP"


class SafetyGuardian(Node):
    """Validates teleop commands and forwards safe ones to the robot."""

    def __init__(self):
        super().__init__("safety_guardian")
        self.declare_parameter("robot_mode", "mock")
        self.declare_parameter("xarm6_ip", "192.168.1.100")
        self.declare_parameter("mjcf_path", "")
        self.declare_parameter("velocity_limit", np.pi)          # rad/s
        self.declare_parameter("position_delta_limit", 0.3)      # rad per step
        self.declare_parameter("control_rate", 60.0)

        mode = self.get_parameter("robot_mode").value
        self._velocity_limit = self.get_parameter("velocity_limit").value
        self._delta_limit = self.get_parameter("position_delta_limit").value
        control_rate = self.get_parameter("control_rate").value

        # Instantiate robot
        if mode == "mock":
            self._robot = MockRobotInterface()
            self.get_logger().info("Safety guardian: mock mode")
        elif mode == "xarm6":
            from stage_1.robot_hal.xarm6_interface import XArm6Interface
            ip = self.get_parameter("xarm6_ip").value
            self._robot = XArm6Interface(ip=ip)
            self.get_logger().info(f"Safety guardian: xarm6 mode (ip={ip})")
        elif mode == "mujoco":
            sys.path.insert(0, "/workspace/umi")
            from stage_2.simulation.mujoco_interface import MujocoRobotInterface
            model_path = self.get_parameter("mjcf_path").value
            self._robot = MujocoRobotInterface(model_path=model_path or None)
            self.get_logger().info(f"Safety guardian: mujoco mode (model={model_path})")
        else:
            raise ValueError(f"Unknown robot_mode: {mode}")

        if not self._robot.connect():
            self.get_logger().error("Failed to connect to robot")

        # State machine
        self._state = STATE_NORMAL
        self._enabled = True
        self._pending_command = None   # (positions, velocity)
        self._pending_gripper = None   # float
        self._last_state = None        # JointState
        self._warning_count = 0
        self._warning_escalate_limit = 3  # consecutive warnings → ESTOP

        # Publishers
        self._pub_joint_state = self.create_publisher(
            JointState, "/teleop/state/joints", 10
        )
        self._pub_gripper_state = self.create_publisher(
            Float64, "/teleop/state/gripper", 10
        )
        self._pub_status = self.create_publisher(String, "/safety/status", 10)

        # Subscribers
        self._sub_joint_cmd = self.create_subscription(
            JointState, "/teleop/command/joints", self._on_joint_cmd, 10
        )
        self._sub_gripper_cmd = self.create_subscription(
            Float64, "/teleop/command/gripper", self._on_gripper_cmd, 10
        )

        # Services
        self._srv_reset = self.create_service(Trigger, "/safety/reset", self._on_reset)
        self._srv_enable = self.create_service(Trigger, "/safety/enable", self._on_enable)
        self._srv_disable = self.create_service(Trigger, "/safety/disable", self._on_disable)

        # Control loop
        self._control_timer = self.create_timer(1.0 / control_rate, self._control_loop)

        self.get_logger().info("Safety guardian ready")

    # ---- Subscriptions ----
    def _on_joint_cmd(self, msg: JointState):
        if not msg.position or len(msg.position) != 6:
            return
        self._pending_command = (
            np.array(msg.position, dtype=np.float64),
            0.5,  # default velocity
        )

    def _on_gripper_cmd(self, msg: Float64):
        self._pending_gripper = float(np.clip(msg.data, 0.0, 1.0))

    # ---- Services ----
    def _on_reset(self, req, resp):
        if self._state == STATE_ESTOP:
            self._state = STATE_NORMAL
            self._warning_count = 0
            self._robot.stop()
            resp.success = True
            resp.message = "Reset from EMERGENCY_STOP to NORMAL"
            self.get_logger().warn("Safety reset to NORMAL")
        else:
            resp.success = True
            resp.message = f"Already in {self._state}"
        return resp

    def _on_enable(self, req, resp):
        self._enabled = True
        resp.success = True
        resp.message = "Commands enabled"
        return resp

    def _on_disable(self, req, resp):
        self._enabled = False
        self._robot.stop()
        resp.success = True
        resp.message = "Commands disabled (soft e-stop)"
        return resp

    # ---- Control loop ----
    def _control_loop(self):
        # Read and publish robot state
        if self._robot is None:
            return
        try:
            joint_state = self._robot.get_joint_state()
        except Exception:
            self.get_logger().error("Failed to read joint state", throttle_duration_sec=2.0)
            return

        self._last_state = joint_state

        state_msg = JointState()
        state_msg.header.stamp = self.get_clock().now().to_msg()
        state_msg.name = joint_state.name
        state_msg.position = joint_state.position.tolist()
        state_msg.velocity = joint_state.velocity.tolist()
        state_msg.effort = joint_state.effort.tolist()
        self._pub_joint_state.publish(state_msg)

        try:
            gripper_state = self._robot.get_gripper_state()
            gs = Float64()
            gs.data = gripper_state.position
            self._pub_gripper_state.publish(gs)
        except Exception:
            pass

        # Process pending command
        if self._pending_command is not None:
            positions, velocity = self._pending_command
            self._pending_command = None

            if self._state == STATE_ESTOP or not self._enabled:
                return

            # Joint limit check
            limited = np.clip(positions, XARM6_JOINT_LIMITS[:, 0], XARM6_JOINT_LIMITS[:, 1])
            if not np.allclose(limited, positions):
                positions = limited
                self._enter_state(STATE_WARNING, "Joint limit violation")

            # Delta check
            if joint_state.position is not None and len(joint_state.position) == 6:
                deltas = np.abs(positions - joint_state.position)
                max_delta = np.max(deltas)
                if max_delta > self._delta_limit:
                    self._enter_state(STATE_WARNING, f"Large delta: {max_delta:.3f} rad")
                    # Clamp to delta limit
                    clamped = joint_state.position + np.clip(
                        positions - joint_state.position,
                        -self._delta_limit,
                        self._delta_limit,
                    )
                    positions = clamped

            # Forward to robot
            try:
                self._robot.move_joints(positions, velocity=velocity, blocking=False)
            except Exception:
                self._enter_state(STATE_ESTOP, "move_joints failed")

        # Process pending gripper command
        if self._pending_gripper is not None:
            gripper = self._pending_gripper
            self._pending_gripper = None

            if self._state == STATE_ESTOP or not self._enabled:
                return

            try:
                self._robot.move_gripper(gripper, blocking=False)
            except Exception:
                self._enter_state(STATE_ESTOP, "move_gripper failed")

        # Publish status
        status = String()
        status.data = self._state
        self._pub_status.publish(status)

    # ---- Internal ----
    def _enter_state(self, new_state: str, reason: str):
        if new_state == self._state:
            return
        old = self._state
        if new_state == STATE_ESTOP:
            self._state = STATE_ESTOP
            self._warning_count = 0
            self._robot.stop()
            self.get_logger().error(f"EMERGENCY_STOP: {reason}")
        elif new_state == STATE_WARNING:
            self._warning_count += 1
            self.get_logger().warn(
                f"WARNING ({self._warning_count}/{self._warning_escalate_limit}): {reason}"
            )
            if self._warning_count >= self._warning_escalate_limit:
                self._state = STATE_ESTOP
                self._robot.stop()
                self.get_logger().error(
                    f"EMERGENCY_STOP: {self._warning_escalate_limit} consecutive warnings"
                )
                return
            self._state = STATE_WARNING
        elif new_state == STATE_NORMAL:
            self._warning_count = 0
            self._state = STATE_NORMAL
        self.get_logger().info(f"State: {old} -> {self._state} ({reason})")

    def destroy_node(self):
        if self._robot is not None:
            self._robot.disconnect()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = SafetyGuardian()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
