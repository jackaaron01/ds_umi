import numpy as np
import pytest
from stage_1.robot_hal.mock_robot import MockRobotInterface


class TestMockRobotInterface:
    def test_connect_disconnect(self):
        robot = MockRobotInterface()
        assert robot.connect()
        assert robot._connected
        assert robot.disconnect()
        assert not robot._connected

    def test_initial_state_zero(self):
        robot = MockRobotInterface()
        robot.connect()
        state = robot.get_joint_state()
        np.testing.assert_array_almost_equal(state.position, np.zeros(6))
        np.testing.assert_array_almost_equal(state.velocity, np.zeros(6))

    def test_blocking_move(self):
        robot = MockRobotInterface()
        robot.connect()
        target = np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.1])
        robot.move_joints(target, velocity=2.0, blocking=True)
        state = robot.get_joint_state()
        np.testing.assert_array_almost_equal(state.position, target, decimal=10)

    def test_blocking_move_returns_false_for_bad_shape(self):
        robot = MockRobotInterface()
        result = robot.move_joints(np.array([1.0, 2.0, 3.0]), blocking=True)
        assert not result

    def test_gripper(self):
        robot = MockRobotInterface()
        robot.connect()
        assert robot.get_gripper_state().position == 0.0
        robot.move_gripper(0.7)
        assert robot.get_gripper_state().position == 0.7

    def test_gripper_clamped(self):
        robot = MockRobotInterface()
        robot.move_gripper(1.5)
        assert robot.get_gripper_state().position == 1.0
        robot.move_gripper(-0.5)
        assert robot.get_gripper_state().position == 0.0

    def test_noise_injection(self):
        robot = MockRobotInterface()
        robot.connect()
        robot.enable_noise(std_pos=0.001, std_vel=0.01)
        robot.move_joints(np.zeros(6), blocking=True)
        state = robot.get_joint_state()
        # Position should be close to zero but not exactly
        assert np.any(np.abs(state.position) > 0)

    def test_set_joint_state(self):
        robot = MockRobotInterface()
        target = np.array([0.5, -0.5, 0.2, 0.3, -0.1, 0.0])
        robot.set_joint_state(target)
        state = robot.get_joint_state()
        np.testing.assert_array_almost_equal(state.position, target, decimal=10)

    def test_disconnect_stops_motion(self):
        robot = MockRobotInterface()
        robot.connect()
        robot.move_joints(np.array([1.0, -1.0, 0.5, -0.5, 0.3, -0.3]), blocking=False)
        robot.disconnect()
        assert not robot._motion_running
