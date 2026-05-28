"""Integration test: mock hand tracker → hand mapper → safety → recorder.

Requires ROS2 to be running. Skip if rclpy cannot be imported.
"""

import os
import tempfile
import time
import pytest

pytest.importorskip("rclpy")


import rclpy
from rclpy.executors import MultiThreadedExecutor
import numpy as np
import h5py


@pytest.mark.integration
class TestMockTeleopPipeline:
    """End-to-end test of the full mock pipeline."""

    def test_pipeline_runs_and_records(self):
        """Launch all nodes, run for 3 seconds, verify HDF5 output."""
        from stage_1.teleop_bridge.mock_hand_tracker import MockHandTracker
        from stage_1.teleop_bridge.hand_mapper import HandMapper
        from stage_1.safety.safety_node import SafetyGuardian
        from stage_1.recorder.recorder_node import RecorderNode

        rclpy.init()

        output_dir = tempfile.mkdtemp()

        # Create nodes
        tracker = MockHandTracker()
        tracker.set_parameters([
            rclpy.parameter.Parameter("amplitude_x", value=0.05),
            rclpy.parameter.Parameter("amplitude_y", value=0.03),
            rclpy.parameter.Parameter("amplitude_z", value=0.03),
            rclpy.parameter.Parameter("offset_z", value=0.5),
        ])
        mapper = HandMapper(
            parameter_overrides=[
                rclpy.parameter.Parameter("scale", value=1.0),
            ]
        )
        safety = SafetyGuardian()
        safety.set_parameters([rclpy.parameter.Parameter("robot_mode", value="mock")])
        recorder = RecorderNode()
        recorder.set_parameters([rclpy.parameter.Parameter("output_dir", value=output_dir)])

        executor = MultiThreadedExecutor(num_threads=4)
        executor.add_node(tracker)
        executor.add_node(mapper)
        executor.add_node(safety)
        executor.add_node(recorder)

        # Start recording
        from std_srvs.srv import Trigger
        req = Trigger.Request()
        recorder._on_start(req, Trigger.Response())

        # Run for 3 seconds
        start = time.time()
        while time.time() - start < 3.0:
            executor.spin_once(timeout_sec=0.1)

        # Stop recording
        recorder._on_stop(req, Trigger.Response())

        # Destroy
        executor.shutdown()
        for node in [tracker, mapper, safety, recorder]:
            node.destroy_node()
        rclpy.shutdown()

        # Verify HDF5 output
        h5_files = [f for f in os.listdir(output_dir) if f.endswith(".h5")]
        assert len(h5_files) > 0, "No HDF5 file was created"

        filepath = os.path.join(output_dir, h5_files[0])
        with h5py.File(filepath, "r") as hf:
            ep = hf["episode_000000"]
            num_steps = ep.attrs["num_steps"]
            assert num_steps > 20, f"Expected >20 steps in 3s, got {num_steps}"

            # Check joint command data exists
            assert "joint_command" in ep, "joint_command group not found in HDF5 episode"
            cmd = ep["joint_command"]["position"][:]
            assert cmd.shape[0] == num_steps
            assert cmd.shape[1] == 6

            # Check joint state data exists
            assert "joint_state" in ep, "joint_state group not found in HDF5 episode"
            state = ep["joint_state"]["position"][:]
            assert state.shape[0] == num_steps
            assert state.shape[1] == 6

        # Cleanup
        os.unlink(filepath)
        os.rmdir(output_dir)
