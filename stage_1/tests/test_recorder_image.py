"""Tests for image recording in the recorder node."""

import os
import tempfile
import time
import pytest
import numpy as np

pytest.importorskip("rclpy")
pytest.importorskip("h5py")

import rclpy
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import Image
import h5py


@pytest.mark.integration
class TestRecorderImage:
    def test_image_written_to_hdf5(self):
        """Verify images are written to HDF5 under sensors/camera/rgb."""
        from stage_1.recorder.recorder_node import RecorderNode
        from std_srvs.srv import Trigger

        rclpy.init()
        output_dir = tempfile.mkdtemp()
        recorder = RecorderNode()
        recorder.set_parameters([
            rclpy.parameter.Parameter("output_dir", value=output_dir),
            rclpy.parameter.Parameter("enable_image_recording", value=True),
        ])

        executor = SingleThreadedExecutor()
        executor.add_node(recorder)

        # Start recording
        req = Trigger.Request()
        recorder._on_start(req, Trigger.Response())

        # Publish a fake joint command and RGB image
        from sensor_msgs.msg import JointState
        cmd = JointState()
        cmd.position = [0.1, -0.2, 0.3, 0.0, 0.5, -0.1]
        recorder._cb_joint_cmd(cmd)

        h, w = 60, 80
        img = Image()
        img.header.stamp = recorder.get_clock().now().to_msg()
        img.header.frame_id = "camera_rgb_frame"
        img.height = h
        img.width = w
        img.encoding = "rgb8"
        img.step = w * 3
        img.data = np.random.randint(0, 255, h * w * 3, dtype=np.uint8).tobytes()

        recorder._cb_rgb(img)

        # Spin to allow flush
        for _ in range(10):
            executor.spin_once(timeout_sec=0.05)
            time.sleep(0.01)

        recorder._on_stop(req, Trigger.Response())
        executor.shutdown()
        recorder.destroy_node()
        rclpy.shutdown()

        # Verify HDF5
        h5_files = [f for f in os.listdir(output_dir) if f.endswith(".h5")]
        assert len(h5_files) > 0
        filepath = os.path.join(output_dir, h5_files[0])
        with h5py.File(filepath, "r") as hf:
            ep = hf["episode_000000"]
            assert "sensors" in ep
            assert "camera" in ep["sensors"]
            rgb_dset = ep["sensors"]["camera"]["rgb"]
            assert rgb_dset.dtype == np.uint8
            assert rgb_dset.shape[1:] == (h, w, 3)

        os.unlink(filepath)
        os.rmdir(output_dir)
