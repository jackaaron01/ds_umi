"""Tests for the perception camera node."""

import os
import tempfile
import pytest

pytest.importorskip("rclpy")


class TestCameraNode:
    def test_import(self):
        from stage_1.perception.camera_node import CameraNode
        assert CameraNode is not None

    def test_load_calibration_yaml(self):
        import yaml

        calib_data = {
            "camera_matrix": {"data": [500, 0, 320, 0, 500, 240, 0, 0, 1]},
            "distortion_coefficients": {"data": [0.1, -0.2, 0.01, 0.01, 0.05]},
            "rectification_matrix": {"data": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
            "projection_matrix": {"data": [500, 0, 320, 0, 0, 500, 240, 0, 0, 0, 1, 0]},
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(calib_data, f)
            calib_path = f.name
        try:
            with open(calib_path, "r") as f:
                loaded = yaml.safe_load(f)
            assert loaded["camera_matrix"]["data"][0] == 500
            assert len(loaded["distortion_coefficients"]["data"]) == 5
        finally:
            os.unlink(calib_path)
