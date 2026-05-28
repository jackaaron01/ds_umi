import os
import tempfile
import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="h5py not installed")
from stage_1.recorder.hdf5_writer import HDF5Writer


class TestHDF5Writer:
    def test_create_and_close(self):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            writer = HDF5Writer(path)
            writer.close()
            assert os.path.exists(path)
            # Verify file is valid HDF5
            with h5py.File(path, "r") as hf:
                assert hf.attrs["format"] == "umi_stage1"
        finally:
            os.unlink(path)

    def test_write_single_step(self):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            writer = HDF5Writer(path)
            writer.start_episode(0, metadata={"robot": "mock"})
            writer.write_step({
                "joint_state/position": np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.1]),
                "joint_command/position": np.array([0.1, -0.2, 0.3, 0.0, 0.5, -0.1]),
                "gripper_command": np.array([0.5]),
                "timestamp": 0.0,
            })
            writer.end_episode()
            writer.close()

            with h5py.File(path, "r") as hf:
                ep = hf["episode_000000"]
                assert ep.attrs["robot"] == "mock"
                assert ep.attrs["num_steps"] == 1
                jp = ep["joint_state"]["position"][:]
                assert jp.shape == (1, 6)
        finally:
            os.unlink(path)

    def test_write_multiple_steps(self):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            writer = HDF5Writer(path)
            writer.start_episode(0)
            for i in range(10):
                writer.write_step({
                    "joint_state/position": np.ones(6) * i * 0.1,
                    "timestamp": i * 0.033,
                })
            writer.end_episode()
            writer.close()

            with h5py.File(path, "r") as hf:
                ep = hf["episode_000000"]
                assert ep.attrs["num_steps"] == 10
                jp = ep["joint_state"]["position"][:]
                assert jp.shape == (10, 6)
                assert jp[0, 0] == 0.0
                assert jp[-1, 0] == 0.9
        finally:
            os.unlink(path)

    def test_multiple_episodes(self):
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            writer = HDF5Writer(path)
            for ep_idx in range(3):
                writer.start_episode(ep_idx)
                for _ in range(5):
                    writer.write_step({
                        "joint_state/position": np.ones(6),
                        "timestamp": 0.0,
                    })
                writer.end_episode()
            writer.close()

            with h5py.File(path, "r") as hf:
                assert "episode_000000" in hf
                assert "episode_000001" in hf
                assert "episode_000002" in hf
                assert hf["episode_000000"].attrs["num_steps"] == 5
        finally:
            os.unlink(path)

    def test_skips_flat_keys(self):
        """Keys without '/' are silently skipped (regression test for recorder bug)."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            writer = HDF5Writer(path)
            writer.start_episode(0)
            writer.write_step({
                "joint_command_position": np.array([0.1, 0.2, 0.3, 0.0, 0.5, -0.1]),
                "joint_state/position": np.array([0.1, 0.2, 0.3, 0.0, 0.5, -0.1]),
                "timestamp": 0.0,
            })
            writer.end_episode()
            writer.close()
            with h5py.File(path, "r") as hf:
                ep = hf["episode_000000"]
                assert "joint_command_position" not in ep, "Flat key should be skipped"
                assert "joint_state" in ep, "Hierarchical key should create group"
                assert ep.attrs["num_steps"] == 1
        finally:
            os.unlink(path)

    def test_group_nesting(self):
        """Verify deeply nested group/dataset paths work."""
        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name
        try:
            writer = HDF5Writer(path)
            writer.start_episode(0)
            writer.write_step({
                "sensors/camera/rgb": np.random.rand(100).astype(np.float32),
                "sensors/camera/depth": np.random.rand(100).astype(np.float32),
                "timestamp": 0.0,
            })
            writer.end_episode()
            writer.close()

            with h5py.File(path, "r") as hf:
                ep = hf["episode_000000"]
                assert "sensors" in ep
                assert "camera" in ep["sensors"]
                assert "rgb" in ep["sensors"]["camera"]
                assert "depth" in ep["sensors"]["camera"]
        finally:
            os.unlink(path)
