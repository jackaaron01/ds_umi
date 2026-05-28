"""Tests for the LeRobot format converter."""

import os
import tempfile
import numpy as np
import pytest

pytest.importorskip("h5py")
pytest.importorskip("pandas")

import h5py
import pandas as pd


class TestLeRobotConverter:
    def test_key_map_has_required_keys(self):
        from stage_1.recorder.lerobot_converter import KEY_MAP

        assert KEY_MAP["joint_command/position"] == "action/joint_position"
        assert KEY_MAP["joint_state/position"] == "observation/joint_position"
        assert KEY_MAP["sensors/camera/rgb"] == "observation/images/camera_rgb"
        assert KEY_MAP["gripper/command"] == "action/gripper"

    def test_convert_single_episode(self):
        from stage_1.recorder.lerobot_converter import (
            convert_episode,
            write_episodes_metadata,
        )
        from stage_1.recorder.hdf5_writer import HDF5Writer

        output_dir = tempfile.mkdtemp()
        input_path = os.path.join(output_dir, "input_episode.h5")

        try:
            # Create a UMI Stage 1 episode
            writer = HDF5Writer(input_path)
            writer.start_episode(0, metadata={"robot": "mock"})
            for i in range(10):
                writer.write_step({
                    "joint_command/position": np.ones(6) * i * 0.1,
                    "joint_state/position": np.ones(6) * i * 0.1,
                    "gripper/command": np.array([0.5]),
                    "timestamp": i * 0.033,
                })
            writer.end_episode()
            writer.close()

            # Convert to LeRobot format
            convert_episode(input_path, output_dir, episode_index=0)
            write_episodes_metadata(output_dir, [(0, 10)])

            # Verify LeRobot output
            out_h5 = os.path.join(output_dir, "data", "episode_000000.h5")
            assert os.path.exists(out_h5)

            with h5py.File(out_h5, "r") as f:
                assert "action" in f, f"Keys in output: {list(f.keys())}"
                assert f["action/joint_position"].shape == (10, 6)
                assert f["action/joint_position"][0, 0] == 0.0
                assert "observation" in f

            # Verify parquet metadata
            pq_path = os.path.join(output_dir, "meta", "episodes.parquet")
            assert os.path.exists(pq_path)
            df = pd.read_parquet(pq_path)
            assert df.iloc[0]["episode_index"] == 0
            assert df.iloc[0]["length"] == 10

        finally:
            import shutil
            shutil.rmtree(output_dir, ignore_errors=True)
