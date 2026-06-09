#!/usr/bin/env python3
"""
Convert UMI Stage 1 HDF5 recordings to LeRobot v3.0 format (Parquet + JSON).

LeRobot v3.0 format spec:
  data/chunk-000/file-000.parquet  — frame data (tabular)
  meta/info.json                   — dataset configuration
  meta/stats.json                  — normalization statistics
  meta/tasks.parquet               — task definitions
  meta/episodes/chunk-000/...      — episode metadata
  videos/...                       — MP4 video files (optional, from camera images)

Usage:
    python3 lerobot_v3_converter.py --input /tmp/umi_recordings --output ./dataset_v3
"""

import argparse
import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Optional

import h5py
import numpy as np
import pandas as pd

# ── Feature definitions for UMI xArm6 teleop ────────────────────────────────
FEATURES = {
    "action.joint_position": {
        "dtype": "float32",
        "shape": [6],
        "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "action.gripper": {
        "dtype": "float32",
        "shape": [1],
        "names": ["gripper"],
    },
    "observation.joint_position": {
        "dtype": "float32",
        "shape": [6],
        "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "observation.joint_velocity": {
        "dtype": "float32",
        "shape": [6],
        "names": ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"],
    },
    "observation.gripper": {
        "dtype": "float32",
        "shape": [1],
        "names": ["gripper"],
    },
    "observation.image_features": {
        "dtype": "float32",
        "shape": [128],
        "names": None,
    },
    "observation.goal_position": {
        "dtype": "float32",
        "shape": [6],
        "names": ["goal_1", "goal_2", "goal_3", "goal_4", "goal_5", "goal_6"],
    },
}

# Mapping from UMI HDF5 keys to LeRobot feature keys
UMI_TO_LEROBOT = {
    "joint_command/position": "action.joint_position",
    "joint_state/position": "observation.joint_position",
    "joint_state/velocity": "observation.joint_velocity",
    "gripper/command": "action.gripper",
    "gripper/state": "observation.gripper",
    "observation/image_features": "observation.image_features",
    "observation/goal_position": "observation.goal_position",
}

CHUNK_SIZE = 1000  # max files per chunk directory
DATA_FILE_SIZE_MB = 100  # target Parquet file size before rotating


@dataclass
class ConversionStats:
    total_episodes: int = 0
    total_frames: int = 0
    episodes: list = field(default_factory=list)


def _ep_idx_from_path(h5_path: str) -> int:
    """Extract episode index from filename (e.g., episode_000003.h5 → 3)."""
    basename = os.path.splitext(os.path.basename(h5_path))[0]
    try:
        parts = basename.split("_")
        if len(parts) >= 2:
            return int(parts[-1])
    except (ValueError, IndexError):
        pass
    return 0


def read_episode_data(h5_path: str, episode_index: int = None) -> tuple:
    """Read a UMI HDF5 episode and return as dict of arrays + metadata."""
    with h5py.File(h5_path, "r") as f:
        eps = [k for k in f.keys() if k.startswith("episode_")]
        if not eps:
            raise ValueError(f"No episode group found in {h5_path}")
        ep = f[eps[0]]

        if episode_index is not None:
            ep_idx = episode_index
        else:
            ep_idx = int(eps[0].split("_")[-1])

        # Determine number of steps from joint_command (primary timeline)
        if "joint_command/position" not in ep:
            raise ValueError(f"No joint_command/position in episode {eps[0]}")
        cmd = ep["joint_command/position"][:]
        n_steps = cmd.shape[0]

        # Build per-feature arrays
        data = {}
        for h5_key, lr_key in UMI_TO_LEROBOT.items():
            if h5_key in ep:
                arr = ep[h5_key][:]
                # Handle shape: (N, 1) → (N,)
                if arr.ndim == 2 and arr.shape[1] == 1:
                    arr = arr.flatten()
                data[lr_key] = arr.astype(np.float32)

        # Timestamps
        timestamps = None
        ts_key = "joint_command/position_timestamp"
        if ts_key in ep:
            timestamps = ep[ts_key][:].flatten()
        elif "timestamp" in ep:
            timestamps = ep["timestamp"][:].flatten()

        # Camera images (check for RGB)
        has_images = "sensors/camera/rgb" in ep

    return ep_idx, n_steps, data, timestamps, has_images


def build_dataframe(
    episode_idx: int,
    n_steps: int,
    data: dict,
    timestamps: Optional[np.ndarray],
    global_frame_start: int,
    task_index: int = 0,
) -> pd.DataFrame:
    """Build a Parquet-ready DataFrame for one episode."""
    rows = []

    for i in range(n_steps):
        row = {
            "episode_index": int(episode_idx),
            "frame_index": int(i),
            "index": int(global_frame_start + i),
            "task_index": int(task_index),
        }
        # Timestamp: seconds since episode start
        if timestamps is not None and i < len(timestamps):
            t0 = timestamps[0]
            row["timestamp"] = np.float32(timestamps[i] - t0)
        else:
            row["timestamp"] = np.float32(0.0)

        # Feature data
        for lr_key in FEATURES:
            if lr_key in data and i < len(data[lr_key]):
                val = data[lr_key][i]
                if np.isscalar(val):
                    row[lr_key] = [float(val)]
                elif isinstance(val, np.ndarray):
                    row[lr_key] = val.astype(np.float32).tolist()
                else:
                    row[lr_key] = list(val)
            else:
                shape = FEATURES[lr_key]["shape"]
                if len(shape) == 1 and shape[0] == 1:
                    row[lr_key] = [0.0]
                else:
                    row[lr_key] = [0.0] * shape[0]

        rows.append(row)

    return pd.DataFrame(rows)


def write_info_json(output_dir: str, stats: ConversionStats, fps: int = 30):
    """Write meta/info.json."""
    info = {
        "codebase_version": "v3.0",
        "robot_type": "xarm6",
        "fps": fps,
        "total_episodes": stats.total_episodes,
        "total_frames": stats.total_frames,
        "chunks_size": CHUNK_SIZE,
        "data_files_size_in_mb": DATA_FILE_SIZE_MB,
        "video_files_size_in_mb": 200,
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": FEATURES,
    }
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)


def compute_and_write_stats(output_dir: str, all_data: list):
    """Compute global normalization stats from collected data and write stats.json."""
    # Accumulate per-feature arrays
    accum = {key: [] for key in FEATURES}
    for df in all_data:
        for key in FEATURES:
            if key in df.columns:
                vals = np.vstack(df[key].values)
                accum[key].append(vals)

    stats = {}
    for key, arrays in accum.items():
        if not arrays:
            continue
        combined = np.concatenate(arrays, axis=0)
        stats[key] = {
            "min": combined.min(axis=0).tolist(),
            "max": combined.max(axis=0).tolist(),
            "mean": combined.mean(axis=0).tolist(),
            "std": combined.std(axis=0).tolist(),
        }

    with open(os.path.join(output_dir, "meta", "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)


def write_episodes_metadata(output_dir: str, episodes: list):
    """Write meta/episodes/ parquet files."""
    meta_dir = os.path.join(output_dir, "meta", "episodes", "chunk-000")
    os.makedirs(meta_dir, exist_ok=True)

    df = pd.DataFrame(episodes, columns=["episode_index", "length", "task"])
    # Add data file location info
    df["data/chunk_index"] = 0
    df["data/file_index"] = 0
    df["task_index"] = 0

    out_path = os.path.join(meta_dir, "file-000.parquet")
    df.to_parquet(out_path, index=False)


def convert_directory(
    input_dir: str, output_dir: str, fps: int = 30, task_yaml: str = None,
    task_index_in_data: bool = False,
) -> ConversionStats:
    """Convert all HDF5 episodes in a directory to LeRobot v3.0 format.

    Args:
        task_index_in_data: If True, read per-episode task_index from HDF5 attrs.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Clear output if exists
    data_dir = os.path.join(output_dir, "data", "chunk-000")
    if os.path.exists(os.path.join(output_dir, "data")):
        shutil.rmtree(os.path.join(output_dir, "data"))
    os.makedirs(data_dir, exist_ok=True)

    # Load task definitions if provided
    task_manager = None
    if task_yaml and os.path.exists(task_yaml):
        from stage_2.task_manager import TaskManager
        task_manager = TaskManager(task_yaml)

    stats = ConversionStats()
    all_dfs = []
    global_frame = 0

    # Sort HDF5 files by episode index
    h5_files = sorted(
        [f for f in os.listdir(input_dir) if f.endswith(".h5")]
    )

    for fname in h5_files:
        h5_path = os.path.join(input_dir, fname)
        ep_idx = _ep_idx_from_path(h5_path)
        _, n_steps, data, timestamps, has_images = read_episode_data(h5_path, episode_index=ep_idx)

        # Get task_index from HDF5 attributes or default
        task_idx = 0
        if task_index_in_data:
            with h5py.File(h5_path, "r") as f:
                eps = [k for k in f.keys() if k.startswith("episode_")]
                if eps and "task_index" in f[eps[0]].attrs:
                    task_idx = int(f[eps[0]].attrs["task_index"])

        # Get task label for this episode
        task_label = ""
        if task_manager is not None:
            task_label = task_manager.get_task(task_manager.default_task_index).description

        df = build_dataframe(ep_idx, n_steps, data, timestamps, global_frame,
                             task_index=task_idx)
        all_dfs.append(df)

        stats.total_episodes += 1
        stats.total_frames += n_steps
        stats.episodes.append((ep_idx, n_steps, task_label))
        global_frame += n_steps

        print(f"  Episode {ep_idx}: {n_steps} frames (task={task_idx})"
              f"{' (+images)' if has_images else ''}")

    # Write all data to a single Parquet file (will split into chunks if > DATA_FILE_SIZE_MB)
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        data_path = os.path.join(data_dir, "file-000.parquet")
        combined.to_parquet(data_path, index=False)
        size_mb = os.path.getsize(data_path) / (1024 * 1024)
        print(f"\nData: {len(combined)} total frames in {size_mb:.1f} MB")

    # Write tasks.parquet if task definitions were loaded
    if task_manager is not None:
        tasks_path = os.path.join(output_dir, "meta", "tasks.parquet")
        task_manager.export_tasks_parquet(tasks_path)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Convert UMI Stage 1 HDF5 to LeRobot v3.0 Parquet format"
    )
    parser.add_argument("--input", "-i", required=True, help="Input HDF5 file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output LeRobot v3.0 dataset directory")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--tasks", default=None, help="Path to tasks YAML file")
    args = parser.parse_args()

    output_dir = args.output
    if os.path.exists(output_dir):
        shutil.rmtree(os.path.join(output_dir, "data"), ignore_errors=True)
        shutil.rmtree(os.path.join(output_dir, "meta"), ignore_errors=True)

    print(f"Converting {args.input} → {output_dir}")
    stats = convert_directory(args.input, output_dir, fps=args.fps, task_yaml=args.tasks)

    # Write metadata
    write_info_json(output_dir, stats, fps=args.fps)
    write_episodes_metadata(output_dir, stats.episodes)

    # Collect all data for stats computation
    data_dir = os.path.join(output_dir, "data", "chunk-000")
    all_dfs = [pd.read_parquet(os.path.join(data_dir, f))
               for f in os.listdir(data_dir) if f.endswith(".parquet")]
    compute_and_write_stats(output_dir, all_dfs)

    print(f"\nConverted {stats.total_episodes} episode(s), {stats.total_frames} total frames")
    print(f"Output: {output_dir}")
    print(f"  {output_dir}/data/chunk-000/file-000.parquet")
    print(f"  {output_dir}/meta/info.json")
    print(f"  {output_dir}/meta/stats.json")
    print(f"  {output_dir}/meta/episodes/chunk-000/file-000.parquet")


if __name__ == "__main__":
    main()
