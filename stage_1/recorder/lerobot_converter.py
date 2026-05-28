#!/usr/bin/env python3
"""Convert UMI Stage 1 HDF5 recordings to LeRobot-compatible format.

Usage as CLI:
    convert_to_lerobot --input episode_000000.h5 --output /path/to/lerobot_dataset/

Usage as library:
    from stage_1.recorder.lerobot_converter import convert_episode
    convert_episode("episode_000000.h5", "/path/to/lerobot_dataset/")
"""

import argparse
import json
import os
import h5py
import numpy as np
import pandas as pd

KEY_MAP = {
    "joint_command/position": "action/joint_position",
    "joint_state/position": "observation/joint_position",
    "joint_state/velocity": "observation/joint_velocity",
    "sensors/camera/rgb": "observation/images/camera_rgb",
    "sensors/camera/depth": "observation/images/camera_depth",
    "gripper/command": "action/gripper",
    "gripper/state": "observation/gripper",
}


def convert_episode(input_path: str, output_dir: str, episode_index: int = None) -> str:
    """Convert a single UMI Stage 1 episode to LeRobot format.

    Args:
        input_path: Path to the UMI Stage 1 HDF5 file.
        output_dir: Root directory of the output LeRobot dataset.
        episode_index: If None, parsed from the episode group name.

    Returns:
        Path to the output HDF5 file.
    """
    os.makedirs(os.path.join(output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)

    with h5py.File(input_path, "r") as src:
        for ep_name in src.keys():
            if not ep_name.startswith("episode_"):
                continue
            ep_group = src[ep_name]
            if episode_index is None:
                ep_idx = int(ep_name.split("_")[-1])
            else:
                ep_idx = episode_index

            output_path = os.path.join(output_dir, "data", f"episode_{ep_idx:06d}.h5")
            num_steps = ep_group.attrs.get("num_steps", 0)

            with h5py.File(output_path, "w") as dst:
                _copy_and_remap(ep_group, dst, KEY_MAP)

    return output_path


def _copy_and_remap(src_group: h5py.Group, dst_file: h5py.File, key_map: dict):
    """Walk source group recursively and copy datasets to destination with key remapping."""

    def _walk(group, prefix=""):
        for name in group:
            item = group[name]
            full_key = f"{prefix}/{name}" if prefix else name
            if isinstance(item, h5py.Dataset):
                mapped = key_map.get(full_key)
                if mapped is not None:
                    parts = mapped.split("/")
                    dset_name = parts[-1]
                    grp_path = "/".join(parts[:-1])
                    if grp_path:
                        dst_grp = dst_file.require_group(grp_path)
                    else:
                        dst_grp = dst_file
                    dst_grp.create_dataset(
                        dset_name,
                        data=item[:],
                        dtype=item.dtype,
                        compression=item.compression,
                        compression_opts=item.compression_opts,
                    )
            elif isinstance(item, h5py.Group):
                _walk(item, full_key)

    _walk(src_group)


def write_episodes_metadata(output_dir: str, episodes: list):
    """Write meta/episodes.parquet with episode indices and lengths.

    Args:
        output_dir: Root of the LeRobot dataset.
        episodes: List of (episode_index, num_steps) tuples.
    """
    df = pd.DataFrame(episodes, columns=["episode_index", "length"])
    meta_dir = os.path.join(output_dir, "meta")
    os.makedirs(meta_dir, exist_ok=True)
    df.to_parquet(os.path.join(meta_dir, "episodes.parquet"), index=False)


def write_features_json(output_dir: str):
    """Write meta/features.json describing the data schema."""
    features = {
        "action/joint_position": {"dtype": "float64", "shape": [6]},
        "observation/joint_position": {"dtype": "float64", "shape": [6]},
        "observation/joint_velocity": {"dtype": "float64", "shape": [6]},
        "action/gripper": {"dtype": "float64", "shape": [1]},
        "observation/gripper": {"dtype": "float64", "shape": [1]},
    }
    os.makedirs(os.path.join(output_dir, "meta"), exist_ok=True)
    with open(os.path.join(output_dir, "meta", "features.json"), "w") as f:
        json.dump(features, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Convert UMI Stage 1 HDF5 to LeRobot format"
    )
    parser.add_argument("--input", "-i", required=True, help="Input HDF5 file or directory")
    parser.add_argument("--output", "-o", required=True, help="Output LeRobot dataset directory")
    parser.add_argument("--features", action="store_true", help="Also write features.json")
    args = parser.parse_args()

    input_path = args.input
    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    episodes_meta = []

    if os.path.isdir(input_path):
        for fname in sorted(os.listdir(input_path)):
            if fname.endswith(".h5"):
                fpath = os.path.join(input_path, fname)
                ep_idx = 0
                base = fname.replace(".h5", "")
                try:
                    parts = base.split("_")
                    if len(parts) >= 2:
                        ep_idx = int(parts[-1])
                except (ValueError, IndexError):
                    pass
                convert_episode(fpath, output_dir, episode_index=ep_idx)
                out_file = os.path.join(output_dir, "data", f"episode_{ep_idx:06d}.h5")
                with h5py.File(out_file, "r") as f:
                    num_steps = len(f["action/joint_position"]) if "action/joint_position" in f else 0
                episodes_meta.append((ep_idx, num_steps))
    else:
        convert_episode(input_path, output_dir, episode_index=0)
        out_file = os.path.join(output_dir, "data", "episode_000000.h5")
        with h5py.File(out_file, "r") as f:
            num_steps = len(f["action/joint_position"]) if "action/joint_position" in f else 0
        episodes_meta.append((0, num_steps))

    write_episodes_metadata(output_dir, episodes_meta)
    if args.features:
        write_features_json(output_dir)

    print(f"Converted {len(episodes_meta)} episode(s) to {output_dir}")


if __name__ == "__main__":
    main()
