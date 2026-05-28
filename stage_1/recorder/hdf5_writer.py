import os
import threading
import h5py
import numpy as np


class HDF5Writer:
    """Thread-safe HDF5 writer for robot teleoperation data.

    Stores episodes under /episode_XXX groups with resizable datasets.
    All I/O is serialized through a lock so concurrent calls are safe.
    """

    def __init__(self, filepath: str, compression: str = "gzip", compression_level: int = 4):
        self._filepath = filepath
        self._lock = threading.Lock()
        self._file = None
        self._episode_group = None
        self._datasets = {}
        self._step_count = 0

        os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
        self._file = h5py.File(filepath, "w")
        self._file.attrs["format"] = "umi_stage1"
        self._compression = compression
        self._compression_opts = compression_level

    # ---- episode lifecycle ----
    def start_episode(self, episode_index: int, metadata: dict = None):
        with self._lock:
            name = f"episode_{episode_index:06d}"
            self._episode_group = self._file.create_group(name)
            self._step_count = 0
            self._datasets.clear()
            if metadata:
                for k, v in metadata.items():
                    self._episode_group.attrs[k] = v

    def end_episode(self):
        with self._lock:
            if self._episode_group is not None:
                self._episode_group.attrs["num_steps"] = self._step_count
                self._episode_group = None
                self._datasets.clear()
                self._step_count = 0
            self._file.flush()

    # ---- step writing ----
    def write_step(self, data: dict):
        """Write one time-aligned step.

        data keys are group/dataset paths like 'joint_state/position'.
        Each value is a 1-D numpy array (will be appended as a row).
        """
        with self._lock:
            for key, value in data.items():
                if "/" not in key:
                    continue
                group_path, dset_name = key.rsplit("/", 1)
                full_path = f"{group_path}/{dset_name}"

                if full_path not in self._datasets:
                    group = self._ensure_group(group_path)
                    val = np.atleast_1d(np.asarray(value))
                    self._datasets[full_path] = group.create_dataset(
                        dset_name,
                        data=val[np.newaxis, :],
                        maxshape=(None,) + val.shape,
                        chunks=(1,) + val.shape,
                        compression=self._compression,
                        compression_opts=self._compression_opts,
                    )
                else:
                    dset = self._datasets[full_path]
                    val = np.atleast_1d(np.asarray(value))
                    dset.resize(dset.shape[0] + 1, axis=0)
                    dset[-1] = val
            self._step_count += 1

    # ---- helpers ----
    def _ensure_group(self, group_path: str):
        parts = group_path.split("/")
        g = self._episode_group
        for p in parts:
            if p not in g:
                g = g.create_group(p)
            else:
                g = g[p]
        return g

    def close(self):
        with self._lock:
            if self._episode_group is not None:
                self.end_episode()
            if self._file is not None:
                self._file.close()
                self._file = None

    @property
    def step_count(self) -> int:
        return self._step_count
