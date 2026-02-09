"""HDF5 episode writer for imitation learning data.

Output format compatible with ACT / Diffusion Policy frameworks.
"""

import os
from pathlib import Path

import h5py
import numpy as np


class HDF5Writer:
    """Collects frames in memory then writes a single HDF5 episode file."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.reset()

    def reset(self):
        """Clear all buffered data."""
        self._qpos = []
        self._qvel = []
        self._gripper = []
        self._color = []
        self._depth = []
        self._timestamps = []
        self._action_qpos = []
        self._action_gripper = []

    def add_frame(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        gripper: float,
        color: np.ndarray,
        depth: np.ndarray,
        timestamp: float,
        action_qpos: np.ndarray | None = None,
        action_gripper: float | None = None,
    ):
        """Buffer a single frame of data.

        Args:
            qpos: (6,) joint positions in radians (observation).
            qvel: (6,) joint velocities.
            gripper: Gripper width in meters (observation).
            color: (H, W, 3) uint8 RGB image.
            depth: (H, W) uint16 depth in mm.
            timestamp: Unix timestamp.
            action_qpos: (6,) action joint positions. If None, uses qpos.
            action_gripper: Action gripper width. If None, uses gripper.
        """
        self._qpos.append(qpos.copy())
        self._qvel.append(qvel.copy())
        self._gripper.append(gripper)
        self._color.append(color.copy())
        self._depth.append(depth.copy())
        self._timestamps.append(timestamp)
        if action_qpos is not None:
            self._action_qpos.append(action_qpos.copy())
        if action_gripper is not None:
            self._action_gripper.append(action_gripper)

    @property
    def num_frames(self) -> int:
        return len(self._timestamps)

    def save(self, task_name: str = "", instruction: str = "") -> str:
        """Write buffered data to an HDF5 file.

        Args:
            task_name: Human-readable task name for this episode.
            instruction: Natural-language instruction for this episode.

        Returns:
            Path to the saved file.
        """
        if self.num_frames == 0:
            raise ValueError("No frames to save")

        episode_idx = self._next_episode_index()
        filename = self.output_dir / f"episode_{episode_idx:04d}.hdf5"

        qpos = np.array(self._qpos, dtype=np.float64)       # (N, 6)
        qvel = np.array(self._qvel, dtype=np.float64)       # (N, 6)
        gripper = np.array(self._gripper, dtype=np.float64).reshape(-1, 1)  # (N, 1)
        color = np.array(self._color, dtype=np.uint8)        # (N, H, W, 3)
        depth = np.array(self._depth, dtype=np.uint16)       # (N, H, W)
        timestamps = np.array(self._timestamps, dtype=np.float64)  # (N,)

        with h5py.File(filename, "w") as f:
            # observations
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=qpos)
            obs.create_dataset("qvel", data=qvel)
            obs.create_dataset("gripper", data=gripper)

            imgs = obs.create_group("images")
            imgs.create_dataset("color", data=color, compression="gzip",
                                compression_opts=4)
            imgs.create_dataset("depth", data=depth, compression="gzip",
                                compression_opts=4)

            # action — separate if master-slave, otherwise same as observation
            act = f.create_group("action")
            if self._action_qpos:
                act_qpos = np.array(self._action_qpos, dtype=np.float64)
                act.create_dataset("qpos", data=act_qpos)
            else:
                act.create_dataset("qpos", data=qpos)
            if self._action_gripper:
                act_grip = np.array(self._action_gripper, dtype=np.float64).reshape(-1, 1)
                act.create_dataset("gripper", data=act_grip)
            else:
                act.create_dataset("gripper", data=gripper)

            # timestamps
            f.create_dataset("timestamps", data=timestamps)

            # metadata
            f.attrs["num_frames"] = len(timestamps)
            f.attrs["fps"] = 30
            duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
            f.attrs["duration_s"] = duration
            f.attrs["task_name"] = task_name
            f.attrs["instruction"] = instruction

        n = self.num_frames
        self.reset()
        print(f"[HDF5Writer] Saved {n} frames -> {filename}")
        return str(filename)

    def _next_episode_index(self) -> int:
        """Find the next available episode index."""
        existing = sorted(self.output_dir.glob("episode_*.hdf5"))
        if not existing:
            return 0
        last = existing[-1].stem  # e.g. "episode_0003"
        return int(last.split("_")[1]) + 1
