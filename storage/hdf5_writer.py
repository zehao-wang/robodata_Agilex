"""Episode writer for folder-based recordings."""

import json
from pathlib import Path

import cv2
import numpy as np


class HDF5Writer:
    """Buffers frames in memory and writes each episode as a folder."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._world_config = None
        self.reset()

    def reset(self):
        """Clear buffered video frames and per-frame metadata."""
        self._frames = []
        self._frame_records = []

    def set_world_config(self, world_config: dict | None) -> None:
        """Set world frame config to be written with each episode."""
        self._world_config = world_config

    def add_frame(
        self,
        qpos: np.ndarray,
        qvel: np.ndarray,
        gripper: float,
        color: np.ndarray,
        depth: np.ndarray,
        timestamp: float,
        arm_timestamp: float | None = None,
        camera_timestamp: float | None = None,
        sync_delta_ms: float | None = None,
        sync_ok: bool | None = None,
        action_qpos: np.ndarray | None = None,
        action_gripper: float | None = None,
        eef_pos: np.ndarray | None = None,
    ):
        """Buffer a single frame of RGB video and aligned arm metadata."""
        frame_index = len(self._frame_records)
        self._frames.append(color.copy())

        record = {
            "frame_index": frame_index,
            "timestamp": float(timestamp),
            "record_timestamp": float(timestamp),
            "qpos": np.asarray(qpos, dtype=np.float64).tolist(),
            "qvel": np.asarray(qvel, dtype=np.float64).tolist(),
            "gripper": float(gripper),
            "depth_available": bool(np.max(depth) > 0),
        }
        if arm_timestamp is not None:
            record["arm_timestamp"] = float(arm_timestamp)
        if camera_timestamp is not None:
            record["camera_timestamp"] = float(camera_timestamp)
        if sync_delta_ms is not None:
            record["sync_delta_ms"] = float(sync_delta_ms)
        if sync_ok is not None:
            record["sync_ok"] = bool(sync_ok)
        if action_qpos is not None:
            record["action_qpos"] = np.asarray(action_qpos, dtype=np.float64).tolist()
        if action_gripper is not None:
            record["action_gripper"] = float(action_gripper)
        if eef_pos is not None:
            record["eef_pos"] = np.asarray(eef_pos, dtype=np.float64).tolist()
        self._frame_records.append(record)

    @property
    def num_frames(self) -> int:
        return len(self._frame_records)

    def save(self, task_name: str = "", instruction: str = "") -> str:
        """Write buffered data to an episode directory."""
        if self.num_frames == 0:
            raise ValueError("No frames to save")

        episode_idx = self._next_episode_index()
        episode_dir = self.output_dir / f"episode_{episode_idx:04d}"
        episode_dir.mkdir(parents=True, exist_ok=False)

        video_path = episode_dir / "camera.mp4"
        metadata_path = episode_dir / "metadata.json"

        self._write_video(video_path)

        timestamps = [record["timestamp"] for record in self._frame_records]
        duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0.0
        sync_deltas = [
            record["sync_delta_ms"]
            for record in self._frame_records
            if "sync_delta_ms" in record
        ]
        sync_ok_count = sum(
            1 for record in self._frame_records if record.get("sync_ok") is True
        )
        metadata = {
            "format": "robodata_episode_v1",
            "task_name": task_name,
            "instruction": instruction,
            "num_frames": self.num_frames,
            "duration_s": float(duration),
            "video_file": video_path.name,
            "world_frame_calibrated": self._world_config is not None,
            "world_config": self._to_jsonable(self._world_config),
            "sync_summary": {
                "frames_with_sync_check": len(sync_deltas),
                "sync_ok_frames": sync_ok_count,
                "sync_ok_ratio": (
                    float(sync_ok_count / len(sync_deltas)) if sync_deltas else None
                ),
                "max_sync_delta_ms": max(sync_deltas) if sync_deltas else None,
                "mean_sync_delta_ms": (
                    float(np.mean(sync_deltas)) if sync_deltas else None
                ),
            },
            "frames": self._frame_records,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        n = self.num_frames
        self.reset()
        print(f"[EpisodeWriter] Saved {n} frames -> {episode_dir}")
        return str(episode_dir)

    def _write_video(self, video_path: Path) -> None:
        first_frame = self._frames[0]
        height, width = first_frame.shape[:2]
        fps = self._infer_fps()
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {video_path}")

        try:
            for frame in self._frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    def _infer_fps(self) -> float:
        if self.num_frames < 2:
            return 30.0
        timestamps = np.asarray(
            [
                record.get("camera_timestamp", record.get("timestamp"))
                for record in self._frame_records
            ],
            dtype=np.float64,
        )
        dt = np.diff(timestamps)
        dt = dt[dt > 1e-6]
        if dt.size == 0:
            return 30.0
        return float(np.clip(1.0 / np.mean(dt), 1.0, 240.0))

    def _next_episode_index(self) -> int:
        """Find the next available episode index."""
        existing = sorted(
            path for path in self.output_dir.glob("episode_*") if path.is_dir()
        )
        if not existing:
            return 0
        last = existing[-1].name
        return int(last.split("_")[1]) + 1

    def _to_jsonable(self, value):
        """Recursively convert numpy values to JSON-serializable Python types."""
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: self._to_jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]
        return value
