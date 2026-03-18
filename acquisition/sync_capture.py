"""Background synchronized capture process."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import asdict

import numpy as np

from camera.manager import CameraSource
from camera.opencv_camera import OpenCVCamera
from camera.realsense import RealsenseCamera
from camera.zed_camera import ZedCamera
from robot.arm_reader import ArmReader, ArmState


class SynchronizedCaptureClient:
    """Owns a child process that captures camera and arm samples."""

    def __init__(
        self,
        *,
        width: int,
        height: int,
        fps: int,
        streams: str,
        max_sync_dt_ms: float,
        no_arm: bool,
        no_camera: bool,
        can_interface: str,
        can_channel: str,
        bitrate: int,
        camera_sources: list[CameraSource],
        initial_source_id: str | None,
    ):
        self._width = width
        self._height = height
        self._fps = fps
        self._streams = streams
        self._max_sync_dt_ms = max_sync_dt_ms
        self._no_arm = no_arm
        self._no_camera = no_camera
        self._can_interface = can_interface
        self._can_channel = can_channel
        self._bitrate = bitrate
        self._camera_sources = camera_sources
        self._source_map = {source.source_id: source for source in camera_sources}
        self._initial_source_id = initial_source_id

        self._ctx = mp.get_context("spawn")
        self._sample_queue = None
        self._command_queue = None
        self._process = None
        self._latest_sample = None
        self._active_source_id = initial_source_id
        self._last_error = None

    @property
    def active_source_id(self) -> str | None:
        return self._active_source_id

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def start(self):
        self._sample_queue = self._ctx.Queue(maxsize=1)
        self._command_queue = self._ctx.Queue(maxsize=16)
        source_payloads = [asdict(source) for source in self._camera_sources]
        self._process = self._ctx.Process(
            target=_capture_worker,
            args=(
                self._sample_queue,
                self._command_queue,
                {
                    "width": self._width,
                    "height": self._height,
                    "fps": self._fps,
                    "streams": self._streams,
                    "max_sync_dt_ms": self._max_sync_dt_ms,
                    "no_arm": self._no_arm,
                    "no_camera": self._no_camera,
                    "can_interface": self._can_interface,
                    "can_channel": self._can_channel,
                    "bitrate": self._bitrate,
                    "camera_sources": source_payloads,
                    "initial_source_id": self._initial_source_id,
                },
            ),
            daemon=True,
        )
        self._process.start()

    def stop(self):
        if self._command_queue is not None:
            self._command_queue.put({"type": "stop"})
        if self._process is not None:
            self._process.join(timeout=3.0)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)
            self._process = None

    def select_camera(self, source_id: str | None) -> bool:
        if self._command_queue is None:
            return False
        self._command_queue.put({"type": "select_camera", "source_id": source_id})
        return True

    def poll_latest_sample(self):
        if self._sample_queue is None:
            return self._latest_sample
        while True:
            try:
                sample = self._sample_queue.get_nowait()
            except queue.Empty:
                break
            self._latest_sample = sample
            self._active_source_id = sample.get("active_source_id")
            self._last_error = sample.get("camera_error")
        return self._latest_sample


def _capture_worker(sample_queue, command_queue, config: dict):
    width = config["width"]
    height = config["height"]
    fps = config["fps"]
    streams = config["streams"]
    max_sync_dt_ms = config["max_sync_dt_ms"]
    no_arm = config["no_arm"]
    no_camera = config["no_camera"]
    source_map = {
        payload["source_id"]: CameraSource(**payload)
        for payload in config["camera_sources"]
    }

    arm_reader = None
    if not no_arm:
        arm_reader = ArmReader(
            can_interface=config["can_interface"],
            can_channel=config["can_channel"],
            bitrate=config["bitrate"],
        )
        arm_reader.start()

    active_source_id = None
    camera = None
    camera_error = None

    def get_arm_state() -> ArmState:
        if arm_reader is None:
            return ArmState()
        return arm_reader.get_state()

    def switch_camera(source_id: str | None):
        nonlocal camera, active_source_id, camera_error
        if camera is not None:
            camera.stop()
            camera = None
        active_source_id = None
        camera_error = None
        if source_id is None or no_camera:
            return
        source = source_map.get(source_id)
        if source is None:
            camera_error = f"Unknown camera source: {source_id}"
            return
        try:
            camera = _build_camera(source, width=width, height=height, fps=fps, streams=streams)
            camera.start()
            active_source_id = source_id
        except Exception as exc:
            camera_error = f"{source.label}: {exc}"
            camera = None

    switch_camera(config["initial_source_id"])

    target_dt = 1.0 / fps
    try:
        while True:
            loop_start = time.time()
            while True:
                try:
                    cmd = command_queue.get_nowait()
                except queue.Empty:
                    break
                if cmd["type"] == "stop":
                    return
                if cmd["type"] == "select_camera":
                    switch_camera(cmd.get("source_id"))

            arm_state = None
            color = np.zeros((height, width, 3), dtype=np.uint8)
            depth = np.zeros((height, width), dtype=np.uint16)
            camera_timestamp = 0.0

            if camera is not None and hasattr(camera, "capture_sync"):
                color, depth, camera_timestamp, arm_state = camera.capture_sync(get_arm_state)
            else:
                arm_state = get_arm_state()
                if camera is not None:
                    color, depth, camera_timestamp = camera.get_frames()

            if arm_state is None:
                arm_state = get_arm_state()

            sync_info = _evaluate_sync(
                arm_timestamp=arm_state.timestamp,
                camera_timestamp=camera_timestamp,
                max_sync_dt_ms=max_sync_dt_ms,
            )
            sample = {
                "color": color,
                "depth": depth,
                "arm_qpos": arm_state.qpos,
                "arm_qvel": arm_state.qvel,
                "arm_gripper": arm_state.gripper,
                "arm_timestamp": sync_info["arm_timestamp"],
                "camera_timestamp": sync_info["camera_timestamp"],
                "sync_delta_ms": sync_info["sync_delta_ms"],
                "sync_ok": sync_info["sync_ok"],
                "active_source_id": active_source_id,
                "camera_error": camera_error,
            }
            _replace_latest(sample_queue, sample)

            elapsed = time.time() - loop_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if camera is not None:
            camera.stop()
        if arm_reader is not None:
            arm_reader.stop()


def _build_camera(source: CameraSource, *, width: int, height: int, fps: int, streams: str):
    if source.backend == "realsense":
        return RealsenseCamera(
            width=width,
            height=height,
            fps=fps,
            streams=streams,
            device_serial=source.serial,
        )
    if source.backend == "opencv":
        return OpenCVCamera(
            camera_index=int(source.camera_index),
            width=width,
            height=height,
            fps=fps,
            streams="rgb",
        )
    if source.backend == "zed":
        return ZedCamera(
            width=1920,
            height=1080,
            fps=30,
            streams="rgb",
        )
    raise ValueError(f"Unsupported camera backend: {source.backend}")


def _replace_latest(sample_queue, sample):
    try:
        sample_queue.put_nowait(sample)
    except queue.Full:
        try:
            sample_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            sample_queue.put_nowait(sample)
        except queue.Full:
            # Main process has not caught up yet; keep running and drop this sample.
            pass


def _evaluate_sync(*, arm_timestamp: float, camera_timestamp: float, max_sync_dt_ms: float) -> dict:
    has_arm = arm_timestamp > 0.0
    has_camera = camera_timestamp > 0.0
    if not has_arm or not has_camera:
        return {
            "arm_timestamp": float(arm_timestamp) if has_arm else None,
            "camera_timestamp": float(camera_timestamp) if has_camera else None,
            "sync_delta_ms": None,
            "sync_ok": None,
        }

    sync_delta_ms = abs(camera_timestamp - arm_timestamp) * 1000.0
    return {
        "arm_timestamp": float(arm_timestamp),
        "camera_timestamp": float(camera_timestamp),
        "sync_delta_ms": float(sync_delta_ms),
        "sync_ok": sync_delta_ms <= max_sync_dt_ms,
    }
