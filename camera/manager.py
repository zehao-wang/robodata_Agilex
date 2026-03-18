"""Camera discovery and runtime switching helpers."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from camera.opencv_camera import OpenCVCamera
from camera.realsense import RealsenseCamera
from camera.zed_camera import ZedCamera


@dataclass(frozen=True)
class CameraSource:
    source_id: str
    label: str
    backend: str
    has_depth: bool = False
    serial: str | None = None
    camera_index: int | None = None


def discover_camera_sources(
    max_opencv_index: int = 10,
    preferred_backend: str = "auto",
) -> list[CameraSource]:
    """Probe connected cameras and return selectable sources."""
    sources: list[CameraSource] = []

    if preferred_backend in ("auto", "zed"):
        for dev in ZedCamera.list_devices():
            label = f"{dev['name']} (HD1080 left @ {dev['fps']}fps)"
            sources.append(
                CameraSource(
                    source_id="zed:left",
                    label=label,
                    backend="zed",
                    has_depth=False,
                    serial=dev.get("serial"),
                )
            )

    if preferred_backend in ("auto", "realsense"):
        for dev in RealsenseCamera.list_devices():
            serial = dev["serial"]
            usb_type = dev.get("usb_type", "unknown")
            label = f"RealSense {dev['name']} ({serial}, USB {usb_type})"
            sources.append(
                CameraSource(
                    source_id=f"realsense:{serial}",
                    label=label,
                    backend="realsense",
                    has_depth=True,
                    serial=serial,
                )
            )

    if preferred_backend in ("auto", "opencv"):
        for camera_index in range(max_opencv_index):
            capture = cv2.VideoCapture(camera_index)
            if not capture.isOpened():
                capture.release()
                continue
            ok, _ = capture.read()
            capture.release()
            if not ok:
                continue
            sources.append(
                CameraSource(
                    source_id=f"opencv:{camera_index}",
                    label=f"System Camera {camera_index}",
                    backend="opencv",
                    has_depth=False,
                    camera_index=camera_index,
                )
            )

    return sources


class CameraManager:
    """Maintains the currently active camera and switches sources on demand."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        streams: str = "rgb",
        max_opencv_index: int = 10,
        preferred_backend: str = "auto",
    ):
        self._width = width
        self._height = height
        self._fps = fps
        self._streams = streams
        self._max_opencv_index = max_opencv_index
        self._preferred_backend = preferred_backend
        self._opencv_zed_mode = True

        self._sources = discover_camera_sources(
            max_opencv_index=max_opencv_index,
            preferred_backend=preferred_backend,
        )
        self._active_camera = None
        self._active_source_id: str | None = None
        self._last_error: str | None = None

    @property
    def sources(self) -> list[CameraSource]:
        return list(self._sources)

    @property
    def active_source_id(self) -> str | None:
        return self._active_source_id

    @property
    def last_error(self) -> str | None:
        return self._last_error

    def source_labels(self) -> list[str]:
        return [source.label for source in self._sources]

    def source_id_to_label(self) -> dict[str, str]:
        return {source.source_id: source.label for source in self._sources}

    @property
    def opencv_zed_mode(self) -> bool:
        return self._opencv_zed_mode

    def start_first_available(self) -> str | None:
        """Start the first camera that can be opened successfully."""
        for source in self._sources:
            if self.select_camera(source.source_id):
                return source.source_id
        return None

    def select_camera(self, source_id: str | None) -> bool:
        """Switch the active camera."""
        if source_id is None:
            self.stop()
            self._last_error = None
            return True

        source = next((s for s in self._sources if s.source_id == source_id), None)
        if source is None:
            self._last_error = f"Unknown camera source: {source_id}"
            return False

        if self._active_source_id == source_id and self._active_camera is not None:
            self._last_error = None
            return True

        previous_camera = self._active_camera
        previous_source_id = self._active_source_id

        self._active_camera = None
        self._active_source_id = None
        if previous_camera is not None:
            previous_camera.stop()

        try:
            camera = self._build_camera(source)
            camera.start()
        except Exception as exc:
            self._last_error = f"{source.label}: {exc}"
            print(f"[CameraManager] Failed to start {source.label}: {exc}")
            if previous_camera is not None and previous_source_id is not None:
                try:
                    previous_camera.start()
                    self._active_camera = previous_camera
                    self._active_source_id = previous_source_id
                except Exception as restore_exc:
                    self._last_error += f" | Restore failed: {restore_exc}"
                    print(f"[CameraManager] Failed to restore previous camera: {restore_exc}")
            return False

        self._active_camera = camera
        self._active_source_id = source.source_id
        self._last_error = None
        print(f"[CameraManager] Active source -> {source.label}")
        return True

    def stop(self):
        if self._active_camera is not None:
            self._active_camera.stop()
            self._active_camera = None
        self._active_source_id = None

    def get_frames(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self._active_camera is None:
            color = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            depth = np.zeros((self._height, self._width), dtype=np.uint16)
            return color, depth, 0.0
        return self._active_camera.get_frames()

    def get_camera_info(self) -> dict | None:
        if self._active_camera is None or not hasattr(self._active_camera, "get_camera_info"):
            return None
        return self._active_camera.get_camera_info()

    def capture_sync(self, arm_state_provider):
        if self._active_camera is None:
            color = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            depth = np.zeros((self._height, self._width), dtype=np.uint16)
            arm_state = arm_state_provider()
            return color, depth, 0.0, arm_state
        if hasattr(self._active_camera, "capture_sync"):
            return self._active_camera.capture_sync(arm_state_provider)
        color, depth, timestamp = self._active_camera.get_frames()
        arm_state = arm_state_provider()
        return color, depth, timestamp, arm_state

    def set_opencv_zed_mode(self, enabled: bool) -> bool:
        enabled = bool(enabled)
        if self._opencv_zed_mode == enabled:
            return True
        self._opencv_zed_mode = enabled

        if (
            self._active_source_id is not None
            and self._active_source_id.startswith("opencv:")
        ):
            return self.select_camera(self._active_source_id)
        return True

    def _build_camera(self, source: CameraSource):
        if source.backend == "realsense":
            return RealsenseCamera(
                width=self._width,
                height=self._height,
                fps=self._fps,
                streams=self._streams,
                device_serial=source.serial,
            )
        if source.backend == "opencv":
            return OpenCVCamera(
                camera_index=int(source.camera_index),
                width=self._width,
                height=self._height,
                fps=self._fps,
                streams="rgb",
                zed_mode=self._opencv_zed_mode,
            )
        if source.backend == "zed":
            return ZedCamera(
                width=1920,
                height=1080,
                fps=30,
                streams="rgb",
            )
        raise ValueError(f"Unsupported camera backend: {source.backend}")
