"""Generic RGB camera capture via OpenCV."""

import time

import cv2
import numpy as np


class OpenCVCamera:
    """Captures RGB frames from a generic system camera."""

    def __init__(
        self,
        camera_index: int,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        streams: str = "rgb",
        zed_mode: bool = False,
    ):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.streams = streams
        self.zed_mode = zed_mode

        self._capture = None
        self._color_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._depth_frame = np.zeros((height, width), dtype=np.uint16)
        self._timestamp = 0.0

    def start(self):
        """Open the camera device."""
        capture = cv2.VideoCapture(self.camera_index)
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open camera index {self.camera_index}")

        requested_width = 3840 if self.zed_mode else self.width
        requested_height = 1080 if self.zed_mode else self.height
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, requested_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, requested_height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)

        ok, frame, timestamp = self._grab_and_retrieve(capture)
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"Camera index {self.camera_index} did not return frames")

        self._validate_raw_resolution(frame)
        frame = self._process_frame(frame)
        self.height, self.width = frame.shape[:2]
        self._color_frame = frame
        self._depth_frame = np.zeros((self.height, self.width), dtype=np.uint16)
        self._timestamp = timestamp
        self._capture = capture
        print(
            f"[OpenCVCamera] Started index={self.camera_index} "
            f"({self.width}x{self.height} @ {self.fps}fps, zed_mode={self.zed_mode})"
        )

    def stop(self):
        """Release the camera device."""
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        print(f"[OpenCVCamera] Stopped index={self.camera_index}")

    def get_frames(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Get the last cached frame."""
        return self._color_frame.copy(), self._depth_frame.copy(), self._timestamp

    def capture_sync(self, arm_state_provider) -> tuple[np.ndarray, np.ndarray, float, object]:
        """Grab a frame, sample arm state, then retrieve the image."""
        if self._capture is None:
            raise RuntimeError("Camera not started")

        grabbed = self._capture.grab()
        camera_timestamp = time.time()
        arm_state = arm_state_provider()
        if not grabbed:
            return self.get_frames()[0], self.get_frames()[1], self._timestamp, arm_state

        ok, frame = self._capture.retrieve()
        if not ok or frame is None:
            return self.get_frames()[0], self.get_frames()[1], self._timestamp, arm_state

        self._validate_raw_resolution(frame)
        frame = self._process_frame(frame)
        self._color_frame = frame
        if self._depth_frame.shape != frame.shape[:2]:
            self._depth_frame = np.zeros(frame.shape[:2], dtype=np.uint16)
        self._timestamp = camera_timestamp
        return self.get_frames()[0], self.get_frames()[1], self._timestamp, arm_state

    def _grab_and_retrieve(self, capture) -> tuple[bool, np.ndarray | None, float]:
        grabbed = capture.grab()
        timestamp = time.time()
        if not grabbed:
            return False, None, timestamp
        ok, frame = capture.retrieve()
        return ok, frame, timestamp

    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.zed_mode:
            frame = frame[:1080, :1920]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def _validate_raw_resolution(self, frame: np.ndarray) -> None:
        if not self.zed_mode:
            return

        raw_h, raw_w = frame.shape[:2]
        if raw_w < 3840 or raw_h < 1080:
            raise RuntimeError(
                "ZED Mode requires raw camera output at least 3840x1080, "
                f"but got {raw_w}x{raw_h}"
            )
