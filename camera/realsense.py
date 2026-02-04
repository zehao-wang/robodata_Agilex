"""Intel RealSense D435i RGBD capture."""

import threading
import time

import numpy as np


class RealsenseCamera:
    """Captures aligned color and depth frames from a D435i."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

        self._pipeline = None
        self._align = None
        self._lock = threading.Lock()
        self._color_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._depth_frame = np.zeros((height, width), dtype=np.uint16)
        self._timestamp = 0.0
        self._running = False
        self._thread = None

    def start(self):
        """Start the RealSense pipeline and background capture thread."""
        import pyrealsense2 as rs

        self._pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.width, self.height,
                             rs.format.rgb8, self.fps)
        config.enable_stream(rs.stream.depth, self.width, self.height,
                             rs.format.z16, self.fps)
        self._pipeline.start(config)

        # Align depth to color
        self._align = rs.align(rs.stream.color)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[RealsenseCamera] Started ({self.width}x{self.height} @ {self.fps}fps)")

    def stop(self):
        """Stop capture and release resources."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None
        print("[RealsenseCamera] Stopped")

    def get_frames(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Get latest color and depth frames.

        Returns:
            (color, depth, timestamp) where color is (H,W,3) uint8 RGB,
            depth is (H,W) uint16 in mm.
        """
        with self._lock:
            return self._color_frame.copy(), self._depth_frame.copy(), self._timestamp

    def _capture_loop(self):
        """Background loop: wait for aligned frames and store them."""
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
            except Exception:
                continue

            aligned = self._align.process(frames)
            color = aligned.get_color_frame()
            depth = aligned.get_depth_frame()
            if not color or not depth:
                continue

            color_arr = np.asanyarray(color.get_data())  # RGB uint8
            depth_arr = np.asanyarray(depth.get_data())  # uint16 mm

            with self._lock:
                self._color_frame = color_arr
                self._depth_frame = depth_arr
                self._timestamp = time.time()
