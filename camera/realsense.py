"""Intel RealSense D435i RGBD capture."""

import threading
import time

import numpy as np


class RealsenseCamera:
    """Captures aligned color and depth frames from a D435i."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30,
                 streams: str = "rgbd"):
        """
        Args:
            streams: "rgb" for color only, "depth" for depth only,
                     "rgbd" for both (default).
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.streams = streams

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
        self._rs = rs

        # Verify device is reachable
        ctx = rs.context()
        devs = ctx.query_devices()
        print(f"[RealsenseCamera] Found {len(devs)} device(s)")
        if len(devs) == 0:
            raise RuntimeError("No RealSense device found")
        usb_type = devs[0].get_info(rs.camera_info.usb_type_descriptor)
        print(f"  - {devs[0].get_info(rs.camera_info.name)}, USB: {usb_type}")

        need_color = self.streams in ("rgb", "rgbd")
        need_depth = self.streams in ("depth", "rgbd")

        self._pipeline = rs.pipeline()
        config = rs.config()
        if need_color:
            config.enable_stream(rs.stream.color, self.width, self.height,
                                 rs.format.rgb8, self.fps)
        if need_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height,
                                 rs.format.z16, self.fps)
        self._pipeline.start(config)

        if need_color and need_depth:
            self._align = rs.align(rs.stream.color)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        print(f"[RealsenseCamera] Started ({self.width}x{self.height} @ {self.fps}fps, streams={self.streams})")

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
        """Background loop: wait for frames and store them."""
        need_color = self.streams in ("rgb", "rgbd")
        need_depth = self.streams in ("depth", "rgbd")
        use_align = need_color and need_depth and self._align is not None
        frame_count = 0
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=5000)
            except Exception as e:
                print(f"[RealsenseCamera] wait_for_frames error: {e}")
                continue

            if use_align:
                frames = self._align.process(frames)

            color_arr = None
            depth_arr = None
            if need_color:
                cf = frames.get_color_frame()
                if cf:
                    color_arr = np.asanyarray(cf.get_data())
            if need_depth:
                df = frames.get_depth_frame()
                if df:
                    depth_arr = np.asanyarray(df.get_data())

            with self._lock:
                if color_arr is not None:
                    self._color_frame = color_arr
                if depth_arr is not None:
                    self._depth_frame = depth_arr
                self._timestamp = time.time()
            frame_count += 1
            if frame_count % 100 == 1:
                print(f"[RealsenseCamera] frame {frame_count}"
                      + (f", color {color_arr.shape} mean={color_arr.mean():.1f}" if color_arr is not None else "")
                      + (f", depth {depth_arr.shape}" if depth_arr is not None else ""))
