"""Intel RealSense D435i RGBD capture."""

import threading
import time

import numpy as np


class RealsenseCamera:
    """Captures aligned color and depth frames from a D435i."""

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        streams: str = "rgbd",
        device_serial: str | None = None,
    ):
        """
        Args:
            streams: "rgb" for color only, "depth" for depth only,
                     "rgbd" for both (default).
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.streams = streams
        self.device_serial = device_serial

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
        devs = self.list_devices()
        print(f"[RealsenseCamera] Found {len(devs)} device(s)")
        if len(devs) == 0:
            raise RuntimeError("No RealSense device found")

        selected = None
        if self.device_serial is not None:
            selected = next((dev for dev in devs if dev["serial"] == self.device_serial), None)
            if selected is None:
                raise RuntimeError(f"Requested RealSense serial not found: {self.device_serial}")
        else:
            selected = devs[0]

        print(f"  - {selected['name']} ({selected['serial']}), USB: {selected['usb_type']}")

        need_color = self.streams in ("rgb", "rgbd")
        need_depth = self.streams in ("depth", "rgbd")

        self._pipeline = rs.pipeline()
        config = rs.config()
        if selected["serial"]:
            config.enable_device(selected["serial"])
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
        print(
            f"[RealsenseCamera] Started ({self.width}x{self.height} @ {self.fps}fps, "
            f"streams={self.streams}, serial={selected['serial']})"
        )

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

    def capture_sync(self, arm_state_provider) -> tuple[np.ndarray, np.ndarray, float, object]:
        """Return the latest RealSense frame and an arm sample taken immediately after."""
        color, depth, timestamp = self.get_frames()
        arm_state = arm_state_provider()
        return color, depth, timestamp, arm_state

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

    @staticmethod
    def list_devices() -> list[dict[str, str]]:
        """Return connected RealSense devices."""
        try:
            import pyrealsense2 as rs
        except ImportError:
            return []

        ctx = rs.context()
        devices = []
        for dev in ctx.query_devices():
            devices.append(
                {
                    "name": dev.get_info(rs.camera_info.name),
                    "serial": dev.get_info(rs.camera_info.serial_number),
                    "usb_type": dev.get_info(rs.camera_info.usb_type_descriptor),
                }
            )
        return devices
