"""Stereolabs ZED 2 RGBD capture via a subprocess bridge.

The bridge (camera/zed_bridge.py) runs under a separate Python interpreter
that has pyzed + numpy<2.0, keeping the main process free to use numpy 2.x.

Configure the bridge interpreter via the environment variable::

    export ZED_BRIDGE_PYTHON=/home/zwa0839/miniconda3/envs/zed_bridge/bin/python3.10

Or pass ``bridge_python`` directly to ZED2CameraConfig().

ZED2i supported resolutions (width x height @ fps):
  2208x1242 @ 15
  1920x1080 @ 15, 30
  1280x720  @ 15, 30, 60
  672x376   @ 15, 30, 100
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from multiprocessing.shared_memory import SharedMemory
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ZED 2 maximum native resolution — shared memory is allocated at this size.
_MAX_W, _MAX_H = 2208, 1242
# Meta layout: [width:i4 4B][height:i4 4B][timestamp:f8 8B][status:u1 1B] = 17B
_META_BYTES = 17
_BRIDGE_SCRIPT = os.path.join(os.path.dirname(__file__), "zed_bridge.py")

logger = logging.getLogger(__name__)


@dataclass
class ZED2CameraConfig:
    """Configuration for the ZED 2 camera subprocess bridge.

    Example configurations:
    ```python
    ZED2CameraConfig(fps=30, width=1280, height=720)               # HD720 color only
    ZED2CameraConfig(fps=30, width=1920, height=1080, use_depth=True)  # HD1080 with depth
    ```

    Attributes:
        fps:           Capture frame rate.
        width:         Frame width. Must match a supported ZED resolution.
        height:        Frame height. Must match a supported ZED resolution.
        use_depth:     Enable depth stream. Defaults to False.
        bridge_python: Path to Python interpreter with pyzed + numpy<2.0.
                       Defaults to ZED_BRIDGE_PYTHON env var, then sys.executable.
    """

    fps: int = 30
    width: int = 1280
    height: int = 720
    use_depth: bool = False
    bridge_python: str = field(
        default_factory=lambda: os.environ.get("ZED_BRIDGE_PYTHON", sys.executable)
    )


class ZED2Camera:
    """Captures color and depth frames from a ZED 2 camera.

    Spawns ``zed_bridge.py`` as a subprocess so pyzed (numpy<2.0) runs
    in isolation. Frames are exchanged via shared memory and surfaced
    through a background polling thread — matching the RealSenseCamera interface.

    Example:
    ```python
    config = ZED2CameraConfig(fps=30, width=1280, height=720, use_depth=True)
    camera = ZED2Camera(config)
    camera.connect()

    color = camera.read()         # blocking, returns latest color frame
    depth = camera.read_depth()   # blocking, returns latest depth frame

    camera.disconnect()
    ```
    """

    def __init__(self, config: ZED2CameraConfig):
        self.config = config
        self.fps = config.fps
        self.use_depth = config.use_depth
        self._bridge_python = config.bridge_python
        self._streams = "rgbd" if config.use_depth else "rgb"

        self._shm_color: SharedMemory | None = None
        self._shm_depth: SharedMemory | None = None
        self._shm_meta: SharedMemory | None = None
        self._proc: subprocess.Popen | None = None

        self._thread: Thread | None = None
        self._stop_event: Event | None = None
        self._frame_lock: Lock = Lock()
        self._new_frame_event: Event = Event()
        self._latest_color_frame: NDArray[Any] | None = None
        self._latest_depth_frame: NDArray[Any] | None = None
        self._latest_timestamp: float | None = None

        self._color_arr: np.ndarray | None = None
        self._depth_arr: np.ndarray | None = None
        self.width: int = 0
        self.height: int = 0
        self._params_file: str | None = None

    def __str__(self) -> str:
        return f"ZED2Camera({self.config.width}x{self.config.height}@{self.fps}fps)"

    @property
    def is_connected(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def connect(self, warmup: bool = True, timeout: float = 30.0) -> None:
        """Spawn the bridge process, wait for camera to open, start poll thread.

        Args:
            warmup:  If True, waits until at least one frame has been captured.
            timeout: Seconds to wait for the bridge to open the camera.

        Raises:
            RuntimeError: If the bridge fails to open the camera.
            TimeoutError: If the bridge does not open the camera within timeout.
        """
        need_depth = self._streams in ("depth", "rgbd")

        self._shm_color = SharedMemory(create=True, size=_MAX_H * _MAX_W * 3)
        self._shm_depth = SharedMemory(create=True, size=_MAX_H * _MAX_W * 2) if need_depth else None
        self._shm_meta = SharedMemory(create=True, size=_META_BYTES)
        self._shm_meta.buf[16] = 0  # status = starting

        self._params_file = tempfile.mktemp(suffix="_zed2_params.json")

        cmd = [
            self._bridge_python, _BRIDGE_SCRIPT,
            "--shm-color",   self._shm_color.name,
            "--shm-meta",    self._shm_meta.name,
            "--fps",         str(self.config.fps),
            "--width",       str(self.config.width),
            "--height",      str(self.config.height),
            "--streams",     self._streams,
            "--params-file", self._params_file,
        ]
        if need_depth:
            cmd += ["--shm-depth", self._shm_depth.name]

        self._proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)

        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self._shm_meta.buf[16]
            if status == 255:
                err = self._proc.stderr.read().decode(errors="replace")
                self.disconnect()
                raise RuntimeError(
                    f"ZED bridge failed to open camera.\n"
                    f"Bridge Python: {self._bridge_python}\n"
                    f"Bridge output:\n{err}\n\n"
                    f"Hint: set ZED_BRIDGE_PYTHON to a Python with pyzed + numpy<2.0"
                )
            if status >= 1:
                break
            time.sleep(0.1)
        else:
            self.disconnect()
            raise TimeoutError(f"ZED bridge did not open camera within {timeout}s")

        self.width  = int(np.frombuffer(self._shm_meta.buf[0:4], dtype=np.int32)[0])
        self.height = int(np.frombuffer(self._shm_meta.buf[4:8], dtype=np.int32)[0])

        if self._streams in ("rgb", "rgbd"):
            self._color_arr = np.ndarray(
                (self.height, self.width, 3), dtype=np.uint8, buffer=self._shm_color.buf
            )
        if need_depth:
            self._depth_arr = np.ndarray(
                (self.height, self.width), dtype=np.uint16, buffer=self._shm_depth.buf
            )

        self._start_poll_thread()

        if warmup:
            self._new_frame_event.wait(timeout=5.0)
            with self._frame_lock:
                if self._latest_color_frame is None:
                    raise RuntimeError(f"{self} failed to capture frames during warmup.")

        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        """Stop the poll thread, terminate the bridge, and release shared memory."""
        self._stop_poll_thread()

        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None

        for shm in (self._shm_color, self._shm_depth, self._shm_meta):
            if shm is not None:
                try:
                    shm.close()
                    shm.unlink()
                except Exception:
                    pass
        self._shm_color = self._shm_depth = self._shm_meta = None

        if self._params_file and os.path.exists(self._params_file):
            try:
                os.unlink(self._params_file)
            except Exception:
                pass
        self._params_file = None

        logger.info(f"{self} disconnected.")

    def get_camera_params(self) -> dict:
        """Return camera intrinsics/calibration written by the bridge on startup.

        Returns an empty dict if the params file is not available.
        """
        if self._params_file and os.path.exists(self._params_file):
            with open(self._params_file) as f:
                return json.load(f)
        return {}

    def read(self) -> NDArray[Any]:
        """Return the next new color frame (blocking).

        Returns:
            np.ndarray: (H, W, 3) uint8 RGB
        """
        return self.async_read(timeout_ms=2000)

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Wait for a new color frame and return it.

        Args:
            timeout_ms: Maximum time to wait in milliseconds.

        Returns:
            np.ndarray: (H, W, 3) uint8 RGB

        Raises:
            TimeoutError: If no new frame arrives within timeout_ms.
        """
        if not self._new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"{self} timed out waiting for frame after {timeout_ms}ms."
            )
        with self._frame_lock:
            frame = self._latest_color_frame
            self._new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"{self}: event set but no frame available.")
        return frame

    def read_depth(self, timeout_ms: float = 2000) -> NDArray[Any]:
        """Wait for a new depth frame and return it.

        Returns:
            np.ndarray: (H, W) uint16 millimetres (0 = invalid)

        Raises:
            RuntimeError: If depth is not enabled in config.
            TimeoutError: If no new frame arrives within timeout_ms.
        """
        if not self.use_depth:
            raise RuntimeError(
                f"{self}: depth is not enabled. Set use_depth=True in ZED2CameraConfig."
            )
        if not self._new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"{self} timed out waiting for depth frame after {timeout_ms}ms."
            )
        with self._frame_lock:
            depth = self._latest_depth_frame
            self._new_frame_event.clear()

        if depth is None:
            raise RuntimeError(f"{self}: event set but no depth frame available.")
        return depth

    def _poll_loop(self) -> None:
        """Background thread: copy frames from shared memory when timestamp changes."""
        last_ts = None
        while not self._stop_event.is_set():
            status = self._shm_meta.buf[16]
            if status != 2:
                time.sleep(0.001)
                continue

            ts = float(np.frombuffer(self._shm_meta.buf[8:16], dtype=np.float64)[0])
            if ts == last_ts:
                time.sleep(0.001)
                continue

            with self._frame_lock:
                if self._color_arr is not None:
                    self._latest_color_frame = self._color_arr.copy()
                if self._depth_arr is not None:
                    self._latest_depth_frame = self._depth_arr.copy()
                self._latest_timestamp = ts

            last_ts = ts
            self._new_frame_event.set()

    def _start_poll_thread(self) -> None:
        self._stop_poll_thread()
        self._stop_event = Event()
        self._thread = Thread(target=self._poll_loop, name=f"{self}_poll", daemon=True)
        self._thread.start()

    def _stop_poll_thread(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None
        self._stop_event = None
        with self._frame_lock:
            self._latest_color_frame = None
            self._latest_depth_frame = None
            self._latest_timestamp = None
            self._new_frame_event.clear()
