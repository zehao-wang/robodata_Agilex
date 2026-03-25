"""Stereolabs ZED 2 RGBD capture via a subprocess bridge.

The bridge (camera/zed_bridge.py) runs under a separate Python interpreter
that has pyzed + numpy<2.0, keeping the main process free to use numpy 2.x.

Configure the bridge interpreter via the environment variable::

    export ZED_BRIDGE_PYTHON=/path/to/python   # e.g. ~/zed_venv/bin/python

Or pass ``bridge_python`` directly to ZED2Camera().
"""
import os
import subprocess
import sys
import threading
import time
from multiprocessing.shared_memory import SharedMemory

import numpy as np

# ZED 2 maximum native resolution — shared memory is allocated at this size.
_MAX_W, _MAX_H = 2208, 1242
# Meta layout: [width:i4 4B][height:i4 4B][timestamp:f8 8B][status:u1 1B] = 17B
_META_BYTES = 17
_BRIDGE_SCRIPT = os.path.join(os.path.dirname(__file__), "zed_bridge.py")


def _default_bridge_python() -> str:
    return os.environ.get("ZED_BRIDGE_PYTHON", sys.executable)


class ZED2Camera:
    """Captures color and depth frames from a ZED 2 camera.

    Spawns ``zed_bridge.py`` as a subprocess so pyzed (numpy<2.0) runs
    in isolation.  Frames are exchanged via shared memory — zero copy.

    Args:
        fps:           Capture frame rate.
        streams:       "rgb", "depth", or "rgbd".
        bridge_python: Path to the Python interpreter that has pyzed + numpy<2.0.
                       Defaults to the ``ZED_BRIDGE_PYTHON`` env var, then
                       the current interpreter (which will fail if numpy>=2.0).
    """

    def __init__(self, width: int = 1280, height: int = 720,
                 fps: int = 30, streams: str = "rgbd",
                 bridge_python: str | None = None):
        self.fps = fps
        self.streams = streams
        self._bridge_python = bridge_python or _default_bridge_python()

        self._shm_color: SharedMemory | None = None
        self._shm_depth: SharedMemory | None = None
        self._shm_meta:  SharedMemory | None = None
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()

        self.width  = 0
        self.height = 0
        self._color_arr: np.ndarray | None = None
        self._depth_arr: np.ndarray | None = None

    def start(self, timeout: float = 30.0):
        """Allocate shared memory, spawn bridge, wait for camera to open."""
        self._shm_color = SharedMemory(create=True, size=_MAX_H * _MAX_W * 3)
        self._shm_depth = SharedMemory(create=True, size=_MAX_H * _MAX_W * 2)
        self._shm_meta  = SharedMemory(create=True, size=_META_BYTES)
        self._shm_meta.buf[16] = 0  # status = starting

        cmd = [
            self._bridge_python, _BRIDGE_SCRIPT,
            "--shm-color", self._shm_color.name,
            "--shm-depth", self._shm_depth.name,
            "--shm-meta",  self._shm_meta.name,
            "--fps",       str(self.fps),
            "--streams",   self.streams,
        ]
        self._proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)

        # Wait for bridge to signal camera open (status 1) or error (255)
        deadline = time.time() + timeout
        while time.time() < deadline:
            status = self._shm_meta.buf[16]
            if status == 255:
                err = self._proc.stderr.read().decode(errors="replace")
                self.stop()
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
            self.stop()
            raise TimeoutError(f"ZED bridge did not open camera within {timeout}s")

        w = int(np.frombuffer(bytes(self._shm_meta.buf[0:4]), dtype=np.int32)[0])
        h = int(np.frombuffer(bytes(self._shm_meta.buf[4:8]), dtype=np.int32)[0])
        self.width  = w
        self.height = h

        self._color_arr = np.ndarray((h, w, 3), dtype=np.uint8,  buffer=self._shm_color.buf)
        self._depth_arr = np.ndarray((h, w),    dtype=np.uint16, buffer=self._shm_depth.buf)

        print(f"[ZED2Camera] Ready: {w}x{h} @ {self.fps}fps, streams={self.streams}")

    def stop(self):
        """Terminate bridge and release shared memory."""
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
        print("[ZED2Camera] Stopped")

    def get_frames(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return latest (color, depth, timestamp).

        color  — (H, W, 3) uint8 RGB
        depth  — (H, W)    uint16 millimetres (0 = invalid)
        timestamp — Unix time of the frame
        """
        with self._lock:
            ts = float(np.frombuffer(bytes(self._shm_meta.buf[8:16]), dtype=np.float64)[0])
            return self._color_arr.copy(), self._depth_arr.copy(), ts
