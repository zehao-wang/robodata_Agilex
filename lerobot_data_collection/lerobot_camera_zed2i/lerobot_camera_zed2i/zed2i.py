"""LeRobot Camera wrapper for the Stereolabs ZED 2i.

Delegates to camera/zed2.py (ZED2Camera) which runs the capture loop in a
subprocess bridge, keeping pyzed (numpy<2.0) isolated from the main process.
"""

import logging
import os
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.cameras.camera import Camera

from .config_zed2i import ZED2iCameraConfig

# ---------------------------------------------------------------------------
# Add project root to sys.path so `camera.zed2` is importable.
# Layout:  <project_root>/lerobot_data_collection/lerobot_camera_zed2i/
#                                                    lerobot_camera_zed2i/zed2i.py (HERE)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))           # .../lerobot_camera_zed2i/
_PKG_ROOT = os.path.dirname(_HERE)                            # .../lerobot_camera_zed2i/  (install root)
_COLLECTION_DIR = os.path.dirname(_PKG_ROOT)                  # .../lerobot_data_collection/
_PROJECT_ROOT = os.path.dirname(_COLLECTION_DIR)              # .../robodata_Agilex/
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

logger = logging.getLogger(__name__)


class ZED2iCamera(Camera):
    """LeRobot-compatible wrapper around ZED2Camera (subprocess bridge).

    Delegates all I/O to the inner ZED2Camera while satisfying lerobot's
    Camera ABC (is_connected, connect, read, async_read, read_latest, disconnect).
    """

    def __init__(self, config: ZED2iCameraConfig):
        super().__init__(config)
        self._config = config
        self._inner = None  # created lazily in connect()

    def _build_inner(self):
        from camera.zed2 import ZED2Camera, ZED2CameraConfig
        return ZED2Camera(
            ZED2CameraConfig(
                fps=self._config.fps,
                width=self._config.width,
                height=self._config.height,
                use_depth=False,
                bridge_python=self._config.bridge_python,
            )
        )

    # ------------------------------------------------------------------
    # Camera ABC
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._inner is not None and self._inner.is_connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        # ZED cameras are enumerated via the bridge; return empty list here.
        return []

    def connect(self, warmup: bool = True) -> None:
        if self._inner is None:
            self._inner = self._build_inner()
        self._inner.connect(warmup=warmup)
        logger.info(f"{self.__class__.__name__} connected ({self._config.width}x{self._config.height}@{self._config.fps}fps)")

    def read(self) -> NDArray[Any]:
        """Blocking read: waits for the next new frame."""
        return self._inner.read()

    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """Wait for a new color frame; raises TimeoutError if it takes too long."""
        return self._inner.async_read(timeout_ms=timeout_ms)

    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent buffered frame without waiting for a new one.

        Non-blocking if any frame has been captured; falls back to async_read
        only if no frame is available yet.
        """
        with self._inner._frame_lock:
            frame = self._inner._latest_color_frame
        if frame is not None:
            return frame.copy()
        # No frame yet — do a single blocking read
        return self._inner.async_read(timeout_ms=float(max_age_ms))

    def get_camera_params(self) -> dict:
        """Return camera intrinsics/calibration (available after connect())."""
        if self._inner is not None:
            return self._inner.get_camera_params()
        return {}

    def disconnect(self) -> None:
        if self._inner is not None:
            self._inner.disconnect()
            self._inner = None
        logger.info(f"{self.__class__.__name__} disconnected.")
