"""ZED 2i left-camera capture via zed-open-capture-mac."""

from __future__ import annotations

import ctypes
import subprocess
import time
from pathlib import Path

import numpy as np


_REPO_ROOT = Path(__file__).resolve().parent.parent
_NATIVE_DIR = Path(__file__).resolve().parent / "native"
_BRIDGE_SRC = _NATIVE_DIR / "zed_capture_bridge.mm"
_BRIDGE_LIB = _NATIVE_DIR / "libzed_capture_bridge.dylib"
_ZED_PREFIX = Path("/opt/stereolabs")
_ZED_INCLUDE = _ZED_PREFIX / "include"
_ZED_LIB_DIR = _ZED_PREFIX / "lib"
_ZED_LIB_FILE = _ZED_LIB_DIR / "libzed-open-capture-mac.dylib"


class _ZedCalibrationC(ctypes.Structure):
    _fields_ = [
        ("stereo_width", ctypes.c_int),
        ("stereo_height", ctypes.c_int),
        ("left_width", ctypes.c_int),
        ("left_height", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("fx", ctypes.c_float),
        ("fy", ctypes.c_float),
        ("cx", ctypes.c_float),
        ("cy", ctypes.c_float),
        ("k1", ctypes.c_float),
        ("k2", ctypes.c_float),
        ("p1", ctypes.c_float),
        ("p2", ctypes.c_float),
        ("k3", ctypes.c_float),
        ("serial", ctypes.c_char * 128),
        ("name", ctypes.c_char * 128),
        ("calibration_section", ctypes.c_char * 128),
    ]


def _decode_c_string(value: bytes) -> str:
    return value.split(b"\0", 1)[0].decode("utf-8", errors="replace")


def _build_bridge() -> None:
    if not _BRIDGE_SRC.exists():
        raise FileNotFoundError(f"Missing ZED bridge source: {_BRIDGE_SRC}")
    if not _ZED_INCLUDE.exists() or not _ZED_LIB_FILE.exists():
        raise FileNotFoundError(
            "Missing zed-open-capture-mac installation under /opt/stereolabs"
        )

    needs_build = not _BRIDGE_LIB.exists()
    if not needs_build:
        needs_build = _BRIDGE_LIB.stat().st_mtime < _BRIDGE_SRC.stat().st_mtime

    if not needs_build:
        return

    cmd = [
        "clang++",
        "-std=c++20",
        "-dynamiclib",
        "-fPIC",
        "-O2",
        str(_BRIDGE_SRC),
        "-o",
        str(_BRIDGE_LIB),
        "-I",
        str(_ZED_INCLUDE),
        "-L",
        str(_ZED_LIB_DIR),
        "-lzed-open-capture-mac",
        "-framework",
        "CoreFoundation",
        "-Wl,-rpath,/opt/stereolabs/lib",
    ]
    subprocess.run(cmd, check=True, cwd=_REPO_ROOT)


def _load_library() -> ctypes.CDLL:
    _build_bridge()
    lib = ctypes.CDLL(str(_BRIDGE_LIB))

    lib.zed_camera_create.restype = ctypes.c_void_p
    lib.zed_camera_destroy.argtypes = [ctypes.c_void_p]
    lib.zed_camera_destroy.restype = None

    for name in ("zed_camera_open_hd1080", "zed_camera_start", "zed_camera_stop"):
        func = getattr(lib, name)
        func.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_size_t]
        func.restype = ctypes.c_int

    lib.zed_camera_get_calibration.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(_ZedCalibrationC),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.zed_camera_get_calibration.restype = ctypes.c_int

    lib.zed_camera_copy_latest_left_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.zed_camera_copy_latest_left_frame.restype = ctypes.c_int

    lib.zed_camera_wait_for_frame.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_size_t,
    ]
    lib.zed_camera_wait_for_frame.restype = ctypes.c_int
    return lib


def _ctypes_call(func, *args):
    error_buf = ctypes.create_string_buffer(1024)
    ok = func(*args, error_buf, len(error_buf))
    if ok:
        return ok
    message = _decode_c_string(error_buf.raw) or f"{func.__name__} failed"
    raise RuntimeError(message)


class ZedCamera:
    """Captures the left RGB stream from a ZED 2i at 1080p."""

    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        fps: int = 30,
        streams: str = "rgb",
    ):
        if streams != "rgb":
            raise ValueError("ZED camera module currently supports RGB left-camera capture only")
        if width != 1920 or height != 1080:
            raise ValueError("ZED camera module is fixed to 1920x1080 left-camera capture")
        if fps != 30:
            raise ValueError("ZED camera module is fixed to 30 FPS")

        self.width = width
        self.height = height
        self.fps = fps
        self.streams = streams

        self._lib = None
        self._handle = None
        self._calibration = None
        self._camera_info = None
        self._color_frame = np.zeros((height, width, 3), dtype=np.uint8)
        self._depth_frame = np.zeros((height, width), dtype=np.uint16)
        self._timestamp = 0.0
        self._frame_buffer = np.empty((height, width, 3), dtype=np.uint8)

    @staticmethod
    def is_supported() -> bool:
        return _ZED_INCLUDE.exists() and _ZED_LIB_FILE.exists()

    @staticmethod
    def list_devices() -> list[dict[str, str]]:
        if not ZedCamera.is_supported():
            return []
        return [
            {
                "name": "ZED 2i Left Camera",
                "serial": None,
                "resolution": "1920x1080",
                "fps": "30",
            }
        ]

    def start(self):
        """Open the ZED stream and start background capture."""
        self._lib = _load_library()
        self._handle = self._lib.zed_camera_create()
        if not self._handle:
            raise RuntimeError("Failed to allocate ZED camera handle")

        try:
            _ctypes_call(self._lib.zed_camera_open_hd1080, self._handle)
            calibration = _ZedCalibrationC()
            _ctypes_call(
                self._lib.zed_camera_get_calibration,
                self._handle,
                ctypes.byref(calibration),
            )
            self._calibration = calibration
            self.width = calibration.left_width
            self.height = calibration.left_height
            self._color_frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            self._depth_frame = np.zeros((self.height, self.width), dtype=np.uint16)
            self._frame_buffer = np.empty((self.height, self.width, 3), dtype=np.uint8)
            self._camera_info = self._build_camera_info(calibration)

            _ctypes_call(self._lib.zed_camera_start, self._handle)
            self._wait_for_first_frame(timeout_s=2.0)
            print(
                "[ZedCamera] Started "
                f"({self.width}x{self.height} @ {self.fps}fps, serial={self._camera_info['serial']})"
            )
        except Exception:
            self.stop()
            raise

    def stop(self):
        """Stop capture and release resources."""
        if self._lib is not None and self._handle is not None:
            try:
                error_buf = ctypes.create_string_buffer(1024)
                self._lib.zed_camera_stop(self._handle, error_buf, len(error_buf))
            finally:
                self._lib.zed_camera_destroy(self._handle)
        self._handle = None
        self._lib = None
        self._timestamp = 0.0
        print("[ZedCamera] Stopped")

    def get_frames(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Return the latest cached RGB frame and an empty depth frame."""
        return self._color_frame.copy(), self._depth_frame.copy(), self._timestamp

    def capture_sync(self, arm_state_provider) -> tuple[np.ndarray, np.ndarray, float, object]:
        """Grab the latest frame and sample the arm immediately after."""
        color, depth, timestamp = self._read_latest_frame()
        arm_state = arm_state_provider()
        return color, depth, timestamp, arm_state

    def get_camera_info(self) -> dict | None:
        """Return calibrated left-camera parameters."""
        if self._camera_info is None:
            return None
        return {
            "backend": self._camera_info["backend"],
            "camera_model": self._camera_info["camera_model"],
            "serial": self._camera_info["serial"],
            "stream": self._camera_info["stream"],
            "resolution": dict(self._camera_info["resolution"]),
            "stereo_resolution": dict(self._camera_info["stereo_resolution"]),
            "fps": self._camera_info["fps"],
            "color_space": self._camera_info["color_space"],
            "calibration_section": self._camera_info["calibration_section"],
            "intrinsics": dict(self._camera_info["intrinsics"]),
            "distortion": dict(self._camera_info["distortion"]),
        }

    def _wait_for_first_frame(self, timeout_s: float) -> None:
        timeout_ms = max(1, int(timeout_s * 1000.0))
        error_buf = ctypes.create_string_buffer(1024)
        ok = self._lib.zed_camera_wait_for_frame(
            self._handle,
            timeout_ms,
            error_buf,
            len(error_buf),
        )
        if ok == 0:
            message = _decode_c_string(error_buf.raw)
            if message:
                raise RuntimeError(message)
            raise RuntimeError("Timed out waiting for the first ZED frame")
        self._read_latest_frame()

    def _read_latest_frame(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self._lib is None or self._handle is None:
            raise RuntimeError("ZED camera not started")

        timestamp = ctypes.c_double(0.0)
        error_buf = ctypes.create_string_buffer(1024)
        ok = self._lib.zed_camera_copy_latest_left_frame(
            self._handle,
            self._frame_buffer.ctypes.data_as(ctypes.c_void_p),
            self._frame_buffer.nbytes,
            ctypes.byref(timestamp),
            error_buf,
            len(error_buf),
        )
        if ok == 0:
            message = _decode_c_string(error_buf.raw)
            if message:
                raise RuntimeError(message)
            raise RuntimeError("No ZED frame available yet")

        color = self._frame_buffer[:, :, ::-1].copy()
        self._color_frame = color
        self._timestamp = float(timestamp.value)
        return self._color_frame.copy(), self._depth_frame.copy(), self._timestamp

    def _build_camera_info(self, calibration: _ZedCalibrationC) -> dict:
        return {
            "backend": "zed-open-capture-mac",
            "camera_model": _decode_c_string(calibration.name),
            "serial": _decode_c_string(calibration.serial),
            "stream": "left",
            "resolution": {
                "width": int(calibration.left_width),
                "height": int(calibration.left_height),
            },
            "stereo_resolution": {
                "width": int(calibration.stereo_width),
                "height": int(calibration.stereo_height),
            },
            "fps": int(self.fps),
            "color_space": "RGB",
            "calibration_section": _decode_c_string(calibration.calibration_section),
            "intrinsics": {
                "fx": float(calibration.fx),
                "fy": float(calibration.fy),
                "cx": float(calibration.cx),
                "cy": float(calibration.cy),
            },
            "distortion": {
                "k1": float(calibration.k1),
                "k2": float(calibration.k2),
                "p1": float(calibration.p1),
                "p2": float(calibration.p2),
                "k3": float(calibration.k3),
            },
        }
