from dataclasses import dataclass

from lerobot.cameras.configs import CameraConfig


@CameraConfig.register_subclass("zed2i")
@dataclass
class ZED2iCameraConfig(CameraConfig):
    """Configuration for Stereolabs ZED 2i camera via subprocess bridge.

    The bridge runs under a separate Python interpreter that has pyzed + numpy<2.0.
    Set ZED_BRIDGE_PYTHON env var to the path of that interpreter, or supply
    bridge_python explicitly in the config dict.

    Supported resolutions (width x height @ fps):
        2208x1242 @ 15
        1920x1080 @ 15, 30
        1280x720  @ 15, 30, 60
        672x376   @ 15, 30, 100

    Example camera config string for lerobot-record:
        "{zed2i: {type: zed2i, width: 1280, height: 720, fps: 15}}"
    """

    # Path to the Python interpreter with pyzed + numpy<2.0 installed.
    # Defaults to $ZED_BRIDGE_PYTHON env var, then sys.executable.
    bridge_python: str | None = None

    def __post_init__(self) -> None:
        if self.bridge_python is None:
            import os
            import sys
            self.bridge_python = os.environ.get("ZED_BRIDGE_PYTHON", sys.executable)
