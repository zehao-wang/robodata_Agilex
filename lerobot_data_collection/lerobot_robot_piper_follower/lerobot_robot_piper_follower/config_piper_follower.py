from dataclasses import dataclass, field

from lerobot.cameras.configs import CameraConfig
from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("piper_follower")
@dataclass
class PiperFollowerConfig(RobotConfig):
    """Configuration for the AgileX PIPER slave (follower) arm.

    In master-slave teleoperation mode the hardware transparently forwards
    master joint commands to the slave arm.  This robot class only reads the
    slave arm's state — it never issues motion commands.

    Args:
        can_interface: CAN backend. Use "socketcan" (Linux) after running
                       setup_can.sh, or "gs_usb" for direct USB access.
        can_channel:   SocketCAN channel name (default "can0").
        cameras:       Dict of camera name → CameraConfig, e.g.
                           {realsense: {type: intelrealsense, ...},
                            zed2i:     {type: zed2i,          ...}}
    """

    can_interface: str = "socketcan"
    can_channel: str = "can0"
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
