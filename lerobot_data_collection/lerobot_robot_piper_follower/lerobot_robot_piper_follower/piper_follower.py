"""LeRobot Robot implementation for the AgileX PIPER slave arm.

In PIPER master-slave mode the hardware automatically forwards the master
arm's joint positions to the slave arm.  This class:
  - Reads slave arm joint positions + gripper via CAN bus (ArmReader)
  - Reads camera images via the lerobot camera system
  - Does NOT send any commands (send_action is a no-op)

Joint positions are returned in radians; gripper width is in metres.
Action keys match SO-ARM convention: joint_1.pos … joint_6.pos, gripper.pos
"""

import logging
import os
import sys

import numpy as np

from lerobot.cameras.utils import make_cameras_from_configs
from lerobot.processor import RobotAction, RobotObservation
from lerobot.robots.robot import Robot

from .config_piper_follower import PiperFollowerConfig

# ---------------------------------------------------------------------------
# Resolve project root so robot.arm_reader is importable.
# Layout:  <project_root>/lerobot_data_collection/lerobot_robot_piper_follower/
#                                                   lerobot_robot_piper_follower/piper_follower.py
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.dirname(_HERE)
_COLLECTION_DIR = os.path.dirname(_PKG_ROOT)
_PROJECT_ROOT = os.path.dirname(_COLLECTION_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from robot.arm_reader import ArmReader, SLAVE_CAN_CONFIG  # noqa: E402

logger = logging.getLogger(__name__)

_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]


class PiperFollower(Robot):
    """Reads the PIPER slave arm state and camera images for lerobot recording.

    Usage (via lerobot-record):
        --robot.type=piper_follower
        --robot.can_interface=socketcan
        --robot.can_channel=can0
        --robot.cameras="{realsense: {type: intelrealsense, ...}, zed2i: {type: zed2i, ...}}"
    """

    config_class = PiperFollowerConfig
    name = "piper_follower"

    def __init__(self, config: PiperFollowerConfig):
        super().__init__(config)
        self.config = config
        self._reader = ArmReader(
            can_interface=config.can_interface,
            can_channel=config.can_channel,
            can_config=SLAVE_CAN_CONFIG,
        )
        self.cameras = make_cameras_from_configs(config.cameras)
        self._connected = False

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    @property
    def observation_features(self) -> dict:
        feats: dict = {f"{j}.pos": float for j in _JOINT_NAMES}
        feats["gripper.pos"] = float
        for cam_key, cam_cfg in self.config.cameras.items():
            feats[cam_key] = (cam_cfg.height, cam_cfg.width, 3)
        return feats

    @property
    def action_features(self) -> dict:
        feats: dict = {f"{j}.pos": float for j in _JOINT_NAMES}
        feats["gripper.pos"] = float
        return feats

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._reader.connect()
        self._reader.start()
        for cam in self.cameras.values():
            cam.connect()
        self._connected = True
        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        self._reader.stop()
        for cam in self.cameras.values():
            cam.disconnect()
        self._connected = False
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------
    # Calibration / configuration (no-ops for PIPER)
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------

    def get_observation(self) -> RobotObservation:
        state = self._reader.get_state()
        obs: RobotObservation = {}
        for i, j in enumerate(_JOINT_NAMES):
            obs[f"{j}.pos"] = float(state.qpos[i])
        obs["gripper.pos"] = float(state.gripper)
        for cam_key, cam in self.cameras.items():
            obs[cam_key] = cam.read_latest()
        return obs

    def send_action(self, action: RobotAction) -> RobotAction:
        # Hardware master-slave handles motion; no command needed.
        return action
