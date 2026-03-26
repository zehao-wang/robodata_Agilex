"""LeRobot Teleoperator that echoes the PIPER slave arm positions as actions.

The slave arm is the one that physically interacts with the environment.
Its joint positions are what a policy must learn to reproduce during rollout.

The master arm has different kinematics — its joint angles are not in the
same space as the slave arm and should NOT be used as action labels.

This teleop reads the same slave CAN feedback frames as PiperFollower
(0x2A5-0x2A8) and returns them as the dataset action, so that:
    action[t] = observation.state[t]  (slave arm joint positions)

lerobot requires either a teleop or a policy to produce actions; this class
satisfies that requirement while keeping the action space correct.

Joint positions are in radians; gripper width is in metres.
Action keys: joint_1.pos … joint_6.pos, gripper.pos
"""

import logging
import os
import sys
from typing import Any

from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator

from .config_piper_leader import PiperLeaderConfig

# ---------------------------------------------------------------------------
# Resolve project root.
# Layout: <project_root>/lerobot_data_collection/lerobot_teleoperator_piper_leader/
#                                                  lerobot_teleoperator_piper_leader/piper_leader.py
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


class PiperLeader(Teleoperator):
    """Echoes PIPER slave arm positions as dataset actions for lerobot-record.

    Opens a second socketcan socket on can0 (Linux allows multiple readers)
    and reads the same slave feedback frames as PiperFollower, so that
    action[t] == observation.state[t] in the recorded dataset.

    Usage (via lerobot-record):
        --teleop.type=piper_slave_echo
        --teleop.can_interface=socketcan
        --teleop.can_channel=can0
    """

    config_class = PiperLeaderConfig
    name = "piper_slave_echo"

    def __init__(self, config: PiperLeaderConfig):
        super().__init__(config)
        self.config = config
        self._reader = ArmReader(
            can_interface=config.can_interface,
            can_channel=config.can_channel,
            can_config=SLAVE_CAN_CONFIG,
        )
        self._connected = False

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    @property
    def action_features(self) -> dict:
        feats: dict = {f"{j}.pos": float for j in _JOINT_NAMES}
        feats["gripper.pos"] = float
        return feats

    @property
    def feedback_features(self) -> dict:
        return {}

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._reader.connect()
        self._reader.start()
        self._connected = True
        logger.info(f"{self} connected.")

    def disconnect(self) -> None:
        self._reader.stop()
        self._connected = False
        logger.info(f"{self} disconnected.")

    # ------------------------------------------------------------------
    # Calibration / configuration (no-ops)
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

    def get_action(self) -> RobotAction:
        state = self._reader.get_state()
        action: RobotAction = {}
        for i, j in enumerate(_JOINT_NAMES):
            action[f"{j}.pos"] = float(state.qpos[i])
        action["gripper.pos"] = float(state.gripper)
        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        pass  # No haptic/force feedback to the master arm
