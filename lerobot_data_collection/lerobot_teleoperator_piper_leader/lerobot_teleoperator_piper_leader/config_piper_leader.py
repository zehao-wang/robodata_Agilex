from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("piper_slave_echo")
@dataclass
class PiperLeaderConfig(TeleoperatorConfig):
    """Teleop that echoes the PIPER slave arm's current joint positions as actions.

    The slave arm is the one that actually interacts with the environment, so
    its joint positions are what the policy should learn to reproduce.  This
    teleop reads the same slave CAN feedback frames as PiperFollower
    (0x2A5-0x2A8) and returns them as the dataset action.

    The master arm's joint positions are NOT used — master and slave have
    different kinematics and their joint angles are not in the same space.

    Args:
        can_interface: CAN backend. Use "socketcan" after running setup_can.sh.
        can_channel:   SocketCAN channel name (default "can0").
    """

    can_interface: str = "socketcan"
    can_channel: str = "can0"
