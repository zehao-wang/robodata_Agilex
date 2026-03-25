"""PIPER arm data reader via CAN bus.

Supports two modes:
- gs_usb: Direct USB access via python-can gs_usb interface (macOS/Linux)
- socketcan: Linux socketcan interface

Supports configurable CAN IDs for master-slave teleoperation mode.
"""

import platform
import struct
import threading
import time
from dataclasses import dataclass, field

import numpy as np


def _patch_gs_usb_reset_macos():
    """Workaround: gs_usb 0.3.0 calls usb device.reset() on start(),
    which invalidates the handle on macOS and causes 'Entity not found'.
    Monkey-patch to skip the reset."""
    if "darwin" not in platform.system().lower():
        return
    try:
        from gs_usb.gs_usb import GsUsb
    except ImportError:
        return

    _original_start = GsUsb.start

    def _patched_start(self, flags=None):
        orig_reset = self.gs_usb.reset
        self.gs_usb.reset = lambda: None
        try:
            if flags is not None:
                _original_start(self, flags)
            else:
                _original_start(self)
        finally:
            self.gs_usb.reset = orig_reset

    GsUsb.start = _patched_start


_patch_gs_usb_reset_macos()

# Conversion factor: raw value to radians (PIPER uses 0.001 degree units)
RAW_TO_RAD = np.pi / 180.0 / 1000.0
# Gripper raw to meters (0.001 mm -> m)
RAW_TO_METER = 1e-6


# ---------------------------------------------------------------------------
# Configurable CAN ID mapping
# ---------------------------------------------------------------------------

@dataclass
class ArmCANConfig:
    """CAN ID mapping for a single arm.

    Attributes:
        joint_ids: Mapping from CAN arbitration ID to (joint_a, joint_b) indices.
        gripper_id: CAN ID for gripper feedback.
        name: Human-readable name for this arm config.
    """
    joint_ids: dict  # {CAN_ID: (joint_a_idx, joint_b_idx)}
    gripper_id: int
    name: str = "arm"


# Default single-arm CAN IDs (from piper_sdk can_id.py)
DEFAULT_CAN_CONFIG = ArmCANConfig(
    joint_ids={0x2A5: (0, 1), 0x2A6: (2, 3), 0x2A7: (4, 5)},
    gripper_id=0x2A8,
    name="default",
)

# Master arm CAN IDs: In master-slave mode, the master (teaching input, 0xFA)
# does NOT broadcast feedback frames. Instead it sends its joint positions as
# control commands on 0x155-0x157 and gripper on 0x159.
MASTER_CAN_CONFIG = ArmCANConfig(
    joint_ids={0x155: (0, 1), 0x156: (2, 3), 0x157: (4, 5)},
    gripper_id=0x159,
    name="master",
)

# Slave arm CAN IDs: The slave (motion output, 0xFC) broadcasts standard
# feedback frames on 0x2A5-0x2A7 and gripper on 0x2A8 at ~200Hz.
SLAVE_CAN_CONFIG = ArmCANConfig(
    joint_ids={0x2A5: (0, 1), 0x2A6: (2, 3), 0x2A7: (4, 5)},
    gripper_id=0x2A8,
    name="slave",
)

# Legacy aliases for backward compatibility
JOINT_FEEDBACK_IDS = DEFAULT_CAN_CONFIG.joint_ids
GRIPPER_FEEDBACK_ID = DEFAULT_CAN_CONFIG.gripper_id


@dataclass
class ArmState:
    """Current state of the arm."""
    qpos: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    qvel: np.ndarray = field(default_factory=lambda: np.zeros(6, dtype=np.float64))
    gripper: float = 0.0
    timestamp: float = 0.0


class ArmReader:
    """Reads arm joint positions and gripper width from CAN bus."""

    def __init__(self, can_interface: str = "gs_usb", can_channel: str = "can0",
                 bitrate: int = 1_000_000, can_config: ArmCANConfig | None = None):
        """
        Args:
            can_interface: 'gs_usb' for USB CAN adapter or 'socketcan' for Linux.
            can_channel: CAN channel name (used for socketcan mode).
            bitrate: CAN bus bitrate.
            can_config: CAN ID mapping. Defaults to DEFAULT_CAN_CONFIG.
        """
        self.can_interface = can_interface
        self.can_channel = can_channel
        self.bitrate = bitrate
        self._can_config = can_config or DEFAULT_CAN_CONFIG

        self._bus = None
        self._state = ArmState()
        self._prev_qpos = np.zeros(6, dtype=np.float64)
        self._prev_time = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def connect(self):
        """Open the CAN bus connection."""
        import can

        if self.can_interface == "gs_usb":
            self._bus = self._open_gs_usb(can, self.bitrate)
        elif self.can_interface == "socketcan":
            self._bus = can.Bus(interface="socketcan", channel=self.can_channel,
                                bitrate=self.bitrate)
        else:
            raise ValueError(f"Unknown CAN interface: {self.can_interface}")

        print(f"[ArmReader] Connected via {self.can_interface}")

    @staticmethod
    def _open_gs_usb(can_module, bitrate: int):
        """Open gs_usb CAN device (candleLight firmware)."""
        import usb.core
        dev = usb.core.find(idVendor=0x1D50, idProduct=0x606F)
        if dev is None:
            raise RuntimeError(
                "CAN adapter not found. Check USB connection and ensure "
                "candleLight firmware is installed."
            )
        return can_module.Bus(
            interface="gs_usb",
            channel=dev.product,
            bus=dev.bus,
            address=dev.address,
            bitrate=bitrate,
        )

    def start(self):
        """Start the background CAN reading thread."""
        if self._bus is None:
            self.connect()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[ArmReader] Reading started")

    def stop(self):
        """Stop the background reading thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None
        print("[ArmReader] Stopped")

    def get_state(self) -> ArmState:
        """Get the latest arm state (thread-safe copy)."""
        with self._lock:
            return ArmState(
                qpos=self._state.qpos.copy(),
                qvel=self._state.qvel.copy(),
                gripper=self._state.gripper,
                timestamp=self._state.timestamp,
            )

    def _read_loop(self):
        """Background loop: read CAN frames and update state."""
        frame_count = 0
        last_print = time.time()
        while self._running:
            msg = self._bus.recv(timeout=0.05)
            if msg is None:
                if time.time() - last_print > 3.0:
                    print(f"[ArmReader] WARNING: no CAN frames received in last 3s "
                          f"(total received: {frame_count})")
                    last_print = time.time()
                continue
            frame_count += 1
            known = msg.arbitration_id in self._can_config.joint_ids \
                    or msg.arbitration_id == self._can_config.gripper_id
            if frame_count <= 5 or (frame_count % 500 == 0):
                tag = "MATCH" if known else "other"
                print(f"[ArmReader] frame #{frame_count}  "
                      f"id=0x{msg.arbitration_id:03X} [{tag}]  "
                      f"data={bytes(msg.data).hex(' ')}")
            last_print = time.time()
            self._parse_frame(msg)

    def _parse_frame(self, msg):
        """Parse a single CAN frame and update internal state."""
        now = time.time()
        joint_ids = self._can_config.joint_ids
        gripper_id = self._can_config.gripper_id

        with self._lock:
            if msg.arbitration_id in joint_ids:
                j0, j1 = joint_ids[msg.arbitration_id]
                # Each joint: 4-byte signed int (big-endian)
                raw0 = struct.unpack(">i", msg.data[0:4])[0]
                raw1 = struct.unpack(">i", msg.data[4:8])[0]
                self._state.qpos[j0] = raw0 * RAW_TO_RAD
                self._state.qpos[j1] = raw1 * RAW_TO_RAD
                self._state.timestamp = now

                # Estimate velocity via finite difference
                if self._prev_time > 0:
                    dt = now - self._prev_time
                    if dt > 0:
                        self._state.qvel = (self._state.qpos - self._prev_qpos) / dt
                self._prev_qpos = self._state.qpos.copy()
                self._prev_time = now

            elif msg.arbitration_id == gripper_id:
                raw_gripper = struct.unpack(">i", msg.data[0:4])[0]
                self._state.gripper = raw_gripper * RAW_TO_METER
                self._state.timestamp = now


# ---------------------------------------------------------------------------
# piper_sdk-based reader (initializes arm + reads via SDK, same interface)
# ---------------------------------------------------------------------------

class PiperArmReader:
    """Reads arm state via piper_sdk (C_PiperInterface_V2).

    Calls ConnectPort() to initialize the arm (which triggers CAN feedback),
    then polls GetArmJointMsgs() / GetArmGripperMsgs() in a background thread.
    Drop-in replacement for ArmReader — same start/stop/get_state interface.
    """

    def __init__(self, can_interface: str = "socketcan", can_channel: str = "can0",
                 bitrate: int = 1_000_000):
        self.can_interface = can_interface
        self.can_channel = can_channel
        self.bitrate = bitrate

        self._piper = None
        self._state = ArmState()
        self._prev_qpos = np.zeros(6, dtype=np.float64)
        self._prev_time = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def start(self):
        from robot.arm_controller import create_piper
        print(f"[PiperArmReader] Connecting via {self.can_interface}...")
        self._piper = create_piper(self.can_interface, self.can_channel, self.bitrate)
        print("[PiperArmReader] Connected. Starting read loop.")
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._piper is not None:
            try:
                self._piper.DisconnectPort()
            except Exception:
                pass
            self._piper = None
        print("[PiperArmReader] Stopped")

    def get_state(self) -> ArmState:
        with self._lock:
            return ArmState(
                qpos=self._state.qpos.copy(),
                qvel=self._state.qvel.copy(),
                gripper=self._state.gripper,
                timestamp=self._state.timestamp,
            )

    def _read_loop(self):
        deg_to_rad = np.pi / 180.0
        frame = 0
        while self._running:
            now = time.time()
            try:
                j = self._piper.GetArmJointMsgs()
                js = j.joint_state
                raw = [js.joint_1, js.joint_2, js.joint_3,
                       js.joint_4, js.joint_5, js.joint_6]
                qpos = np.array(raw, dtype=np.float64) * 0.001 * deg_to_rad

                g = self._piper.GetArmGripperMsgs()
                gripper_m = g.gripper_state.grippers_angle * 1e-6

                if frame < 10 or frame % 200 == 0:
                    print(f"[PiperArmReader] frame={frame} "
                          f"raw_joints={raw} "
                          f"gripper_raw={g.gripper_state.grippers_angle} "
                          f"qpos_deg={[round(v*180/np.pi,2) for v in qpos]}")

                with self._lock:
                    if self._prev_time > 0:
                        dt = now - self._prev_time
                        if dt > 0:
                            self._state.qvel = (qpos - self._prev_qpos) / dt
                    self._state.qpos = qpos
                    self._state.gripper = gripper_m
                    self._state.timestamp = now
                    self._prev_qpos = qpos.copy()
                    self._prev_time = now
            except Exception as e:
                print(f"[PiperArmReader] read error: {e}")
            frame += 1
            time.sleep(0.005)  # ~200 Hz poll
