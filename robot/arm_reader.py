"""PIPER arm data reader via CAN bus.

Supports two modes:
- gs_usb: Direct USB access via python-can gs_usb interface (macOS/Linux)
- socketcan: Linux socketcan interface
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

# PIPER CAN protocol constants (from piper_sdk can_id.py)
# Joint feedback CAN IDs: two joints per frame, big-endian int32, 0.001 degree units
JOINT_FEEDBACK_IDS = {0x2A5: (0, 1), 0x2A6: (2, 3), 0x2A7: (4, 5)}
# Gripper feedback CAN ID: bytes 0:4 = angle (int32, 0.001mm), bytes 4:6 = effort (int16, 0.001 Nm)
GRIPPER_FEEDBACK_ID = 0x2A8

# Conversion factor: raw value to radians (PIPER uses 0.001 degree units)
RAW_TO_RAD = np.pi / 180.0 / 1000.0
# Gripper raw to meters (0.001 mm -> m)
RAW_TO_METER = 1e-6


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
                 bitrate: int = 1_000_000):
        """
        Args:
            can_interface: 'gs_usb' for USB CAN adapter or 'socketcan' for Linux.
            can_channel: CAN channel name (used for socketcan mode).
            bitrate: CAN bus bitrate.
        """
        self.can_interface = can_interface
        self.can_channel = can_channel
        self.bitrate = bitrate

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
        while self._running:
            msg = self._bus.recv(timeout=0.05)
            if msg is None:
                continue
            self._parse_frame(msg)

    def _parse_frame(self, msg):
        """Parse a single CAN frame and update internal state."""
        now = time.time()

        with self._lock:
            if msg.arbitration_id in JOINT_FEEDBACK_IDS:
                j0, j1 = JOINT_FEEDBACK_IDS[msg.arbitration_id]
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

            elif msg.arbitration_id == GRIPPER_FEEDBACK_ID:
                raw_gripper = struct.unpack(">i", msg.data[0:4])[0]
                self._state.gripper = raw_gripper * RAW_TO_METER
                self._state.timestamp = now
