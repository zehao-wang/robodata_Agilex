"""Dual-arm CAN reader for master-slave teleoperation.

In PIPER's master-slave mode on a single CAN bus:
- Master arm (teaching input, 0xFA) sends its joint positions as control
  commands on 0x155/0x156/0x157 (~3Hz) and gripper on 0x159.
- Slave arm (motion output, 0xFC) broadcasts standard feedback frames
  on 0x2A5/0x2A6/0x2A7 (~200Hz) and gripper on 0x2A8.

No CAN ID offset is needed — master uses control frame IDs, slave uses
feedback frame IDs, so they don't collide.
"""

import struct
import threading
import time

import numpy as np

from .arm_reader import (
    ArmState,
    MASTER_CAN_CONFIG,
    SLAVE_CAN_CONFIG,
    RAW_TO_RAD,
    RAW_TO_METER,
    _patch_gs_usb_reset_macos,
)

_patch_gs_usb_reset_macos()

# CAN IDs for master arm (control commands from master firmware)
_MASTER_JOINT_IDS = {0x155: (0, 1), 0x156: (2, 3), 0x157: (4, 5)}
_MASTER_GRIPPER_ID = 0x159

# CAN IDs for slave arm (standard feedback frames)
_SLAVE_JOINT_IDS = {0x2A5: (0, 1), 0x2A6: (2, 3), 0x2A7: (4, 5)}
_SLAVE_GRIPPER_ID = 0x2A8


class DualArmReader:
    """Reads master and slave arm states from a single CAN bus.

    Master arm positions come from joint control commands (0x155-0x157).
    Slave arm positions come from joint feedback frames (0x2A5-0x2A7).
    """

    def __init__(
        self,
        can_interface: str = "gs_usb",
        can_channel: str = "can0",
        bitrate: int = 1_000_000,
    ):
        self.can_interface = can_interface
        self.can_channel = can_channel
        self.bitrate = bitrate

        self._bus = None
        self._master_state = ArmState()
        self._slave_state = ArmState()
        self._master_prev_qpos = np.zeros(6, dtype=np.float64)
        self._slave_prev_qpos = np.zeros(6, dtype=np.float64)
        self._master_prev_time = 0.0
        self._slave_prev_time = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None

    def connect(self):
        """Open the CAN bus connection."""
        import can

        if self.can_interface == "gs_usb":
            import usb.core
            dev = usb.core.find(idVendor=0x1D50, idProduct=0x606F)
            if dev is None:
                raise RuntimeError("CAN adapter not found.")
            self._bus = can.Bus(
                interface="gs_usb",
                channel=dev.product,
                bus=dev.bus,
                address=dev.address,
                bitrate=self.bitrate,
            )
        elif self.can_interface == "socketcan":
            self._bus = can.Bus(
                interface="socketcan",
                channel=self.can_channel,
                bitrate=self.bitrate,
            )
        else:
            raise ValueError(f"Unknown CAN interface: {self.can_interface}")
        print(f"[DualArmReader] Connected via {self.can_interface}")

    def start(self):
        """Start the background CAN reading thread."""
        if self._bus is None:
            self.connect()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[DualArmReader] Reading started")

    def stop(self):
        """Stop the background reading thread."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._bus is not None:
            self._bus.shutdown()
            self._bus = None
        print("[DualArmReader] Stopped")

    def get_master_state(self) -> ArmState:
        """Get the latest master arm state (thread-safe copy)."""
        with self._lock:
            return ArmState(
                qpos=self._master_state.qpos.copy(),
                qvel=self._master_state.qvel.copy(),
                gripper=self._master_state.gripper,
                timestamp=self._master_state.timestamp,
            )

    def get_slave_state(self) -> ArmState:
        """Get the latest slave arm state (thread-safe copy)."""
        with self._lock:
            return ArmState(
                qpos=self._slave_state.qpos.copy(),
                qvel=self._slave_state.qvel.copy(),
                gripper=self._slave_state.gripper,
                timestamp=self._slave_state.timestamp,
            )

    def _read_loop(self):
        """Background loop: read CAN frames and dispatch to master/slave."""
        while self._running:
            msg = self._bus.recv(timeout=0.05)
            if msg is None:
                continue
            self._parse_frame(msg)

    def _parse_frame(self, msg):
        """Parse a single CAN frame and update the appropriate arm state."""
        cid = msg.arbitration_id
        now = time.time()

        # --- Master arm: joint control commands from master firmware ---
        if cid in _MASTER_JOINT_IDS:
            j0, j1 = _MASTER_JOINT_IDS[cid]
            raw0 = struct.unpack(">i", msg.data[0:4])[0]
            raw1 = struct.unpack(">i", msg.data[4:8])[0]
            with self._lock:
                self._master_state.qpos[j0] = raw0 * RAW_TO_RAD
                self._master_state.qpos[j1] = raw1 * RAW_TO_RAD
                self._master_state.timestamp = now
                if self._master_prev_time > 0:
                    dt = now - self._master_prev_time
                    if dt > 0:
                        self._master_state.qvel = (
                            self._master_state.qpos - self._master_prev_qpos
                        ) / dt
                self._master_prev_qpos[:] = self._master_state.qpos
                self._master_prev_time = now
            return

        if cid == _MASTER_GRIPPER_ID:
            raw_gripper = struct.unpack(">i", msg.data[0:4])[0]
            with self._lock:
                self._master_state.gripper = raw_gripper * RAW_TO_METER
                self._master_state.timestamp = now
            return

        # --- Slave arm: standard feedback frames ---
        if cid in _SLAVE_JOINT_IDS:
            j0, j1 = _SLAVE_JOINT_IDS[cid]
            raw0 = struct.unpack(">i", msg.data[0:4])[0]
            raw1 = struct.unpack(">i", msg.data[4:8])[0]
            with self._lock:
                self._slave_state.qpos[j0] = raw0 * RAW_TO_RAD
                self._slave_state.qpos[j1] = raw1 * RAW_TO_RAD
                self._slave_state.timestamp = now
                if self._slave_prev_time > 0:
                    dt = now - self._slave_prev_time
                    if dt > 0:
                        self._slave_state.qvel = (
                            self._slave_state.qpos - self._slave_prev_qpos
                        ) / dt
                self._slave_prev_qpos[:] = self._slave_state.qpos
                self._slave_prev_time = now
            return

        if cid == _SLAVE_GRIPPER_ID:
            raw_gripper = struct.unpack(">i", msg.data[0:4])[0]
            with self._lock:
                self._slave_state.gripper = raw_gripper * RAW_TO_METER
                self._slave_state.timestamp = now
            return
