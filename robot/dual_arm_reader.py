"""Dual-arm CAN reader for master-slave teleoperation.

Uses a single CAN bus connection and dispatches frames to master/slave
ArmState instances based on CAN ID configuration.
"""

import struct
import threading
import time

import numpy as np

from .arm_reader import (
    ArmCANConfig,
    ArmState,
    MASTER_CAN_CONFIG,
    SLAVE_CAN_CONFIG,
    RAW_TO_RAD,
    RAW_TO_METER,
    _patch_gs_usb_reset_macos,
)

_patch_gs_usb_reset_macos()


class DualArmReader:
    """Reads master and slave arm states from a single CAN bus."""

    def __init__(
        self,
        can_interface: str = "gs_usb",
        can_channel: str = "can0",
        bitrate: int = 1_000_000,
        master_config: ArmCANConfig | None = None,
        slave_config: ArmCANConfig | None = None,
    ):
        self.can_interface = can_interface
        self.can_channel = can_channel
        self.bitrate = bitrate
        self._master_cfg = master_config or MASTER_CAN_CONFIG
        self._slave_cfg = slave_config or SLAVE_CAN_CONFIG

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

        # Build combined CAN ID lookup: arbitration_id -> (config, "master"/"slave")
        self._id_lookup = {}
        for can_id in self._master_cfg.joint_ids:
            self._id_lookup[can_id] = ("master", "joint", self._master_cfg.joint_ids[can_id])
        self._id_lookup[self._master_cfg.gripper_id] = ("master", "gripper", None)
        for can_id in self._slave_cfg.joint_ids:
            self._id_lookup[can_id] = ("slave", "joint", self._slave_cfg.joint_ids[can_id])
        self._id_lookup[self._slave_cfg.gripper_id] = ("slave", "gripper", None)

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
        entry = self._id_lookup.get(msg.arbitration_id)
        if entry is None:
            return

        arm_name, msg_type, joint_pair = entry
        now = time.time()

        with self._lock:
            if arm_name == "master":
                state = self._master_state
                prev_qpos = self._master_prev_qpos
                prev_time = self._master_prev_time
            else:
                state = self._slave_state
                prev_qpos = self._slave_prev_qpos
                prev_time = self._slave_prev_time

            if msg_type == "joint":
                j0, j1 = joint_pair
                raw0 = struct.unpack(">i", msg.data[0:4])[0]
                raw1 = struct.unpack(">i", msg.data[4:8])[0]
                state.qpos[j0] = raw0 * RAW_TO_RAD
                state.qpos[j1] = raw1 * RAW_TO_RAD
                state.timestamp = now

                if prev_time > 0:
                    dt = now - prev_time
                    if dt > 0:
                        state.qvel = (state.qpos - prev_qpos) / dt
                prev_qpos[:] = state.qpos
                if arm_name == "master":
                    self._master_prev_time = now
                else:
                    self._slave_prev_time = now

            elif msg_type == "gripper":
                raw_gripper = struct.unpack(">i", msg.data[0:4])[0]
                state.gripper = raw_gripper * RAW_TO_METER
                state.timestamp = now
