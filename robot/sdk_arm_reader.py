"""PIPER arm reader using the official piper_sdk feedback API.

This follows the same feedback path used by the PushT service:
``C_PiperInterface_V2`` -> ``GetArmEndPoseMsgs().end_pose``.
"""

import threading
import time

import numpy as np

from .arm_controller import RAW_TO_DEG, RAW_TO_MM, RAW_TO_RAD, create_piper
from .arm_reader import ArmState

_MM_TO_M = 1e-3
_GRIPPER_RAW_TO_M = 1e-6


class PiperSDKArmReader:
    """Read joints, gripper, and Cartesian EEF pose through piper_sdk."""

    def __init__(
        self,
        can_interface: str = "gs_usb",
        can_channel: str = "can0",
        bitrate: int = 1_000_000,
        poll_hz: float = 100.0,
        feedback_timeout: float = 3.0,
    ):
        self.can_interface = can_interface
        self.can_channel = can_channel
        self.bitrate = bitrate
        self.poll_hz = float(poll_hz)
        self.feedback_timeout = float(feedback_timeout)

        self._piper = None
        self._state = ArmState()
        self._prev_qpos = np.zeros(6, dtype=np.float64)
        self._prev_time = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._last_error = ""

    def connect(self) -> None:
        """Open the SDK CAN port."""
        self._piper = create_piper(
            interface=self.can_interface,
            channel=self.can_channel,
            bitrate=self.bitrate,
        )
        print(
            f"[PiperSDKArmReader] Connected via {self.can_interface} "
            f"on {self.can_channel}"
        )
        if not self._wait_for_pose_feedback(self.feedback_timeout):
            print("[PiperSDKArmReader] WARNING: no SDK EEF feedback received yet")

    def start(self) -> None:
        """Start the background SDK polling thread."""
        if self._piper is None:
            self.connect()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()
        print("[PiperSDKArmReader] Reading started")

    def stop(self) -> None:
        """Stop reading and close the SDK CAN port."""
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
        print("[PiperSDKArmReader] Stopped")

    def get_state(self) -> ArmState:
        """Return the latest SDK-backed arm state."""
        with self._lock:
            return ArmState(
                qpos=self._state.qpos.copy(),
                qvel=self._state.qvel.copy(),
                gripper=self._state.gripper,
                timestamp=self._state.timestamp,
                eef_pos_m=(
                    None if self._state.eef_pos_m is None else self._state.eef_pos_m.copy()
                ),
                eef_euler_deg=(
                    None
                    if self._state.eef_euler_deg is None
                    else self._state.eef_euler_deg.copy()
                ),
                pose_timestamp=self._state.pose_timestamp,
                pose_source=self._state.pose_source,
            )

    @property
    def last_error(self) -> str:
        return self._last_error

    def _wait_for_pose_feedback(self, timeout: float) -> bool:
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self._read_once():
                return True
            time.sleep(0.05)
        return False

    def _read_loop(self) -> None:
        target_dt = 1.0 / max(self.poll_hz, 1.0)
        while self._running:
            t0 = time.perf_counter()
            self._read_once()
            sleep_time = target_dt - (time.perf_counter() - t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _read_once(self) -> bool:
        if self._piper is None:
            return False

        now = time.time()
        try:
            pose = self._piper.GetArmEndPoseMsgs().end_pose
            joints = self._piper.GetArmJointMsgs().joint_state
            gripper_msg = self._piper.GetArmGripperMsgs().gripper_state
        except Exception as exc:
            self._last_error = str(exc)
            return False

        qpos = np.array(
            [
                joints.joint_1,
                joints.joint_2,
                joints.joint_3,
                joints.joint_4,
                joints.joint_5,
                joints.joint_6,
            ],
            dtype=np.float64,
        ) * RAW_TO_RAD

        qvel = np.zeros(6, dtype=np.float64)
        if self._prev_time > 0.0:
            dt = now - self._prev_time
            if dt > 0.0:
                qvel = (qpos - self._prev_qpos) / dt
        self._prev_qpos = qpos.copy()
        self._prev_time = now

        has_pose = any([pose.X_axis, pose.Y_axis, pose.Z_axis])
        eef_pos_m = None
        eef_euler_deg = None
        pose_timestamp = 0.0
        if has_pose:
            eef_pos_m = np.array(
                [
                    pose.X_axis * RAW_TO_MM * _MM_TO_M,
                    pose.Y_axis * RAW_TO_MM * _MM_TO_M,
                    pose.Z_axis * RAW_TO_MM * _MM_TO_M,
                ],
                dtype=np.float64,
            )
            eef_euler_deg = np.array(
                [
                    pose.RX_axis * RAW_TO_DEG,
                    pose.RY_axis * RAW_TO_DEG,
                    pose.RZ_axis * RAW_TO_DEG,
                ],
                dtype=np.float64,
            )
            pose_timestamp = now

        gripper_m = float(gripper_msg.grippers_angle * _GRIPPER_RAW_TO_M)

        with self._lock:
            self._state = ArmState(
                qpos=qpos,
                qvel=qvel,
                gripper=gripper_m,
                timestamp=now,
                eef_pos_m=eef_pos_m,
                eef_euler_deg=eef_euler_deg,
                pose_timestamp=pose_timestamp,
                pose_source="piper-sdk" if has_pose else "",
            )
        self._last_error = ""
        return has_pose
