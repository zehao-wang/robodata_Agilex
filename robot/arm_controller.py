"""Extracted motion primitives for the PIPER arm.

Provides low-level arm control functions (enable, move, read joints)
using the piper_sdk C_PiperInterface_V2.  Includes macOS gs_usb patches.
"""

import math
import platform
import time

import can
import numpy as np


# ---------------------------------------------------------------------------
# macOS gs_usb monkey-patches
# ---------------------------------------------------------------------------

def _patch_gs_usb_reset_macos():
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


def _patch_std_can_for_gs_usb():
    """Patch C_STD_CAN.Init to support gs_usb bustype on macOS."""
    from piper_sdk import C_STD_CAN

    _original_init = C_STD_CAN.Init

    def _patched_init(self):
        if self.bustype == "gs_usb":
            if self.bus is not None:
                return self.CAN_STATUS.INIT_CAN_BUS_IS_EXIST
            try:
                import usb.core
                dev = usb.core.find(idVendor=0x1D50, idProduct=0x606F)
                if dev is None:
                    self.bus = None
                    return self.CAN_STATUS.INIT_CAN_BUS_OPENED_FAILED
                self.bus = can.interface.Bus(
                    interface="gs_usb",
                    channel=dev.bus,
                    bus=dev.bus,
                    address=dev.address,
                    bitrate=self.expected_bitrate,
                )
                return self.CAN_STATUS.INIT_CAN_BUS_OPENED_SUCCESS
            except can.CanError:
                self.bus = None
                return self.CAN_STATUS.INIT_CAN_BUS_OPENED_FAILED
        return _original_init(self)

    C_STD_CAN.Init = _patched_init


_patch_gs_usb_reset_macos()
_patch_std_can_for_gs_usb()


# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------

MM_TO_RAW = 1000     # mm -> 0.001mm raw unit
DEG_TO_RAW = 1000    # deg -> 0.001deg raw unit
RAW_TO_MM = 0.001
RAW_TO_DEG = 0.001
RAW_TO_RAD = math.pi / 180.0 / 1000.0
RAD_TO_RAW = 1.0 / RAW_TO_RAD

# Minimum joint-space distance (0.001deg units) between recorded waypoints.
WAYPOINT_MIN_DIST = 2000

# Safe EEF home position (raw units: 0.001mm / 0.001deg).
HOME_EEF = (57000, 0, 215000, 0, 85000, 0)


# ---------------------------------------------------------------------------
# Piper creation and basic control
# ---------------------------------------------------------------------------

def list_can_devices():
    import usb.core
    return list(usb.core.find(find_all=True, idVendor=0x1D50, idProduct=0x606F))


def create_piper(interface="gs_usb", channel="can0", bitrate=1_000_000):
    """Create and connect a C_PiperInterface_V2 instance."""
    from piper_sdk import C_PiperInterface_V2

    if interface == "gs_usb":
        piper = C_PiperInterface_V2(can_name=channel, judge_flag=False,
                                     can_auto_init=False)
        piper.CreateCanBus(can_name=channel, bustype="gs_usb",
                           expected_bitrate=bitrate, judge_flag=False)
        piper.ConnectPort(can_init=False)
    else:
        piper = C_PiperInterface_V2(can_name=channel, judge_flag=True,
                                     can_auto_init=True)
        piper.ConnectPort()
    return piper


def enable_arm(piper, timeout=5.0):
    """Enable arm with timeout to avoid infinite retries."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if piper.EnablePiper():
            return True
        time.sleep(0.05)
    return False


def wait_for_feedback(piper, timeout=2.0):
    """Wait until the SDK receives valid end-effector feedback."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        pose = piper.GetArmEndPoseMsgs().end_pose
        if any([pose.X_axis, pose.Y_axis, pose.Z_axis]):
            return {
                "x": pose.X_axis * RAW_TO_MM,
                "y": pose.Y_axis * RAW_TO_MM,
                "z": pose.Z_axis * RAW_TO_MM,
                "rx": pose.RX_axis * RAW_TO_DEG,
                "ry": pose.RY_axis * RAW_TO_DEG,
                "rz": pose.RZ_axis * RAW_TO_DEG,
            }
        time.sleep(0.05)
    return None


def read_joints(piper):
    """Read current 6-joint positions as a list of raw ints (0.001deg)."""
    js = piper.GetArmJointMsgs().joint_state
    return [js.joint_1, js.joint_2, js.joint_3,
            js.joint_4, js.joint_5, js.joint_6]


def joint_distance(a, b):
    """Max absolute joint difference (raw 0.001deg units)."""
    return max(abs(ai - bi) for ai, bi in zip(a, b))


def wait_motion_done(piper, timeout=5.0, settle_time=0.5, tol=200):
    """Wait for motion complete signal (status or joint settle)."""
    t0 = time.time()
    last_j = read_joints(piper)
    stable_since = None

    while time.time() - t0 < timeout:
        try:
            st = piper.GetArmStatus().arm_status
            st_str = str(st.arm_status)
            if any(k in st_str for k in ["IDLE", "STANDBY", "READY", "FINISH"]):
                return True
        except Exception:
            pass

        cur_j = read_joints(piper)
        if joint_distance(cur_j, last_j) <= tol:
            if stable_since is None:
                stable_since = time.time()
            elif time.time() - stable_since >= settle_time:
                return True
        else:
            stable_since = None
        last_j = cur_j
        time.sleep(0.02)
    return False


# ---------------------------------------------------------------------------
# Motion primitives
# ---------------------------------------------------------------------------

def move_to_joint_waypoint(piper, target_j, speed, timeout=10.0, tol=3000,
                           soft_tol=7000):
    """Move to a single joint waypoint via MOVE_J. Returns True on success."""
    target_arr = np.array(target_j, dtype=np.float64)
    t0 = time.time()
    err = float("inf")
    status_str = ""
    while time.time() - t0 < timeout:
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        piper.JointCtrl(*target_j)
        arm_st = piper.GetArmStatus().arm_status
        status_str = str(arm_st.arm_status)
        if "LIMIT" in status_str:
            print(f"\n[arm_controller] ERROR: exceeds joint limits ({status_str})")
            return False
        cur = read_joints(piper)
        err = np.max(np.abs(np.array(cur, dtype=np.float64) - target_arr))
        if err < tol:
            return True
        time.sleep(0.01)
    if err < soft_tol and "LIMIT" not in status_str:
        return True
    return False


def move_to_joint_waypoint_record(piper, target_j, speed, waypoints,
                                  timeout=10.0, tol=3000, soft_tol=7000):
    """MOVE_J to target while recording joint waypoints."""
    target_arr = np.array(target_j, dtype=np.float64)
    t0 = time.time()
    err = float("inf")
    status_str = ""
    while time.time() - t0 < timeout:
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)
        piper.JointCtrl(*target_j)
        arm_st = piper.GetArmStatus().arm_status
        status_str = str(arm_st.arm_status)
        if "LIMIT" in status_str:
            print(f"\n[arm_controller] ERROR: exceeds joint limits ({status_str})")
            return False
        cur = read_joints(piper)
        if waypoints and joint_distance(cur, waypoints[-1]) >= WAYPOINT_MIN_DIST:
            waypoints.append(cur)
        err = np.max(np.abs(np.array(cur, dtype=np.float64) - target_arr))
        if err < tol:
            final_j = read_joints(piper)
            if not waypoints or joint_distance(final_j, waypoints[-1]) > 0:
                waypoints.append(final_j)
            return True
        time.sleep(0.01)
    if err < soft_tol and "LIMIT" not in status_str:
        return True
    return False


def move_joints_path(piper, start_j, target_j, speed, waypoints,
                     steps=10, timeout_per_step=6.0, tol=3000, soft_tol=4500):
    """Interpolate joint path to reduce limit/timeout issues."""
    start = np.array(start_j, dtype=np.float64)
    target = np.array(target_j, dtype=np.float64)
    for i in range(1, steps + 1):
        alpha = i / steps
        wp = (start + alpha * (target - start)).astype(np.int64).tolist()
        ok = move_to_joint_waypoint_record(
            piper, wp, speed, waypoints,
            timeout=timeout_per_step, tol=tol, soft_tol=soft_tol
        )
        if not ok:
            print(f"[arm_controller] MOVE_J step {i}/{steps} failed.")
            return False
    return True


def move_to_eef(piper, eef_raw, speed, waypoints, move_mode=0x00,
                timeout=10.0, tol_mm=5.0):
    """Move to an EEF position via MOVE_P/J/L, recording joint waypoints."""
    x, y, z, rx, ry, rz = eef_raw
    target_xyz = np.array([x * RAW_TO_MM, y * RAW_TO_MM, z * RAW_TO_MM])
    limit_count = 0
    t0 = time.time()
    grace_period = 2.0

    while time.time() - t0 < timeout:
        piper.MotionCtrl_2(0x01, move_mode, speed, 0x00)
        piper.EndPoseCtrl(x, y, z, rx, ry, rz)

        elapsed = time.time() - t0
        arm_st = piper.GetArmStatus().arm_status
        status_str = str(arm_st.arm_status)
        if elapsed > grace_period and "LIMIT" in status_str:
            limit_count += 1
            if limit_count >= 100:
                return False
        else:
            limit_count = 0

        cur_j = read_joints(piper)
        if waypoints and joint_distance(cur_j, waypoints[-1]) >= WAYPOINT_MIN_DIST:
            waypoints.append(cur_j)

        pose = piper.GetArmEndPoseMsgs().end_pose
        cur_xyz = np.array([pose.X_axis * RAW_TO_MM,
                            pose.Y_axis * RAW_TO_MM,
                            pose.Z_axis * RAW_TO_MM])
        error = np.linalg.norm(cur_xyz - target_xyz)
        if error < tol_mm:
            final_j = read_joints(piper)
            if not waypoints or joint_distance(final_j, waypoints[-1]) > 0:
                waypoints.append(final_j)
            return True
        time.sleep(0.01)
    return False
