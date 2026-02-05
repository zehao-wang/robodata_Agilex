"""Move the PIPER arm end-effector to a target 3D position.

Uses the official piper_sdk (C_PiperInterface_V2) with macOS gs_usb support.

Flow:
  1. Record initial joint positions
  2. Move to zero/home pose (EEF home by default; optional joint home)
  3. Solve IK for target EEF -> joint positions (SDK IK)
  4. MOVE_J to target joints
  5. Record joint waypoints during all forward motion
  6. Return to home via recorded reverse path
  7. Return to initial pose via recorded reverse path

Usage:
    python move_arm.py --x 57 --y 0 --z 215 --rx 0 --ry 85 --rz 0
    python move_arm.py --x 57 --y 0 --z 215 --rx 0 --ry 85 --rz 0 --speed 30
    python move_arm.py --x 57 --y 0 --z 215 --rx 0 --ry 85 --rz 0 --home-joints "0,0,0,0,0,0"
    python move_arm.py --list-devices

Units: position in mm, rotation in degrees.
"""

import argparse
import platform
import time
import inspect
import math

import can
import numpy as np


# ---------------------------------------------------------------------------
# macOS gs_usb monkey-patch
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
# Helpers
# ---------------------------------------------------------------------------

MM_TO_RAW = 1000    # mm -> 0.001mm raw unit
DEG_TO_RAW = 1000   # deg -> 0.001deg raw unit
RAW_TO_MM = 0.001
RAW_TO_DEG = 0.001
RAW_TO_RAD = math.pi / 180.0 / 1000.0
RAD_TO_RAW = 1.0 / RAW_TO_RAD

# Minimum joint-space distance (0.001deg units) between recorded waypoints.
# ~2 degrees — avoids recording redundant points while stationary.
WAYPOINT_MIN_DIST = 2000

# Safe EEF home position (raw units: 0.001mm / 0.001deg).
# This is the official demo position — well within joint limits.
HOME_EEF = (57000, 0, 215000, 0, 85000, 0)  # x, y, z, rx, ry, rz

# Debug flag (set from CLI)
IK_DEBUG = False


def debug_print(msg):
    if IK_DEBUG:
        print(msg)


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


def move_to_joint_waypoint(piper, target_j, speed, timeout=10.0, tol=3000,
                           soft_tol=7000):
    """Move to a single joint waypoint via MOVE_J. Returns True on success."""
    target_arr = np.array(target_j, dtype=np.float64)
    t0 = time.time()
    while time.time() - t0 < timeout:
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)  # MOVE_J
        piper.JointCtrl(*target_j)
        arm_st = piper.GetArmStatus().arm_status
        status_str = str(arm_st.arm_status)
        if "LIMIT" in status_str:
            print(f"\n[move_arm] ERROR: exceeds joint limits ({status_str})")
            return False
        cur = read_joints(piper)
        err = np.max(np.abs(np.array(cur, dtype=np.float64) - target_arr))
        if err < tol:
            return True
        time.sleep(0.01)
    if err < soft_tol and "LIMIT" not in status_str:
        print(f"\n[move_arm] WARN: MOVE_J timeout but within soft tol "
              f"(max_err={err * RAW_TO_DEG:.2f} deg, status={status_str})")
        return True
    print(f"\n[move_arm] ERROR: MOVE_J timeout "
          f"(max_err={err * RAW_TO_DEG:.2f} deg, status={status_str})")
    return False


def move_to_joint_waypoint_record(piper, target_j, speed, waypoints,
                                  timeout=10.0, tol=3000, soft_tol=7000):
    """MOVE_J to target while recording joint waypoints."""
    target_arr = np.array(target_j, dtype=np.float64)
    t0 = time.time()
    while time.time() - t0 < timeout:
        piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)  # MOVE_J
        piper.JointCtrl(*target_j)
        arm_st = piper.GetArmStatus().arm_status
        status_str = str(arm_st.arm_status)
        if "LIMIT" in status_str:
            print(f"\n[move_arm] ERROR: exceeds joint limits ({status_str})")
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
        print(f"\n[move_arm] WARN: MOVE_J timeout but within soft tol "
              f"(max_err={err * RAW_TO_DEG:.2f} deg, status={status_str})")
        return True
    cur_j = read_joints(piper)
    cur_deg = [v * RAW_TO_DEG for v in cur_j]
    tgt_deg = [v * RAW_TO_DEG for v in target_j]
    print(f"\n[move_arm] ERROR: MOVE_J timeout "
          f"(max_err={err * RAW_TO_DEG:.2f} deg, status={status_str})")
    print("[move_arm] Current joints (deg): "
          + " ".join(f"{v:.1f}" for v in cur_deg))
    print("[move_arm] Target  joints (deg): "
          + " ".join(f"{v:.1f}" for v in tgt_deg))
    return False


def move_joints_path(piper, start_j, target_j, speed, waypoints,
                     steps=10, timeout_per_step=6.0, tol=3000, soft_tol=4500):
    """Interpolate joint path to reduce limit/timeout issues."""
    start = np.array(start_j, dtype=np.float64)
    target = np.array(target_j, dtype=np.float64)
    for i in range(1, steps + 1):
        alpha = i / steps
        wp = (start + alpha * (target - start)).astype(np.int64).tolist()
        wp_deg = [v * RAW_TO_DEG for v in wp]
        print(f"[move_arm] MOVE_J step {i}/{steps}: "
              + " ".join(f"{v:.1f}" for v in wp_deg))
        ok = move_to_joint_waypoint_record(
            piper, wp, speed, waypoints,
            timeout=timeout_per_step, tol=tol, soft_tol=soft_tol
        )
        if not ok:
            print(f"[move_arm] MOVE_J step {i} failed.")
            return False
    return True


def enable_arm(piper, timeout=5.0):
    """Enable arm with timeout to avoid infinite retries."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        if piper.EnablePiper():
            return True
        time.sleep(0.05)
    return False


def wait_motion_done(piper, timeout=5.0, settle_time=0.5, tol=200):
    """Wait for motion complete signal (status or joint settle)."""
    t0 = time.time()
    last_j = read_joints(piper)
    stable_since = None

    while time.time() - t0 < timeout:
        # Try to detect idle/standby status
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


def _call_ik_fn(fn, eef_raw, eef_mmdeg, seed_j):
    """Try calling an IK function with common signatures."""
    x_raw, y_raw, z_raw, rx_raw, ry_raw, rz_raw = eef_raw
    x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = eef_mmdeg
    candidates = [
        (x_raw, y_raw, z_raw, rx_raw, ry_raw, rz_raw),
        (x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg),
    ]
    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        pass

    for args in candidates:
        # Try without seed
        try:
            if sig is None or len(sig.parameters) in (6, 0):
                return fn(*args)
        except Exception:
            pass
        # Try with seed
        if seed_j is not None:
            try:
                if sig is None or len(sig.parameters) in (7, 0):
                    return fn(*args, seed_j)
            except Exception:
                pass
    return None


def solve_ik(piper, eef_raw, seed_j=None):
    """Solve IK via SDK if available. Returns joint list or None."""
    eef_mmdeg = (
        eef_raw[0] * RAW_TO_MM,
        eef_raw[1] * RAW_TO_MM,
        eef_raw[2] * RAW_TO_MM,
        eef_raw[3] * RAW_TO_DEG,
        eef_raw[4] * RAW_TO_DEG,
        eef_raw[5] * RAW_TO_DEG,
    )
    ik_method_names = [
        "GetInverseKinematics",
        "InverseKinematics",
        "SolveIK",
        "GetIK",
        "IK",
        "CalcIK",
        "GetIKSolution",
        "GetArmInverseKinematics",
    ]
    for name in ik_method_names:
        fn = getattr(piper, name, None)
        if fn is None:
            continue
        result = _call_ik_fn(fn, eef_raw, eef_mmdeg, seed_j)
        if result is None:
            continue
        # Normalize possible return types
        if isinstance(result, (list, tuple)) and len(result) >= 6:
            joints = list(result)[:6]
            if max(abs(v) for v in joints) <= 360:
                joints = [v * DEG_TO_RAW for v in joints]
            return [int(round(v)) for v in joints]
        if hasattr(result, "joint"):
            j = getattr(result, "joint")
            if isinstance(j, (list, tuple)) and len(j) >= 6:
                joints = list(j)[:6]
                if max(abs(v) for v in joints) <= 360:
                    joints = [v * DEG_TO_RAW for v in joints]
                return [int(round(v)) for v in joints]
    return None


def _fk_pose_from_joints(fk, joints_rad):
    """Return EEF pose [x,y,z,rx,ry,rz] from FK (mm/deg)."""
    pose_all = fk.CalFK(joints_rad)
    return pose_all[-1]


def solve_ik_numerical(eef_raw, seed_j_raw, verbose=True, allow_pos_only_fallback=True):
    """Numerical IK using SDK forward kinematics. Returns joints in raw units."""
    from piper_sdk import C_PiperForwardKinematics

    target = np.array([
        eef_raw[0] * RAW_TO_MM,
        eef_raw[1] * RAW_TO_MM,
        eef_raw[2] * RAW_TO_MM,
        eef_raw[3] * RAW_TO_DEG,
        eef_raw[4] * RAW_TO_DEG,
        eef_raw[5] * RAW_TO_DEG,
    ], dtype=np.float64)

    fk = C_PiperForwardKinematics()  # uses DH offset by default
    q = np.array(seed_j_raw, dtype=np.float64) * RAW_TO_RAD

    max_iters = 400
    tol_pos = 1.5   # mm
    tol_rot = 3.0   # deg
    damp = 0.4
    step_limit = 0.1  # rad
    rot_w = 0.3  # deg -> mm weight for balancing
    dq_eps = 1e-3

    def _solve(target_vec):
        q_local = q.copy()
        for _ in range(max_iters):
            cur = np.array(_fk_pose_from_joints(fk, q_local.tolist()),
                           dtype=np.float64)
            err = target_vec - cur
            pos_err = np.linalg.norm(err[:3])
            rot_err = np.linalg.norm(err[3:])
            if pos_err < tol_pos and rot_err < tol_rot:
                return q_local, pos_err, rot_err

            # Weighted error vector
            e = err.copy()
            e[3:] *= rot_w

            # Numerical Jacobian (6x6)
            J = np.zeros((6, 6), dtype=np.float64)
            for i in range(6):
                q_pert = q_local.copy()
                q_pert[i] += dq_eps
                pose_pert = np.array(_fk_pose_from_joints(fk, q_pert.tolist()),
                                     dtype=np.float64)
                d = (pose_pert - cur) / dq_eps
                d[3:] *= rot_w
                J[:, i] = d

            # Damped least squares: dq = J^T (J J^T + λ^2 I)^-1 e
            A = J @ J.T + (damp ** 2) * np.eye(6)
            try:
                dq = J.T @ np.linalg.solve(A, e)
            except np.linalg.LinAlgError:
                dq = np.linalg.lstsq(J, e, rcond=None)[0]

            dq_norm = np.linalg.norm(dq)
            if dq_norm > step_limit:
                dq = dq * (step_limit / dq_norm)
            q_local = q_local + dq
        return None, pos_err, rot_err

    sol, pos_err, rot_err = _solve(target)
    if sol is not None:
        if verbose:
            print(f"[move_arm] IK solved: pos_err={pos_err:.2f}mm "
                  f"rot_err={rot_err:.2f}deg")
        return [int(round(v * RAD_TO_RAW)) for v in sol]

    if allow_pos_only_fallback:
        # fallback: keep current orientation, solve position only
        cur_pose = np.array(_fk_pose_from_joints(fk, q.tolist()),
                            dtype=np.float64)
        target_pos_only = target.copy()
        target_pos_only[3:] = cur_pose[3:]
        sol2, pos_err2, rot_err2 = _solve(target_pos_only)
        if sol2 is not None:
            if verbose:
                print(f"[move_arm] IK pos-only solved: pos_err={pos_err2:.2f}mm "
                      f"rot_err={rot_err2:.2f}deg (orientation held)")
            return [int(round(v * RAD_TO_RAW)) for v in sol2]

    if verbose:
        print(f"[move_arm] IK failed: pos_err={pos_err:.2f}mm "
              f"rot_err={rot_err:.2f}deg")
    return None


def move_to_eef(piper, eef_raw, speed, waypoints, move_mode=0x00,
                timeout=10.0, tol_mm=5.0):
    """Move to an EEF position via MOVE_P/J/L, recording joint waypoints.

    Args:
        eef_raw: tuple of (x, y, z, rx, ry, rz) in raw units (0.001mm/0.001deg)
        move_mode: 0x00=MOVE_P, 0x01=MOVE_J, 0x02=MOVE_L
        waypoints: list to append recorded joint waypoints to

    Returns True if target reached within tolerance.
    """
    x, y, z, rx, ry, rz = eef_raw
    target_xyz = np.array([x * RAW_TO_MM, y * RAW_TO_MM, z * RAW_TO_MM])
    limit_count = 0
    t0 = time.time()
    grace_period = 2.0  # ignore LIMIT for first 2s while firmware plans

    while time.time() - t0 < timeout:
        piper.MotionCtrl_2(0x01, move_mode, speed, 0x00)
        piper.EndPoseCtrl(x, y, z, rx, ry, rz)

        # Check limits (only after grace period)
        elapsed = time.time() - t0
        arm_st = piper.GetArmStatus().arm_status
        status_str = str(arm_st.arm_status)
        if elapsed > grace_period and "LIMIT" in status_str:
            limit_count += 1
            if limit_count >= 100:
                print(f"\n[move_arm] ERROR: exceeds joint limits ({status_str})")
                return False
        else:
            limit_count = 0

        # Record waypoint
        cur_j = read_joints(piper)
        if waypoints and joint_distance(cur_j, waypoints[-1]) >= WAYPOINT_MIN_DIST:
            waypoints.append(cur_j)

        # Check convergence
        pose = piper.GetArmEndPoseMsgs().end_pose
        cur_xyz = np.array([pose.X_axis * RAW_TO_MM,
                            pose.Y_axis * RAW_TO_MM,
                            pose.Z_axis * RAW_TO_MM])
        error = np.linalg.norm(cur_xyz - target_xyz)
        print(f"\r[move_arm] Error: {error:.1f} mm  "
              f"(x={cur_xyz[0]:.1f} y={cur_xyz[1]:.1f} z={cur_xyz[2]:.1f})",
              end="", flush=True)
        if error < tol_mm:
            final_j = read_joints(piper)
            if not waypoints or joint_distance(final_j, waypoints[-1]) > 0:
                waypoints.append(final_j)
            print()  # newline after \r progress
            return True

        time.sleep(0.01)
    print()  # newline after \r progress
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Move PIPER arm end-effector to a target 3D position."
    )
    parser.add_argument("--x", type=float, default=None, help="Target X (mm)")
    parser.add_argument("--y", type=float, default=None, help="Target Y (mm)")
    parser.add_argument("--z", type=float, default=None, help="Target Z (mm)")
    parser.add_argument("--rx", type=float, default=0.0, help="Target RX (degrees, default: 0)")
    parser.add_argument("--ry", type=float, default=85.0, help="Target RY (degrees, default: 85)")
    parser.add_argument("--rz", type=float, default=0.0, help="Target RZ (degrees, default: 0)")
    parser.add_argument("--speed", type=int, default=50,
                        help="Speed percentage 0-100 (default: 50)")
    parser.add_argument("--joint-steps", type=int, default=20,
                        help="Number of joint interpolation steps (default: 20)")
    parser.add_argument("--step-timeout", type=float, default=10.0,
                        help="Timeout per joint step in seconds (default: 10.0)")
    parser.add_argument("--joint-tol-deg", type=float, default=5.0,
                        help="Joint tolerance in degrees (default: 5.0)")
    parser.add_argument("--joint-soft-tol-deg", type=float, default=7.0,
                        help="Soft joint tolerance in degrees (default: 7.0)")
    parser.add_argument("--home-joints", default=None,
                        help="Home joint pose (deg), comma-separated 6 values. "
                             "If set, home motion uses MOVE_J.")
    parser.add_argument("--interface", choices=["gs_usb", "socketcan"],
                        default="gs_usb", help="CAN interface (default: gs_usb)")
    parser.add_argument("--channel", default="can0",
                        help="CAN channel for socketcan (default: can0)")
    parser.add_argument("--bitrate", type=int, default=1_000_000,
                        help="CAN bitrate (default: 1000000)")
    parser.add_argument("--timeout", type=float, default=15.0,
                        help="Motion timeout in seconds (default: 15)")
    parser.add_argument("--tolerance", type=float, default=5.0,
                        help="Position tolerance in mm (default: 5.0)")
    parser.add_argument("--list-devices", action="store_true",
                        help="List USB CAN adapters and exit.")
    args = parser.parse_args()

    if args.list_devices:
        try:
            devices = list_can_devices()
        except ImportError:
            print("pyusb not installed. Cannot enumerate USB devices.")
            return
        if not devices:
            print("No candleLight USB CAN adapters found.")
        else:
            print(f"Found {len(devices)} adapter(s):")
            for i, d in enumerate(devices):
                print(f"  [{i}] bus={d.bus} address={d.address}")
        return

    if args.x is None or args.y is None or args.z is None:
        parser.error("--x, --y, --z are required for movement commands.")

    # Connect
    print(f"[move_arm] Connecting via {args.interface}...")
    piper = create_piper(args.interface, args.channel, args.bitrate)
    print("[move_arm] Connected.")

    waypoints = []  # recorded joint waypoints during forward motion
    reached_target = False
    home_wp_index = None

    try:
        # Wait for feedback
        print("[move_arm] Waiting for arm feedback...")
        time.sleep(1.0)

        current = wait_for_feedback(piper, timeout=3.0)
        if current is None:
            print("[move_arm] WARNING: No pose feedback received.")
        else:
            print(f"[move_arm] Current EEF: "
                  f"x={current['x']:.1f} y={current['y']:.1f} "
                  f"z={current['z']:.1f} mm")

        # Record initial joint positions (first waypoint)
        init_j = read_joints(piper)
        waypoints.append(init_j)
        print(f"[move_arm] Initial joints recorded")

        print(f"[move_arm] Target: x={args.x:.1f} y={args.y:.1f} "
              f"z={args.z:.1f} mm, rx={args.rx:.1f} ry={args.ry:.1f} "
              f"rz={args.rz:.1f} deg")

        # Enable
        print("[move_arm] Enabling arm...")
        if not enable_arm(piper, timeout=5.0):
            print("[move_arm] ERROR: Enable arm failed (CAN send not OK).")
            print("[move_arm] Please check CAN connection and retry.")
            return
        print("[move_arm] Arm enabled.")

        # --- Phase 1: Go to home (EEF home by default or joint home if provided) ---
        if args.home_joints:
            home_joints = [int(round(float(v) * DEG_TO_RAW))
                           for v in args.home_joints.split(",")]
            if len(home_joints) != 6:
                raise ValueError("--home-joints must have 6 comma-separated values.")
            print("[move_arm] Phase 1: Moving to home joints (MOVE_J)...")
            ok = move_to_joint_waypoint_record(
                piper, home_joints, args.speed, waypoints,
                timeout=args.timeout, tol=3000
            )
        else:
            print("[move_arm] Phase 1: Moving to home "
                  f"(EEF {HOME_EEF[0]*RAW_TO_MM:.0f}, "
                  f"{HOME_EEF[1]*RAW_TO_MM:.0f}, "
                  f"{HOME_EEF[2]*RAW_TO_MM:.0f} mm)...")
            ok = move_to_eef(piper, HOME_EEF, args.speed, waypoints,
                             move_mode=0x00, timeout=args.timeout,
                             tol_mm=args.tolerance)
        if ok:
            print("[move_arm] Home position reached.")
        else:
            print("[move_arm] WARNING: Home position not fully reached.")
            cur_j = read_joints(piper)
            if waypoints and joint_distance(cur_j, waypoints[-1]) > 0:
                waypoints.append(cur_j)
        home_wp_index = len(waypoints) - 1

        # Transition through STANDBY and wait for status to clear
        piper.MotionCtrl_2(0x00, 0x00, 0, 0x00)
        time.sleep(0.5)
        # Keep sending STANDBY until LIMIT status clears
        for _ in range(100):
            piper.MotionCtrl_2(0x00, 0x00, 0, 0x00)
            st = piper.GetArmStatus().arm_status
            if "LIMIT" not in str(st.arm_status):
                break
            time.sleep(0.02)

        # --- Phase 2: Home -> Target via IK (joint control) ---
        print(f"[move_arm] Phase 2: Solve IK for target, then MOVE_J, "
              f"speed={args.speed}%")
        target_eef_raw = (
            int(round(args.x * MM_TO_RAW)),
            int(round(args.y * MM_TO_RAW)),
            int(round(args.z * MM_TO_RAW)),
            int(round(args.rx * DEG_TO_RAW)),
            int(round(args.ry * DEG_TO_RAW)),
            int(round(args.rz * DEG_TO_RAW)),
        )

        seed_j = read_joints(piper)
        target_joints = solve_ik(piper, target_eef_raw, seed_j=seed_j)
        if target_joints is None:
            print("[move_arm] SDK IK not available; using numerical IK...")
            target_joints = solve_ik_numerical(target_eef_raw, seed_j)
        if target_joints is None:
            print("\n[move_arm] ERROR: IK failed or SDK does not expose IK.")
        else:
            print("[move_arm] Target joints (deg): "
                  + " ".join(f"{v * RAW_TO_DEG:.1f}" for v in target_joints))
            ok = move_joints_path(
                piper, read_joints(piper), target_joints, args.speed, waypoints,
                steps=args.joint_steps, timeout_per_step=args.step_timeout,
                tol=int(round(args.joint_tol_deg * DEG_TO_RAW)),
                soft_tol=int(round(args.joint_soft_tol_deg * DEG_TO_RAW))
            )
            if ok:
                reached_target = True
                print("[move_arm] Target reached. Waiting for motion done...")
                wait_motion_done(piper, timeout=5.0)
            else:
                print("\n[move_arm] Phase 2 failed (timeout or limit).")
        print(f"[move_arm] Recorded {len(waypoints)} waypoints.")

    except KeyboardInterrupt:
        # Record current position for safe return
        try:
            cur_j = read_joints(piper)
            if waypoints and joint_distance(cur_j, waypoints[-1]) > 0:
                waypoints.append(cur_j)
        except Exception:
            pass
        print("\n[move_arm] Interrupted by user.")
    finally:
        # --- Reverse path: replay waypoints in reverse order ---
        returned_home = False
        try:
            # Transition through STANDBY before reverse motion
            piper.MotionCtrl_2(0x00, 0x00, 0, 0x00)
            time.sleep(0.5)
            if len(waypoints) > 1:
                reversed_wps = list(reversed(waypoints))
                # Stage A: return to home waypoint (if recorded)
                if home_wp_index is not None:
                    print(f"[move_arm] Returning to home via reverse path...")
                    home_rev_idx = len(waypoints) - 1 - home_wp_index
                    for i, wp in enumerate(reversed_wps):
                        print(f"\r[move_arm] Waypoint {i+1}/{len(reversed_wps)}",
                              end="", flush=True)
                        ok = move_to_joint_waypoint(piper, wp, args.speed,
                                                    timeout=args.timeout,
                                                    tol=3000)
                        if not ok:
                            print(f"\n[move_arm] Waypoint {i+1} timeout, "
                                  f"continuing...")
                        # Stop once we've reached the recorded home waypoint
                        if i >= home_rev_idx:
                            print("\n[move_arm] Home reached on reverse path.")
                            break
                    # Continue remaining waypoints to return to initial pose
                    remaining = list(reversed(waypoints[:home_wp_index]))
                    if remaining:
                        print("[move_arm] Returning to initial via reverse path...")
                        for i, wp in enumerate(remaining):
                            print(f"\r[move_arm] Waypoint {i+1}/{len(remaining)}",
                                  end="", flush=True)
                            ok = move_to_joint_waypoint(piper, wp, args.speed,
                                                        timeout=args.timeout,
                                                        tol=3000)
                            if not ok:
                                print(f"\n[move_arm] Waypoint {i+1} timeout, "
                                      f"continuing...")
                else:
                    print(f"[move_arm] Returning via {len(reversed_wps)} "
                          f"waypoints (reverse path)...")
                    for i, wp in enumerate(reversed_wps):
                        print(f"\r[move_arm] Waypoint {i+1}/{len(reversed_wps)}",
                              end="", flush=True)
                        ok = move_to_joint_waypoint(piper, wp, args.speed,
                                                    timeout=args.timeout,
                                                    tol=3000)
                        if not ok:
                            print(f"\n[move_arm] Waypoint {i+1} timeout, "
                                  f"continuing...")
                # Verify we are back at the initial joint position
                cur_j = read_joints(piper)
                if joint_distance(cur_j, waypoints[0]) < 5000:  # 5 deg
                    returned_home = True
                    print(f"\n[move_arm] Return complete.")
                else:
                    print(f"\n[move_arm] Return inaccurate "
                          f"(max joint err: "
                          f"{joint_distance(cur_j, waypoints[0]) * RAW_TO_DEG:.1f} deg)")
            elif len(waypoints) <= 1:
                # Never moved, safe to disable
                returned_home = True
        except Exception as e:
            print(f"\n[move_arm] Return path error: {e}")

        if returned_home:
            print("[move_arm] Returned to initial pose.")
        else:
            print("[move_arm] WARNING: Arm did NOT return to initial pose!")
        print("[move_arm] Motors kept ENABLED (holding position).")

        try:
            piper.DisconnectPort()
        except Exception:
            pass
        print("[move_arm] Done.")


if __name__ == "__main__":
    main()
