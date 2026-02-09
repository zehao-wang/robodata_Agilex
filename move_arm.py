"""Move the PIPER arm end-effector to a target 3D position.

Uses the official piper_sdk (C_PiperInterface_V2) with macOS gs_usb support.
Now uses pyroki for IK solving and extracted motion primitives.

Flow:
  1. Record initial joint positions
  2. Move to zero/home pose (EEF home by default; optional joint home)
  3. Solve IK for target EEF -> joint positions (pyroki IK)
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
import time

import numpy as np

from robot.arm_controller import (
    create_piper,
    enable_arm,
    wait_for_feedback,
    read_joints,
    joint_distance,
    wait_motion_done,
    move_to_joint_waypoint,
    move_to_joint_waypoint_record,
    move_joints_path,
    move_to_eef,
    list_can_devices,
    MM_TO_RAW,
    DEG_TO_RAW,
    RAW_TO_MM,
    RAW_TO_DEG,
    RAW_TO_RAD,
    RAD_TO_RAW,
    HOME_EEF,
)


def solve_ik_pyroki(target_mm_deg, seed_j_raw=None):
    """Solve IK using pyroki. Returns joints in raw units (0.001deg) or None."""
    try:
        from solver.pyroki_ik import PiperIKSolver
        from utils.urdf_loader import euler_deg_to_wxyz
    except ImportError as e:
        print(f"[move_arm] pyroki IK not available: {e}")
        return None

    x_mm, y_mm, z_mm, rx_deg, ry_deg, rz_deg = target_mm_deg
    target_pos = np.array([x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0])  # to meters
    target_wxyz = euler_deg_to_wxyz(rx_deg, ry_deg, rz_deg)

    solver = PiperIKSolver()
    cfg = solver.solve_from_can(target_pos, target_wxyz)
    if cfg is not None:
        # Convert radians to raw units
        joints_raw = [int(round(v * RAD_TO_RAW)) for v in cfg]
        return joints_raw
    return None


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

    waypoints = []
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

        # --- Phase 1: Go to home ---
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

        # Transition through STANDBY
        piper.MotionCtrl_2(0x00, 0x00, 0, 0x00)
        time.sleep(0.5)
        for _ in range(100):
            piper.MotionCtrl_2(0x00, 0x00, 0, 0x00)
            st = piper.GetArmStatus().arm_status
            if "LIMIT" not in str(st.arm_status):
                break
            time.sleep(0.02)

        # --- Phase 2: Home -> Target via IK ---
        print(f"[move_arm] Phase 2: Solve IK for target, then MOVE_J, "
              f"speed={args.speed}%")

        target_mm_deg = (args.x, args.y, args.z, args.rx, args.ry, args.rz)

        seed_j = read_joints(piper)
        target_joints = solve_ik_pyroki(target_mm_deg, seed_j)

        if target_joints is None:
            print("\n[move_arm] ERROR: IK failed. Target may be unreachable.")
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
        try:
            cur_j = read_joints(piper)
            if waypoints and joint_distance(cur_j, waypoints[-1]) > 0:
                waypoints.append(cur_j)
        except Exception:
            pass
        print("\n[move_arm] Interrupted by user.")
    finally:
        # --- Reverse path ---
        returned_home = False
        try:
            piper.MotionCtrl_2(0x00, 0x00, 0, 0x00)
            time.sleep(0.5)
            if len(waypoints) > 1:
                reversed_wps = list(reversed(waypoints))
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
                        if i >= home_rev_idx:
                            print("\n[move_arm] Home reached on reverse path.")
                            break
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
                cur_j = read_joints(piper)
                if joint_distance(cur_j, waypoints[0]) < 5000:
                    returned_home = True
                    print(f"\n[move_arm] Return complete.")
                else:
                    print(f"\n[move_arm] Return inaccurate "
                          f"(max joint err: "
                          f"{joint_distance(cur_j, waypoints[0]) * RAW_TO_DEG:.1f} deg)")
            elif len(waypoints) <= 1:
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
