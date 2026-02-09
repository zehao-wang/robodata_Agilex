#!/usr/bin/env python3
"""Interactive arm control via viser web GUI.

Opens a web GUI at http://localhost:8080 with:
- 3D arm visualization with URDF
- Interactive drag control (IK target handle)
- Manual EEF pose input
- Motion execution with reverse path safety

Usage:
    # With physical arm
    python control_arm.py

    # GUI-only mode (no physical arm)
    python control_arm.py --no-arm

    # Custom port
    python control_arm.py --port 8081 --speed 30
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Interactive PIPER arm control")
    parser.add_argument("--interface", choices=["gs_usb", "socketcan"],
                        default="gs_usb", help="CAN interface (default: gs_usb)")
    parser.add_argument("--channel", default="can0",
                        help="CAN channel for socketcan (default: can0)")
    parser.add_argument("--bitrate", type=int, default=1_000_000,
                        help="CAN bitrate (default: 1000000)")
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser server port (default: 8080)")
    parser.add_argument("--speed", type=int, default=50,
                        help="Default motion speed 1-100 (default: 50)")
    parser.add_argument("--no-arm", action="store_true",
                        help="Run without physical arm (GUI preview only)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: no hardware required (implies --no-arm)")
    args = parser.parse_args()

    # Demo mode enables no-hardware flags
    if args.demo:
        args.no_arm = True
        print("[Demo Mode] Running without hardware - IK preview and visualization only")

    from gui.arm_control_app import ArmControlApp

    piper = None
    if not args.no_arm:
        from robot.arm_controller import create_piper, enable_arm, wait_for_feedback
        import time

        print(f"[control_arm] Connecting via {args.interface}...")
        piper = create_piper(args.interface, args.channel, args.bitrate)
        print("[control_arm] Connected.")

        print("[control_arm] Waiting for arm feedback...")
        time.sleep(1.0)
        feedback = wait_for_feedback(piper, timeout=3.0)
        if feedback:
            print(f"[control_arm] Current EEF: x={feedback['x']:.1f} "
                  f"y={feedback['y']:.1f} z={feedback['z']:.1f} mm")
        else:
            print("[control_arm] WARNING: No pose feedback received.")

        print("[control_arm] Enabling arm...")
        if not enable_arm(piper, timeout=5.0):
            print("[control_arm] WARNING: Enable arm may have failed.")
        else:
            print("[control_arm] Arm enabled.")
    else:
        print("[control_arm] Running without physical arm (--no-arm mode)")

    app = ArmControlApp(
        piper=piper,
        port=args.port,
        speed=args.speed,
        demo_mode=args.demo,
    )

    try:
        app.run()
    finally:
        if piper is not None:
            try:
                piper.DisconnectPort()
            except Exception:
                pass
        print("[control_arm] Done.")


if __name__ == "__main__":
    main()
