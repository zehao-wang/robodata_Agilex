#!/usr/bin/env python3
"""Viser-based data collection for PIPER arm.

Opens a web GUI at http://localhost:8080 with:
- 3D arm visualization (URDF-based)
- Camera feeds (color + depth)
- Recording controls

Usage:
    python collect_viser.py --output_dir ./data

    # Without hardware (for testing GUI)
    python collect_viser.py --no-arm --no-camera
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Viser-based PIPER data collection")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Directory to save episode HDF5 files")
    parser.add_argument("--can-interface", type=str, default="gs_usb",
                        choices=["gs_usb", "socketcan"],
                        help="CAN bus interface type")
    parser.add_argument("--can-channel", type=str, default="can0",
                        help="CAN channel (for socketcan mode)")
    parser.add_argument("--bitrate", type=int, default=1_000_000,
                        help="CAN bus bitrate")
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser server port")
    parser.add_argument("--width", type=int, default=640,
                        help="Camera frame width")
    parser.add_argument("--height", type=int, default=480,
                        help="Camera frame height")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target capture frame rate")
    parser.add_argument("--no-camera", action="store_true",
                        help="Run without camera (dummy black frames)")
    parser.add_argument("--no-arm", action="store_true",
                        help="Run without arm (dummy zero state)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: no hardware required (implies --no-arm --no-camera)")
    args = parser.parse_args()

    # Demo mode enables all no-hardware flags
    if args.demo:
        args.no_arm = True
        args.no_camera = True
        print("[Demo Mode] Running without hardware - GUI and visualization only")

    # Import here to delay loading until args are parsed
    from storage.hdf5_writer import HDF5Writer
    from gui.viser_collector import ViserDataCollectorApp

    # Create arm reader (reads slave arm feedback from CAN bus)
    if args.no_arm:
        arm_reader = None
        print("[Arm] Skipped (--no-arm mode)")
    else:
        from robot.arm_reader import ArmReader
        arm_reader = ArmReader(
            can_interface=args.can_interface,
            can_channel=args.can_channel,
            bitrate=args.bitrate,
        )
        arm_reader.start()

    # Create camera
    if args.no_camera:
        camera = None
        print("[Camera] Skipped (--no-camera mode)")
    else:
        from camera.realsense import RealsenseCamera
        camera = RealsenseCamera(
            width=args.width,
            height=args.height,
            fps=args.fps,
        )
        camera.start()

    # Create writer
    writer = HDF5Writer(output_dir=args.output_dir)

    # Create and run the viser app
    app = ViserDataCollectorApp(
        arm_reader=arm_reader,
        camera=camera,
        writer=writer,
        port=args.port,
        fps=args.fps,
        frame_w=args.width,
        frame_h=args.height,
        demo_mode=args.demo,
    )

    try:
        app.run()
    finally:
        if arm_reader is not None:
            arm_reader.stop()
        if camera is not None:
            camera.stop()
        print("[collect_viser] Done.")


if __name__ == "__main__":
    main()
