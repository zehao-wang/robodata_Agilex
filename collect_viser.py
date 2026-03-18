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
    parser.add_argument("--output_dir", type=str, default="./data/records",
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
    parser.add_argument("--camera-backend", type=str, default="auto",
                        choices=["auto", "opencv", "realsense", "zed"],
                        help="Preferred camera backend")
    parser.add_argument("--max-sync-dt-ms", type=float, default=50.0,
                        help="Maximum allowed arm/camera timestamp delta in milliseconds")
    parser.add_argument("--streams", type=str, default="rgb",
                        choices=["rgb", "depth", "rgbd"],
                        help="Camera streams: rgb, depth, or rgbd (default: rgb)")
    parser.add_argument("--no-camera", action="store_true",
                        help="Run without camera (dummy black frames)")
    parser.add_argument("--no-arm", action="store_true",
                        help="Run without arm (dummy zero state)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: no hardware required (implies --no-arm --no-camera)")
    parser.add_argument("--world-config", type=str, default="./data/world_config.json",
                        help="Path to world frame calibration JSON")
    args = parser.parse_args()

    # Demo mode enables all no-hardware flags
    if args.demo:
        args.no_arm = True
        args.no_camera = True
        print("[Demo Mode] Running without hardware - GUI and visualization only")
    elif args.camera_backend == "zed":
        args.width = 1920
        args.height = 1080
        args.fps = 30
        args.streams = "rgb"

    # Import here to delay loading until args are parsed
    from storage.hdf5_writer import HDF5Writer
    from gui.viser_collector import ViserDataCollectorApp

    # Create arm reader
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
        camera_sources = []
        print("[Camera] Skipped (--no-camera mode)")
    else:
        from camera import CameraManager

        camera = CameraManager(
            width=args.width,
            height=args.height,
            fps=args.fps,
            streams=args.streams,
            preferred_backend=args.camera_backend,
        )
        camera_sources = camera.sources
        print(f"[Camera] Discovered {len(camera_sources)} candidate source(s)")
        for source in camera_sources:
            print(f"  - {source.label}")
        active_source_id = None
        if args.camera_backend != "auto":
            preferred_source = next(
                (source for source in camera_sources if source.backend == args.camera_backend),
                None,
            )
            if preferred_source is None:
                print(f"[Camera] Preferred backend '{args.camera_backend}' not found")
            else:
                if camera.select_camera(preferred_source.source_id):
                    active_source_id = preferred_source.source_id
                else:
                    print(f"[Camera] Failed to start preferred backend '{args.camera_backend}'")
        if active_source_id is None:
            active_source_id = camera.start_first_available()
        if active_source_id is None:
            print("[Camera] No available camera could be started, using blank frames")
        else:
            active_label = camera.source_id_to_label()[active_source_id]
            print(f"[Camera] Auto-selected: {active_label}")
            camera_info = camera.get_camera_info()
            if camera_info is not None:
                writer_info = {
                    key: value
                    for key, value in camera_info.items()
                }
                print(f"[Camera] Params: {writer_info}")
                resolution = camera_info.get("resolution")
                if isinstance(resolution, dict):
                    args.width = int(resolution.get("width", args.width))
                    args.height = int(resolution.get("height", args.height))
            else:
                writer_info = None
        if active_source_id is None:
            writer_info = None

    # Create writer
    writer = HDF5Writer(output_dir=args.output_dir)
    if not args.no_camera:
        writer.set_camera_info(writer_info)

    # Load world frame config
    from utils.world_frame import load_world_config
    world_config = load_world_config(args.world_config)
    if world_config is not None:
        print(f"[WorldFrame] Loaded calibration from {args.world_config}")
    else:
        print(f"[WorldFrame] No calibration found at {args.world_config} — using base frame")

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
        world_config=world_config,
        streams=args.streams,
        output_dir=args.output_dir,
        camera_sources=camera_sources,
        max_sync_dt_ms=args.max_sync_dt_ms,
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
