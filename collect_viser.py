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
from pathlib import Path

from camera_ipc import DEFAULT_POINTGREY_SHM_PREFIX, DEFAULT_POINTGREY_SOCKET_PATH


def main():
    parser = argparse.ArgumentParser(description="Viser-based PIPER data collection")
    parser.add_argument("--output_dir", type=str, default="./data/records",
                        help="Directory to save episode HDF5 files")
    parser.add_argument("--can-interface", type=str, default="socketcan",
                        choices=["gs_usb", "socketcan"],
                        help="CAN bus interface type")
    parser.add_argument("--can-channel", type=str, default="can0",
                        help="CAN channel (for socketcan mode)")
    parser.add_argument("--bitrate", type=int, default=1_000_000,
                        help="CAN bus bitrate")
    parser.add_argument("--arm-backend", type=str, default="http",
                        choices=["http", "sdk", "can"],
                        help=(
                            "Robot feedback backend. 'http' polls the PushT "
                            "service API; 'sdk' uses local piper_sdk; 'can' "
                            "uses the raw python-can decoder."
                        ))
    parser.add_argument("--arm-server", type=str, default="http://127.0.0.1:8012",
                        help="PushT service API URL for --arm-backend http")
    parser.add_argument("--arm-http-connect", dest="arm_http_connect", action="store_true",
                        default=True,
                        help=(
                            "Ask the PushT service client to connect the arm at startup "
                            "(default for --arm-backend http)."
                        ))
    parser.add_argument("--no-arm-http-connect", dest="arm_http_connect", action="store_false",
                        help=(
                            "Do not connect the PushT service arm at startup; only read "
                            "from an already connected service."
                        ))
    parser.add_argument("--arm-poll-hz", type=float, default=30.0,
                        help="Arm feedback polling rate for http/sdk backends")
    parser.add_argument("--arm-profile", type=str, default="auto",
                        choices=["auto", "default", "master", "master-control",
                                 "master-feedback", "slave", "arm3"],
                        help=(
                            "CAN ID profile for arm feedback. 'auto' listens for "
                            "the common feedback/control pose and joint IDs."
                        ))
    parser.add_argument("--port", type=int, default=8080,
                        help="Viser server port")
    parser.add_argument("--control-api-host", type=str, default="127.0.0.1",
                        help="Host for the external trajectory control HTTP API")
    parser.add_argument("--control-api-port", type=int, default=8765,
                        help="Port for the external trajectory control HTTP API")
    parser.add_argument("--width", type=int, default=2448,
                        help="Camera frame width")
    parser.add_argument("--height", type=int, default=2048,
                        help="Camera frame height")
    parser.add_argument("--fps", type=int, default=30,
                        help="Target capture frame rate")
    parser.add_argument("--camera-backend", type=str, default="pointgrey",
                        choices=["auto", "opencv", "realsense", "pointgrey", "zed"],
                        help="Preferred camera backend")
    parser.add_argument("--pointgrey-python", type=str, default=None,
                        help="Python interpreter for the standalone PointGrey capture service")
    parser.add_argument("--pointgrey-service-script", type=str, default=None,
                        help="Path to pointgrey_capture_service.py (defaults to repo copy)")
    parser.add_argument("--pointgrey-socket", type=str, default=DEFAULT_POINTGREY_SOCKET_PATH,
                        help="Unix socket path used by the PointGrey capture service")
    parser.add_argument("--pointgrey-shm-prefix", type=str, default=DEFAULT_POINTGREY_SHM_PREFIX,
                        help="Shared-memory name prefix used by the PointGrey capture service")
    parser.add_argument("--pointgrey-serial", type=str, default=None,
                        help="Optional PointGrey camera serial to open in the external service")
    parser.add_argument("--pointgrey-calibration", type=str, default=None,
                        help="Path to a saved PointGrey intrinsics JSON to merge into camera_info")
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
    parser.add_argument("--inference-server", type=str, default="http://127.0.0.1:8000",
                        help="Base URL for the online inference server")
    parser.add_argument("--inference-max-new-tokens", type=int, default=100,
                        help="Maximum number of tokens to request from the inference server")
    parser.add_argument("--inference-temperature", type=float, default=1.0,
                        help="Sampling temperature for online inference")
    parser.add_argument("--inference-top-k", type=int, default=None,
                        help="Optional top-k sampling limit for online inference")
    parser.add_argument("--inference-no-sample", action="store_true",
                        help="Disable sampling for online inference")
    parser.add_argument("--inference-forbidden-tokens", type=str, default="",
                        help="Comma-separated tokens that the server should forbid")
    parser.add_argument("--inference-require-state-after-movement", action="store_true",
                        help="Require the model to emit a state token after movement tokens")
    parser.add_argument("--inference-state-after-movement-prob", type=float, default=0.0,
                        help="Probability bias for inserting state after movement tokens")
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

    if args.pointgrey_calibration is None:
        default_pointgrey_calibration = Path("./data/pointgrey_calibration.json")
        if default_pointgrey_calibration.exists():
            args.pointgrey_calibration = str(default_pointgrey_calibration)
            print(
                f"[Camera] Auto-loading PointGrey calibration from "
                f"{default_pointgrey_calibration}"
            )

    # Import here to delay loading until args are parsed
    from storage.hdf5_writer import HDF5Writer
    from gui.viser_collector import ViserDataCollectorApp

    # Create arm reader
    if args.no_arm:
        arm_reader = None
        print("[Arm] Skipped (--no-arm mode)")
    elif args.arm_backend == "http":
        from robot.http_arm_reader import PushTHTTPArmReader
        arm_reader = PushTHTTPArmReader(
            service_url=args.arm_server,
            poll_hz=args.arm_poll_hz,
            connect_on_start=args.arm_http_connect,
            can_interface=args.can_interface,
            can_channel=args.can_channel,
        )
        arm_reader.start()
        print(f"[Arm] Using PushT HTTP service: {args.arm_server}")
    elif args.arm_backend == "sdk":
        from robot.sdk_arm_reader import PiperSDKArmReader
        arm_reader = PiperSDKArmReader(
            can_interface=args.can_interface,
            can_channel=args.can_channel,
            bitrate=args.bitrate,
            poll_hz=args.arm_poll_hz,
        )
        arm_reader.start()
        print("[Arm] Using piper_sdk feedback: GetArmEndPoseMsgs().end_pose")
    else:
        from robot.arm_reader import ArmReader, get_arm_can_config
        arm_can_config = get_arm_can_config(args.arm_profile)
        arm_reader = ArmReader(
            can_interface=args.can_interface,
            can_channel=args.can_channel,
            bitrate=args.bitrate,
            can_config=arm_can_config,
        )
        arm_reader.start()
        print(f"[Arm] Using CAN profile: {arm_can_config.name}")

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
            pointgrey_socket_path=args.pointgrey_socket,
            pointgrey_shm_prefix=args.pointgrey_shm_prefix,
            pointgrey_service_python=args.pointgrey_python,
            pointgrey_service_script=args.pointgrey_service_script,
            pointgrey_serial=args.pointgrey_serial,
            pointgrey_calibration_path=args.pointgrey_calibration,
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
        inference_server=args.inference_server,
        inference_max_new_tokens=args.inference_max_new_tokens,
        inference_temperature=args.inference_temperature,
        inference_top_k=args.inference_top_k,
        inference_do_sample=not args.inference_no_sample,
        inference_forbidden_tokens=args.inference_forbidden_tokens,
        inference_require_state_after_movement=args.inference_require_state_after_movement,
        inference_state_after_movement_prob=args.inference_state_after_movement_prob,
        control_api_host=args.control_api_host,
        control_api_port=args.control_api_port,
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
