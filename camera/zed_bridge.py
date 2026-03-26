#!/usr/bin/env python3
"""ZED2 capture bridge process.

Must be run with a Python interpreter that has pyzed + numpy<2.0.
Communicates with the parent via three shared memory blocks:
  shm_color  — (H, W, 3) uint8 RGB
  shm_depth  — (H, W)    uint16 millimetres
  shm_meta   — 17 bytes: [width:i4][height:i4][timestamp:f8][status:u1]
                status: 0=starting, 1=camera open, 2=frame ready, 255=error
"""
import argparse
import signal
import sys
import time

import numpy as np
from multiprocessing.shared_memory import SharedMemory

_RESOLUTION_MAP = {
    (2208, 1242): None,  # filled after pyzed import
    (1920, 1080): None,
    (1280,  720): None,
    ( 672,  376): None,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shm-color", required=True)
    parser.add_argument("--shm-depth", default=None)
    parser.add_argument("--shm-meta",  required=True)
    parser.add_argument("--fps",         type=int, default=30)
    parser.add_argument("--width",       type=int, default=1280)
    parser.add_argument("--height",      type=int, default=720)
    parser.add_argument("--streams",     default="rgbd", choices=["rgb", "depth", "rgbd"])
    parser.add_argument("--params-file", default=None,
                        help="If given, write camera intrinsics JSON to this path.")
    args = parser.parse_args()

    shm_color = SharedMemory(name=args.shm_color)
    shm_depth = SharedMemory(name=args.shm_depth) if args.shm_depth else None
    shm_meta  = SharedMemory(name=args.shm_meta)

    try:
        import pyzed.sl as sl
    except Exception as e:
        print(f"[zed_bridge] Failed to import pyzed: {e}", file=sys.stderr)
        shm_meta.buf[16] = 255
        sys.exit(1)

    resolution_map = {
        (2208, 1242): sl.RESOLUTION.HD2K,
        (1920, 1080): sl.RESOLUTION.HD1080,
        (1280,  720): sl.RESOLUTION.HD720,
        ( 672,  376): sl.RESOLUTION.VGA,
    }
    resolution = resolution_map.get((args.width, args.height))
    if resolution is None:
        print(f"[zed_bridge] Unsupported resolution {args.width}x{args.height}, "
              f"supported: {list(resolution_map.keys())}", file=sys.stderr)
        shm_meta.buf[16] = 255
        sys.exit(1)

    need_color = args.streams in ("rgb", "rgbd")
    need_depth = args.streams in ("depth", "rgbd")

    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution
    init_params.camera_fps = args.fps
    init_params.depth_mode = sl.DEPTH_MODE.PERFORMANCE if need_depth else sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.open_timeout_sec = 15.0

    zed = sl.Camera()
    status_code = sl.ERROR_CODE.FAILURE
    for attempt in range(1, 4):
        status_code = zed.open(init_params)
        if status_code == sl.ERROR_CODE.SUCCESS:
            break
        print(f"[zed_bridge] Open attempt {attempt} failed ({status_code}), retrying...",
              file=sys.stderr)
        zed.close()
        time.sleep(5)

    if status_code != sl.ERROR_CODE.SUCCESS:
        print(f"[zed_bridge] Could not open ZED camera: {status_code}", file=sys.stderr)
        shm_meta.buf[16] = 255
        sys.exit(1)

    cam_info = zed.get_camera_information()
    cam_res = cam_info.camera_configuration.resolution
    w, h = cam_res.width, cam_res.height

    # Optionally write camera intrinsics/calibration to a JSON file
    if args.params_file:
        import json
        calib = cam_info.camera_configuration.calibration_parameters
        left  = calib.left_cam
        right = calib.right_cam
        params = {
            "type":             "zed2i",
            "serial_number":    cam_info.serial_number,
            "model":            str(cam_info.camera_model),
            "firmware_version": cam_info.camera_configuration.firmware_version,
            "width":            w,
            "height":           h,
            "fps":              args.fps,
            "left": {
                "fx":               float(left.fx),
                "fy":               float(left.fy),
                "cx":               float(left.cx),
                "cy":               float(left.cy),
                "h_fov_deg":        float(left.h_fov),
                "v_fov_deg":        float(left.v_fov),
                "d_fov_deg":        float(left.d_fov),
                "focal_length_mm":  float(left.focal_length_metric),
                "distortion_coeffs": left.disto.tolist(),
            },
            "right": {
                "fx":               float(right.fx),
                "fy":               float(right.fy),
                "cx":               float(right.cx),
                "cy":               float(right.cy),
                "h_fov_deg":        float(right.h_fov),
                "v_fov_deg":        float(right.v_fov),
                "d_fov_deg":        float(right.d_fov),
                "focal_length_mm":  float(right.focal_length_metric),
                "distortion_coeffs": right.disto.tolist(),
            },
            "baseline_mm": float(calib.get_camera_baseline()),
        }
        with open(args.params_file, "w") as _f:
            json.dump(params, _f, indent=2)

    # Write static width/height + initial status once
    shm_meta.buf[0:4]  = np.int32(w).tobytes()
    shm_meta.buf[4:8]  = np.int32(h).tobytes()
    shm_meta.buf[16]   = 1  # camera open
    print(f"[zed_bridge] Camera open: {w}x{h} @ {args.fps}fps", file=sys.stderr)

    color_arr = np.ndarray((h, w, 3), dtype=np.uint8,  buffer=shm_color.buf) if need_color else None
    depth_arr = np.ndarray((h, w), dtype=np.uint16, buffer=shm_depth.buf) if need_depth else None

    running = True

    def _stop(sig, frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    rt_params = sl.RuntimeParameters()
    color_mat = sl.Mat()
    depth_mat = sl.Mat()

    while running:
        if zed.grab(rt_params) != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.001)
            continue

        if need_color:
            zed.retrieve_image(color_mat, sl.VIEW.LEFT)
            color_arr[:] = color_mat.get_data()[..., [2, 1, 0]]  # BGRA -> RGB

        if need_depth:
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth_f32 = np.nan_to_num(depth_mat.get_data(), nan=0.0, posinf=0.0, neginf=0.0)
            depth_arr[:] = np.clip(depth_f32, 0, 65535).astype(np.uint16)

        # Only timestamp and status change per frame
        shm_meta.buf[8:16] = np.float64(time.time()).tobytes()
        shm_meta.buf[16]   = 2  # frame ready

    zed.close()
    shm_color.close()
    if shm_depth is not None:
        shm_depth.close()
    shm_meta.close()
    print("[zed_bridge] Exited cleanly.", file=sys.stderr)


if __name__ == "__main__":
    main()
