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


def _write_meta(buf, width, height, timestamp, status):
    buf[0:4]  = np.int32(width).tobytes()
    buf[4:8]  = np.int32(height).tobytes()
    buf[8:16] = np.float64(timestamp).tobytes()
    buf[16]   = status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shm-color", required=True)
    parser.add_argument("--shm-depth", required=True)
    parser.add_argument("--shm-meta",  required=True)
    parser.add_argument("--fps",     type=int, default=30)
    parser.add_argument("--streams", default="rgbd", choices=["rgb", "depth", "rgbd"])
    args = parser.parse_args()

    shm_color = SharedMemory(name=args.shm_color)
    shm_depth = SharedMemory(name=args.shm_depth)
    shm_meta  = SharedMemory(name=args.shm_meta)

    try:
        import pyzed.sl as sl
    except Exception as e:
        print(f"[zed_bridge] Failed to import pyzed: {e}", file=sys.stderr)
        shm_meta.buf[16] = 255
        sys.exit(1)

    need_color = args.streams in ("rgb", "rgbd")
    need_depth = args.streams in ("depth", "rgbd")

    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_fps = args.fps
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA if need_depth else sl.DEPTH_MODE.NONE
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.open_timeout_sec = 15.0

    status_code = sl.ERROR_CODE.FAILURE
    for attempt in range(1, 4):
        zed = sl.Camera()
        status_code = zed.open(init_params)
        if status_code == sl.ERROR_CODE.SUCCESS:
            break
        print(f"[zed_bridge] Open attempt {attempt} failed ({status_code}), retrying...",
              file=sys.stderr)
        time.sleep(5)

    if status_code != sl.ERROR_CODE.SUCCESS:
        print(f"[zed_bridge] Could not open ZED camera: {status_code}", file=sys.stderr)
        shm_meta.buf[16] = 255
        sys.exit(1)

    info = zed.get_camera_information()
    cam_res = info.camera_configuration.resolution
    w, h = cam_res.width, cam_res.height

    color_arr = np.ndarray((h, w, 3), dtype=np.uint8,  buffer=shm_color.buf)
    depth_arr = np.ndarray((h, w),    dtype=np.uint16, buffer=shm_depth.buf)

    _write_meta(shm_meta.buf, w, h, 0.0, 1)  # camera open
    print(f"[zed_bridge] Camera open: {w}x{h} @ {args.fps}fps", file=sys.stderr)

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
        err = zed.grab(rt_params)
        if err != sl.ERROR_CODE.SUCCESS:
            time.sleep(0.001)
            continue

        if need_color:
            zed.retrieve_image(color_mat, sl.VIEW.LEFT)
            bgra = color_mat.get_data()
            color_arr[:] = bgra[..., [2, 1, 0]]  # BGRA -> RGB

        if need_depth:
            zed.retrieve_measure(depth_mat, sl.MEASURE.DEPTH)
            depth_f32 = depth_mat.get_data()
            depth_f32 = np.nan_to_num(depth_f32, nan=0.0, posinf=0.0, neginf=0.0)
            depth_arr[:] = np.clip(depth_f32, 0, 65535).astype(np.uint16)

        _write_meta(shm_meta.buf, w, h, time.time(), 2)  # frame ready

    zed.close()
    shm_color.close()
    shm_depth.close()
    shm_meta.close()
    print("[zed_bridge] Exited cleanly.", file=sys.stderr)


if __name__ == "__main__":
    main()
