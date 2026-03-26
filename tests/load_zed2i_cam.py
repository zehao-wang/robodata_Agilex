"""Test script for ZED 2i camera — streams color and depth via rerun.

Usage::

    ZED_BRIDGE_PYTHON=/home/zwa0839/miniconda3/envs/zed_bridge/bin/python3.10 \\
        python tests/load_zed2i_cam.py [--width W] [--height H] [--fps F] [--enable-depth]

Or set the env var permanently::

    export ZED_BRIDGE_PYTHON=/home/zwa0839/miniconda3/envs/zed_bridge/bin/python3.10

The bridge env (zed_bridge) must have pyzed + numpy<2.0 installed.

ZED2i supported param
  ┌────────────┬───────────────┬─────────────┐
  │ Resolution │ Actual pixels │     FPS     │
  ├────────────┼───────────────┼─────────────┤
  │ HD2K       │ 2208×1242     │ 15 only*    │
  ├────────────┼───────────────┼─────────────┤
  │ HD1080     │ 1920×1080     │ 15, 30      │
  ├────────────┼───────────────┼─────────────┤
  │ HD720      │ 1280×720      │ 15, 30, 60  │
  ├────────────┼───────────────┼─────────────┤
  │ VGA        │ 672×376       │ 15, 30, 100 │
  └────────────┴───────────────┴─────────────┘
"""
import argparse
import sys
import os

import rerun as rr
import rerun.blueprint as rrb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from camera.zed2 import ZED2Camera, ZED2CameraConfig

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=1920)
parser.add_argument("--height", type=int, default=1080)
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--enable-depth", action="store_true")
args = parser.parse_args()

config = ZED2CameraConfig(
    fps=args.fps,
    width=args.width,
    height=args.height,
    use_depth=args.enable_depth,
)

camera = ZED2Camera(config)
camera.connect()

if args.enable_depth:
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial2DView(name="Color", origin="color/image"),
            rrb.Spatial2DView(name="Depth", origin="depth/image"),
        )
    )
else:
    blueprint = rrb.Blueprint(
        rrb.Spatial2DView(name="Color", origin="color/image"),
    )

rr.init("zed2i_stream", spawn=True)
rr.send_blueprint(blueprint)

try:
    frame_idx = 0
    while True:
        color_frame = camera.read()

        rr.set_time_sequence("frame", frame_idx)
        rr.log("color/image", rr.Image(color_frame))

        if args.enable_depth:
            depth_map = camera.read_depth()
            rr.log("depth/image", rr.DepthImage(depth_map))

        frame_idx += 1
except KeyboardInterrupt:
    pass
finally:
    camera.disconnect()
