
'''
RealSense D435if supported param
  ┌────────────┬───────────────┐
  │ Resolution │      FPS      │
  ├────────────┼───────────────┤
  │ 424×240    │ 6, 15, 30, 60 │
  ├────────────┼───────────────┤
  │ 640×480    │ 6, 15, 30     │
  ├────────────┼───────────────┤
  │ 1280×720   │ 6, 10, 15     │
  ├────────────┼───────────────┤
  │ 1920×1080  │ 8             │
  └────────────┴───────────────┘

'''
import argparse
import rerun as rr
import rerun.blueprint as rrb
import numpy as np
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import RealSenseCamera
from lerobot.cameras.configs import ColorMode, Cv2Rotation

parser = argparse.ArgumentParser()
parser.add_argument("--width", type=int, default=1280)
parser.add_argument("--height", type=int, default=720)
parser.add_argument("--fps", type=int, default=15)
parser.add_argument("--enable-depth", action="store_true")
args = parser.parse_args()

# Create a `RealSenseCameraConfig` specifying your camera's serial number and enabling depth.
config = RealSenseCameraConfig(
    serial_number_or_name="332322070769",
    fps=args.fps,
    width=args.width,
    height=args.height,
    color_mode=ColorMode.RGB,
    use_depth=args.enable_depth,
    rotation=Cv2Rotation.NO_ROTATION
)

# Instantiate and connect a `RealSenseCamera` with warm-up read (default).
camera = RealSenseCamera(config)
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

rr.init("realsense_stream", spawn=True)
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
