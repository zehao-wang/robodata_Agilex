"""Minimal RealSense test — run with: sudo /path/to/python test_realsense.py"""
import pyrealsense2 as rs
import time

ctx = rs.context()
devs = ctx.query_devices()
print(f"Devices: {len(devs)}")
if len(devs) == 0:
    raise SystemExit("No device found")
dev = devs[0]
print(f"  Name: {dev.get_info(rs.camera_info.name)}")
print(f"  USB:  {dev.get_info(rs.camera_info.usb_type_descriptor)}")

# Test 1: depth only
print("\n--- Test 1: Depth only 640x480 @ 15fps ---")
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
try:
    pipe.start(cfg)
    print("  Pipeline started")
    for i in range(5):
        try:
            frames = pipe.wait_for_frames(timeout_ms=3000)
            print(f"  Frame {i}: depth={frames.get_depth_frame()}")
            break
        except RuntimeError as e:
            print(f"  Timeout {i}: {e}")
    pipe.stop()
    print("  Stopped OK")
except RuntimeError as e:
    print(f"  FAILED: {e}")

time.sleep(1)

# Test 2: color only
print("\n--- Test 2: Color only 640x480 @ 15fps ---")
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
try:
    pipe.start(cfg)
    print("  Pipeline started")
    for i in range(5):
        try:
            frames = pipe.wait_for_frames(timeout_ms=3000)
            print(f"  Frame {i}: color={frames.get_color_frame()}")
            break
        except RuntimeError as e:
            print(f"  Timeout {i}: {e}")
    pipe.stop()
    print("  Stopped OK")
except RuntimeError as e:
    print(f"  FAILED: {e}")

time.sleep(1)

# Test 3: both streams
print("\n--- Test 3: Depth + Color 640x480 @ 15fps ---")
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 15)
try:
    pipe.start(cfg)
    print("  Pipeline started")
    for i in range(5):
        try:
            frames = pipe.wait_for_frames(timeout_ms=3000)
            print(f"  Frame {i}: depth={frames.get_depth_frame()}, color={frames.get_color_frame()}")
            break
        except RuntimeError as e:
            print(f"  Timeout {i}: {e}")
    pipe.stop()
    print("  Stopped OK")
except RuntimeError as e:
    print(f"  FAILED: {e}")

time.sleep(1)

# Test 4: both streams lower res
print("\n--- Test 4: Depth + Color 480x270 @ 15fps ---")
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 480, 270, rs.format.z16, 15)
cfg.enable_stream(rs.stream.color, 424, 240, rs.format.rgb8, 15)
try:
    pipe.start(cfg)
    print("  Pipeline started")
    for i in range(5):
        try:
            frames = pipe.wait_for_frames(timeout_ms=3000)
            print(f"  Frame {i}: depth={frames.get_depth_frame()}, color={frames.get_color_frame()}")
            break
        except RuntimeError as e:
            print(f"  Timeout {i}: {e}")
    pipe.stop()
    print("  Stopped OK")
except RuntimeError as e:
    print(f"  FAILED: {e}")

# Test 5: auto config (let camera decide)
print("\n--- Test 5: Auto config (no explicit streams) ---")
pipe = rs.pipeline()
cfg = rs.config()
try:
    pipe.start(cfg)
    print("  Pipeline started")
    for i in range(5):
        try:
            frames = pipe.wait_for_frames(timeout_ms=3000)
            print(f"  Frame {i}: got {frames.size()} frames")
            break
        except RuntimeError as e:
            print(f"  Timeout {i}: {e}")
    pipe.stop()
    print("  Stopped OK")
except RuntimeError as e:
    print(f"  FAILED: {e}")

print("\nDone.")
