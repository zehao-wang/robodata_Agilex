#!/usr/bin/env python3
"""Agilex PIPER data collection script.

Simultaneously captures:
- Operator arm joint positions (6 DOF) and gripper width via CAN bus
- D435i RGBD video stream via pyrealsense2

Data is saved in HDF5 format compatible with ACT / Diffusion Policy.

Usage:
    # macOS (gs_usb)
    python collect.py --output_dir ./data

    # Linux (socketcan)
    python collect.py --output_dir ./data --can-interface socketcan

Controls (OpenCV window must be focused):
    SPACE  - Start / stop recording an episode
    q      - Quit
"""

import argparse
import signal
import time

import cv2
import numpy as np

from robot.arm_reader import ArmReader, ArmState
from storage.hdf5_writer import HDF5Writer
from utils.annotation_dialog import ask_annotation
from utils.arm_visualizer import ArmVisualizer


class DataCollector:
    """Main controller: coordinates arm, camera, and storage."""

    WINDOW_NAME = "Agilex PIPER Data Collection"

    def __init__(self, args: argparse.Namespace):
        self._no_arm = args.no_arm
        if not self._no_arm:
            self.arm = ArmReader(
                can_interface=args.can_interface,
                can_channel=args.can_channel,
                bitrate=args.bitrate,
            )
        else:
            self.arm = None

        self._no_camera = args.no_camera
        if not self._no_camera:
            from camera.realsense import RealsenseCamera
            self.camera = RealsenseCamera(
                width=args.width,
                height=args.height,
                fps=args.fps,
            )
        else:
            self.camera = None
            self._frame_w = args.width
            self._frame_h = args.height

        self.writer = HDF5Writer(output_dir=args.output_dir)
        self.arm_viz = ArmVisualizer(size=args.height)

        self._recording = False
        self._running = True
        self._target_dt = 1.0 / args.fps
        self._task_name = ""
        self._instruction = ""

    def run(self):
        """Main entry point."""
        if self.arm is not None:
            self.arm.start()
        else:
            print("[Arm] Skipped (--no-arm mode, using dummy state)")
        if self.camera is not None:
            self.camera.start()
        else:
            print("[Camera] Skipped (--no-camera mode, using dummy frames)")

        signal.signal(signal.SIGINT, lambda *_: self._shutdown())

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 1920, 540)
        print("\n=== Agilex PIPER Data Collection ===")
        print("Press SPACE to start/stop recording, 'q' to quit.\n")

        try:
            self._main_loop()
        finally:
            self._cleanup()

    def _main_loop(self):
        while self._running:
            loop_start = time.time()

            arm_state = self.arm.get_state() if self.arm is not None else ArmState()
            if self.camera is not None:
                color, depth, _ = self.camera.get_frames()
            else:
                color = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
                depth = np.zeros((self._frame_h, self._frame_w), dtype=np.uint16)

            if self._recording:
                timestamp = time.time()
                self.writer.add_frame(
                    qpos=arm_state.qpos,
                    qvel=arm_state.qvel,
                    gripper=arm_state.gripper,
                    color=color,
                    depth=depth,
                    timestamp=timestamp,
                )

            # Build display frame
            display = self._build_display(color, depth, arm_state)
            cv2.imshow(self.WINDOW_NAME, display)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                self._toggle_recording()
            elif key == ord("q"):
                self._shutdown()

            # Maintain target frame rate
            elapsed = time.time() - loop_start
            sleep_time = self._target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _build_display(self, color: np.ndarray, depth: np.ndarray, arm_state) -> np.ndarray:
        """Compose a side-by-side color + depth display with overlays."""
        # Color: RGB -> BGR for OpenCV
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        # Depth: colorize with JET colormap
        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
        depth_u8 = depth_norm.astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)

        # 3D arm visualization panel
        arm_img = self.arm_viz.render(arm_state.qpos, arm_state.gripper)

        # Side-by-side: color | depth | arm 3D
        display = np.hstack([color_bgr, depth_color, arm_img])

        # --- Status overlay (top-left) ---
        if self._recording:
            n = self.writer.num_frames
            duration = n / 30.0
            status = f"REC  {n} frames ({duration:.1f}s)"
            status_color = (0, 0, 255)  # red
        else:
            status = "IDLE"
            status_color = (0, 200, 0)  # green

        cv2.putText(display, status, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # --- Annotation overlay (below status, only while recording) ---
        if self._recording and (self._task_name or self._instruction):
            y = 58
            if self._task_name:
                cv2.putText(display, f"Task: {self._task_name}", (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y += 22
            if self._instruction:
                cv2.putText(display, f"Instr: {self._instruction}", (12, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # --- Joint angles + gripper (bottom) ---
        h = display.shape[0]
        qpos_str = "Joints: " + "  ".join(f"{a:.2f}" for a in arm_state.qpos)
        gripper_str = f"Gripper: {arm_state.gripper:.4f} m"
        cv2.putText(display, qpos_str, (12, h - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(display, gripper_str, (12, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

        return display

    def _toggle_recording(self):
        if not self._recording:
            # Ask for annotation before starting
            result = ask_annotation()
            if result is None:
                print(">> Recording cancelled.")
                return
            self._task_name, self._instruction = result
            self.writer.reset()
            self._recording = True
            print(f">> Recording STARTED  task={self._task_name!r}")
        else:
            self._recording = False
            print(">> Recording STOPPED")
            if self.writer.num_frames > 0:
                path = self.writer.save(
                    task_name=self._task_name,
                    instruction=self._instruction,
                )
                print(f">> Episode saved: {path}")
            else:
                print(">> No frames captured, nothing saved.")

    def _shutdown(self):
        """Signal the main loop to exit."""
        print("\n>> Shutting down...")
        if self._recording:
            self._recording = False
            if self.writer.num_frames > 0:
                path = self.writer.save(
                    task_name=self._task_name,
                    instruction=self._instruction,
                )
                print(f">> Episode saved before exit: {path}")
        self._running = False

    def _cleanup(self):
        if self.arm is not None:
            self.arm.stop()
        if self.camera is not None:
            self.camera.stop()
        cv2.destroyAllWindows()
        print(">> Done.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agilex PIPER data collection")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Directory to save episode HDF5 files")
    parser.add_argument("--can-interface", type=str, default="gs_usb",
                        choices=["gs_usb", "socketcan"],
                        help="CAN bus interface type")
    parser.add_argument("--can-channel", type=str, default="can0",
                        help="CAN channel (for socketcan mode)")
    parser.add_argument("--bitrate", type=int, default=1_000_000,
                        help="CAN bus bitrate")
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collector = DataCollector(args)
    collector.run()
