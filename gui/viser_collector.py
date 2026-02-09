"""Viser-based web GUI for data collection.

Replaces the OpenCV+Tkinter interface with a viser web app combining:
- 3D arm visualization (URDF-based via ViserUrdf)
- Camera feed display (color + depth)
- Recording controls and status in the sidebar
"""

import math
import time

import numpy as np
import viser
from viser.extras import ViserUrdf

from robot.arm_reader import ArmReader, ArmState
from robot.dual_arm_reader import DualArmReader
from storage.hdf5_writer import HDF5Writer
from utils.urdf_loader import load_piper_urdf, can_qpos_to_urdf_cfg_with_gripper


class DemoArmSimulator:
    """Simulates arm movement for demo mode without hardware."""

    def __init__(self):
        self._t = 0.0
        # Joint limits (radians) from URDF
        self._limits = [
            (-2.618, 2.618),   # joint1
            (0, 3.14),         # joint2
            (-2.967, 0),       # joint3
            (-1.745, 1.745),   # joint4
            (-1.22, 1.22),     # joint5
            (-2.0944, 2.0944), # joint6
        ]

    def get_state(self) -> ArmState:
        """Generate a smoothly animated arm state."""
        self._t += 0.03  # ~30 FPS
        qpos = np.zeros(6, dtype=np.float64)

        # Animate each joint with different frequencies
        for i, (lo, hi) in enumerate(self._limits):
            mid = (lo + hi) / 2
            amp = (hi - lo) / 4  # Use 1/4 of range for smooth motion
            freq = 0.3 + i * 0.1  # Different frequency per joint
            qpos[i] = mid + amp * math.sin(self._t * freq)

        # Animate gripper
        gripper = 0.035 * (0.5 + 0.5 * math.sin(self._t * 0.5))

        return ArmState(
            qpos=qpos,
            qvel=np.zeros(6, dtype=np.float64),
            gripper=gripper,
            timestamp=time.time(),
        )


class ViserDataCollectorApp:
    """Viser web app for PIPER data collection."""

    def __init__(
        self,
        arm_reader: ArmReader | DualArmReader | None,
        camera=None,
        writer: HDF5Writer | None = None,
        port: int = 8080,
        fps: int = 30,
        frame_w: int = 640,
        frame_h: int = 480,
        demo_mode: bool = False,
    ):
        self._arm_reader = arm_reader
        self._camera = camera
        self._writer = writer or HDF5Writer("./data")
        self._port = port
        self._fps = fps
        self._frame_w = frame_w
        self._frame_h = frame_h
        self._recording = False
        self._task_name = ""
        self._instruction = ""
        self._is_dual = isinstance(arm_reader, DualArmReader)
        self._demo_mode = demo_mode
        self._demo_sim = DemoArmSimulator() if demo_mode else None

    def run(self):
        """Start the viser server and run the main loop."""
        server = viser.ViserServer(port=self._port)
        self._server = server

        # --- 3D Scene ---
        server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        urdf = load_piper_urdf()
        self._urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

        # --- Sidebar GUI ---
        with server.gui.add_folder("Camera"):
            self._color_handle = server.gui.add_image(
                np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8),
                label="Color",
            )
            self._depth_handle = server.gui.add_image(
                np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8),
                label="Depth",
            )

        with server.gui.add_folder("Recording"):
            self._task_input = server.gui.add_text("Task Name", initial_value="")
            self._instr_input = server.gui.add_text("Instruction", initial_value="")
            self._record_btn = server.gui.add_button("Start Recording")
            self._status_md = server.gui.add_markdown("**Status:** IDLE")

        self._record_btn.on_click(self._on_record_click)

        print(f"[ViserCollector] Server started at http://localhost:{self._port}")
        print("Press Ctrl+C to stop.\n")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[ViserCollector] Shutting down...")
            if self._recording:
                self._stop_recording()

    def _main_loop(self):
        target_dt = 1.0 / self._fps
        while True:
            loop_start = time.time()

            # Read arm state
            if self._demo_mode and self._demo_sim is not None:
                # Demo mode: use animated simulation
                display_state = self._demo_sim.get_state()
                master_state = None
                slave_state = None
            elif self._is_dual:
                slave_state = self._arm_reader.get_slave_state()
                master_state = self._arm_reader.get_master_state()
                display_state = slave_state
            elif self._arm_reader is not None:
                display_state = self._arm_reader.get_state()
                master_state = None
                slave_state = None
            else:
                display_state = ArmState()
                master_state = None
                slave_state = None

            # Update 3D arm visualization
            cfg = can_qpos_to_urdf_cfg_with_gripper(display_state.qpos, display_state.gripper)
            self._urdf_vis.update_cfg(cfg)

            # Read camera frames
            if self._camera is not None:
                color, depth, _ = self._camera.get_frames()
            else:
                color = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
                depth = np.zeros((self._frame_h, self._frame_w), dtype=np.uint16)

            # Update camera displays
            self._color_handle.image = color

            # Colorize depth for display
            if depth.max() > 0:
                import cv2
                depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                depth_u8 = depth_norm.astype(np.uint8)
                depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
                depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
            else:
                depth_color = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
            self._depth_handle.image = depth_color

            # Record if active
            if self._recording:
                timestamp = time.time()
                if self._is_dual:
                    # Slave = observation, Master = action
                    self._writer.add_frame(
                        qpos=slave_state.qpos,
                        qvel=slave_state.qvel,
                        gripper=slave_state.gripper,
                        color=color,
                        depth=depth,
                        timestamp=timestamp,
                        action_qpos=master_state.qpos,
                        action_gripper=master_state.gripper,
                    )
                else:
                    self._writer.add_frame(
                        qpos=display_state.qpos,
                        qvel=display_state.qvel,
                        gripper=display_state.gripper,
                        color=color,
                        depth=depth,
                        timestamp=timestamp,
                    )
                n = self._writer.num_frames
                duration = n / self._fps
                self._status_md.content = (
                    f"**Status:** 🔴 REC | {n} frames ({duration:.1f}s)"
                )

            elapsed = time.time() - loop_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _on_record_click(self, _event):
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self._task_name = self._task_input.value
        self._instruction = self._instr_input.value
        self._writer.reset()
        self._recording = True
        self._record_btn.name = "Stop Recording"
        self._status_md.content = "**Status:** 🔴 REC | 0 frames"
        print(f">> Recording STARTED  task={self._task_name!r}")

    def _stop_recording(self):
        self._recording = False
        self._record_btn.name = "Start Recording"
        if self._writer.num_frames > 0:
            path = self._writer.save(
                task_name=self._task_name,
                instruction=self._instruction,
            )
            self._status_md.content = f"**Status:** IDLE | Saved: {path}"
            print(f">> Episode saved: {path}")
        else:
            self._status_md.content = "**Status:** IDLE | No frames captured"
            print(">> No frames captured, nothing saved.")
