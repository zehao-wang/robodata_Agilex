"""Viser-based web GUI for data collection.

Replaces the OpenCV+Tkinter interface with a viser web app combining:
- 3D arm visualization (URDF-based via ViserUrdf)
- Camera feed display (color + depth)
- Recording controls and status in the sidebar
- Replay of recorded episodes
"""

import glob as globmod
import math
import os
import time

import cv2
import h5py
import numpy as np
import viser
from viser.extras import ViserUrdf

from robot.arm_reader import ArmReader, ArmState
from storage.hdf5_writer import HDF5Writer
from utils.arm_visualizer import forward_kinematics, fingertip_center_from_T_ee
from utils.urdf_loader import load_piper_urdf, can_qpos_to_urdf_cfg_with_gripper
from utils.world_frame import point_base_to_world, add_world_frame_visual


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
        arm_reader: ArmReader | None,
        camera=None,
        writer: HDF5Writer | None = None,
        port: int = 8080,
        fps: int = 30,
        frame_w: int = 640,
        frame_h: int = 480,
        demo_mode: bool = False,
        world_config: dict | None = None,
        streams: str = "rgb",
        output_dir: str = "./data",
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
        self._demo_mode = demo_mode
        self._demo_sim = DemoArmSimulator() if demo_mode else None
        self._world_config = world_config
        self._streams = streams
        self._output_dir = output_dir
        self._has_depth = streams in ("depth", "rgbd")
        # Replay state
        self._replaying = False
        self._replay_stop_requested = False
        self._replay_data = None
        self._replay_idx = 0
        if world_config is not None:
            self._T_world_from_base = np.asarray(
                world_config["T_world_from_base"], dtype=np.float64
            )
            self._writer.set_world_config(world_config)
        else:
            self._T_world_from_base = None

    def run(self):
        """Start the viser server and run the main loop."""
        server = viser.ViserServer(port=self._port)
        self._server = server

        # --- 3D Scene ---
        server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        urdf = load_piper_urdf()
        self._urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

        # World frame calibration visualization
        if self._world_config is not None:
            add_world_frame_visual(server, self._world_config)

        # --- Sidebar GUI ---
        # Task/Instruction inputs at top level
        self._task_input = server.gui.add_text("Task Name", initial_value="")
        self._instr_input = server.gui.add_text("Instruction", initial_value="")

        # Camera folder
        with server.gui.add_folder("Camera"):
            self._color_handle = server.gui.add_image(
                np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8),
                label="Color",
            )
            if self._has_depth:
                self._depth_handle = server.gui.add_image(
                    np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8),
                    label="Depth",
                )
            else:
                self._depth_handle = None

        eef_label = "EEF Position (World)" if self._world_config is not None else "EEF Position (Base)"
        with server.gui.add_folder("Arm State"):
            self._eef_md = server.gui.add_markdown(f"**{eef_label}:**\n\nX: ---  Y: ---  Z: ---")
            self._qpos_md = server.gui.add_markdown(
                "**Joint Positions (deg):**\n\n"
                "J1: ---  J2: ---  J3: ---\n\nJ4: ---  J5: ---  J6: ---"
            )
            self._gripper_md = server.gui.add_markdown("**Gripper:** ---")

        with server.gui.add_folder("Recording"):
            self._record_btn = server.gui.add_button("Start Recording", color="blue")
            self._status_md = server.gui.add_markdown("**Status:** IDLE")

        self._record_btn.on_click(self._on_record_click)

        # Replay folder
        with server.gui.add_folder("Replay"):
            self._replay_dropdown = server.gui.add_dropdown(
                "Episode",
                options=self._list_episodes(),
            )
            self._replay_btn = server.gui.add_button("Replay")
            self._stop_replay_btn = server.gui.add_button("Stop Replay", visible=False)

        self._replay_btn.on_click(self._on_replay_click)
        self._stop_replay_btn.on_click(self._on_stop_replay_click)

        print(f"[ViserCollector] Server started at http://localhost:{self._port}")
        print("Press Ctrl+C to stop.\n")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[ViserCollector] Shutting down...")
            if self._recording:
                self._stop_recording()
            if self._replaying:
                self._finish_replay_cleanup()

    def _list_episodes(self) -> list[str]:
        """List available episode HDF5 files in output_dir."""
        pattern = os.path.join(self._output_dir, "episode_*.hdf5")
        files = sorted(globmod.glob(pattern))
        if not files:
            return ["(none)"]
        return [os.path.basename(f) for f in files]

    def _refresh_episode_list(self):
        """Update the replay dropdown with current episodes."""
        episodes = self._list_episodes()
        self._replay_dropdown.options = episodes

    def _main_loop(self):
        target_dt = 1.0 / self._fps
        while True:
            loop_start = time.time()

            if self._replaying and self._replay_data is not None:
                # Check if stop was requested (from button callback thread)
                if self._replay_stop_requested:
                    self._finish_replay_cleanup()
                    elapsed = time.time() - loop_start
                    sleep_time = target_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                # --- Replay mode: use recorded data ---
                rd = self._replay_data
                idx = self._replay_idx

                if idx >= rd["num_frames"]:
                    # Replay finished
                    self._finish_replay_cleanup()
                    elapsed = time.time() - loop_start
                    sleep_time = target_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                # Build display state from recorded data
                display_state = ArmState(
                    qpos=rd["qpos"][idx],
                    qvel=np.zeros(6, dtype=np.float64),
                    gripper=rd["gripper"][idx],
                    timestamp=time.time(),
                )
                color = rd["color"][idx]

                # Update OpenCV replay window
                color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                cv2.imshow("Replay", color_bgr)
                cv2.waitKey(1)

                self._replay_idx += 1
                progress = f"{idx + 1}/{rd['num_frames']}"
                self._status_md.content = f"**Status:** REPLAY | Frame {progress}"
            else:
                # --- Live mode ---
                # Read arm state
                if self._demo_mode and self._demo_sim is not None:
                    display_state = self._demo_sim.get_state()
                elif self._arm_reader is not None:
                    display_state = self._arm_reader.get_state()
                else:
                    display_state = ArmState()

                # Read camera frames
                if self._camera is not None:
                    color, depth, _ = self._camera.get_frames()
                else:
                    color = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
                    depth = np.zeros((self._frame_h, self._frame_w), dtype=np.uint16)

                # Update depth display
                if self._depth_handle is not None:
                    if depth.max() > 0:
                        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                        depth_u8 = depth_norm.astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
                        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
                    else:
                        depth_color = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
                    self._depth_handle.image = depth_color

            # --- Common updates (both live and replay) ---
            # Update 3D arm visualization
            cfg = can_qpos_to_urdf_cfg_with_gripper(display_state.qpos, display_state.gripper)
            self._urdf_vis.update_cfg(cfg)

            # Update camera display
            self._color_handle.image = color

            # Compute fingertip center position via FK
            _, T_ee = forward_kinematics(display_state.qpos)
            eef_pos_base = fingertip_center_from_T_ee(T_ee)
            if self._T_world_from_base is not None:
                eef_pos = point_base_to_world(eef_pos_base, self._T_world_from_base)
            else:
                eef_pos = eef_pos_base

            # Update arm state display
            frame_label = "World" if self._world_config is not None else "Base"
            self._eef_md.content = (
                f"**EEF Position ({frame_label}):**\n\n"
                f"X: {eef_pos[0]:.4f}  Y: {eef_pos[1]:.4f}  Z: {eef_pos[2]:.4f} m"
            )
            qd = np.degrees(display_state.qpos)
            self._qpos_md.content = (
                f"**Joint Positions (deg):**\n\n"
                f"J1: {qd[0]:+7.2f}  J2: {qd[1]:+7.2f}  J3: {qd[2]:+7.2f}\n\n"
                f"J4: {qd[3]:+7.2f}  J5: {qd[4]:+7.2f}  J6: {qd[5]:+7.2f}"
            )
            self._gripper_md.content = f"**Gripper:** {display_state.gripper*1000:.1f} mm"

            # Record if active (only in live mode)
            if self._recording and not self._replaying:
                self._writer.add_frame(
                    qpos=display_state.qpos,
                    qvel=display_state.qvel,
                    gripper=display_state.gripper,
                    color=color,
                    depth=depth,
                    timestamp=time.time(),
                    eef_pos=eef_pos,
                )
                n = self._writer.num_frames
                duration = n / self._fps
                self._status_md.content = (
                    f"**Status:** REC | {n} frames ({duration:.1f}s)"
                )

            elapsed = time.time() - loop_start
            sleep_time = target_dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    # --- Recording callbacks ---

    def _on_record_click(self, _event):
        if self._replaying:
            return  # Ignore during replay
        if not self._recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self._task_name = self._task_input.value
        self._instruction = self._instr_input.value
        self._writer.reset()
        self._recording = True
        self._record_btn.label = "Stop Recording"
        self._record_btn.color = "red"
        self._status_md.content = "**Status:** REC | 0 frames"
        print(f">> Recording STARTED  task={self._task_name!r}")

    def _stop_recording(self):
        self._recording = False
        self._record_btn.label = "Start Recording"
        self._record_btn.color = "blue"
        if self._writer.num_frames > 0:
            path = self._writer.save(
                task_name=self._task_name,
                instruction=self._instruction,
            )
            self._status_md.content = f"**Status:** IDLE | Saved: {path}"
            print(f">> Episode saved: {path}")
            self._refresh_episode_list()
        else:
            self._status_md.content = "**Status:** IDLE | No frames captured"
            print(">> No frames captured, nothing saved.")

    # --- Replay callbacks ---

    def _on_replay_click(self, _event):
        if self._recording or self._replaying:
            return
        selected = self._replay_dropdown.value
        if selected == "(none)":
            return
        filepath = os.path.join(self._output_dir, selected)
        if not os.path.exists(filepath):
            self._status_md.content = f"**Status:** File not found: {selected}"
            return
        self._start_replay(filepath)

    def _on_stop_replay_click(self, _event):
        if self._replaying:
            self._replay_stop_requested = True

    def _start_replay(self, filepath: str):
        """Load HDF5 episode and begin playback."""
        print(f">> Replay STARTED: {filepath}")
        with h5py.File(filepath, "r") as f:
            self._replay_data = {
                "qpos": f["observations/qpos"][:],
                "gripper": f["observations/gripper"][:].flatten(),
                "color": f["observations/images/color"][:],
                "num_frames": int(f.attrs["num_frames"]),
                "fps": int(f.attrs.get("fps", self._fps)),
            }
        self._replay_idx = 0
        self._replaying = True
        self._replay_btn.visible = False
        self._stop_replay_btn.visible = True
        self._record_btn.disabled = True
        self._status_md.content = f"**Status:** REPLAY | Frame 0/{self._replay_data['num_frames']}"

    def _finish_replay_cleanup(self):
        """Clean up replay state. Must be called from the main thread."""
        self._replaying = False
        self._replay_stop_requested = False
        self._replay_idx = 0
        # Free large arrays before cv2 cleanup
        del self._replay_data
        self._replay_data = None
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Flush macOS event queue
        self._replay_btn.visible = True
        self._stop_replay_btn.visible = False
        self._record_btn.disabled = False
        self._status_md.content = "**Status:** IDLE"
        print(">> Replay STOPPED")
