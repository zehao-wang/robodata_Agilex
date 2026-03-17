"""Viser-based web GUI for data collection.

Replaces the OpenCV+Tkinter interface with a viser web app combining:
- 3D arm visualization (URDF-based via ViserUrdf)
- Camera feed display (color + depth)
- Recording controls and status in the sidebar
- Replay of recorded episodes
"""

import glob as globmod
import json
import math
import os
import time

import cv2
import numpy as np
import viser
from viser.extras import ViserUrdf

from robot.arm_reader import ArmReader, ArmState
from storage.hdf5_writer import HDF5Writer
from utils.urdf_loader import (
    load_piper_urdf,
    can_qpos_to_urdf_cfg_with_gripper,
    fingertip_center_from_urdf_cfg,
)
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
        camera_sources: list | None = None,
        max_sync_dt_ms: float = 50.0,
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
        self._camera_sources = camera_sources or []
        self._max_sync_dt_ms = max_sync_dt_ms
        self._camera_options = ["(none)"] + [source.label for source in self._camera_sources]
        self._camera_label_to_id = {source.label: source.source_id for source in self._camera_sources}
        self._camera_id_to_label = {source.source_id: source.label for source in self._camera_sources}
        self._last_camera_selection = self._get_current_camera_label()
        self._last_sync_info = self._empty_sync_info()
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
        self._urdf = urdf
        self._urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
        self._eef_marker = server.scene.add_icosphere(
            "/eef_marker",
            radius=0.01,
            color=(255, 80, 80),
            position=(0.0, 0.0, 0.0),
        )

        # World frame calibration visualization
        if self._world_config is not None:
            add_world_frame_visual(server, self._world_config)

        # --- Sidebar GUI ---
        # Task/Instruction inputs at top level
        self._task_input = server.gui.add_text("Task Name", initial_value="")
        self._instr_input = server.gui.add_text("Instruction", initial_value="")

        # Camera folder
        with server.gui.add_folder("Camera"):
            self._camera_dropdown = server.gui.add_dropdown(
                "Source",
                options=self._camera_options,
                initial_value=self._last_camera_selection,
            )
            self._zed_mode_checkbox = server.gui.add_checkbox(
                "ZED Mode",
                initial_value=bool(
                    getattr(self._camera, "opencv_zed_mode", False)
                ),
            )
            self._camera_status_md = server.gui.add_markdown(
                self._camera_status_text(self._last_camera_selection)
            )
            self._camera_backend_md = server.gui.add_markdown(
                self._camera_backend_text(self._last_camera_selection)
            )
            self._camera_resolution_md = server.gui.add_markdown(
                "**Resolution:** ---"
            )
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
        """List available episode directories in output_dir."""
        pattern = os.path.join(self._output_dir, "episode_*")
        paths = sorted(
            path for path in globmod.glob(pattern) if os.path.isdir(path)
        )
        episodes = [
            os.path.basename(path)
            for path in paths
            if os.path.exists(os.path.join(path, "metadata.json"))
            and os.path.exists(os.path.join(path, "camera.mp4"))
        ]
        if not episodes:
            return ["(none)"]
        return episodes

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
                self._sync_camera_selection()
                self._sync_opencv_zed_mode()
                if self._camera is not None:
                    color, depth, camera_timestamp, display_state = self._camera.capture_sync(
                        self._get_current_arm_state
                    )
                    self._last_sync_info = self._evaluate_sync(
                        arm_timestamp=display_state.timestamp,
                        camera_timestamp=camera_timestamp,
                    )
                elif self._demo_mode and self._demo_sim is not None:
                    display_state = self._demo_sim.get_state()
                    color = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
                    depth = np.zeros((self._frame_h, self._frame_w), dtype=np.uint16)
                    self._last_sync_info = self._evaluate_sync(
                        arm_timestamp=display_state.timestamp,
                        camera_timestamp=0.0,
                    )
                else:
                    if self._arm_reader is not None:
                        display_state = self._arm_reader.get_state()
                    else:
                        display_state = ArmState()
                    color = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
                    depth = np.zeros((self._frame_h, self._frame_w), dtype=np.uint16)
                    self._last_sync_info = self._evaluate_sync(
                        arm_timestamp=display_state.timestamp,
                        camera_timestamp=0.0,
                    )

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
            self._camera_resolution_md.content = self._format_camera_resolution(
                color=color,
                depth=depth,
            )

            # Compute fingertip endpoint center via URDF FK (link7/link8 tip midpoint)
            eef_pos_base = fingertip_center_from_urdf_cfg(self._urdf, cfg)
            self._eef_marker.position = tuple(eef_pos_base)
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
                sync_info = self._last_sync_info or self._empty_sync_info()
                record_timestamp = time.time()
                self._writer.add_frame(
                    qpos=display_state.qpos,
                    qvel=display_state.qvel,
                    gripper=display_state.gripper,
                    color=color,
                    depth=depth,
                    timestamp=record_timestamp,
                    arm_timestamp=sync_info["arm_timestamp"],
                    camera_timestamp=sync_info["camera_timestamp"],
                    sync_delta_ms=sync_info["sync_delta_ms"],
                    sync_ok=sync_info["sync_ok"],
                    eef_pos=eef_pos,
                )
                n = self._writer.num_frames
                duration = n / self._fps
                self._status_md.content = self._format_recording_status(n, duration)

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
        self._last_sync_info = self._empty_sync_info()
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
        episode_dir = os.path.join(self._output_dir, selected)
        if not os.path.isdir(episode_dir):
            self._status_md.content = f"**Status:** File not found: {selected}"
            return
        self._start_replay(episode_dir)

    def _on_stop_replay_click(self, _event):
        if self._replaying:
            self._replay_stop_requested = True

    def _start_replay(self, episode_dir: str):
        """Load folder-based episode data and begin playback."""
        print(f">> Replay STARTED: {episode_dir}")
        metadata_path = os.path.join(episode_dir, "metadata.json")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        video_path = os.path.join(episode_dir, metadata.get("video_file", "camera.mp4"))
        color_frames = self._load_video_frames(video_path)
        frame_records = metadata.get("frames", [])
        num_frames = min(len(frame_records), len(color_frames))
        if num_frames == 0:
            self._status_md.content = "**Status:** Replay data is empty"
            return

        self._replay_data = {
            "qpos": np.asarray(
                [frame["qpos"] for frame in frame_records[:num_frames]], dtype=np.float64
            ),
            "gripper": np.asarray(
                [frame["gripper"] for frame in frame_records[:num_frames]], dtype=np.float64
            ),
            "color": color_frames[:num_frames],
            "num_frames": num_frames,
            "fps": int(round(self._infer_replay_fps(frame_records[:num_frames]))),
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

    def _load_video_frames(self, video_path: str) -> list[np.ndarray]:
        capture = cv2.VideoCapture(video_path)
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open replay video: {video_path}")

        frames = []
        try:
            while True:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()
        return frames

    def _infer_replay_fps(self, frame_records: list[dict]) -> float:
        if len(frame_records) < 2:
            return float(self._fps)
        timestamps = np.asarray(
            [
                frame.get("camera_timestamp", frame.get("timestamp"))
                for frame in frame_records
            ],
            dtype=np.float64,
        )
        dt = np.diff(timestamps)
        dt = dt[dt > 1e-6]
        if dt.size == 0:
            return float(self._fps)
        return float(np.clip(1.0 / np.mean(dt), 1.0, 240.0))

    def _evaluate_sync(self, arm_timestamp: float, camera_timestamp: float) -> dict:
        has_arm = arm_timestamp > 0.0
        has_camera = camera_timestamp > 0.0
        if not has_arm or not has_camera:
            return {
                "arm_timestamp": float(arm_timestamp) if has_arm else None,
                "camera_timestamp": float(camera_timestamp) if has_camera else None,
                "sync_delta_ms": None,
                "sync_ok": None,
            }

        sync_delta_ms = abs(camera_timestamp - arm_timestamp) * 1000.0
        return {
            "arm_timestamp": float(arm_timestamp),
            "camera_timestamp": float(camera_timestamp),
            "sync_delta_ms": float(sync_delta_ms),
            "sync_ok": sync_delta_ms <= self._max_sync_dt_ms,
        }

    def _format_recording_status(self, num_frames: int, duration_s: float) -> str:
        status = f"**Status:** REC | {num_frames} frames ({duration_s:.1f}s)"
        if self._last_sync_info is None:
            return status

        sync_delta_ms = self._last_sync_info["sync_delta_ms"]
        sync_ok = self._last_sync_info["sync_ok"]
        if sync_delta_ms is None:
            return status + " | Sync: N/A"

        sync_label = "OK" if sync_ok else "OUT"
        return status + f" | Sync: {sync_label} ({sync_delta_ms:.1f} ms)"

    def _get_current_arm_state(self) -> ArmState:
        if self._demo_mode and self._demo_sim is not None:
            return self._demo_sim.get_state()
        if self._arm_reader is not None:
            return self._arm_reader.get_state()
        return ArmState()

    def _empty_sync_info(self) -> dict:
        return {
            "arm_timestamp": None,
            "camera_timestamp": None,
            "sync_delta_ms": None,
            "sync_ok": None,
        }

    def _format_camera_resolution(self, color: np.ndarray, depth: np.ndarray) -> str:
        if color.ndim >= 2 and color.shape[0] > 1 and color.shape[1] > 1:
            color_text = f"Color {color.shape[1]}x{color.shape[0]}"
        else:
            color_text = "Color ---"

        if self._depth_handle is not None and depth.ndim >= 2 and depth.shape[0] > 1 and depth.shape[1] > 1:
            depth_text = f"Depth {depth.shape[1]}x{depth.shape[0]}"
            return f"**Resolution:** {color_text} | {depth_text}"

        return f"**Resolution:** {color_text}"

    def _get_current_camera_label(self) -> str:
        camera_owner = self._camera
        if camera_owner is None or not hasattr(camera_owner, "active_source_id"):
            return "(none)"
        source_id = camera_owner.active_source_id
        if source_id is None:
            return "(none)"
        return self._camera_id_to_label.get(source_id, "(none)")

    def _camera_status_text(self, selection_label: str) -> str:
        if selection_label == "(none)":
            return "**Camera:** disabled"
        camera_owner = self._camera
        if camera_owner is not None and getattr(camera_owner, "last_error", None):
            return f"**Camera:** {camera_owner.last_error}"
        return f"**Camera:** {selection_label}"

    def _sync_camera_selection(self):
        selection_label = self._camera_dropdown.value
        if selection_label != self._last_camera_selection:
            camera_owner = self._camera
            if camera_owner is not None and hasattr(camera_owner, "select_camera"):
                source_id = self._camera_label_to_id.get(selection_label)
                if selection_label == "(none)":
                    camera_owner.select_camera(None)
                else:
                    ok = camera_owner.select_camera(source_id)
                    if not ok:
                        selection_label = self._get_current_camera_label()
                        self._camera_dropdown.value = selection_label
                self._last_camera_selection = selection_label
            else:
                self._camera_dropdown.value = self._last_camera_selection

        current_label = self._get_current_camera_label()
        if current_label != self._camera_dropdown.value:
            self._camera_dropdown.value = current_label
        self._last_camera_selection = current_label
        self._camera_status_md.content = self._camera_status_text(current_label)
        self._camera_backend_md.content = self._camera_backend_text(current_label)

    def _camera_backend_text(self, selection_label: str) -> str:
        if selection_label == "(none)":
            return "**Module:** none"

        source_id = self._camera_label_to_id.get(selection_label)
        if source_id is None:
            return "**Module:** unknown"

        if source_id.startswith("opencv:"):
            backend = "OpenCV (ZED Mode)" if self._zed_mode_checkbox.value else "OpenCV"
        elif source_id.startswith("realsense:"):
            backend = "RealSense"
        else:
            backend = "unknown"
        return f"**Module:** {backend}"

    def _sync_opencv_zed_mode(self):
        if self._camera is None or not hasattr(self._camera, "set_opencv_zed_mode"):
            return
        enabled = bool(self._zed_mode_checkbox.value)
        if getattr(self._camera, "opencv_zed_mode", False) != enabled:
            self._camera.set_opencv_zed_mode(enabled)
            current_label = self._get_current_camera_label()
            self._camera_backend_md.content = self._camera_backend_text(current_label)
