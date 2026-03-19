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
from pathlib import Path

import cv2
import numpy as np
import viser
from scipy.spatial.transform import Rotation
from viser.extras import ViserUrdf

try:
    from pupil_apriltags import Detector
except ImportError:
    Detector = None

from robot.arm_reader import ArmReader, ArmState
from storage.hdf5_writer import HDF5Writer
from utils.urdf_loader import (
    load_piper_urdf,
    can_qpos_to_urdf_cfg_with_gripper,
    eef_pose_from_urdf_cfg,
    fingertip_center_from_urdf_cfg,
)
from utils.arm_world_calibration import (
    ArmWorldCalibrationResult,
    ArmWorldCalibrationSample,
    load_arm_world_calibration,
    save_arm_world_calibration,
    solve_arm_world_calibration,
)
from utils.apriltag_reconstruction import (
    BACKGROUND_TAG_IDS,
    AprilTagAnchorGeometry,
    AprilTagPoseKalmanSmoother,
    AprilTagPoseKalmanSmootherConfig,
    AprilTagStaticReconstructionModel,
    camera_calibration_from_info,
    ConstantVelocityKalmanConfig,
    load_static_reconstruction_model,
    MeshVertexKalmanSmoother,
    MeshVertexKalmanSmootherConfig,
    normalized_detections,
    reconstruct_object_mesh_from_detections,
    reconstruct_anchor_geometry_from_frames,
    save_geometry_point_cloud_ply,
    save_geometry_points_json,
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

    _APRILTAG_SIZE_M = 0.018
    _ARM_WORLD_TARGETS = (
        ("tag100_c2", 100, 2),
        ("tag99_c0", 99, 0),
        ("tag98_c0", 98, 0),
    )

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
        self._updating_zed_controls = False
        self._latest_color_frame = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
        self._apriltag_enabled = True
        self._apriltag_family = "tag36h11"
        self._apriltag_nthreads = max(1, min(4, os.cpu_count() or 1))
        self._apriltag_quad_decimate = 1.0
        self._apriltag_quad_sigma = 1.0
        self._apriltag_refine_edges = True
        self._apriltag_decode_sharpening = 0.25
        self._apriltag_status = "Initializing"
        self._apriltag_recon_status = "Waiting for frame"
        self._apriltag_points_text = "Waiting for frame"
        self._apriltag_show_raw_mesh = True
        self._apriltag_show_filtered_mesh = True
        self._apriltag_world_filter_strength = 0.85
        self._apriltag_mesh_filter_strength = 0.60
        self._apriltag_applied_world_filter_strength = self._apriltag_world_filter_strength
        self._apriltag_applied_mesh_filter_strength = self._apriltag_mesh_filter_strength
        self._apriltag_anchor_geometry: AprilTagAnchorGeometry | None = None
        self._apriltag_static_model: AprilTagStaticReconstructionModel | None = None
        self._apriltag_static_model_error: str | None = None
        self._arm_world_calibration_path = Path(self._output_dir) / "arm_world_calibration.json"
        self._arm_world_samples = {name: [] for name, _, _ in self._ARM_WORLD_TARGETS}
        (
            self._arm_world_result,
            loaded_arm_world_samples,
        ) = load_arm_world_calibration(
            self._arm_world_calibration_path,
            return_samples=True,
        )
        for target_name in self._arm_world_samples:
            self._arm_world_samples[target_name] = list(loaded_arm_world_samples.get(target_name, []))
        self._arm_world_status = (
            f"Loaded {self._arm_world_calibration_path.name}"
            if self._arm_world_result is not None
            else "Idle"
        )
        self._latest_arm_cfg = np.zeros(8, dtype=np.float64)
        self._latest_eef_position_base = np.zeros(3, dtype=np.float64)
        self._latest_eef_rotation_base = np.eye(3, dtype=np.float64)
        self._latest_eef_wxyz_base = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._latest_arm_pose_valid = False
        self._latest_tblock_pose_world = None
        self._apriltag_camera_world_smoother = self._build_apriltag_camera_world_smoother(
            self._apriltag_world_filter_strength
        )
        self._apriltag_mesh_vertex_smoother = self._build_apriltag_mesh_vertex_smoother(
            self._apriltag_mesh_filter_strength
        )
        self._pre_reconstruct_active = False
        self._pre_reconstruct_detections: list[list] = []
        self._pre_reconstruct_last_capture_t = 0.0
        self._pre_reconstruct_last_center = None
        self._pre_reconstruct_last_bbox_diag = None
        self._pre_reconstruct_status = "Idle"
        self._pre_reconstruct_last_saved_path: Path | None = None
        self._apriltag_background_handles = {}
        self._apriltag_camera_handle = None
        self._apriltag_raw_mesh_handle = None
        self._apriltag_filtered_mesh_handle = None
        self._replay_tblock_root_handle = None
        self._replay_tblock_mesh_handle = None
        self._apriltag_filter_divergence_start_s: float | None = None
        self._base_arm_root_handle = None
        self._arm_world_root_handle = None
        self._urdf_world_vis = None
        self._stick_world_handle = None
        self._stick_tip_world_handle = None
        self._apriltag_detector = self._create_apriltag_detector()
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

    def _build_apriltag_camera_world_smoother(self, strength: float) -> AprilTagPoseKalmanSmoother:
        strength = float(np.clip(strength, 0.0, 1.0))
        return AprilTagPoseKalmanSmoother(
            AprilTagPoseKalmanSmootherConfig(
                translation=ConstantVelocityKalmanConfig(
                    process_accel_std=float(np.interp(strength, [0.0, 1.0], [2.0, 0.25])),
                    measurement_std=float(np.interp(strength, [0.0, 1.0], [0.004, 0.028])),
                    initial_position_std=0.03,
                    initial_velocity_std=0.10,
                ),
                rotation=ConstantVelocityKalmanConfig(
                    process_accel_std=float(np.interp(strength, [0.0, 1.0], [10.0, 2.5])),
                    measurement_std=float(np.interp(strength, [0.0, 1.0], [0.08, 0.20])),
                    initial_position_std=0.2,
                    initial_velocity_std=0.6,
                ),
                max_dt_s=float(np.interp(strength, [0.0, 1.0], [0.15, 0.28])),
                reset_timeout_s=0.8,
                reproj_error_scale=float(np.interp(strength, [0.0, 1.0], [0.12, 0.24])),
                max_reproj_error_px_for_update=6.5,
                snap_reproj_error_px_threshold=float(np.interp(strength, [0.0, 1.0], [2.6, 1.8])),
                snap_translation_error_m=float(np.interp(strength, [0.0, 1.0], [0.03, 0.06])),
                snap_rotation_error_rad=float(np.interp(strength, [0.0, 1.0], [0.30, 0.50])),
                relock_reproj_error_px_threshold=float(np.interp(strength, [0.0, 1.0], [4.5, 3.8])),
                relock_translation_consistency_m=float(np.interp(strength, [0.0, 1.0], [0.025, 0.012])),
                relock_rotation_consistency_rad=float(np.interp(strength, [0.0, 1.0], [0.20, 0.12])),
                relock_required_frames=int(round(np.interp(strength, [0.0, 1.0], [2, 4]))),
            )
        )

    def _build_apriltag_mesh_vertex_smoother(self, strength: float) -> MeshVertexKalmanSmoother:
        strength = float(np.clip(strength, 0.0, 1.0))
        return MeshVertexKalmanSmoother(
            MeshVertexKalmanSmootherConfig(
                vertex=ConstantVelocityKalmanConfig(
                    process_accel_std=float(np.interp(strength, [0.0, 1.0], [2.0, 0.45])),
                    measurement_std=float(np.interp(strength, [0.0, 1.0], [0.004, 0.020])),
                    initial_position_std=0.020,
                    initial_velocity_std=0.15,
                ),
                max_dt_s=float(np.interp(strength, [0.0, 1.0], [0.12, 0.22])),
                reset_timeout_s=0.7,
                reproj_error_scale=float(np.interp(strength, [0.0, 1.0], [0.10, 0.24])),
            )
        )

    def _sync_apriltag_filter_strength(self):
        world_strength = float(self._apriltag_world_filter_strength_slider.value)
        mesh_strength = float(self._apriltag_mesh_filter_strength_slider.value)

        if abs(world_strength - self._apriltag_applied_world_filter_strength) > 1e-6:
            self._apriltag_world_filter_strength = world_strength
            self._apriltag_applied_world_filter_strength = world_strength
            self._apriltag_camera_world_smoother = self._build_apriltag_camera_world_smoother(
                world_strength
            )
            self._apriltag_filter_divergence_start_s = None

        if abs(mesh_strength - self._apriltag_applied_mesh_filter_strength) > 1e-6:
            self._apriltag_mesh_filter_strength = mesh_strength
            self._apriltag_applied_mesh_filter_strength = mesh_strength
            self._apriltag_mesh_vertex_smoother = self._build_apriltag_mesh_vertex_smoother(
                mesh_strength
            )
            self._apriltag_filter_divergence_start_s = None

    def _arm_world_result_to_config(self) -> dict | None:
        if self._arm_world_result is None:
            return None
        return {
            "type": "arm_world_from_apriltag_points_v1",
            "tip_position_in_eef_m": self._arm_world_result.tip_position_in_eef_m.copy(),
            "T_world_from_base": self._arm_world_result.T_world_from_base.copy(),
            "T_base_from_world": self._arm_world_result.T_base_from_world.copy(),
            "rmse_m": float(self._arm_world_result.rmse_m),
            "max_error_m": float(self._arm_world_result.max_error_m),
            "sample_counts_by_target": dict(self._arm_world_result.sample_counts_by_target),
        }

    def _set_arm_world_status(self, message: str) -> None:
        self._arm_world_status = message
        if hasattr(self, "_arm_world_status_md"):
            self._arm_world_status_md.content = f"**Arm World:** {message}"

    def _update_arm_world_target_status(self) -> None:
        if not hasattr(self, "_arm_world_target_status_md"):
            return
        for target_name, tag_id, corner_idx in self._ARM_WORLD_TARGETS:
            count = len(self._arm_world_samples[target_name])
            self._arm_world_target_status_md[target_name].content = (
                f"**Tag {tag_id} C{corner_idx}:** {count} samples"
            )

    def _arm_world_target_point(self, tag_id: int, corner_idx: int) -> np.ndarray:
        self._ensure_apriltag_static_model_scene()
        if self._apriltag_static_model is None:
            raise RuntimeError("Static AprilTag model is unavailable.")
        return np.asarray(
            self._apriltag_static_model.corner_points_by_tag[tag_id][corner_idx],
            dtype=np.float64,
        )

    def _record_arm_world_sample(self, target_name: str, tag_id: int, corner_idx: int) -> None:
        if not self._latest_arm_pose_valid:
            self._set_arm_world_status("Waiting for a live arm pose before recording.")
            return
        world_point = self._arm_world_target_point(tag_id, corner_idx)
        sample = ArmWorldCalibrationSample(
            target_name=target_name,
            world_point=world_point,
            eef_position_base=self._latest_eef_position_base.copy(),
            eef_rotation_base=self._latest_eef_rotation_base.copy(),
            qpos_rad=self._latest_arm_cfg[:6].copy(),
            gripper_m=float(self._latest_arm_cfg[6] - self._latest_arm_cfg[7]),
            timestamp_s=time.time(),
        )
        self._arm_world_samples[target_name].append(sample)
        self._update_arm_world_target_status()
        self._set_arm_world_status(
            f"Recorded {target_name} | total={sum(len(v) for v in self._arm_world_samples.values())}"
        )

    def _reset_arm_world_calibration(self) -> None:
        self._arm_world_samples = {name: [] for name, _, _ in self._ARM_WORLD_TARGETS}
        self._arm_world_result = None
        self._writer.set_world_config(self._arm_world_result_to_config())
        self._update_arm_world_target_status()
        self._set_arm_world_status("Reset")
        self._set_base_arm_visual_enabled(True)
        if self._stick_world_handle is not None:
            self._stick_world_handle.remove()
            self._stick_world_handle = None
        if self._stick_tip_world_handle is not None:
            self._stick_tip_world_handle.remove()
            self._stick_tip_world_handle = None
        if self._arm_world_root_handle is not None:
            self._arm_world_root_handle.remove()
            self._arm_world_root_handle = None
        self._urdf_world_vis = None

    def _ensure_world_arm_visual(self) -> None:
        if self._arm_world_result is None:
            return
        if self._arm_world_root_handle is None:
            self._arm_world_root_handle = self._server.scene.add_frame(
                "/arm_world",
                axes_length=0.0,
                axes_radius=0.0,
                origin_radius=0.0,
            )
        if self._urdf_world_vis is None:
            self._urdf_world_vis = ViserUrdf(self._server, self._urdf, root_node_name="/arm_world/base")

    def _update_world_arm_visual(self, cfg: np.ndarray) -> None:
        if self._arm_world_result is None:
            return
        self._ensure_world_arm_visual()
        T_world_from_base = self._arm_world_result.T_world_from_base
        xyzw = Rotation.from_matrix(T_world_from_base[:3, :3]).as_quat()
        self._arm_world_root_handle.position = tuple(T_world_from_base[:3, 3])
        self._arm_world_root_handle.wxyz = (
            float(xyzw[3]),
            float(xyzw[0]),
            float(xyzw[1]),
            float(xyzw[2]),
        )
        self._urdf_world_vis.update_cfg(cfg)

        eef_pos_base = self._latest_eef_position_base
        tip_offset_eef = self._arm_world_result.tip_position_in_eef_m
        tip_pos_base = eef_pos_base + self._latest_eef_rotation_base @ tip_offset_eef

        if self._stick_world_handle is not None:
            self._stick_world_handle.remove()
        self._stick_world_handle = self._server.scene.add_line_segments(
            "/arm_world/stick",
            points=np.asarray([[eef_pos_base, tip_pos_base]], dtype=np.float32),
            colors=np.array([255, 80, 80], dtype=np.uint8),
            line_width=4.0,
        )
        if self._stick_tip_world_handle is not None:
            self._stick_tip_world_handle.remove()
        self._stick_tip_world_handle = self._server.scene.add_icosphere(
            "/arm_world/stick_tip",
            radius=0.006,
            color=(255, 220, 80),
            position=tuple(tip_pos_base),
        )

    def _set_base_arm_visual_enabled(self, enabled: bool) -> None:
        if not hasattr(self, "_server"):
            return
        if not enabled:
            if self._base_arm_root_handle is not None:
                self._base_arm_root_handle.remove()
                self._base_arm_root_handle = None
            self._urdf_vis = None
            self._eef_marker = None
            return

        if self._base_arm_root_handle is not None:
            return
        self._base_arm_root_handle = self._server.scene.add_frame(
            "/arm_base_live",
            axes_length=0.0,
            axes_radius=0.0,
            origin_radius=0.0,
        )
        self._urdf_vis = ViserUrdf(self._server, self._urdf, root_node_name="/arm_base_live/base")
        self._eef_marker = self._server.scene.add_icosphere(
            "/arm_base_live/eef_marker",
            radius=0.01,
            color=(255, 80, 80),
            position=(0.0, 0.0, 0.0),
        )

    def _on_arm_world_calibrate(self, _event) -> None:
        samples = [
            sample
            for sample_list in self._arm_world_samples.values()
            for sample in sample_list
        ]
        if not self._latest_arm_pose_valid:
            self._set_arm_world_status("Waiting for a live arm pose before calibrating.")
            return
        try:
            self._arm_world_result = solve_arm_world_calibration(samples)
        except Exception as exc:
            self._set_arm_world_status(f"Calibration failed: {exc}")
            return

        save_path = save_arm_world_calibration(
            self._arm_world_result,
            self._arm_world_calibration_path,
            samples_by_target=self._arm_world_samples,
        )
        self._writer.set_world_config(self._arm_world_result_to_config())
        self._set_base_arm_visual_enabled(False)
        self._set_arm_world_status(
            f"RMSE={self._arm_world_result.rmse_m*1000:.1f}mm | "
            f"max={self._arm_world_result.max_error_m*1000:.1f}mm | saved {save_path.name}"
        )

    def _save_arm_world_calibration_result(self) -> None:
        if self._arm_world_result is None:
            self._set_arm_world_status("No arm calibration result to save.")
            return
        save_path = save_arm_world_calibration(
            self._arm_world_result,
            self._arm_world_calibration_path,
            samples_by_target=self._arm_world_samples,
        )
        self._writer.set_world_config(self._arm_world_result_to_config())
        self._set_arm_world_status(f"Saved {save_path.name}")

    def _load_arm_world_calibration_result(self) -> None:
        loaded, samples_by_target = load_arm_world_calibration(
            self._arm_world_calibration_path,
            return_samples=True,
        )
        if loaded is None:
            self._set_arm_world_status(f"Calibration file not found: {self._arm_world_calibration_path.name}")
            return
        self._arm_world_result = loaded
        self._arm_world_samples = {name: [] for name, _, _ in self._ARM_WORLD_TARGETS}
        for target_name in self._arm_world_samples:
            self._arm_world_samples[target_name] = list(samples_by_target.get(target_name, []))
        self._update_arm_world_target_status()
        self._writer.set_world_config(self._arm_world_result_to_config())
        self._set_base_arm_visual_enabled(False)
        self._set_arm_world_status(
            f"Loaded {self._arm_world_calibration_path.name} | "
            f"RMSE={loaded.rmse_m*1000:.1f}mm | max={loaded.max_error_m*1000:.1f}mm"
        )

    def _bind_arm_world_control_events(self) -> None:
        if not hasattr(self, "_arm_world_target_buttons"):
            return
        for target_name, tag_id, corner_idx in self._ARM_WORLD_TARGETS:
            self._arm_world_target_buttons[target_name].on_click(
                lambda _event, name=target_name, tid=tag_id, cid=corner_idx: self._record_arm_world_sample(
                    name,
                    tid,
                    cid,
                )
            )
        self._arm_world_calibrate_btn.on_click(self._on_arm_world_calibrate)
        self._arm_world_save_btn.on_click(lambda _event: self._save_arm_world_calibration_result())
        self._arm_world_load_btn.on_click(lambda _event: self._load_arm_world_calibration_result())
        self._arm_world_reset_btn.on_click(lambda _event: self._reset_arm_world_calibration())

    def run(self):
        """Start the viser server and run the main loop."""
        server = viser.ViserServer(port=self._port)
        self._server = server
        server.gui.configure_theme(control_width="large")
        server.gui.add_html(
            """
            <style>
              :root {
                --app-shell-navbar-width: 760px !important;
                --app-shell-navbar-offset: 760px !important;
              }
            </style>
            """
        )

        # --- 3D Scene ---
        server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)

        urdf = load_piper_urdf()
        self._urdf = urdf
        self._set_base_arm_visual_enabled(self._arm_world_result is None)

        # World frame calibration visualization
        if self._world_config is not None:
            add_world_frame_visual(server, self._world_config, show_axes=False)

        self._ensure_apriltag_static_model_scene()

        # --- Sidebar GUI ---
        # Task/Instruction inputs at top level
        self._task_input = server.gui.add_text("Task Name", initial_value="")
        self._instr_input = server.gui.add_text("Instruction", initial_value="")

        with server.gui.add_folder("RGB Viewer"):
            self._color_handle = server.gui.add_image(
                np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8),
                label="Color",
            )
            self._save_screenshot_btn = server.gui.add_button("Save Screenshot")

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
            with server.gui.add_folder("AprilTag"):
                self._apriltag_enable_checkbox = server.gui.add_checkbox(
                    "Enable Overlay", initial_value=self._apriltag_enabled
                )
                self._apriltag_family_dropdown = server.gui.add_dropdown(
                    "Family",
                    options=("tag36h11", "tag25h9", "tag16h5", "tagStandard41h12"),
                    initial_value=self._apriltag_family,
                )
                self._apriltag_nthreads_slider = server.gui.add_slider(
                    "Threads",
                    min=1,
                    max=max(1, min(16, os.cpu_count() or 1)),
                    step=1,
                    initial_value=self._apriltag_nthreads,
                )
                self._apriltag_quad_decimate_slider = server.gui.add_slider(
                    "Quad Decimate",
                    min=1.0,
                    max=4.0,
                    step=0.1,
                    initial_value=self._apriltag_quad_decimate,
                )
                self._apriltag_quad_sigma_slider = server.gui.add_slider(
                    "Quad Sigma",
                    min=0.0,
                    max=2.0,
                    step=0.1,
                    initial_value=self._apriltag_quad_sigma,
                )
                self._apriltag_refine_edges_checkbox = server.gui.add_checkbox(
                    "Refine Edges", initial_value=self._apriltag_refine_edges
                )
                self._apriltag_decode_sharpening_slider = server.gui.add_slider(
                    "Decode Sharpening",
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    initial_value=self._apriltag_decode_sharpening,
                )
                self._apriltag_apply_btn = server.gui.add_button("Apply AprilTag Settings")
                self._apriltag_status_md = server.gui.add_markdown(
                    f"**AprilTag:** {self._apriltag_status}"
                )
                self._apriltag_points_visible_checkbox = server.gui.add_checkbox(
                    "Show 3D Point List",
                    initial_value=False,
                )
                self._apriltag_show_raw_mesh_checkbox = server.gui.add_checkbox(
                    "Show Raw Mesh",
                    initial_value=self._apriltag_show_raw_mesh,
                )
                self._apriltag_show_filtered_mesh_checkbox = server.gui.add_checkbox(
                    "Show Filtered Mesh",
                    initial_value=self._apriltag_show_filtered_mesh,
                )
                self._apriltag_world_filter_strength_slider = server.gui.add_slider(
                    "World Camera Filter",
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    initial_value=self._apriltag_world_filter_strength,
                )
                self._apriltag_mesh_filter_strength_slider = server.gui.add_slider(
                    "Mesh Filter",
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    initial_value=self._apriltag_mesh_filter_strength,
                )
                self._apriltag_points_md = server.gui.add_markdown(
                    f"**AprilTag 3D Points:**\n\n{self._apriltag_points_text}",
                    visible=False,
                )
                self._pre_reconstruct_btn = server.gui.add_button("Start Pre-Reconstruct")
                self._pre_reconstruct_reset_btn = server.gui.add_button("Reset Pre-Reconstruct")
                self._pre_reconstruct_status_md = server.gui.add_markdown(
                    f"**Pre-Reconstruct:** {self._pre_reconstruct_status}"
                )
                with server.gui.add_folder("Arm To World"):
                    self._arm_world_target_buttons = {}
                    self._arm_world_target_status_md = {}
                    for target_name, tag_id, corner_idx in self._ARM_WORLD_TARGETS:
                        self._arm_world_target_status_md[target_name] = server.gui.add_markdown(
                            f"**Tag {tag_id} C{corner_idx}:** 0 samples"
                        )
                        self._arm_world_target_buttons[target_name] = server.gui.add_button(
                            f"Record Tag {tag_id} C{corner_idx}"
                        )
                    self._arm_world_calibrate_btn = server.gui.add_button("Calibrate Arm")
                    self._arm_world_save_btn = server.gui.add_button("Save Arm Calibration")
                    self._arm_world_load_btn = server.gui.add_button("Load Arm Calibration")
                    self._arm_world_reset_btn = server.gui.add_button("Reset Arm Calibration")
                    self._arm_world_status_md = server.gui.add_markdown(
                        f"**Arm World:** {self._arm_world_status}"
                    )
            self._camera_resolution_md = server.gui.add_markdown(
                "**Resolution:** ---"
            )
            self._zed_brightness_slider = server.gui.add_slider(
                "ZED Brightness", min=0, max=8, step=1, initial_value=4, visible=False
            )
            self._zed_contrast_slider = server.gui.add_slider(
                "ZED Contrast", min=0, max=8, step=1, initial_value=4, visible=False
            )
            self._zed_hue_slider = server.gui.add_slider(
                "ZED Hue", min=0, max=8, step=1, initial_value=0, visible=False
            )
            self._zed_saturation_slider = server.gui.add_slider(
                "ZED Saturation", min=0, max=8, step=1, initial_value=4, visible=False
            )
            self._zed_sharpness_slider = server.gui.add_slider(
                "ZED Sharpness", min=0, max=8, step=1, initial_value=6, visible=False
            )
            self._zed_auto_wb_checkbox = server.gui.add_checkbox(
                "ZED Auto WB", initial_value=True, visible=False
            )
            self._zed_wb_slider = server.gui.add_slider(
                "ZED WB Temp", min=2800, max=6500, step=100, initial_value=4600, visible=False
            )
            self._zed_led_checkbox = server.gui.add_checkbox(
                "ZED LED", initial_value=False, visible=False
            )
            if self._has_depth:
                self._depth_handle = server.gui.add_image(
                    np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8),
                    label="Depth",
                )
            else:
                self._depth_handle = None

        eef_label = (
            "EEF Position (AprilTag World)"
            if self._arm_world_result is not None
            else ("EEF Position (World)" if self._world_config is not None else "EEF Position (Base)")
        )
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
        self._save_screenshot_btn.on_click(self._on_save_screenshot_click)
        self._bind_zed_control_events()
        self._bind_apriltag_control_events()
        self._bind_arm_world_control_events()
        self._update_arm_world_target_status()
        self._writer.set_world_config(self._arm_world_result_to_config())

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

    def _detect_apriltags(self, color: np.ndarray) -> list:
        if self._apriltag_detector is None or color.ndim != 3 or color.shape[2] != 3:
            return []
        if color.shape[0] < 2 or color.shape[1] < 2:
            return []

        try:
            gray = cv2.cvtColor(color, cv2.COLOR_RGB2GRAY)
            detections = self._apriltag_detector.detect(gray)
        except Exception as exc:
            self._apriltag_status = f"Detect failed: {exc}"
            self._apriltag_recon_status = "Detection failed"
            self._clear_apriltag_live_scene()
            return []

        self._apriltag_status = f"Detected {len(detections)} tag(s)"
        return detections

    def _clear_apriltag_scene(self):
        for handle in self._apriltag_background_handles.values():
            handle.remove()
        self._apriltag_background_handles = {}
        self._clear_apriltag_live_scene()

    def _clear_apriltag_live_scene(self, *, reset_divergence_timer: bool = True):
        if self._apriltag_camera_handle is not None:
            self._apriltag_camera_handle.remove()
            self._apriltag_camera_handle = None
        if self._apriltag_raw_mesh_handle is not None:
            self._apriltag_raw_mesh_handle.remove()
            self._apriltag_raw_mesh_handle = None
        if self._apriltag_filtered_mesh_handle is not None:
            self._apriltag_filtered_mesh_handle.remove()
            self._apriltag_filtered_mesh_handle = None
        if reset_divergence_timer:
            self._apriltag_filter_divergence_start_s = None

    def _ensure_apriltag_static_model_scene(self):
        if self._apriltag_static_model is None and self._apriltag_static_model_error is None:
            try:
                self._apriltag_static_model = load_static_reconstruction_model()
            except Exception as exc:
                self._apriltag_static_model_error = str(exc)

        if self._apriltag_static_model is None:
            self._apriltag_recon_status = (
                f"Model init failed: {self._apriltag_static_model_error}"
            )
            self._apriltag_points_text = self._apriltag_recon_status
            return

        if self._apriltag_background_handles:
            return

        tag_colors = {
            98: (255, 140, 0),
            99: (0, 180, 255),
            100: (80, 255, 80),
            76: (255, 255, 255),
            53: (255, 80, 80),
            101: (255, 220, 80),
            77: (170, 120, 255),
        }
        for tag_id in BACKGROUND_TAG_IDS:
            tag_points = self._apriltag_static_model.corner_points_by_tag.get(tag_id)
            if tag_points is None:
                continue
            for corner_idx, point_world in enumerate(tag_points):
                handle_name = f"/apriltag_background/tag_{tag_id}_corner_{corner_idx}"
                self._apriltag_background_handles[(tag_id, corner_idx)] = (
                    self._server.scene.add_icosphere(
                        handle_name,
                        radius=0.0035,
                        color=tag_colors.get(tag_id, (220, 220, 220)),
                        position=tuple(point_world),
                    )
                )
        self._apriltag_recon_status = "Static model ready"
        self._apriltag_points_text = self._format_live_reconstruction_text()

    def _format_live_reconstruction_text(
        self,
        live_result=None,
        filtered_transform=None,
        filtered_camera_from_world=None,
        filtered_mesh_center_world=None,
        center_delta_m: float | None = None,
        filter_reinitialized: bool = False,
    ) -> str:
        if self._apriltag_static_model is None:
            return self._apriltag_static_model_error or "Static model unavailable"

        tag_100 = self._apriltag_static_model.corner_points_by_tag[100]
        lines = [
            f"- Static model: `{self._apriltag_static_model.json_path.name}` + `{self._apriltag_static_model.mesh_path.name}`",
            f"- T1 tag100 C2: `[{tag_100[2][0]:+.4f}, {tag_100[2][1]:+.4f}, {tag_100[2][2]:+.4f}] m`",
            f"- T1 tag100 C1: `[{tag_100[1][0]:+.4f}, {tag_100[1][1]:+.4f}, {tag_100[1][2]:+.4f}] m`",
            f"- T1 tag100 C3: `[{tag_100[3][0]:+.4f}, {tag_100[3][1]:+.4f}, {tag_100[3][2]:+.4f}] m`",
            f"- Background tags in C1: `{', '.join(str(tag_id) for tag_id in BACKGROUND_TAG_IDS)}`",
        ]
        if live_result is None:
            lines.append(
                f"- Filter strength: world=`{self._apriltag_world_filter_strength:.2f}`, mesh=`{self._apriltag_mesh_filter_strength:.2f}`"
            )
            lines.append(f"- Live status: `{self._apriltag_recon_status}`")
            return "\n".join(lines)

        raw_translation = live_result.T_world_from_object.translation
        raw_camera_center = live_result.camera_from_world.inverse().translation
        raw_camera_object_translation = live_result.camera_from_object.translation
        lines.extend(
            [
                f"- World PnP tags: `{', '.join(str(tag_id) for tag_id in live_result.visible_background_tag_ids)}`",
                f"- Object PnP tags: `{', '.join(str(tag_id) for tag_id in live_result.visible_object_tag_ids)}`",
                f"- Raw camera(world) center: `[{raw_camera_center[0]:+.4f}, {raw_camera_center[1]:+.4f}, {raw_camera_center[2]:+.4f}] m`",
                f"- Raw camera(object) tvec: `[{raw_camera_object_translation[0]:+.4f}, {raw_camera_object_translation[1]:+.4f}, {raw_camera_object_translation[2]:+.4f}] m`",
                f"- Raw object translation: `[{raw_translation[0]:+.4f}, {raw_translation[1]:+.4f}, {raw_translation[2]:+.4f}] m`",
                f"- Reproj: world=`{live_result.world_reproj_error_px:.2f} px`, object=`{live_result.object_reproj_error_px:.2f} px`",
                f"- Filter strength: world=`{self._apriltag_world_filter_strength:.2f}`, mesh=`{self._apriltag_mesh_filter_strength:.2f}`",
            ]
        )
        if filtered_transform is not None and filtered_camera_from_world is not None:
            filtered_translation = filtered_transform.translation
            delta = filtered_translation - raw_translation
            filtered_camera_center = filtered_camera_from_world.inverse().translation
            lines.extend(
                [
                    f"- Filtered camera(world) center: `[{filtered_camera_center[0]:+.4f}, {filtered_camera_center[1]:+.4f}, {filtered_camera_center[2]:+.4f}] m`",
                    f"- World-camera-filtered translation: `[{filtered_translation[0]:+.4f}, {filtered_translation[1]:+.4f}, {filtered_translation[2]:+.4f}] m`",
                    f"- World-camera delta: `[{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}] m`",
                    "- Pose smoothing: `camera_from_world Kalman + mesh-vertex Kalman`",
                ]
            )
        if filtered_mesh_center_world is not None:
            lines.append(
                f"- Filtered mesh center: `[{filtered_mesh_center_world[0]:+.4f}, {filtered_mesh_center_world[1]:+.4f}, {filtered_mesh_center_world[2]:+.4f}] m`"
            )
        if center_delta_m is not None:
            lines.append(f"- Raw/filtered center delta: `{center_delta_m:.4f} m`")
        if filter_reinitialized:
            lines.append("- Filter reset: `reinitialized from raw mesh vertices`")
            return "\n".join(lines)

    def _serialize_tblock_pose_world(self, transform) -> dict | None:
        if transform is None:
            return None
        xyzw = Rotation.from_matrix(transform.rotation).as_quat()
        return {
            "translation_m": np.asarray(transform.translation, dtype=np.float64).copy(),
            "wxyz": np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64),
        }

    def _parse_tblock_pose_world(self, value) -> dict | None:
        if not isinstance(value, dict):
            return None
        try:
            translation = np.asarray(value.get("translation_m"), dtype=np.float64).reshape(3)
            wxyz = np.asarray(value.get("wxyz"), dtype=np.float64).reshape(4)
        except (TypeError, ValueError):
            return None
        if not (np.all(np.isfinite(translation)) and np.all(np.isfinite(wxyz))):
            return None
        return {
            "translation_m": translation,
            "wxyz": wxyz,
        }

    def _remove_replay_tblock_visual(self) -> None:
        if self._replay_tblock_root_handle is not None:
            self._replay_tblock_root_handle.remove()
            self._replay_tblock_root_handle = None
        self._replay_tblock_mesh_handle = None

    def _update_replay_tblock_visual(self, pose_world: dict | None) -> bool:
        pose_world = self._parse_tblock_pose_world(pose_world)
        if pose_world is None:
            self._remove_replay_tblock_visual()
            return False

        self._ensure_apriltag_static_model_scene()
        if self._apriltag_static_model is None:
            self._remove_replay_tblock_visual()
            return False

        if self._replay_tblock_root_handle is None:
            import trimesh

            self._replay_tblock_root_handle = self._server.scene.add_frame(
                "/apriltag_replay/tblock",
                axes_length=0.0,
                axes_radius=0.0,
                origin_radius=0.0,
            )
            replay_mesh = trimesh.Trimesh(
                vertices=self._apriltag_static_model.mesh_vertices,
                faces=self._apriltag_static_model.mesh_faces,
                process=False,
            )
            replay_mesh.visual.face_colors = np.tile(
                np.array([[60, 180, 255, 180]], dtype=np.uint8),
                (len(self._apriltag_static_model.mesh_faces), 1),
            )
            self._replay_tblock_mesh_handle = self._server.scene.add_mesh_trimesh(
                "/apriltag_replay/tblock/mesh",
                replay_mesh,
            )

        self._replay_tblock_root_handle.position = tuple(pose_world["translation_m"])
        self._replay_tblock_root_handle.wxyz = tuple(pose_world["wxyz"])
        return True

    def _update_apriltag_reconstruction(
        self,
        detections: list,
        camera_info: dict | None,
        timestamp_s: float | None = None,
    ):
        self._latest_tblock_pose_world = None
        self._ensure_apriltag_static_model_scene()
        if self._apriltag_static_model is None:
            self._clear_apriltag_live_scene()
            return

        if not detections:
            self._apriltag_recon_status = "No tags"
            self._apriltag_points_text = self._format_live_reconstruction_text()
            self._clear_apriltag_live_scene()
            return

        camera_matrix, dist_coeffs, calib_error = camera_calibration_from_info(camera_info)
        if calib_error is not None:
            self._apriltag_recon_status = calib_error
            self._apriltag_points_text = self._format_live_reconstruction_text()
            self._clear_apriltag_live_scene()
            return

        reconstruction, recon_status = reconstruct_object_mesh_from_detections(
            detections=normalized_detections(detections),
            model=self._apriltag_static_model,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
        )
        if reconstruction is None:
            self._apriltag_recon_status = recon_status
            self._apriltag_points_text = self._format_live_reconstruction_text()
            self._clear_apriltag_live_scene()
            return

        pose_timestamp_s = time.time() if timestamp_s is None or timestamp_s <= 0.0 else float(timestamp_s)
        filtered_camera_from_world = self._apriltag_camera_world_smoother.update(
            reconstruction.camera_from_world,
            timestamp_s=pose_timestamp_s,
            reproj_error_px=float(reconstruction.world_reproj_error_px),
            visible_tag_count=len(reconstruction.visible_background_tag_ids),
        )
        world_camera_filtered_transform = filtered_camera_from_world.inverse().compose(
            reconstruction.camera_from_object,
        )
        world_camera_filtered_mesh_vertices_world = world_camera_filtered_transform.apply_points(
            self._apriltag_static_model.mesh_vertices
        )
        filtered_mesh_vertices_world = self._apriltag_mesh_vertex_smoother.update(
            world_camera_filtered_mesh_vertices_world,
            timestamp_s=pose_timestamp_s,
            reproj_error_px=float(reconstruction.object_reproj_error_px),
        )
        raw_mesh_vertices_world = reconstruction.mesh_vertices_world
        raw_center_world = np.mean(raw_mesh_vertices_world, axis=0)
        filtered_center_world = np.mean(filtered_mesh_vertices_world, axis=0)
        center_delta_m = float(np.linalg.norm(filtered_center_world - raw_center_world))
        filter_reinitialized = False

        if center_delta_m > 0.006:
            if self._apriltag_filter_divergence_start_s is None:
                self._apriltag_filter_divergence_start_s = pose_timestamp_s
            elif pose_timestamp_s - self._apriltag_filter_divergence_start_s >= 1.0:
                self._apriltag_camera_world_smoother.reset()
                self._apriltag_mesh_vertex_smoother.reset()
                filtered_camera_from_world = self._apriltag_camera_world_smoother.update(
                    reconstruction.camera_from_world,
                    timestamp_s=pose_timestamp_s,
                    reproj_error_px=float(reconstruction.world_reproj_error_px),
                    visible_tag_count=len(reconstruction.visible_background_tag_ids),
                )
                world_camera_filtered_transform = filtered_camera_from_world.inverse().compose(
                    reconstruction.camera_from_object,
                )
                world_camera_filtered_mesh_vertices_world = world_camera_filtered_transform.apply_points(
                    self._apriltag_static_model.mesh_vertices
                )
                filtered_mesh_vertices_world = self._apriltag_mesh_vertex_smoother.update(
                    world_camera_filtered_mesh_vertices_world,
                    timestamp_s=pose_timestamp_s,
                    reproj_error_px=float(reconstruction.object_reproj_error_px),
                )
                filtered_center_world = np.mean(filtered_mesh_vertices_world, axis=0)
                center_delta_m = float(np.linalg.norm(filtered_center_world - raw_center_world))
                self._apriltag_filter_divergence_start_s = None
                filter_reinitialized = True
        else:
            self._apriltag_filter_divergence_start_s = None

        self._clear_apriltag_live_scene(reset_divergence_timer=False)
        self._apriltag_camera_handle = self._server.scene.add_icosphere(
            "/apriltag_camera_center",
            radius=0.006,
            color=(255, 80, 80),
            position=tuple(filtered_camera_from_world.inverse().translation),
        )
        import trimesh

        raw_mesh = trimesh.Trimesh(
            vertices=raw_mesh_vertices_world,
            faces=reconstruction.mesh_faces,
            process=False,
        )
        raw_mesh.visual.face_colors = np.tile(
            np.array([[255, 140, 0, 90]], dtype=np.uint8),
            (len(reconstruction.mesh_faces), 1),
        )
        filtered_mesh = trimesh.Trimesh(
            vertices=filtered_mesh_vertices_world,
            faces=reconstruction.mesh_faces,
            process=False,
        )
        filtered_mesh.visual.face_colors = np.tile(
            np.array([[60, 180, 255, 180]], dtype=np.uint8),
            (len(reconstruction.mesh_faces), 1),
        )
        if self._apriltag_show_raw_mesh:
            self._apriltag_raw_mesh_handle = self._server.scene.add_mesh_trimesh(
                "/apriltag_reconstruction/object_mesh_raw",
                raw_mesh,
            )
        if self._apriltag_show_filtered_mesh:
            self._apriltag_filtered_mesh_handle = self._server.scene.add_mesh_trimesh(
                "/apriltag_reconstruction/object_mesh_filtered",
                filtered_mesh,
            )
        self._latest_tblock_pose_world = self._serialize_tblock_pose_world(
            world_camera_filtered_transform
        )
        self._apriltag_recon_status = (
            f"World PnP={reconstruction.world_reproj_error_px:.2f}px | "
            f"Object PnP={reconstruction.object_reproj_error_px:.2f}px"
        )
        self._apriltag_points_text = self._format_live_reconstruction_text(
            reconstruction,
            filtered_transform=world_camera_filtered_transform,
            filtered_camera_from_world=filtered_camera_from_world,
            filtered_mesh_center_world=filtered_center_world,
            center_delta_m=center_delta_m,
            filter_reinitialized=filter_reinitialized,
        )


    def _create_apriltag_detector(self):
        if not self._apriltag_enabled:
            self._apriltag_status = "Disabled"
            self._apriltag_recon_status = "Disabled"
            self._apriltag_points_text = "Disabled"
            self._clear_apriltag_live_scene()
            return None
        if Detector is None:
            self._apriltag_status = "Module missing: install pupil-apriltags"
            self._apriltag_recon_status = "Detector unavailable"
            self._apriltag_points_text = "Detector unavailable"
            self._clear_apriltag_live_scene()
            print("[AprilTag] pupil_apriltags not installed, GUI overlay disabled")
            return None

        try:
            detector = Detector(
                families=self._apriltag_family,
                nthreads=int(self._apriltag_nthreads),
                quad_decimate=float(self._apriltag_quad_decimate),
                quad_sigma=float(self._apriltag_quad_sigma),
                refine_edges=int(bool(self._apriltag_refine_edges)),
                decode_sharpening=float(self._apriltag_decode_sharpening),
            )
            self._apriltag_status = (
                f"Ready | {self._apriltag_family} | decimate={self._apriltag_quad_decimate:.2f}"
            )
            return detector
        except Exception as exc:
            self._apriltag_status = f"Init failed: {exc}"
            print(f"[AprilTag] Failed to initialize detector: {exc}")
            return None

    def _bind_apriltag_control_events(self):
        self._apriltag_apply_btn.on_click(self._on_apriltag_apply)
        self._pre_reconstruct_btn.on_click(self._on_pre_reconstruct_click)
        self._pre_reconstruct_reset_btn.on_click(self._on_pre_reconstruct_reset_click)

    def _on_apriltag_apply(self, _event):
        self._apriltag_enabled = bool(self._apriltag_enable_checkbox.value)
        self._apriltag_family = str(self._apriltag_family_dropdown.value)
        self._apriltag_nthreads = int(self._apriltag_nthreads_slider.value)
        self._apriltag_quad_decimate = float(self._apriltag_quad_decimate_slider.value)
        self._apriltag_quad_sigma = float(self._apriltag_quad_sigma_slider.value)
        self._apriltag_refine_edges = bool(self._apriltag_refine_edges_checkbox.value)
        self._apriltag_decode_sharpening = float(self._apriltag_decode_sharpening_slider.value)
        self._apriltag_show_raw_mesh = bool(self._apriltag_show_raw_mesh_checkbox.value)
        self._apriltag_show_filtered_mesh = bool(self._apriltag_show_filtered_mesh_checkbox.value)
        self._apriltag_world_filter_strength = float(self._apriltag_world_filter_strength_slider.value)
        self._apriltag_mesh_filter_strength = float(self._apriltag_mesh_filter_strength_slider.value)
        self._apriltag_applied_world_filter_strength = self._apriltag_world_filter_strength
        self._apriltag_applied_mesh_filter_strength = self._apriltag_mesh_filter_strength
        self._apriltag_camera_world_smoother = self._build_apriltag_camera_world_smoother(
            self._apriltag_world_filter_strength
        )
        self._apriltag_mesh_vertex_smoother = self._build_apriltag_mesh_vertex_smoother(
            self._apriltag_mesh_filter_strength
        )
        self._apriltag_filter_divergence_start_s = None
        self._apriltag_detector = self._create_apriltag_detector()
        self._apriltag_status_md.content = f"**AprilTag:** {self._apriltag_status}"

    def _on_pre_reconstruct_click(self, _event):
        if not self._pre_reconstruct_active:
            self._pre_reconstruct_active = True
            self._pre_reconstruct_detections = []
            self._pre_reconstruct_last_capture_t = 0.0
            self._pre_reconstruct_last_center = None
            self._pre_reconstruct_last_bbox_diag = None
            self._pre_reconstruct_status = "Capturing views"
            self._pre_reconstruct_btn.label = "Stop Pre-Reconstruct"
            return

        self._pre_reconstruct_active = False
        self._pre_reconstruct_btn.label = "Start Pre-Reconstruct"
        self._pre_reconstruct_status = (
            f"Optimizing ({len(self._pre_reconstruct_detections)} captured views)..."
        )
        self._pre_reconstruct_status_md.content = (
            f"**Pre-Reconstruct:** {self._pre_reconstruct_status}"
        )
        camera_info = self._camera.get_camera_info() if self._camera is not None else None
        camera_matrix, dist_coeffs, calib_error = camera_calibration_from_info(camera_info)
        if calib_error is not None:
            self._pre_reconstruct_status = calib_error
            self._apriltag_recon_status = calib_error
            self._apriltag_points_text = calib_error
            self._pre_reconstruct_status_md.content = (
                f"**Pre-Reconstruct:** {self._pre_reconstruct_status}"
            )
            return

        def progress_update(message: str):
            self._pre_reconstruct_status = message
            self._pre_reconstruct_status_md.content = (
                f"**Pre-Reconstruct:** {self._pre_reconstruct_status}"
            )

        geometry = reconstruct_anchor_geometry_from_frames(
            detection_frames=self._pre_reconstruct_detections,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            tag_size_m=self._APRILTAG_SIZE_M,
            progress_callback=progress_update,
        )
        if geometry is None:
            self._pre_reconstruct_status = (
                f"Failed ({len(self._pre_reconstruct_detections)} captured views) | insufficient geometry/anchors"
            )
            self._apriltag_anchor_geometry = None
            self._clear_apriltag_live_scene()
            self._pre_reconstruct_status_md.content = (
                f"**Pre-Reconstruct:** {self._pre_reconstruct_status}"
            )
            return

        self._apriltag_anchor_geometry = geometry
        reconstruction_dir = Path(self._output_dir) / "apriltag_reconstruction"
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        ply_path = reconstruction_dir / f"apriltag_corners_{timestamp}.ply"
        json_path = reconstruction_dir / f"apriltag_corners_{timestamp}.json"
        saved_ply_path = save_geometry_point_cloud_ply(geometry, ply_path)
        saved_json_path = save_geometry_points_json(geometry, json_path)
        self._pre_reconstruct_last_saved_path = saved_ply_path
        self._pre_reconstruct_status = (
            f"Solved {geometry.optimized_views} views | mean={geometry.mean_reproj_error_px:.2f}px | "
            f"max={geometry.max_reproj_error_px:.2f}px | saved {saved_ply_path.name}, {saved_json_path.name}"
        )
        self._pre_reconstruct_status_md.content = (
            f"**Pre-Reconstruct:** {self._pre_reconstruct_status}"
        )

    def _on_pre_reconstruct_reset_click(self, _event):
        self._pre_reconstruct_active = False
        self._pre_reconstruct_detections = []
        self._pre_reconstruct_last_capture_t = 0.0
        self._pre_reconstruct_last_center = None
        self._pre_reconstruct_last_bbox_diag = None
        self._pre_reconstruct_status = "Idle"
        self._pre_reconstruct_last_saved_path = None
        self._pre_reconstruct_btn.label = "Start Pre-Reconstruct"
        self._apriltag_anchor_geometry = None
        self._clear_apriltag_live_scene()
        self._pre_reconstruct_status_md.content = (
            f"**Pre-Reconstruct:** {self._pre_reconstruct_status}"
        )

    def _format_anchor_geometry_text(self, geometry: AprilTagAnchorGeometry) -> str:
        lines = [
            f"- Scale factor: `{geometry.scale_factor:.4f}`",
            f"- Mean reproj error: `{geometry.mean_reproj_error_px:.3f} px`",
            f"- Max reproj error: `{geometry.max_reproj_error_px:.3f} px`",
            f"- Optimized views: `{geometry.optimized_views}`",
        ]
        for tag_id in sorted(geometry.corner_points_by_tag):
            tag_points = geometry.corner_points_by_tag[tag_id]
            for corner_idx, point_world in enumerate(tag_points):
                count = geometry.sample_counts_by_corner.get((tag_id, corner_idx), 0)
                lines.append(
                    f"- Tag {tag_id} C{corner_idx}: "
                    f"`[{point_world[0]:+.4f}, {point_world[1]:+.4f}, {point_world[2]:+.4f}] m` "
                    f"(samples={count})"
                )
        return "\n".join(lines)

    def _maybe_collect_pre_reconstruct_frame(self, detections: list):
        if not self._pre_reconstruct_active:
            return

        normalized = normalized_detections(detections)
        if len(normalized) < 2:
            self._pre_reconstruct_status = (
                f"Capturing views ({len(self._pre_reconstruct_detections)} saved) | need >=2 tags"
            )
            return

        total_corners = sum(detection.corners.shape[0] for detection in normalized)
        if total_corners < 8:
            self._pre_reconstruct_status = (
                f"Capturing views ({len(self._pre_reconstruct_detections)} saved) | need >=8 corners"
            )
            return

        now = time.time()
        all_corners = np.concatenate([detection.corners for detection in normalized], axis=0)
        center = np.mean(all_corners, axis=0)
        bbox_min = np.min(all_corners, axis=0)
        bbox_max = np.max(all_corners, axis=0)
        bbox_diag = float(np.linalg.norm(bbox_max - bbox_min))
        elapsed_since_last = now - self._pre_reconstruct_last_capture_t
        if self._pre_reconstruct_last_capture_t > 0.0 and elapsed_since_last < 1.0:
            self._pre_reconstruct_status = (
                f"Capturing views ({len(self._pre_reconstruct_detections)} saved) | waiting {1.0 - elapsed_since_last:.1f}s"
            )
            return
        moved_enough = (
            self._pre_reconstruct_last_center is None
            or float(np.linalg.norm(center - self._pre_reconstruct_last_center)) >= 60.0
        )
        scale_changed_enough = (
            self._pre_reconstruct_last_bbox_diag is None
            or abs(bbox_diag - self._pre_reconstruct_last_bbox_diag) >= 40.0
        )
        if not (moved_enough or scale_changed_enough):
            self._pre_reconstruct_status = (
                f"Capturing views ({len(self._pre_reconstruct_detections)} saved)"
            )
            return

        self._pre_reconstruct_detections.append(normalized)
        self._pre_reconstruct_last_capture_t = now
        self._pre_reconstruct_last_center = center
        self._pre_reconstruct_last_bbox_diag = bbox_diag
        self._pre_reconstruct_status = (
            f"Capturing views ({len(self._pre_reconstruct_detections)} saved)"
        )

    def _sync_apriltag_point_list_visibility(self):
        self._apriltag_points_md.visible = bool(self._apriltag_points_visible_checkbox.value)
        self._apriltag_show_raw_mesh = bool(self._apriltag_show_raw_mesh_checkbox.value)
        self._apriltag_show_filtered_mesh = bool(self._apriltag_show_filtered_mesh_checkbox.value)
        self._sync_apriltag_filter_strength()

    def _render_apriltag_overlay(self, color: np.ndarray, detections: list | None = None) -> np.ndarray:
        if self._apriltag_detector is None or color.ndim != 3 or color.shape[2] != 3:
            return color
        if color.shape[0] < 2 or color.shape[1] < 2:
            return color

        if detections is None:
            detections = self._detect_apriltags(color)
        if not detections:
            return color

        preview = color.copy()
        for detection in detections:
            corners = np.round(detection.corners).astype(int)
            center = tuple(np.round(detection.center).astype(int))

            for i in range(4):
                pt1 = tuple(corners[i])
                pt2 = tuple(corners[(i + 1) % 4])
                cv2.line(preview, pt1, pt2, (0, 255, 0), 2)

            corner_colors = [
                (255, 0, 0),
                (255, 165, 0),
                (0, 255, 0),
                (0, 0, 255),
            ]
            for corner_idx, corner in enumerate(corners):
                corner_color = corner_colors[corner_idx % len(corner_colors)]
                corner_xy = tuple(corner)
                cv2.circle(
                    preview,
                    corner_xy,
                    4,
                    corner_color,
                    -1,
                )
                cv2.putText(
                    preview,
                    str(corner_idx),
                    (corner_xy[0] + 6, corner_xy[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    corner_color,
                    2,
                    cv2.LINE_AA,
                )

            cv2.circle(preview, center, 5, (0, 128, 255), -1)
            cv2.putText(
                preview,
                f"ID: {detection.tag_id}",
                (center[0] - 24, center[1] - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 128, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.putText(
            preview,
            f"AprilTags: {len(detections)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        return preview

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

                preview_color = color
                self._clear_apriltag_live_scene()
                has_tblock_pose = self._update_replay_tblock_visual(
                    rd["tblock_pose_world"][idx]
                )
                self._apriltag_status = "Replay mode: overlay disabled"
                if has_tblock_pose:
                    self._apriltag_recon_status = "Replay mode: recorded Tblock pose"
                    self._apriltag_points_text = "Replay mode: recorded Tblock pose"
                else:
                    self._apriltag_recon_status = "Replay mode: no recorded Tblock pose"
                    self._apriltag_points_text = "Replay mode: no recorded Tblock pose"
                self._replay_idx += 1
                progress = f"{idx + 1}/{rd['num_frames']}"
                self._status_md.content = f"**Status:** REPLAY | Frame {progress}"
            else:
                # --- Live mode ---
                self._sync_camera_selection()
                self._sync_opencv_zed_mode()
                self._sync_zed_controls()
                self._sync_apriltag_point_list_visibility()
                if self._camera is not None and hasattr(self._camera, "get_camera_info"):
                    self._writer.set_camera_info(self._camera.get_camera_info())
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

                camera_info = self._camera.get_camera_info() if self._camera is not None else None
                detections = self._detect_apriltags(color)
                self._maybe_collect_pre_reconstruct_frame(detections)
                pose_timestamp_s = camera_timestamp if self._camera is not None else time.time()
                self._update_apriltag_reconstruction(
                    detections,
                    camera_info,
                    timestamp_s=pose_timestamp_s,
                )
                preview_color = self._render_apriltag_overlay(color, detections)

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
            self._latest_arm_cfg = cfg.copy()
            (
                self._latest_eef_position_base,
                self._latest_eef_rotation_base,
                self._latest_eef_wxyz_base,
            ) = eef_pose_from_urdf_cfg(self._urdf, cfg)
            self._latest_arm_pose_valid = True
            if self._urdf_vis is not None:
                self._urdf_vis.update_cfg(cfg)
            self._update_world_arm_visual(cfg)

            # Update camera display
            self._latest_color_frame = color.copy()
            self._color_handle.image = preview_color
            self._apriltag_status_md.content = f"**AprilTag:** {self._apriltag_status}"
            self._pre_reconstruct_status_md.content = (
                f"**Pre-Reconstruct:** {self._pre_reconstruct_status}"
            )
            self._apriltag_points_md.content = (
                f"**AprilTag 3D Points:**\n\n{self._apriltag_points_text}"
            )
            if hasattr(self, "_arm_world_status_md"):
                self._arm_world_status_md.content = f"**Arm World:** {self._arm_world_status}"
            self._camera_resolution_md.content = self._format_camera_resolution(
                color=color,
                depth=depth,
            )

            # Compute fingertip endpoint center via URDF FK (link7/link8 tip midpoint)
            eef_pos_base = fingertip_center_from_urdf_cfg(self._urdf, cfg)
            if self._eef_marker is not None:
                self._eef_marker.position = tuple(eef_pos_base)
            if self._arm_world_result is not None:
                eef_pos = point_base_to_world(
                    eef_pos_base,
                    self._arm_world_result.T_world_from_base,
                )
                frame_label = "AprilTag World"
            elif self._T_world_from_base is not None:
                eef_pos = point_base_to_world(eef_pos_base, self._T_world_from_base)
                frame_label = "World"
            else:
                eef_pos = eef_pos_base
                frame_label = "Base"

            # Update arm state display
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
                    tblock_pose_world=self._latest_tblock_pose_world,
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

    def _on_save_screenshot_click(self, _event):
        frame = self._latest_color_frame
        if frame is None or frame.size == 0:
            self._status_md.content = "**Status:** No camera frame available"
            return

        screenshot_dir = Path(self._output_dir) / "screenshots"
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        screenshot_path = screenshot_dir / f"camera_{timestamp}.png"
        ok = cv2.imwrite(str(screenshot_path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            self._status_md.content = "**Status:** Failed to save screenshot"
            return
        self._status_md.content = f"**Status:** Screenshot saved: {screenshot_path}"
        print(f">> Screenshot saved: {screenshot_path}")

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
            "tblock_pose_world": [
                frame.get("tblock_pose_world")
                for frame in frame_records[:num_frames]
            ],
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
        self._remove_replay_tblock_visual()
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
        self._update_zed_control_visibility(current_label)

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
        elif source_id.startswith("zed:"):
            backend = "ZED Open Capture"
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

    def _bind_zed_control_events(self):
        self._zed_brightness_slider.on_update(lambda _: self._apply_zed_control("brightness", self._zed_brightness_slider.value))
        self._zed_contrast_slider.on_update(lambda _: self._apply_zed_control("contrast", self._zed_contrast_slider.value))
        self._zed_hue_slider.on_update(lambda _: self._apply_zed_control("hue", self._zed_hue_slider.value))
        self._zed_saturation_slider.on_update(lambda _: self._apply_zed_control("saturation", self._zed_saturation_slider.value))
        self._zed_sharpness_slider.on_update(lambda _: self._apply_zed_control("sharpness", self._zed_sharpness_slider.value))
        self._zed_auto_wb_checkbox.on_update(lambda _: self._apply_zed_control("auto_white_balance", self._zed_auto_wb_checkbox.value))
        self._zed_wb_slider.on_update(lambda _: self._apply_zed_control("white_balance_temperature", self._zed_wb_slider.value))
        self._zed_led_checkbox.on_update(lambda _: self._apply_zed_control("led", self._zed_led_checkbox.value))

    def _apply_zed_control(self, name: str, value) -> None:
        if self._updating_zed_controls:
            return
        if self._camera is None or not hasattr(self._camera, "set_control"):
            return
        current_label = self._get_current_camera_label()
        source_id = self._camera_label_to_id.get(current_label)
        if source_id is None or not source_id.startswith("zed:"):
            return
        try:
            self._camera.set_control(name, value)
        except Exception as exc:
            self._camera_status_md.content = f"**Camera:** {exc}"
            return
        self._sync_zed_controls()

    def _sync_zed_controls(self) -> None:
        current_label = self._get_current_camera_label()
        source_id = self._camera_label_to_id.get(current_label)
        is_zed = source_id is not None and source_id.startswith("zed:")
        if not is_zed or self._camera is None or not hasattr(self._camera, "get_control_state"):
            return
        try:
            control_state = self._camera.get_control_state()
        except Exception as exc:
            self._camera_status_md.content = f"**Camera:** {exc}"
            return
        if control_state is None:
            return
        self._updating_zed_controls = True
        try:
            self._zed_brightness_slider.value = int(control_state["brightness"])
            self._zed_contrast_slider.value = int(control_state["contrast"])
            self._zed_hue_slider.value = int(control_state["hue"])
            self._zed_saturation_slider.value = int(control_state["saturation"])
            self._zed_sharpness_slider.value = int(control_state["sharpness"])
            self._zed_auto_wb_checkbox.value = bool(control_state["auto_white_balance"])
            self._zed_wb_slider.value = int(control_state["white_balance_temperature"])
            self._zed_wb_slider.disabled = bool(control_state["auto_white_balance"])
            self._zed_led_checkbox.value = bool(control_state["led"])
        finally:
            self._updating_zed_controls = False

    def _update_zed_control_visibility(self, selection_label: str) -> None:
        source_id = self._camera_label_to_id.get(selection_label)
        is_zed = source_id is not None and source_id.startswith("zed:")
        for handle in (
            self._zed_brightness_slider,
            self._zed_contrast_slider,
            self._zed_hue_slider,
            self._zed_saturation_slider,
            self._zed_sharpness_slider,
            self._zed_auto_wb_checkbox,
            self._zed_wb_slider,
            self._zed_led_checkbox,
        ):
            handle.visible = bool(is_zed)
