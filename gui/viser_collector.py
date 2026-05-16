"""Viser-based web GUI for data collection.

Replaces the OpenCV+Tkinter interface with a viser web app combining:
- 3D arm visualization (URDF-based via ViserUrdf)
- Camera feed display (color + depth)
- Recording controls and status in the sidebar
- Replay of recorded episodes
"""

import base64
import glob as globmod
import http.server
import json
import math
import os
import threading
import time
import zlib
from collections import deque
from dataclasses import replace
from functools import partial
from pathlib import Path
from urllib import error, parse, request

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
    euler_deg_to_wxyz,
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
    RigidTransform,
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
from utils.world_frame import point_base_to_world, point_world_to_base, add_world_frame_visual
from utils.spatial_language import (
    build_voxel_mesh,
    build_composited_sentence,
    compute_spatial_language,
    default_bbox,
    project_point_to_spatial_coord,
    render_projected_voxels_image,
    unavailable_result,
)


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
    _MOVEMENT_ONLY_FORBIDDEN_TOKENS = (
        "state",
        "pusher",
        "tbar",
        "goal",
        "end",
        "left",
        "right",
        "up",
        "down",
    )
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
        inference_server: str = "http://127.0.0.1:8000",
        inference_max_new_tokens: int = 100,
        inference_temperature: float = 1.0,
        inference_top_k: int | None = None,
        inference_do_sample: bool = True,
        inference_forbidden_tokens: str = "",
        inference_require_state_after_movement: bool = False,
        inference_state_after_movement_prob: float = 0.0,
        control_api_host: str = "127.0.0.1",
        control_api_port: int = 8765,
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
        self._gui_preview_max_width = 960
        self._gui_preview_max_height = 720
        self._cached_camera_calibration_text: str | None = None
        self._cached_camera_resolution_text: str | None = None
        self._last_camera_info_signature = None
        self._last_gui_text_values: dict[str, str] = {}
        self._gui_fast_text_update_interval_frames = 10
        self._gui_slow_text_update_interval_frames = 30
        self._apriltag_enabled = True
        self._apriltag_family = "tag36h11"
        self._apriltag_nthreads = 16
        self._apriltag_quad_decimate = 2.0
        self._apriltag_quad_sigma = 1.0
        self._apriltag_refine_edges = True
        self._apriltag_decode_sharpening = 0.25
        self._apriltag_status = "Initializing"
        self._apriltag_recon_status = "Waiting for frame"
        self._apriltag_points_text = "Waiting for frame"
        self._apriltag_show_raw_mesh = True
        self._apriltag_show_filtered_mesh = True
        self._apriltag_world_filter_strength = 0.20
        self._apriltag_mesh_filter_strength = 0.40
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
        self._arm_lock_busy = False
        self._arm_lock_status = "Idle"
        self._arm_move_busy = False
        self._arm_move_status = "Idle"
        self._arm_move_z_index = 3
        self._arm_move_buttons = {}
        self._latest_tblock_pose_world = None
        self._latest_tblock_pose_world_raw = None
        self._apriltag_camera_world_smoother = self._build_apriltag_camera_world_smoother(
            self._apriltag_world_filter_strength
        )
        self._apriltag_tblock_pose_smoother = self._build_apriltag_camera_world_smoother(
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
        self._live_pusher_world_history = deque(maxlen=256)
        self._live_spatial_result_history = deque(maxlen=8)
        self._latest_live_spatial_result = None
        self._latest_available_live_spatial_result = None
        self._online_inference_server = str(inference_server).rstrip("/")
        self._online_inference_max_new_tokens = int(inference_max_new_tokens)
        self._online_inference_temperature = float(inference_temperature)
        self._online_inference_top_k = (
            None if inference_top_k in ("", None) else int(inference_top_k)
        )
        self._online_inference_do_sample = bool(inference_do_sample)
        self._online_inference_forbidden_tokens = str(inference_forbidden_tokens)
        self._online_inference_require_state_after_movement = bool(
            inference_require_state_after_movement
        )
        self._online_inference_state_after_movement_prob = float(
            inference_state_after_movement_prob
        )
        self._control_api_host = str(control_api_host)
        self._control_api_port = int(control_api_port)
        self._control_api_server = None
        self._control_api_thread: threading.Thread | None = None
        self._trajectory_lock = threading.Lock()
        self._trajectory_runner_thread: threading.Thread | None = None
        self._trajectory_session = {
            "run_id": "",
            "running": False,
            "status": "idle",
            "step_index": -1,
            "total_steps": 0,
            "error": "",
            "output_json": "",
            "started_at": None,
            "finished_at": None,
            "trajectory": [],
            "goal": [],
            "frames": [],
        }
        self._episode_session = {
            "episode_id": time.strftime("%Y%m%d_%H%M%S"),
            "started_at": time.time(),
            "finished_at": None,
            "movements": [],
            "output_json": "",
        }
        self._online_inference_movement_only = True
        self._online_inference_busy = False
        self._online_inference_status = "Idle"
        self._online_inference_prompt_text = "waiting for request"
        self._spatial_composited_sentence_text = "waiting for request"
        self._online_inference_output_text = "waiting for response"
        self._online_inference_raw_response_text = "waiting for response"
        self._online_inference_current_pusher_coord_text = "---"
        self._online_inference_vis_image = np.full((16, 16, 3), 255, dtype=np.uint8)
        self._online_inference_details_url = ""
        self._online_inference_details_text = "No detail page yet."
        self._online_inference_details_dir = Path(self._output_dir) / "inference_details"
        self._online_inference_details_httpd = None
        self._online_inference_details_server_thread: threading.Thread | None = None
        self._online_inference_details_base_url = ""
        self._online_inference_active_spatial_result = None
        self._online_inference_prediction_visual_dirty = False
        self._online_inference_pending_spatial_result = None
        self._online_inference_goal_coords = np.zeros((0, 2), dtype=np.int32)
        self._online_inference_goal_dataset_path: Path | None = None
        self._online_inference_goal_source = "No dataset goal loaded"
        self._online_inference_current_pusher_handle = None
        self._online_inference_prediction_path_handle = None
        self._online_inference_prediction_point_handles = []
        self._apriltag_background_handles = {}
        self._apriltag_camera_handle = None
        self._apriltag_raw_mesh_handle = None
        self._apriltag_filtered_mesh_handle = None
        self._replay_tblock_root_handle = None
        self._replay_tblock_mesh_handle = None
        self._spatial_bbox_handle = None
        self._spatial_tblock_voxel_handle = None
        self._spatial_pusher_voxel_handle = None
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
        self._replay_filter_enabled = True
        self._replay_paused = False
        self._replay_step_delta = 0
        self._dataset_selected_episodes: list[str] = []
        self._dataset_global_goal_episode: str | None = None
        bbox_min, bbox_max = default_bbox()
        self._spatial_bbox_min = bbox_min
        self._spatial_bbox_max = bbox_max
        self._spatial_resolution_xyz = np.array([128, 128, 12], dtype=np.int32)
        self._online_inference_vis_image = self._blank_online_inference_vis_image()
        self._pipeline_stage_labels = {
            "settings_sync": "Settings Sync",
            "controls_sync": "Controls Sync",
            "camera_capture": "Camera Capture",
            "apriltag_detect": "AprilTag Detect",
            "pre_reconstruct": "Pre-Reconstruct",
            "apriltag_reconstruct": "AprilTag Reconstruct",
            "overlay_render": "Overlay Render",
            "spatial_gui": "Spatial GUI",
            "depth_display": "Depth Display",
            "replay_frame": "Replay Frame",
            "replay_window": "Replay Window",
            "replay_tblock_visual": "Replay TBlock Visual",
            "replay_spatial": "Replay Spatial",
            "arm_visual": "Arm Visual",
            "gui_image_upload": "GUI Image Upload",
            "gui_text_update": "GUI Text Update",
            "record_write": "Record Write",
            "total": "Total Frame",
        }
        self._pipeline_stage_order = [
            "settings_sync",
            "controls_sync",
            "camera_capture",
            "apriltag_detect",
            "pre_reconstruct",
            "apriltag_reconstruct",
            "overlay_render",
            "spatial_gui",
            "depth_display",
            "replay_frame",
            "replay_window",
            "replay_tblock_visual",
            "replay_spatial",
            "arm_visual",
            "gui_image_upload",
            "gui_text_update",
            "record_write",
            "total",
        ]
        self._pipeline_timing_stats: dict[str, dict[str, float | int | None]] = {}
        self._pipeline_last_timings_ms: dict[str, float] = {}
        self._pipeline_last_total_ms: float | None = None
        self._pipeline_last_mode = "Live"
        self._pipeline_frame_counter = 0
        self._reset_pipeline_timing_stats()
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
                    process_accel_std=float(np.interp(strength, [0.0, 1.0], [1.4, 0.18])),
                    measurement_std=float(np.interp(strength, [0.0, 1.0], [0.008, 0.030])),
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
                reproj_error_scale=float(np.interp(strength, [0.0, 1.0], [0.12, 0.24])),
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
            self._apriltag_tblock_pose_smoother = self._build_apriltag_camera_world_smoother(
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

    def _normalize_coord_list(
        self,
        coords,
    ) -> list[tuple[int, int]]:
        normalized: list[tuple[int, int]] = []
        if coords is None:
            return normalized
        for item in coords:
            if item is None:
                continue
            try:
                x_idx = int(item[0])
                y_idx = int(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            normalized.append((x_idx, y_idx))
        return normalized

    def _spatial_result_payload(
        self,
        spatial_result,
        *,
        goal_coords: list[tuple[int, int]] | None = None,
        movement_coords: list[tuple[int, int]] | None = None,
    ) -> dict:
        payload = {
            "available": bool(getattr(spatial_result, "available", False)),
            "status": str(getattr(spatial_result, "status", "")),
            "processing_time_ms": (
                None
                if getattr(spatial_result, "processing_time_ms", None) is None
                else float(spatial_result.processing_time_ms)
            ),
            "resolution_xyz": np.asarray(
                getattr(spatial_result, "resolution_xyz", self._spatial_resolution_xyz),
                dtype=np.int32,
            ).reshape(3).tolist(),
            "bbox_min": np.asarray(
                getattr(spatial_result, "bbox_min", self._spatial_bbox_min),
                dtype=np.float64,
            ).reshape(3).tolist(),
            "bbox_max": np.asarray(
                getattr(spatial_result, "bbox_max", self._spatial_bbox_max),
                dtype=np.float64,
            ).reshape(3).tolist(),
            "pusher_coords": [],
            "tblock_coords": [],
            "tblock_coords_full": [],
            "goal_coords": [],
            "movement_coords": [],
            "sentence": "",
        }
        if not payload["available"]:
            return payload

        pusher_coords = [
            (int(coord[0]), int(coord[1]))
            for coord in np.asarray(spatial_result.pusher_voxels_2d, dtype=np.int32).reshape(-1, 2).tolist()
        ]
        tblock_coords = [
            (int(coord[0]), int(coord[1]))
            for coord in np.asarray(spatial_result.tblock_voxels_2d, dtype=np.int32).reshape(-1, 2).tolist()
        ]
        tblock_coords_full = [
            (int(coord[0]), int(coord[1]))
            for coord in np.asarray(spatial_result.tblock_voxels_2d_full, dtype=np.int32).reshape(-1, 2).tolist()
        ]
        goal_norm = self._normalize_coord_list(goal_coords)
        movement_norm = self._normalize_coord_list(movement_coords)
        sentence = build_composited_sentence(
            pusher_coords=pusher_coords,
            tbar_coords=tblock_coords,
            goal_coords=goal_norm,
            movement_coords=movement_norm,
        )
        payload["pusher_coords"] = [[x_idx, y_idx] for x_idx, y_idx in pusher_coords]
        payload["tblock_coords"] = [[x_idx, y_idx] for x_idx, y_idx in tblock_coords]
        payload["tblock_coords_full"] = [[x_idx, y_idx] for x_idx, y_idx in tblock_coords_full]
        payload["goal_coords"] = [[x_idx, y_idx] for x_idx, y_idx in goal_norm]
        payload["movement_coords"] = [[x_idx, y_idx] for x_idx, y_idx in movement_norm]
        payload["sentence"] = sentence
        return payload

    def _scene_snapshot_payload(self) -> dict:
        # Use the main-loop cached spatial result first to avoid transient
        # mismatches between HTTP request timing and live reconstruction updates.
        spatial_result = self._latest_live_spatial_result
        if spatial_result is None:
            spatial_result = self._compute_live_spatial_result()
        if not getattr(spatial_result, "available", False):
            cached_available = self._latest_available_live_spatial_result
            if cached_available is not None:
                spatial_result = cached_available
        with self._trajectory_lock:
            goal_coords = self._normalize_coord_list(self._trajectory_session.get("goal", []))
        return {
            "timestamp": time.time(),
            "recording": bool(self._recording),
            "replaying": bool(self._replaying),
            "arm_world_ready": bool(self._arm_world_result is not None),
            "arm_pose_ready": bool(self._latest_arm_pose_valid),
            "spatial": (
                self._spatial_result_payload(
                    spatial_result,
                    goal_coords=goal_coords,
                )
                | self._tblock_apriltag_projection_payload()
            ),
        }

    def _tblock_apriltag_projection_payload(self) -> dict:
        payload = {
            "tblock_apriltag_points_world": [],
            "tblock_apriltag_coords_2d": [],
        }
        # Project the static tag corners from the reconstructed T-block pose.
        # This does not depend on which tags were detected in the current frame.
        selected_tag_ids = (48, 49, 73)
        pose_world = self._parse_tblock_pose_world(self._latest_tblock_pose_world)
        if pose_world is None or self._apriltag_static_model is None:
            return payload

        rotation = self._rotation_from_wxyz(pose_world["wxyz"])
        translation = np.asarray(pose_world["translation_m"], dtype=np.float64).reshape(1, 3)
        coords_2d: list[tuple[int, int]] = []
        points_payload: list[dict] = []
        for tag_id in selected_tag_ids:
            tag_points_object = self._apriltag_static_model.corner_points_by_tag.get(int(tag_id))
            if tag_points_object is None:
                continue
            tag_points_world = np.asarray(tag_points_object, dtype=np.float64).reshape(-1, 3) @ rotation.as_matrix().T + translation
            for corner_idx, point_world in enumerate(tag_points_world):
                coord_xy = project_point_to_spatial_coord(
                    point_world,
                    bbox_min=self._spatial_bbox_min,
                    bbox_max=self._spatial_bbox_max,
                    resolution_xyz=self._spatial_resolution_xyz,
                )
                if coord_xy is not None:
                    coords_2d.append((int(coord_xy[0]), int(coord_xy[1])))
                points_payload.append(
                    {
                        "tag_id": int(tag_id),
                        "corner_idx": int(corner_idx),
                        "xyz_m": [
                            float(point_world[0]),
                            float(point_world[1]),
                            float(point_world[2]),
                        ],
                        "coord_xy": (
                            None
                            if coord_xy is None
                            else [int(coord_xy[0]), int(coord_xy[1])]
                        ),
                    }
                )
        payload["tblock_apriltag_points_world"] = points_payload
        payload["tblock_apriltag_coords_2d"] = [
            [int(x_idx), int(y_idx)]
            for x_idx, y_idx in sorted(set(coords_2d))
        ]
        return payload

    def _latest_rgb_snapshot_jpeg(self) -> tuple[bytes, float] | None:
        frame = getattr(self, "_latest_color_frame", None)
        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            return None
        frame_bgr = cv2.cvtColor(np.array(frame, copy=True), cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            raise RuntimeError("Failed to encode RGB snapshot as JPEG.")
        return encoded.tobytes(), time.time()

    def _snapshot_trajectory_session(self) -> dict:
        with self._trajectory_lock:
            return {
                "run_id": str(self._trajectory_session.get("run_id", "")),
                "running": bool(self._trajectory_session.get("running", False)),
                "status": str(self._trajectory_session.get("status", "idle")),
                "step_index": int(self._trajectory_session.get("step_index", -1)),
                "total_steps": int(self._trajectory_session.get("total_steps", 0)),
                "error": str(self._trajectory_session.get("error", "")),
                "output_json": str(self._trajectory_session.get("output_json", "")),
                "started_at": self._trajectory_session.get("started_at"),
                "finished_at": self._trajectory_session.get("finished_at"),
                "trajectory": [list(coord) for coord in self._trajectory_session.get("trajectory", [])],
                "goal": [list(coord) for coord in self._trajectory_session.get("goal", [])],
                "recorded_frames": len(self._trajectory_session.get("frames", [])),
            }

    def _update_trajectory_session(self, **kwargs) -> None:
        with self._trajectory_lock:
            self._trajectory_session.update(kwargs)

    def _new_episode_id(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1.0) * 1000):03d}"

    def _episode_total_frames(self) -> int:
        return int(
            sum(
                len(movement.get("frames", []))
                for movement in self._episode_session.get("movements", [])
            )
        )

    def _episode_movement_summaries(self) -> list[dict]:
        summaries: list[dict] = []
        movements = list(self._episode_session.get("movements", []))
        for idx, movement in enumerate(movements):
            trajectory = list(movement.get("trajectory", []))
            frames = list(movement.get("frames", []))
            goal = list(movement.get("goal", []))
            summaries.append(
                {
                    "index": int(idx),
                    "run_id": str(movement.get("run_id", "")),
                    "timestamp": movement.get("timestamp"),
                    "trajectory_steps": int(len(trajectory)),
                    "frame_count": int(len(frames)),
                    "goal_cells": int(len(goal)),
                }
            )
        return summaries

    def _snapshot_episode_session(self) -> dict:
        with self._trajectory_lock:
            return {
                "episode_id": str(self._episode_session.get("episode_id", "")),
                "started_at": self._episode_session.get("started_at"),
                "finished_at": self._episode_session.get("finished_at"),
                "movement_count": len(self._episode_session.get("movements", [])),
                "frame_count": self._episode_total_frames(),
                "movements": self._episode_movement_summaries(),
            }

    def _reset_episode_session(self) -> None:
        self._episode_session = {
            "episode_id": self._new_episode_id(),
            "started_at": time.time(),
            "finished_at": None,
            "movements": [],
            "output_json": "",
        }

    def _append_completed_movement_to_episode(
        self,
        *,
        run_id: str,
        trajectory: list[tuple[int, int]],
        goal: list[tuple[int, int]],
        frames: list[dict],
    ) -> None:
        warning_messages: list[str] = []
        for frame_idx, frame in enumerate(frames):
            spatial = frame.get("spatial", {})
            pusher_coords = list(spatial.get("pusher_coords", []))
            tblock_coords = list(spatial.get("tblock_coords", []))
            if len(pusher_coords) == 0 or len(tblock_coords) == 0:
                missing_parts = []
                if len(pusher_coords) == 0:
                    missing_parts.append("pusher")
                if len(tblock_coords) == 0:
                    missing_parts.append("tblock")
                warning_messages.append(
                    f"run_id={run_id} frame={frame_idx + 1} empty {'/'.join(missing_parts)}"
                )
        movement_payload = {
            "run_id": str(run_id),
            "timestamp": float(time.time()),
            "trajectory": [[int(x_idx), int(y_idx)] for x_idx, y_idx in trajectory],
            "goal": [[int(x_idx), int(y_idx)] for x_idx, y_idx in goal],
            "frames": frames,
            "warnings": warning_messages,
        }
        with self._trajectory_lock:
            self._episode_session["movements"].append(movement_payload)
            self._episode_session["output_json"] = ""

    def _build_episode_payload(self) -> dict:
        with self._trajectory_lock:
            movements = self._to_jsonable_value(self._episode_session.get("movements", []))
            episode_id = str(self._episode_session.get("episode_id", self._new_episode_id()))
            started_at = self._episode_session.get("started_at")
            if len(movements) == 0:
                raise ValueError("Current episode has no movement data to save.")

        payload = {
            "format": "robodata_spatial_episode_v1",
            "episode_id": episode_id,
            "started_at": started_at,
            "finished_at": time.time(),
            "spatial_config": {
                "bbox_min": self._spatial_bbox_min.tolist(),
                "bbox_max": self._spatial_bbox_max.tolist(),
                "resolution_xyz": self._spatial_resolution_xyz.tolist(),
            },
            "movement_count": len(movements),
            "frame_count": int(sum(len(m.get("frames", [])) for m in movements)),
            "movements": movements,
        }

        with self._trajectory_lock:
            self._episode_session["finished_at"] = payload["finished_at"]
        return payload

    def _finish_episode_and_restart(self) -> tuple[int, dict]:
        with self._trajectory_lock:
            if bool(self._trajectory_session.get("running", False)):
                return 409, {"ok": False, "error": "trajectory runner is busy."}
        try:
            episode_payload = self._build_episode_payload()
        except Exception as exc:
            return 400, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        previous_episode = self._snapshot_episode_session()
        self._reset_episode_session()
        current_episode = self._snapshot_episode_session()
        return 200, {
            "ok": True,
            "episode_payload": episode_payload,
            "previous_episode": previous_episode,
            "current_episode": current_episode,
        }

    def _start_new_episode(self) -> tuple[int, dict]:
        with self._trajectory_lock:
            if bool(self._trajectory_session.get("running", False)):
                return 409, {"ok": False, "error": "trajectory runner is busy."}
        previous_episode = self._snapshot_episode_session()
        self._reset_episode_session()
        current_episode = self._snapshot_episode_session()
        return 200, {
            "ok": True,
            "previous_episode": previous_episode,
            "current_episode": current_episode,
        }

    def _trajectory_target_to_robot(
        self,
        target_coord: tuple[int, int],
        *,
        z_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        target_points = self._pusher_discrete_coord_to_eef_base_point(
            target_coord,
            z_index=z_index,
        )
        if target_points is None:
            raise RuntimeError("Cannot convert target coord to EEF base frame.")
        return target_points

    def _move_robot_to_discrete_coord(
        self,
        target_coord: tuple[int, int],
        *,
        z_index: int,
        timesteps: int,
        speed: int,
        timeout_s: float,
    ) -> dict:
        if self._arm_reader is None:
            raise RuntimeError("Arm reader is unavailable.")
        can_move_to_position = hasattr(self._arm_reader, "move_to_base_position")
        can_step_eef = hasattr(self._arm_reader, "step_eef")
        can_move_by_delta = hasattr(self._arm_reader, "move_by_base_delta")
        if not (can_move_to_position or can_step_eef or can_move_by_delta):
            raise RuntimeError(
                "Current arm backend does not support movement commands."
            )
        if self._arm_world_result is None:
            raise RuntimeError("Calibrate Arm To World first.")
        if not self._latest_arm_pose_valid:
            raise RuntimeError("Waiting for live arm pose.")

        target_world, target_pusher_base, target_eef_base = self._trajectory_target_to_robot(
            target_coord,
            z_index=z_index,
        )
        delta_base = None
        if not can_move_to_position and (can_step_eef or can_move_by_delta):
            current_world = self._get_live_pusher_world_point()
            current_base = (
                None
                if current_world is None
                else point_world_to_base(current_world, self._arm_world_result.T_base_from_world)
            )
            delta_base = (
                None
                if current_base is None
                else (target_pusher_base - current_base).astype(np.float64)
            )
            if delta_base is None:
                raise RuntimeError("Cannot compute relative base-frame delta.")

        if can_move_to_position:
            raw_result = self._arm_reader.move_to_base_position(
                target_eef_base,
                timesteps=int(timesteps),
                speed=int(speed),
                timeout_s=float(timeout_s),
            )
        elif can_step_eef:
            raw_result = self._arm_reader.step_eef(
                delta_base,
                timesteps=int(timesteps),
                speed=int(speed),
                timeout_s=float(timeout_s),
            )
        else:
            raw_result = self._arm_reader.move_by_base_delta(
                delta_base,
                timesteps=int(timesteps),
                speed=int(speed),
                timeout_s=float(timeout_s),
            )

        return {
            "target_coord": [int(target_coord[0]), int(target_coord[1]), int(z_index)],
            "target_world_m": np.asarray(target_world, dtype=np.float64).reshape(3).tolist(),
            "target_pusher_base_m": np.asarray(target_pusher_base, dtype=np.float64).reshape(3).tolist(),
            "target_eef_base_m": np.asarray(target_eef_base, dtype=np.float64).reshape(3).tolist(),
            "delta_base_m": None
            if delta_base is None
            else np.asarray(delta_base, dtype=np.float64).reshape(3).tolist(),
            "result": self._to_jsonable_value(raw_result),
        }

    def _save_trajectory_session_json(
        self,
        *,
        run_id: str,
        trajectory: list[tuple[int, int]],
        goal: list[tuple[int, int]],
        frames: list[dict],
    ) -> str:
        output_dir = Path(self._output_dir) / "trajectory_runs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"spatial_trajectory_{run_id}.json"
        payload = {
            "format": "robodata_spatial_trajectory_v1",
            "run_id": run_id,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "spatial_config": {
                "bbox_min": self._spatial_bbox_min.tolist(),
                "bbox_max": self._spatial_bbox_max.tolist(),
                "resolution_xyz": self._spatial_resolution_xyz.tolist(),
            },
            "trajectory": [[int(x_idx), int(y_idx)] for x_idx, y_idx in trajectory],
            "goal": [[int(x_idx), int(y_idx)] for x_idx, y_idx in goal],
            "frames": frames,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(output_path)

    def _to_jsonable_value(self, value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {key: self._to_jsonable_value(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._to_jsonable_value(item) for item in value]
        return value

    def _execute_external_trajectory(
        self,
        *,
        run_id: str,
        trajectory: list[tuple[int, int]],
        goal: list[tuple[int, int]],
        z_index: int,
        timesteps: int,
        speed: int,
        timeout_s: float,
        settle_s: float,
    ) -> dict:
        frames: list[dict] = []
        movement_warnings: list[str] = []
        try:
            self._arm_move_busy = True
            self._arm_move_status = f"External trajectory running ({len(trajectory)} steps)"
            for step_index, target_coord in enumerate(trajectory):
                move_payload = self._move_robot_to_discrete_coord(
                    target_coord,
                    z_index=z_index,
                    timesteps=timesteps,
                    speed=speed,
                    timeout_s=timeout_s,
                )
                if settle_s > 0.0:
                    time.sleep(settle_s)
                spatial_result = self._compute_live_spatial_result()
                frame_payload = {
                    "step_index": int(step_index),
                    "timestamp": float(time.time()),
                    "target_coord": [int(target_coord[0]), int(target_coord[1])],
                    "move": move_payload,
                    "spatial": self._spatial_result_payload(
                        spatial_result,
                        goal_coords=goal,
                        movement_coords=trajectory[: step_index + 1],
                    ),
                }
                spatial_payload = frame_payload["spatial"]
                if (
                    len(spatial_payload.get("pusher_coords", [])) == 0
                    or len(spatial_payload.get("tblock_coords", [])) == 0
                ):
                    missing_parts = []
                    if len(spatial_payload.get("pusher_coords", [])) == 0:
                        missing_parts.append("pusher")
                    if len(spatial_payload.get("tblock_coords", [])) == 0:
                        missing_parts.append("tblock")
                    warning_msg = (
                        "Recording warning: empty "
                        f"{'/'.join(missing_parts)} at step {step_index + 1}/{len(trajectory)} "
                        f"(run_id={run_id}). Continuing."
                    )
                    print(f">> {warning_msg}")
                    frame_payload["warning"] = warning_msg
                    movement_warnings.append(warning_msg)
                frames.append(frame_payload)
            self._arm_move_status = (
                f"External trajectory done | run_id={run_id} steps={len(trajectory)} "
                f"frames={len(frames)} warnings={len(movement_warnings)}"
            )
            return {
                "run_id": str(run_id),
                "timestamp": float(time.time()),
                "trajectory": [[int(x_idx), int(y_idx)] for x_idx, y_idx in trajectory],
                "goal": [[int(x_idx), int(y_idx)] for x_idx, y_idx in goal],
                "frames": frames,
                "warnings": movement_warnings,
            }
        except Exception as exc:
            self._arm_move_status = f"External trajectory failed: {exc}"
            raise
        finally:
            self._arm_move_busy = False

    def _start_external_trajectory(self, payload: dict) -> tuple[int, dict]:
        if not isinstance(payload, dict):
            return 400, {"ok": False, "error": "JSON object required."}

        trajectory = self._normalize_coord_list(payload.get("trajectory"))
        goal = self._normalize_coord_list(payload.get("goal"))
        if len(trajectory) == 0:
            return 400, {"ok": False, "error": "trajectory must include at least one (x,y) voxel."}
        if self._recording:
            return 409, {"ok": False, "error": "stop recording before running external trajectory."}
        if self._replaying:
            return 409, {"ok": False, "error": "stop replay before running external trajectory."}
        if self._arm_move_busy:
            return 409, {"ok": False, "error": "trajectory runner is busy."}

        z_index = int(payload.get("z_index", self._arm_move_z_index))
        timesteps = int(payload.get("timesteps", 15))
        speed = int(payload.get("speed", 1))
        timeout_s = float(payload.get("timeout_s", 8.0))
        settle_s = float(payload.get("settle_s", 0.12))
        z_max = max(int(self._spatial_resolution_xyz[2]) - 1, 0)
        z_index = int(np.clip(z_index, 0, z_max))
        if timesteps < 1:
            timesteps = 1
        if speed < 1:
            speed = 1
        settle_s = max(0.0, settle_s)

        run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1.0) * 1000):03d}"
        try:
            movement_payload = self._execute_external_trajectory(
                run_id=run_id,
                trajectory=trajectory,
                goal=goal,
                z_index=z_index,
                timesteps=timesteps,
                speed=speed,
                timeout_s=timeout_s,
                settle_s=settle_s,
            )
        except Exception as exc:
            return 500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return 200, {
            "ok": True,
            "run_id": run_id,
            "movement": movement_payload,
            "total_steps": len(trajectory),
            "z_index": z_index,
        }

    def _start_control_api_server(self) -> None:
        if self._control_api_port <= 0:
            print("[ViserCollector] Control API disabled (port <= 0).")
            return
        if self._control_api_server is not None:
            return

        app = self

        class _ControlHandler(http.server.BaseHTTPRequestHandler):
            def log_message(self, _format, *_args):
                return

            def _send_json(self, status_code: int, payload: dict) -> None:
                body = json.dumps(payload).encode("utf-8")
                self.send_response(int(status_code))
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _send_bytes(
                self,
                status_code: int,
                body: bytes,
                *,
                content_type: str,
                extra_headers: dict[str, str] | None = None,
            ) -> None:
                self.send_response(int(status_code))
                self.send_header("Content-Type", str(content_type))
                self.send_header("Content-Length", str(len(body)))
                for key, value in (extra_headers or {}).items():
                    self.send_header(str(key), str(value))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = parse.urlparse(self.path).path
                if path == "/health":
                    self._send_json(200, {"ok": True})
                    return
                if path == "/scene":
                    try:
                        payload = app._scene_snapshot_payload()
                        self._send_json(200, {"ok": True, "scene": payload})
                    except Exception as exc:
                        self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                    return
                if path == "/camera/rgb.jpg":
                    try:
                        snapshot = app._latest_rgb_snapshot_jpeg()
                        if snapshot is None:
                            self._send_json(503, {"ok": False, "error": "no camera frame available"})
                            return
                        jpeg_bytes, timestamp = snapshot
                        self._send_bytes(
                            200,
                            jpeg_bytes,
                            content_type="image/jpeg",
                            extra_headers={"X-Timestamp": str(float(timestamp))},
                        )
                    except Exception as exc:
                        self._send_json(500, {"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                    return
                self._send_json(404, {"ok": False, "error": "unknown endpoint"})

            def do_POST(self):
                path = parse.urlparse(self.path).path
                if path != "/trajectory/execute":
                    self._send_json(404, {"ok": False, "error": "unknown endpoint"})
                    return
                try:
                    content_length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    content_length = 0
                raw_body = self.rfile.read(max(content_length, 0))
                try:
                    payload = json.loads(raw_body.decode("utf-8")) if raw_body else {}
                except Exception as exc:
                    self._send_json(400, {"ok": False, "error": f"invalid JSON: {exc}"})
                    return
                status_code, result = app._start_external_trajectory(payload)
                self._send_json(status_code, result)

        try:
            self._control_api_server = http.server.ThreadingHTTPServer(
                (self._control_api_host, self._control_api_port),
                _ControlHandler,
            )
        except OSError as exc:
            self._control_api_server = None
            print(
                "[ViserCollector] Failed to start control API on "
                f"{self._control_api_host}:{self._control_api_port}: {exc}"
            )
            return

        self._control_api_thread = threading.Thread(
            target=self._control_api_server.serve_forever,
            daemon=True,
        )
        self._control_api_thread.start()

    def _stop_control_api_server(self) -> None:
        if self._control_api_server is None:
            return
        try:
            self._control_api_server.shutdown()
            self._control_api_server.server_close()
        finally:
            self._control_api_server = None
        if self._control_api_thread is not None:
            self._control_api_thread.join(timeout=1.0)
            self._control_api_thread = None

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
                    max=max(16, os.cpu_count() or 1),
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
            self._camera_calibration_md = server.gui.add_markdown(
                self._camera_calibration_text()
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
            self._pusher_md = server.gui.add_markdown(
                "**Pusher Coord (Discrete):** ---"
            )
            self._arm_feedback_md = server.gui.add_markdown("**Arm Feedback:** waiting")
            self._arm_lock_btn = server.gui.add_button("Lock Robot Arm")
            self._arm_lock_btn.disabled = not hasattr(self._arm_reader, "lock_pose")
            self._arm_lock_status_md = server.gui.add_markdown(
                f"**Arm Lock:** {self._arm_lock_status}"
            )
            self._arm_move_status_md = server.gui.add_markdown(
                f"**Pusher Move:** {self._arm_move_status}"
            )
            for label, delta in (
                ("NW", (-1, 1)),
                ("N", (0, 1)),
                ("NE", (1, 1)),
                ("W", (-1, 0)),
                ("E", (1, 0)),
                ("SW", (-1, -1)),
                ("S", (0, -1)),
                ("SE", (1, -1)),
            ):
                btn = server.gui.add_button(label)
                btn.disabled = not (
                    hasattr(self._arm_reader, "step_eef")
                    or hasattr(self._arm_reader, "move_by_base_delta")
                    or hasattr(self._arm_reader, "move_to_base_position")
                )
                btn.on_click(partial(self._on_pusher_nudge_click, delta=delta, label=label))
                self._arm_move_buttons[label] = btn
            self._qpos_md = server.gui.add_markdown(
                "**Joint Positions (deg):**\n\n"
                "J1: ---  J2: ---  J3: ---\n\nJ4: ---  J5: ---  J6: ---"
            )
            self._gripper_md = server.gui.add_markdown("**Gripper:** ---")

        with server.gui.add_folder("Recording"):
            self._record_btn = server.gui.add_button("Start Recording", color="blue")
            self._status_md = server.gui.add_markdown("**Status:** IDLE")

        with server.gui.add_folder("Pipeline Timing"):
            self._pipeline_timing_md = server.gui.add_markdown(
                self._format_pipeline_timing_markdown()
            )

        with server.gui.add_folder("Spatial Language"):
            self._spatial_resolution_x = server.gui.add_slider(
                "Voxel Resolution X",
                min=4,
                max=150,
                step=1,
                initial_value=int(self._spatial_resolution_xyz[0]),
            )
            self._spatial_resolution_y = server.gui.add_slider(
                "Voxel Resolution Y",
                min=4,
                max=150,
                step=1,
                initial_value=int(self._spatial_resolution_xyz[1]),
            )
            self._spatial_resolution_z = server.gui.add_slider(
                "Voxel Resolution Z",
                min=4,
                max=150,
                step=1,
                initial_value=int(self._spatial_resolution_xyz[2]),
            )
            self._spatial_show_3d_checkbox = server.gui.add_checkbox(
                "Show 3D voxels",
                initial_value=False,
            )
            self._spatial_show_2d_checkbox = server.gui.add_checkbox(
                "Show 2D voxels",
                initial_value=False,
            )
            with server.gui.add_folder("Bounding Box"):
                self._spatial_bbox_min_x = server.gui.add_number(
                    "Min X (m)",
                    initial_value=float(self._spatial_bbox_min[0]),
                    step=0.01,
                )
                self._spatial_bbox_min_y = server.gui.add_number(
                    "Min Y (m)",
                    initial_value=float(self._spatial_bbox_min[1]),
                    step=0.01,
                )
                self._spatial_bbox_min_z = server.gui.add_number(
                    "Min Z (m)",
                    initial_value=float(self._spatial_bbox_min[2]),
                    step=0.01,
                )
                self._spatial_bbox_max_x = server.gui.add_number(
                    "Max X (m)",
                    initial_value=float(self._spatial_bbox_max[0]),
                    step=0.01,
                )
                self._spatial_bbox_max_y = server.gui.add_number(
                    "Max Y (m)",
                    initial_value=float(self._spatial_bbox_max[1]),
                    step=0.01,
                )
                self._spatial_bbox_max_z = server.gui.add_number(
                    "Max Z (m)",
                    initial_value=float(self._spatial_bbox_max[2]),
                    step=0.01,
                )
            self._spatial_2d_handle = server.gui.add_image(
                np.full(
                    (
                        int(self._spatial_resolution_xyz[1]) * 16,
                        int(self._spatial_resolution_xyz[0]) * 16,
                        3,
                    ),
                    255,
                    dtype=np.uint8,
                ),
                label="TBlock / Pusher XY Projection",
            )
            self._spatial_2d_handle.visible = bool(self._spatial_show_2d_checkbox.value)
            with server.gui.add_folder("Online Inference"):
                self._online_inference_server_input = server.gui.add_text(
                    "Server URL",
                    initial_value=self._online_inference_server,
                )
                self._online_inference_max_new_tokens_input = server.gui.add_number(
                    "Max New Tokens",
                    initial_value=self._online_inference_max_new_tokens,
                    step=1,
                )
                self._online_inference_temperature_input = server.gui.add_number(
                    "Temperature",
                    initial_value=self._online_inference_temperature,
                    step=0.1,
                )
                self._online_inference_top_k_input = server.gui.add_text(
                    "Top-k",
                    initial_value=(
                        "" if self._online_inference_top_k is None else str(self._online_inference_top_k)
                    ),
                )
                self._online_inference_forbidden_tokens_input = server.gui.add_text(
                    "Forbidden Tokens",
                    initial_value=self._online_inference_forbidden_tokens,
                )
                self._online_inference_do_sample_checkbox = server.gui.add_checkbox(
                    "Do Sample",
                    initial_value=self._online_inference_do_sample,
                )
                self._online_inference_movement_only_checkbox = server.gui.add_checkbox(
                    "Movement-only decoding",
                    initial_value=self._online_inference_movement_only,
                )
                self._online_inference_require_state_checkbox = server.gui.add_checkbox(
                    "Require State After Movement",
                    initial_value=self._online_inference_require_state_after_movement,
                )
                self._online_inference_state_after_prob_input = server.gui.add_number(
                    "State-after prob",
                    initial_value=self._online_inference_state_after_movement_prob,
                    step=0.05,
                )
                self._online_inference_btn = server.gui.add_button("Run Online Inference")
                self._online_inference_details_md = server.gui.add_markdown(
                    self._format_online_inference_details_link()
                )
                self._online_inference_current_pusher_coord_handle = server.gui.add_text(
                    "Current Pusher Coord",
                    initial_value=self._online_inference_current_pusher_coord_text,
                )
                self._online_inference_current_pusher_coord_handle.disabled = True
                self._online_inference_vis_handle = server.gui.add_image(
                    self._online_inference_vis_image,
                    label="Inference Sequence XY",
                )

        self._record_btn.on_click(self._on_record_click)
        self._arm_lock_btn.on_click(self._on_arm_lock_click)
        self._save_screenshot_btn.on_click(self._on_save_screenshot_click)
        self._online_inference_btn.on_click(self._on_online_inference_click)
        self._bind_zed_control_events()
        self._bind_apriltag_control_events()
        self._bind_arm_world_control_events()
        self._update_arm_world_target_status()
        self._reload_online_inference_goal()
        self._writer.set_world_config(self._arm_world_result_to_config())

        # Replay folder
        with server.gui.add_folder("Replay"):
            self._replay_dropdown = server.gui.add_dropdown(
                "Episode",
                options=self._list_episodes(),
            )
            self._replay_filter_checkbox = server.gui.add_checkbox(
                "Filter Tblock pose",
                initial_value=self._replay_filter_enabled,
            )
            self._replay_btn = server.gui.add_button("Replay")
            self._replay_pause_btn = server.gui.add_button("Pause", visible=False)
            self._replay_prev_btn = server.gui.add_button("Prev Frame", visible=False)
            self._replay_next_btn = server.gui.add_button("Next Frame", visible=False)
            self._stop_replay_btn = server.gui.add_button("Stop Replay", visible=False)
            self._dataset_add_btn = server.gui.add_button("Add To Dataset")
            self._dataset_remove_btn = server.gui.add_button("Remove From Dataset")
            self._dataset_add_all_btn = server.gui.add_button("Add All Episodes")
            self._dataset_clear_btn = server.gui.add_button("Clear Dataset Selection")
            self._dataset_set_goal_btn = server.gui.add_button("Use Selected Episode Goal")
            self._dataset_clear_goal_btn = server.gui.add_button("Use Per-Episode Goal")
            self._dataset_export_btn = server.gui.add_button("Export Selected Dataset")
            self._dataset_selected_md = server.gui.add_markdown("**Dataset Episodes:** none")
            self._dataset_goal_md = server.gui.add_markdown("**Global Goal:** per-episode")
            self._dataset_status_md = server.gui.add_markdown("**Dataset Export:** idle")

        self._replay_btn.on_click(self._on_replay_click)
        self._replay_pause_btn.on_click(self._on_replay_pause_click)
        self._replay_prev_btn.on_click(self._on_replay_prev_click)
        self._replay_next_btn.on_click(self._on_replay_next_click)
        self._stop_replay_btn.on_click(self._on_stop_replay_click)
        self._dataset_add_btn.on_click(self._on_dataset_add_click)
        self._dataset_remove_btn.on_click(self._on_dataset_remove_click)
        self._dataset_add_all_btn.on_click(self._on_dataset_add_all_click)
        self._dataset_clear_btn.on_click(self._on_dataset_clear_click)
        self._dataset_set_goal_btn.on_click(self._on_dataset_set_goal_click)
        self._dataset_clear_goal_btn.on_click(self._on_dataset_clear_goal_click)
        self._dataset_export_btn.on_click(self._on_dataset_export_click)
        self._update_dataset_selection_ui()
        self._update_spatial_bbox_visual()
        self._start_control_api_server()

        print(f"[ViserCollector] Server started at http://localhost:{self._port}")
        print(
            "[ViserCollector] Control API at "
            f"http://{self._control_api_host}:{self._control_api_port}"
        )
        print("Press Ctrl+C to stop.\n")

        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("\n[ViserCollector] Shutting down...")
            if self._recording:
                self._stop_recording()
            if self._replaying:
                self._finish_replay_cleanup()
        finally:
            self._stop_control_api_server()

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
                    f"- Filtered object translation: `[{filtered_translation[0]:+.4f}, {filtered_translation[1]:+.4f}, {filtered_translation[2]:+.4f}] m`",
                    f"- Object translation delta: `[{delta[0]:+.4f}, {delta[1]:+.4f}, {delta[2]:+.4f}] m`",
                    "- Pose smoothing: `object-pose Kalman/ESKF`",
                ]
            )
        if filtered_mesh_center_world is not None:
            lines.append(
                f"- Filtered mesh center: `[{filtered_mesh_center_world[0]:+.4f}, {filtered_mesh_center_world[1]:+.4f}, {filtered_mesh_center_world[2]:+.4f}] m`"
            )
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

    def _rotation_from_wxyz(self, wxyz: np.ndarray) -> Rotation:
        wxyz = np.asarray(wxyz, dtype=np.float64).reshape(4)
        return Rotation.from_quat([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

    def _pose_dict_from_components(
        self,
        translation_m: np.ndarray,
        rotation: Rotation,
    ) -> dict:
        xyzw = rotation.as_quat()
        return {
            "translation_m": np.asarray(translation_m, dtype=np.float64).reshape(3).copy(),
            "wxyz": np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64),
        }

    def _filter_replay_tblock_poses(
        self,
        poses_world: list[dict | None],
        timestamps_s: np.ndarray,
    ) -> tuple[list[dict | None], int]:
        smoother = self._build_apriltag_camera_world_smoother(
            self._apriltag_world_filter_strength
        )
        filtered_poses: list[dict | None] = []
        last_filtered_pose: dict | None = None
        corrected_count = 0

        for pose_world, timestamp_s in zip(poses_world, timestamps_s, strict=False):
            parsed_pose = self._parse_tblock_pose_world(pose_world)
            timestamp_s = float(timestamp_s)

            if parsed_pose is None:
                filtered_poses.append(last_filtered_pose)
                if last_filtered_pose is not None:
                    corrected_count += 1
                continue

            measured_translation = np.asarray(parsed_pose["translation_m"], dtype=np.float64)
            measured_rotation = self._rotation_from_wxyz(parsed_pose["wxyz"])
            filtered_transform = smoother.update(
                RigidTransform(
                    rotation=measured_rotation.as_matrix(),
                    translation=measured_translation,
                ),
                timestamp_s=timestamp_s,
                reproj_error_px=0.0,
                visible_tag_count=4,
            )
            filtered_pose = self._pose_dict_from_components(
                filtered_transform.translation,
                Rotation.from_matrix(filtered_transform.rotation),
            )
            filtered_poses.append(filtered_pose)
            translation_delta = float(np.linalg.norm(filtered_transform.translation - measured_translation))
            rotation_delta = float(
                (measured_rotation.inv() * Rotation.from_matrix(filtered_transform.rotation)).magnitude()
            )
            if translation_delta > 1e-6 or rotation_delta > 1e-6:
                corrected_count += 1
            last_filtered_pose = filtered_pose

        return filtered_poses, corrected_count

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

    def _sync_spatial_language_settings(self) -> None:
        bbox_min = np.array(
            [
                self._spatial_bbox_min_x.value,
                self._spatial_bbox_min_y.value,
                self._spatial_bbox_min_z.value,
            ],
            dtype=np.float64,
        )
        bbox_max = np.array(
            [
                self._spatial_bbox_max_x.value,
                self._spatial_bbox_max_y.value,
                self._spatial_bbox_max_z.value,
            ],
            dtype=np.float64,
        )
        bbox_max = np.maximum(bbox_max, bbox_min + 1e-3)
        self._spatial_bbox_max_x.value = float(bbox_max[0])
        self._spatial_bbox_max_y.value = float(bbox_max[1])
        self._spatial_bbox_max_z.value = float(bbox_max[2])
        self._spatial_bbox_min = bbox_min
        self._spatial_bbox_max = bbox_max
        self._spatial_resolution_xyz = np.array(
            [
                int(self._spatial_resolution_x.value),
                int(self._spatial_resolution_y.value),
                int(self._spatial_resolution_z.value),
            ],
            dtype=np.int32,
        )
        self._spatial_2d_handle.visible = bool(self._spatial_show_2d_checkbox.value)
        self._update_spatial_bbox_visual()

    def _reset_pipeline_timing_stats(self) -> None:
        self._pipeline_timing_stats = {
            stage: {
                "count": 0,
                "sum_ms": 0.0,
                "min_ms": None,
                "max_ms": None,
            }
            for stage in self._pipeline_stage_order
        }

    def _record_pipeline_timing(self, stage: str, duration_ms: float) -> None:
        stats = self._pipeline_timing_stats.setdefault(
            stage,
            {
                "count": 0,
                "sum_ms": 0.0,
                "min_ms": None,
                "max_ms": None,
            },
        )
        stats["count"] = int(stats["count"]) + 1
        stats["sum_ms"] = float(stats["sum_ms"]) + float(duration_ms)
        min_ms = stats["min_ms"]
        max_ms = stats["max_ms"]
        stats["min_ms"] = float(duration_ms) if min_ms is None else min(float(min_ms), float(duration_ms))
        stats["max_ms"] = float(duration_ms) if max_ms is None else max(float(max_ms), float(duration_ms))

    def _update_pipeline_timing_snapshot(
        self,
        mode: str,
        frame_timings_ms: dict[str, float],
        total_ms: float,
    ) -> None:
        self._pipeline_last_mode = mode
        self._pipeline_last_timings_ms = {
            stage: float(duration_ms)
            for stage, duration_ms in frame_timings_ms.items()
        }
        self._pipeline_last_total_ms = float(total_ms)
        self._pipeline_frame_counter += 1
        for stage, duration_ms in self._pipeline_last_timings_ms.items():
            self._record_pipeline_timing(stage, duration_ms)
        self._record_pipeline_timing("total", total_ms)

    def _format_pipeline_timing_markdown(self) -> str:
        lines = [
            f"**Pipeline Timing:** mode=`{self._pipeline_last_mode}` frame=`{self._pipeline_frame_counter}`",
            "",
            "```text",
            "Stage                   cur     avg     min     max",
        ]
        for stage in self._pipeline_stage_order:
            current_ms = self._pipeline_last_total_ms if stage == "total" else self._pipeline_last_timings_ms.get(stage)
            stats = self._pipeline_timing_stats.get(stage)
            if current_ms is None and (stats is None or int(stats["count"]) == 0):
                continue
            avg_ms = (
                float(stats["sum_ms"]) / int(stats["count"])
                if stats is not None and int(stats["count"]) > 0
                else None
            )
            min_ms = None if stats is None else stats["min_ms"]
            max_ms = None if stats is None else stats["max_ms"]
            label = self._pipeline_stage_labels.get(stage, stage)
            cur_text = "---" if current_ms is None else f"{float(current_ms):6.1f}"
            avg_text = "---" if avg_ms is None else f"{float(avg_ms):6.1f}"
            min_text = "---" if min_ms is None else f"{float(min_ms):6.1f}"
            max_text = "---" if max_ms is None else f"{float(max_ms):6.1f}"
            lines.append(f"{label:<22} {cur_text} {avg_text} {min_text} {max_text}")
        lines.append("```")
        return "\n".join(lines)

    def _set_markdown_if_changed(self, key: str, handle, content: str) -> None:
        previous = self._last_gui_text_values.get(key)
        if previous == content:
            return
        handle.content = content
        self._last_gui_text_values[key] = content

    def _make_camera_info_signature(self, camera_info) -> tuple | None:
        if not isinstance(camera_info, dict):
            return None
        intrinsics = camera_info.get("intrinsics")
        metadata = camera_info.get("calibration_metadata") or {}
        resolution = camera_info.get("resolution")
        return (
            camera_info.get("backend"),
            tuple(sorted(intrinsics.items())) if isinstance(intrinsics, dict) else None,
            metadata.get("calibration_path"),
            tuple(sorted(metadata.items())) if isinstance(metadata, dict) else None,
            tuple(sorted(resolution.items())) if isinstance(resolution, dict) else None,
        )

    def _update_camera_info_cache(self, camera_info) -> None:
        signature = self._make_camera_info_signature(camera_info)
        if signature == self._last_camera_info_signature:
            return
        self._last_camera_info_signature = signature
        self._cached_camera_calibration_text = self._camera_calibration_text_from_info(camera_info)

    def _prepare_gui_preview(self, image: np.ndarray) -> np.ndarray:
        if image.ndim < 2:
            return image
        height, width = image.shape[:2]
        scale = min(
            1.0,
            self._gui_preview_max_width / max(width, 1),
            self._gui_preview_max_height / max(height, 1),
        )
        if scale >= 0.999:
            return image
        resized_w = max(1, int(round(width * scale)))
        resized_h = max(1, int(round(height * scale)))
        return cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

    def _update_spatial_bbox_visual(self) -> None:
        if not hasattr(self, "_server"):
            return
        if self._spatial_bbox_handle is not None:
            self._spatial_bbox_handle.remove()
            self._spatial_bbox_handle = None
        min_corner = self._spatial_bbox_min
        max_corner = self._spatial_bbox_max
        corners = np.array(
            [
                [min_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], min_corner[1], min_corner[2]],
                [max_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], max_corner[1], min_corner[2]],
                [min_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], min_corner[1], max_corner[2]],
                [max_corner[0], max_corner[1], max_corner[2]],
                [min_corner[0], max_corner[1], max_corner[2]],
            ],
            dtype=np.float32,
        )
        edges = np.asarray(
            [
                [corners[0], corners[1]],
                [corners[1], corners[2]],
                [corners[2], corners[3]],
                [corners[3], corners[0]],
                [corners[4], corners[5]],
                [corners[5], corners[6]],
                [corners[6], corners[7]],
                [corners[7], corners[4]],
                [corners[0], corners[4]],
                [corners[1], corners[5]],
                [corners[2], corners[6]],
                [corners[3], corners[7]],
            ],
            dtype=np.float32,
        )
        self._spatial_bbox_handle = self._server.scene.add_line_segments(
            "/spatial_language/bbox",
            points=edges,
            colors=np.array([255, 220, 80], dtype=np.uint8),
            line_width=2.5,
        )

    def _clear_spatial_voxel_visuals(self) -> None:
        if self._spatial_tblock_voxel_handle is not None:
            self._spatial_tblock_voxel_handle.remove()
            self._spatial_tblock_voxel_handle = None
        if self._spatial_pusher_voxel_handle is not None:
            self._spatial_pusher_voxel_handle.remove()
            self._spatial_pusher_voxel_handle = None

    def _update_spatial_gui(self, spatial_result) -> None:
        show_2d = bool(self._spatial_show_2d_checkbox.value)
        self._spatial_2d_handle.visible = show_2d
        if show_2d:
            self._spatial_2d_handle.image = render_projected_voxels_image(spatial_result)
        if not self._spatial_show_3d_checkbox.value:
            self._clear_spatial_voxel_visuals()
            return

        self._clear_spatial_voxel_visuals()
        tblock_mesh = build_voxel_mesh(
            spatial_result.tblock_voxels_3d,
            bbox_min=spatial_result.bbox_min,
            voxel_size_xyz=spatial_result.voxel_size_xyz,
            color_rgba=(80, 170, 255, 150),
        )
        if tblock_mesh is not None:
            self._spatial_tblock_voxel_handle = self._server.scene.add_mesh_trimesh(
                "/spatial_language/tblock_voxels",
                tblock_mesh,
            )
        pusher_mesh = build_voxel_mesh(
            spatial_result.pusher_voxels_3d,
            bbox_min=spatial_result.bbox_min,
            voxel_size_xyz=spatial_result.voxel_size_xyz,
            color_rgba=(255, 110, 80, 220),
        )
        if pusher_mesh is not None:
            self._spatial_pusher_voxel_handle = self._server.scene.add_mesh_trimesh(
                "/spatial_language/pusher_voxels",
                pusher_mesh,
            )

    def _format_spatial_sentence_box(self, sentence: str) -> str:
        return self._format_scroll_box("Sentence", sentence, height_px=132)

    def _format_scroll_box(self, title: str, text: str, *, height_px: int = 132) -> str:
        escaped = (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return (
            f"**{title}:**\n\n"
            f"<div style=\"height: {int(height_px)}px; overflow-y: auto; padding: 10px; "
            "border: 1px solid rgba(128,128,128,0.35); border-radius: 8px; "
            "background: rgba(0,0,0,0.03);\">"
            f"<code style=\"white-space: pre-wrap; word-break: break-word;\">{escaped}</code>"
            "</div>"
        )

    def _escape_html_text(self, text: str) -> str:
        return (
            str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )

    def _format_online_inference_details_link(self) -> str:
        if self._online_inference_details_url:
            return (
                "**Inference Details:** "
                f"<a href=\"{self._online_inference_details_url}\" target=\"_blank\" "
                "rel=\"noopener noreferrer\">Open In New Window</a>"
            )
        return f"**Inference Details:** {self._escape_html_text(self._online_inference_details_text)}"

    def _build_online_inference_details_url(
        self,
        *,
        mode_label: str,
        status_text: str,
        prompt_text: str,
        output_text: str,
        raw_response_text: str,
        ) -> str:
        self._ensure_online_inference_details_server()
        title = f"Online Inference Details - {mode_label}"
        sections = [
            ("Mode", mode_label),
            ("Status", status_text),
            ("Prompt", prompt_text),
            ("Output", output_text),
            ("Raw Response", raw_response_text),
        ]
        section_html = []
        for heading, body in sections:
            section_html.append(
                "<section>"
                f"<h2>{self._escape_html_text(heading)}</h2>"
                "<pre>"
                f"{self._escape_html_text(body)}"
                "</pre>"
                "</section>"
            )
        html_doc = (
            "<!DOCTYPE html><html><head><meta charset=\"utf-8\">"
            f"<title>{self._escape_html_text(title)}</title>"
            "<style>"
            "body{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;"
            "margin:24px;background:#f7f7f5;color:#1f2328;line-height:1.45;}"
            "h1{font-size:20px;margin:0 0 16px 0;}h2{font-size:14px;margin:0 0 8px 0;}"
            "section{margin:0 0 18px 0;padding:14px;border:1px solid #d0d7de;"
            "border-radius:10px;background:#fff;}"
            "pre{white-space:pre-wrap;word-break:break-word;margin:0;}"
            "</style></head><body>"
            f"<h1>{self._escape_html_text(title)}</h1>"
            f"{''.join(section_html)}"
            "</body></html>"
        )
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"inference_{timestamp}_{int(time.time() * 1000) % 1000:03d}.html"
        output_path = self._online_inference_details_dir / filename
        output_path.write_text(html_doc, encoding="utf-8")
        return f"{self._online_inference_details_base_url}/{filename}"

    def _ensure_online_inference_details_server(self) -> None:
        if self._online_inference_details_httpd is not None:
            return

        self._online_inference_details_dir.mkdir(parents=True, exist_ok=True)

        class _QuietHandler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, format: str, *args) -> None:
                return

        handler = partial(
            _QuietHandler,
            directory=str(self._online_inference_details_dir),
        )
        httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        thread = threading.Thread(
            target=httpd.serve_forever,
            name="online-inference-details-http",
            daemon=True,
        )
        thread.start()
        self._online_inference_details_httpd = httpd
        self._online_inference_details_server_thread = thread
        host, port = httpd.server_address[:2]
        self._online_inference_details_base_url = f"http://{host}:{port}"

    def _blank_online_inference_vis_image(self) -> np.ndarray:
        return np.full(
            (
                int(self._spatial_resolution_xyz[1]) * 16,
                int(self._spatial_resolution_xyz[0]) * 16,
                3,
            ),
            255,
            dtype=np.uint8,
        )

    def _parse_spatial_coord_token(self, token: str) -> tuple[int, int] | None:
        token = str(token).strip()
        if not (token.startswith("(") and token.endswith(")")):
            return None
        body = token[1:-1]
        parts = body.split(",", maxsplit=1)
        if len(parts) != 2:
            return None
        try:
            return (int(parts[0].strip()), int(parts[1].strip()))
        except ValueError:
            return None

    def _parse_spatial_sequence_text(self, text: str) -> dict[str, object]:
        parsed: dict[str, object] = {
            "pusher": [],
            "tbar": [],
            "goal": [],
            "moveto": [],
        }
        mode: str | None = None
        for token in str(text).split():
            if token == "state":
                mode = None
                continue
            if token in ("pusher", "tbar", "goal"):
                mode = token
                continue
            if token == "moveto":
                mode = "moveto"
                continue
            if token == "end":
                mode = None
                continue
            coord = self._parse_spatial_coord_token(token)
            if coord is None:
                continue
            if mode == "moveto":
                cast_list = parsed["moveto"]
                assert isinstance(cast_list, list)
                cast_list.append(coord)
                mode = None
                continue
            if mode in ("pusher", "tbar", "goal"):
                cast_list = parsed[mode]
                assert isinstance(cast_list, list)
                cast_list.append(coord)
        return parsed

    def _future_output_moveto_coords(
        self,
        *,
        prompt_text: str,
        output_text: str,
    ) -> list[tuple[int, int]]:
        prompt_parsed = self._parse_spatial_sequence_text(prompt_text)
        output_parsed = self._parse_spatial_sequence_text(output_text)
        moveto_coords = list(output_parsed["moveto"])
        prompt_moveto_coords = list(prompt_parsed["moveto"])
        if (
            len(prompt_moveto_coords) > 0
            and moveto_coords[: len(prompt_moveto_coords)] == prompt_moveto_coords
        ):
            moveto_coords = moveto_coords[len(prompt_moveto_coords):]
        return moveto_coords

    def _representative_pusher_coord(self, spatial_result) -> tuple[int, int] | None:
        pusher_coords = np.asarray(spatial_result.pusher_voxels_2d, dtype=np.int32).reshape(-1, 2)
        if len(pusher_coords) == 0:
            return None
        coord = self._representative_action_coord(pusher_coords)
        return (int(coord[0]), int(coord[1]))

    def _representative_pusher_z_index(self, spatial_result) -> int | None:
        pusher_voxels_3d = np.asarray(spatial_result.pusher_voxels_3d, dtype=np.int32).reshape(-1, 3)
        if len(pusher_voxels_3d) == 0:
            return None
        return int(np.median(pusher_voxels_3d[:, 2]))

    def _spatial_coord_to_world_point(
        self,
        coord_xy: tuple[int, int],
        *,
        spatial_result,
        z_index: int | None = None,
        force_z_world: float | None = None,
    ) -> np.ndarray | None:
        resolution_xyz = np.maximum(np.asarray(spatial_result.resolution_xyz, dtype=np.int32).reshape(3), 1)
        bbox_min = np.asarray(spatial_result.bbox_min, dtype=np.float64).reshape(3)
        voxel_size_xyz = np.asarray(spatial_result.voxel_size_xyz, dtype=np.float64).reshape(3)
        x_idx = int(coord_xy[0])
        y_idx = int(coord_xy[1])
        if not (0 <= x_idx < int(resolution_xyz[0]) and 0 <= y_idx < int(resolution_xyz[1])):
            return None
        if z_index is None:
            z_index = self._representative_pusher_z_index(spatial_result)
        if z_index is None:
            return None
        z_idx = int(np.clip(z_index, 0, int(resolution_xyz[2]) - 1))
        indices_xyz = np.array([x_idx, y_idx, z_idx], dtype=np.float64)
        point_world = bbox_min + (indices_xyz + 0.5) * voxel_size_xyz
        if force_z_world is not None:
            point_world[2] = float(force_z_world)
        return point_world

    def _spatial_config_coord_to_world_point(
        self,
        coord_xyz: tuple[int, int, int],
    ) -> np.ndarray | None:
        resolution_xyz = np.maximum(
            np.asarray(self._spatial_resolution_xyz, dtype=np.int32).reshape(3), 1
        )
        indices_xyz = np.asarray(coord_xyz, dtype=np.int32).reshape(3)
        if np.any(indices_xyz < 0) or np.any(indices_xyz >= resolution_xyz):
            return None
        bbox_min = np.asarray(self._spatial_bbox_min, dtype=np.float64).reshape(3)
        bbox_max = np.asarray(self._spatial_bbox_max, dtype=np.float64).reshape(3)
        voxel_size_xyz = (bbox_max - bbox_min) / resolution_xyz.astype(np.float64)
        return bbox_min + (indices_xyz.astype(np.float64) + 0.5) * voxel_size_xyz

    def _current_pusher_discrete_coord(self) -> tuple[int, int] | None:
        live_coord = self._live_pusher_discrete_coord_from_arm()
        if live_coord is not None:
            return live_coord

        spatial_result = self._latest_live_spatial_result
        if spatial_result is not None and getattr(spatial_result, "available", False):
            coord = self._representative_pusher_coord(spatial_result)
            if coord is not None:
                return coord

        pusher_world = self._get_live_pusher_world_point()
        if pusher_world is None:
            return None
        coord = project_point_to_spatial_coord(
            pusher_world,
            bbox_min=self._spatial_bbox_min,
            bbox_max=self._spatial_bbox_max,
            resolution_xyz=self._spatial_resolution_xyz,
        )
        if coord is None:
            return None
        return (int(coord[0]), int(coord[1]))

    def _target_pusher_world_from_discrete(
        self,
        coord_xy: tuple[int, int],
        *,
        z_index: int,
    ) -> np.ndarray | None:
        spatial_result = self._latest_live_spatial_result
        if spatial_result is not None and getattr(spatial_result, "available", False):
            return self._spatial_coord_to_world_point(
                coord_xy,
                spatial_result=spatial_result,
                z_index=z_index,
            )
        return self._spatial_config_coord_to_world_point(
            (int(coord_xy[0]), int(coord_xy[1]), int(z_index))
        )

    def _target_pusher_base_from_discrete(
        self,
        coord_xy: tuple[int, int],
        *,
        z_index: int,
    ) -> np.ndarray | None:
        if self._arm_world_result is None:
            return None
        target_world = self._target_pusher_world_from_discrete(coord_xy, z_index=z_index)
        if target_world is None:
            return None
        return point_world_to_base(target_world, self._arm_world_result.T_base_from_world)

    def _pusher_discrete_coord_to_eef_base_point(
        self,
        coord_xy: tuple[int, int],
        *,
        z_index: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Convert pusher discrete target to world/base pusher points and base EEF point."""
        if self._arm_world_result is None or not self._latest_arm_pose_valid:
            return None
        pusher_world = self._target_pusher_world_from_discrete(coord_xy, z_index=z_index)
        if pusher_world is None:
            return None
        pusher_base = point_world_to_base(
            pusher_world,
            self._arm_world_result.T_base_from_world,
        )
        tip_offset_base = (
            self._latest_eef_rotation_base
            @ self._arm_world_result.tip_position_in_eef_m
        )
        eef_base = pusher_base - tip_offset_base
        return pusher_world, pusher_base, eef_base

    def _clear_online_inference_prediction_visuals(self) -> None:
        if self._online_inference_current_pusher_handle is not None:
            self._online_inference_current_pusher_handle.remove()
            self._online_inference_current_pusher_handle = None
        if self._online_inference_prediction_path_handle is not None:
            self._online_inference_prediction_path_handle.remove()
            self._online_inference_prediction_path_handle = None
        for handle in self._online_inference_prediction_point_handles:
            handle.remove()
        self._online_inference_prediction_point_handles = []

    def _update_online_inference_prediction_visuals(self) -> None:
        self._clear_online_inference_prediction_visuals()
        spatial_result = self._online_inference_active_spatial_result
        if spatial_result is None:
            return

        current_coord = self._representative_pusher_coord(spatial_result)
        self._online_inference_current_pusher_coord_text = (
            "---" if current_coord is None else f"({current_coord[0]},{current_coord[1]})"
        )
        if current_coord is None:
            return

        z_index = self._representative_pusher_z_index(spatial_result)
        current_world = self._spatial_coord_to_world_point(
            current_coord,
            spatial_result=spatial_result,
            z_index=z_index,
        )
        if current_world is None:
            return

        self._online_inference_current_pusher_handle = self._server.scene.add_icosphere(
            "/online_inference_prediction/current_pusher",
            radius=0.008,
            color=(255, 110, 80),
            position=tuple(current_world.tolist()),
        )

        future_coords = self._future_output_moveto_coords(
            prompt_text=self._online_inference_prompt_text,
            output_text=self._online_inference_output_text,
        )
        if len(future_coords) == 0:
            return

        future_points_world: list[np.ndarray] = []
        for idx, coord_xy in enumerate(future_coords):
            point_world = self._spatial_coord_to_world_point(
                coord_xy,
                spatial_result=spatial_result,
                z_index=z_index,
                force_z_world=0.0,
            )
            if point_world is None:
                continue
            future_points_world.append(point_world)
            handle = self._server.scene.add_icosphere(
                f"/online_inference_prediction/future_{idx}",
                radius=0.007,
                color=(170, 60, 220),
                position=tuple(point_world.tolist()),
            )
            self._online_inference_prediction_point_handles.append(handle)

        if len(future_points_world) == 0:
            return

        path_points = [current_world] + future_points_world
        if len(path_points) >= 2:
            segments = np.asarray(
                [[path_points[idx], path_points[idx + 1]] for idx in range(len(path_points) - 1)],
                dtype=np.float32,
            )
            self._online_inference_prediction_path_handle = self._server.scene.add_line_segments(
                "/online_inference_prediction/path",
                points=segments,
                colors=np.asarray([170, 60, 220], dtype=np.uint8),
                line_width=3.0,
            )

    def _render_online_inference_sequence_image(
        self,
        *,
        prompt_text: str,
        output_text: str,
        scale: int = 16,
    ) -> np.ndarray:
        prompt_parsed = self._parse_spatial_sequence_text(prompt_text)
        output_parsed = self._parse_spatial_sequence_text(output_text)

        pusher_coords = output_parsed["pusher"] or prompt_parsed["pusher"]
        tbar_coords = output_parsed["tbar"] or prompt_parsed["tbar"]
        goal_coords = output_parsed["goal"] or prompt_parsed["goal"]
        moveto_coords = self._future_output_moveto_coords(
            prompt_text=prompt_text,
            output_text=output_text,
        )

        resolution_x = max(int(self._spatial_resolution_xyz[0]), 1)
        resolution_y = max(int(self._spatial_resolution_xyz[1]), 1)
        scale = max(int(scale), 1)
        image = np.full((resolution_y * scale, resolution_x * scale, 3), 255, dtype=np.uint8)

        def fill_cells(coords: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
            for x_idx, y_idx in coords:
                if not (0 <= int(x_idx) < resolution_x and 0 <= int(y_idx) < resolution_y):
                    continue
                row0 = (resolution_y - 1 - int(y_idx)) * scale
                col0 = int(x_idx) * scale
                image[row0:row0 + scale, col0:col0 + scale] = np.asarray(color, dtype=np.uint8)

        fill_cells(goal_coords, (130, 210, 120))
        fill_cells(tbar_coords, (80, 170, 255))
        fill_cells(pusher_coords, (255, 110, 80))

        if moveto_coords:
            centers: list[tuple[int, int]] = []
            for x_idx, y_idx in moveto_coords:
                if not (0 <= int(x_idx) < resolution_x and 0 <= int(y_idx) < resolution_y):
                    continue
                row = (resolution_y - 1 - int(y_idx)) * scale + scale // 2
                col = int(x_idx) * scale + scale // 2
                centers.append((col, row))
            for start, end in zip(centers[:-1], centers[1:], strict=False):
                cv2.line(image, start, end, (170, 60, 220), thickness=max(scale // 4, 1))
            for center in centers:
                cv2.circle(image, center, radius=max(scale // 3, 1), color=(170, 60, 220), thickness=-1)

        for x in range(0, resolution_x * scale + 1, scale):
            cv2.line(image, (x, 0), (x, resolution_y * scale), (230, 230, 230), thickness=1)
        for y in range(0, resolution_y * scale + 1, scale):
            cv2.line(image, (0, y), (resolution_x * scale, y), (230, 230, 230), thickness=1)
        return image

    def _get_live_pusher_world_point(self) -> np.ndarray | None:
        if self._arm_world_result is None or not self._latest_arm_pose_valid:
            return None
        tip_offset_eef = self._arm_world_result.tip_position_in_eef_m
        pusher_base = self._latest_eef_position_base + self._latest_eef_rotation_base @ tip_offset_eef
        return point_base_to_world(
            pusher_base,
            self._arm_world_result.T_world_from_base,
        )

    def _live_pusher_discrete_from_arm(self) -> tuple[tuple[int, int], int] | None:
        pusher_world = self._get_live_pusher_world_point()
        if pusher_world is None:
            return None
        bbox_min = np.asarray(self._spatial_bbox_min, dtype=np.float64).reshape(3)
        bbox_max = np.asarray(self._spatial_bbox_max, dtype=np.float64).reshape(3)
        resolution_xyz = np.maximum(
            np.asarray(self._spatial_resolution_xyz, dtype=np.int32).reshape(3),
            1,
        )
        extent = bbox_max - bbox_min
        if np.any(extent <= 0.0):
            return None
        voxel_size_xyz = extent / resolution_xyz.astype(np.float64)
        rel = (np.asarray(pusher_world, dtype=np.float64).reshape(3) - bbox_min) / voxel_size_xyz
        idx = np.floor(rel).astype(np.int32)
        if np.any(idx < 0) or np.any(idx >= resolution_xyz):
            return None
        return (int(idx[0]), int(idx[1])), int(idx[2])

    def _live_pusher_discrete_coord_from_arm(self) -> tuple[int, int] | None:
        projected = self._live_pusher_discrete_from_arm()
        if projected is None:
            return None
        return projected[0]

    def _format_pusher_discrete_coord(self) -> str:
        if not self._latest_arm_pose_valid:
            return "**Pusher Coord (Discrete):** waiting for arm pose"
        if self._arm_world_result is None:
            return "**Pusher Coord (Discrete):** unavailable - calibrate Arm To World"

        live_projected = self._live_pusher_discrete_from_arm()
        if live_projected is not None:
            coord, z_index = live_projected
            return (
                "**Pusher Coord (Discrete):** "
                f"({coord[0]},{coord[1]})  Z: {z_index}"
            )

        spatial_result = self._latest_live_spatial_result
        if spatial_result is not None and getattr(spatial_result, "available", False):
            coord = self._representative_pusher_coord(spatial_result)
            z_index = self._representative_pusher_z_index(spatial_result)
            if coord is not None:
                z_text = "" if z_index is None else f"  Z: {z_index}"
                return (
                    "**Pusher Coord (Discrete):** "
                    f"({coord[0]},{coord[1]}){z_text}"
                )

        pusher_world = self._get_live_pusher_world_point()
        coord = (
            None
            if pusher_world is None
            else project_point_to_spatial_coord(
                pusher_world,
                bbox_min=self._spatial_bbox_min,
                bbox_max=self._spatial_bbox_max,
                resolution_xyz=self._spatial_resolution_xyz,
            )
        )
        if coord is None:
            return "**Pusher Coord (Discrete):** outside grid"
        return f"**Pusher Coord (Discrete):** ({coord[0]},{coord[1]})"

    def _format_pusher_position(self) -> str:
        return self._format_pusher_discrete_coord()

    def _record_live_pusher_history(self) -> None:
        if self._replaying:
            return
        pusher_world = self._get_live_pusher_world_point()
        if pusher_world is None:
            return
        self._live_pusher_world_history.append(np.asarray(pusher_world, dtype=np.float64).copy())

    def _latest_pusher_moveto_coords(
        self,
        current_coord: tuple[int, int] | None,
        *,
        max_count: int = 2,
    ) -> list[tuple[int, int]]:
        coords_reversed: list[tuple[int, int]] = []
        if current_coord is not None:
            coords_reversed.append(tuple(current_coord))

        history_snapshot = list(self._live_pusher_world_history)
        for point_world in reversed(history_snapshot):
            coord = project_point_to_spatial_coord(
                point_world,
                bbox_min=self._spatial_bbox_min,
                bbox_max=self._spatial_bbox_max,
                resolution_xyz=self._spatial_resolution_xyz,
            )
            if coord is None:
                continue
            if coords_reversed and coord == coords_reversed[-1]:
                continue
            coords_reversed.append(coord)
            if len(coords_reversed) >= max_count:
                break

        return list(reversed(coords_reversed[:max_count]))

    def _spatial_results_equivalent(self, left, right) -> bool:
        if left is None or right is None:
            return False
        return (
            np.array_equal(
                np.asarray(left.pusher_voxels_2d, dtype=np.int32),
                np.asarray(right.pusher_voxels_2d, dtype=np.int32),
            )
            and np.array_equal(
                np.asarray(left.tblock_voxels_2d, dtype=np.int32),
                np.asarray(right.tblock_voxels_2d, dtype=np.int32),
            )
        )

    def _record_live_spatial_history(self) -> None:
        if self._replaying:
            return
        spatial_result = self._compute_live_spatial_result()
        self._latest_live_spatial_result = spatial_result
        if not spatial_result.available:
            return
        self._latest_available_live_spatial_result = spatial_result
        if (
            len(self._live_spatial_result_history) > 0
            and self._spatial_results_equivalent(self._live_spatial_result_history[-1], spatial_result)
        ):
            return
        self._live_spatial_result_history.append(spatial_result)

    def _movement_coords_from_spatial_results(
        self,
        spatial_results: list,
    ) -> list[tuple[int, int]]:
        coords: list[tuple[int, int]] = []
        for spatial_result in spatial_results:
            pusher_coords = np.asarray(spatial_result.pusher_voxels_2d, dtype=np.int32).reshape(-1, 2)
            if len(pusher_coords) == 0:
                continue
            coord = tuple(self._representative_action_coord(pusher_coords))
            if coords and coord == coords[-1]:
                continue
            coords.append(coord)
        return coords

    def _compress_spatial_results_by_pusher_runs(
        self,
        spatial_results: list,
    ) -> list:
        compressed: list = []
        last_coord: tuple[int, int] | None = None
        for spatial_result in spatial_results:
            if spatial_result is None or not spatial_result.available:
                continue
            coord = self._representative_pusher_coord(spatial_result)
            if coord is None:
                continue
            if compressed and coord == last_coord:
                compressed[-1] = spatial_result
            else:
                compressed.append(spatial_result)
                last_coord = coord
        return compressed

    def _build_temporal_online_inference_prompt_from_results(
        self,
        spatial_results: list,
    ) -> tuple[str, object]:
        if not spatial_results:
            raise RuntimeError("No spatial results available for online inference.")
        current_spatial_result = spatial_results[-1]
        if not current_spatial_result.available:
            raise RuntimeError(current_spatial_result.status)

        available_results = [result for result in spatial_results if result is not None and result.available]
        if not available_results:
            raise RuntimeError(current_spatial_result.status)

        first_movement_pos = max(0, len(available_results) - 2)
        state_pos = max(0, first_movement_pos - 1)
        state_spatial_result = available_results[state_pos]
        moveto_coords = self._movement_coords_from_spatial_results(
            available_results[first_movement_pos:]
        )
        if state_pos == first_movement_pos and len(moveto_coords) > 0:
            state_anchor = self._representative_pusher_coord(state_spatial_result)
            if state_anchor is not None and moveto_coords[0] == state_anchor:
                moveto_coords = moveto_coords[1:]

        prompt_text = self._build_online_inference_prompt_from_spatial_result(
            state_spatial_result,
            movement_coords=moveto_coords,
        )
        return prompt_text, current_spatial_result

    def _get_replay_spatial_result(self, frame_idx: int):
        if self._replay_data is None:
            raise RuntimeError("Replay is not active.")
        num_frames = int(self._replay_data["num_frames"])
        if num_frames <= 0:
            raise RuntimeError("Replay data is empty.")
        frame_idx = int(np.clip(frame_idx, 0, num_frames - 1))
        spatial_result = self._replay_data["spatial_language"][frame_idx]
        if spatial_result is None:
            spatial_result = self._compute_replay_spatial_result(
                self._replay_data["qpos"][frame_idx],
                self._replay_data["gripper"][frame_idx],
                self._replay_data["tblock_pose_world"][frame_idx],
            )
            self._replay_data["spatial_language"][frame_idx] = spatial_result
        return spatial_result

    def _compute_live_spatial_result(self):
        start_t = time.perf_counter()
        if self._arm_world_result is None:
            result = unavailable_result(
                "Load or calibrate `Arm To World` before online inference.",
                bbox_min=self._spatial_bbox_min,
                bbox_max=self._spatial_bbox_max,
                resolution_xyz=self._spatial_resolution_xyz,
            )
            return replace(
                result,
                processing_time_ms=(time.perf_counter() - start_t) * 1000.0,
            )
        if not self._latest_arm_pose_valid:
            result = unavailable_result(
                "Waiting for a live arm pose.",
                bbox_min=self._spatial_bbox_min,
                bbox_max=self._spatial_bbox_max,
                resolution_xyz=self._spatial_resolution_xyz,
            )
            return replace(
                result,
                processing_time_ms=(time.perf_counter() - start_t) * 1000.0,
            )
        pose_world = self._parse_tblock_pose_world(self._latest_tblock_pose_world)
        if pose_world is None:
            result = unavailable_result(
                "Waiting for a live Tblock pose from AprilTag reconstruction.",
                bbox_min=self._spatial_bbox_min,
                bbox_max=self._spatial_bbox_max,
                resolution_xyz=self._spatial_resolution_xyz,
            )
            return replace(
                result,
                processing_time_ms=(time.perf_counter() - start_t) * 1000.0,
            )
        if self._apriltag_static_model is None:
            result = unavailable_result(
                "Static Tblock model is unavailable.",
                bbox_min=self._spatial_bbox_min,
                bbox_max=self._spatial_bbox_max,
                resolution_xyz=self._spatial_resolution_xyz,
            )
            return replace(
                result,
                processing_time_ms=(time.perf_counter() - start_t) * 1000.0,
            )

        pusher_world = self._get_live_pusher_world_point()
        if pusher_world is None:
            result = unavailable_result(
                "Waiting for a live pusher point.",
                bbox_min=self._spatial_bbox_min,
                bbox_max=self._spatial_bbox_max,
                resolution_xyz=self._spatial_resolution_xyz,
            )
            return replace(
                result,
                processing_time_ms=(time.perf_counter() - start_t) * 1000.0,
            )

        full_rotation = self._rotation_from_wxyz(pose_world["wxyz"])
        yaw_deg = full_rotation.as_euler("xyz", degrees=True)[2]
        rotation = Rotation.from_euler("z", yaw_deg, degrees=True)
        mesh_vertices_world = (
            self._apriltag_static_model.mesh_vertices @ rotation.as_matrix().T
            + np.asarray(pose_world["translation_m"], dtype=np.float64).reshape(1, 3)
        )
        result = compute_spatial_language(
            mesh_vertices_world=mesh_vertices_world,
            mesh_faces=self._apriltag_static_model.mesh_faces,
            pusher_point_world=pusher_world,
            bbox_min=self._spatial_bbox_min,
            bbox_max=self._spatial_bbox_max,
            resolution_xyz=self._spatial_resolution_xyz,
        )
        return replace(
            result,
            processing_time_ms=(time.perf_counter() - start_t) * 1000.0,
        )

    def _build_online_inference_prompt(self) -> tuple[str, object]:
        self._reload_online_inference_goal()
        if len(self._online_inference_goal_coords) == 0:
            raise RuntimeError(
                f"Dataset goal unavailable. {self._online_inference_goal_source}"
            )
        current_spatial_result = self._compute_live_spatial_result()
        if not current_spatial_result.available:
            raise RuntimeError(current_spatial_result.status)

        recent_results = list(self._live_spatial_result_history)
        if len(recent_results) == 0 or not self._spatial_results_equivalent(
            recent_results[-1],
            current_spatial_result,
        ):
            recent_results.append(current_spatial_result)
        recent_results = self._compress_spatial_results_by_pusher_runs(recent_results)
        recent_results = recent_results[-3:]
        return self._build_temporal_online_inference_prompt_from_results(recent_results)

    def _build_online_inference_prompt_from_spatial_result(
        self,
        spatial_result,
        *,
        movement_coords: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
    ) -> str:
        prompt_text = build_composited_sentence(
            pusher_coords=[tuple(coord) for coord in spatial_result.pusher_voxels_2d.tolist()],
            tbar_coords=[tuple(coord) for coord in spatial_result.tblock_voxels_2d.tolist()],
            goal_coords=[tuple(coord) for coord in self._online_inference_goal_coords.tolist()],
            movement_coords=movement_coords,
        )
        if not prompt_text:
            raise RuntimeError("Unable to compose a grammar-compliant prompt.")
        return prompt_text

    def _build_replay_online_inference_prompt(self) -> tuple[str, object]:
        if not self._replaying or self._replay_data is None:
            raise RuntimeError("Replay is not active.")
        if not self._replay_paused:
            raise RuntimeError("Pause replay on the target frame before online inference.")

        self._reload_online_inference_goal()
        if len(self._online_inference_goal_coords) == 0:
            raise RuntimeError(
                f"Dataset goal unavailable. {self._online_inference_goal_source}"
            )

        idx = int(np.clip(self._replay_idx, 0, self._replay_data["num_frames"] - 1))
        replay_results = [
            self._get_replay_spatial_result(frame_idx)
            for frame_idx in range(max(0, idx - 2), idx + 1)
        ]
        return self._build_temporal_online_inference_prompt_from_results(replay_results)

    def _get_online_inference_status_payload(self) -> dict:
        req = request.Request(
            url=f"{self._online_inference_server}/api/status",
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="GET",
        )
        try:
            with request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {text}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Failed to reach server: {exc}") from exc

    def _post_online_inference(self, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self._online_inference_server}/api/infer",
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {text}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Failed to reach server: {exc}") from exc

    def _run_online_inference_request(self, payload: dict) -> None:
        mode_label = "Replay" if self._replaying else "Live"
        try:
            status_payload = self._get_online_inference_status_payload()
            if not status_payload.get("status", {}).get("ready", False):
                raise RuntimeError("Server is not ready. Load a model on the server first.")
            result = self._post_online_inference(payload)
            self._online_inference_output_text = result.get("result", {}).get("output_text", "")
            if not self._online_inference_output_text:
                self._online_inference_output_text = "(empty response)"
            self._online_inference_raw_response_text = json.dumps(
                result,
                indent=2,
                ensure_ascii=False,
            )
            self._online_inference_vis_image = self._render_online_inference_sequence_image(
                prompt_text=self._online_inference_prompt_text,
                output_text=self._online_inference_output_text,
            )
            self._online_inference_prediction_visual_dirty = True
            self._online_inference_status = "Inference completed."
            self._online_inference_details_url = self._build_online_inference_details_url(
                mode_label=mode_label,
                status_text=self._online_inference_status,
                prompt_text=self._online_inference_prompt_text,
                output_text=self._online_inference_output_text,
                raw_response_text=self._online_inference_raw_response_text,
            )
            self._online_inference_details_text = "Ready"
        except Exception as exc:
            self._online_inference_output_text = f"Error: {exc}"
            self._online_inference_raw_response_text = f"Error: {exc}"
            self._online_inference_status = f"Error: {exc}"
            self._online_inference_details_url = self._build_online_inference_details_url(
                mode_label=mode_label,
                status_text=self._online_inference_status,
                prompt_text=self._online_inference_prompt_text,
                output_text=self._online_inference_output_text,
                raw_response_text=self._online_inference_raw_response_text,
            )
            self._online_inference_details_text = "Ready"
        finally:
            self._online_inference_busy = False

    def _on_online_inference_click(self, _event) -> None:
        if self._online_inference_busy:
            return
        self._online_inference_server = self._online_inference_server_input.value.strip().rstrip("/")
        if not self._online_inference_server:
            self._online_inference_status = "Error: server URL is empty."
            return
        try:
            if self._replaying:
                prompt_text, spatial_result = self._build_replay_online_inference_prompt()
            else:
                prompt_text, spatial_result = self._build_online_inference_prompt()
        except Exception as exc:
            self._online_inference_status = f"Error: {exc}"
            self._online_inference_output_text = f"Error: {exc}"
            self._online_inference_raw_response_text = f"Error: {exc}"
            return

        self._online_inference_pending_spatial_result = spatial_result
        self._online_inference_active_spatial_result = spatial_result
        self._online_inference_prompt_text = prompt_text
        self._spatial_composited_sentence_text = prompt_text
        self._online_inference_output_text = "waiting for response"
        self._online_inference_raw_response_text = "waiting for response"
        self._online_inference_vis_image = self._render_online_inference_sequence_image(
            prompt_text=prompt_text,
            output_text="",
        )
        current_pusher_coord = self._representative_pusher_coord(spatial_result)
        self._online_inference_current_pusher_coord_text = (
            "---"
            if current_pusher_coord is None
            else f"({current_pusher_coord[0]},{current_pusher_coord[1]})"
        )
        self._clear_online_inference_prediction_visuals()
        self._online_inference_prediction_visual_dirty = True
        self._online_inference_details_url = ""
        self._online_inference_details_text = "Detail page will be available after inference completes."
        self._online_inference_status = "Working..."
        try:
            top_k_text = self._online_inference_top_k_input.value.strip()
            self._online_inference_max_new_tokens = int(self._online_inference_max_new_tokens_input.value)
            self._online_inference_temperature = float(self._online_inference_temperature_input.value)
            self._online_inference_top_k = int(top_k_text) if top_k_text else None
            self._online_inference_do_sample = bool(self._online_inference_do_sample_checkbox.value)
            self._online_inference_movement_only = bool(
                self._online_inference_movement_only_checkbox.value
            )
            self._online_inference_forbidden_tokens = self._effective_forbidden_tokens_text(
                self._online_inference_forbidden_tokens_input.value.strip(),
                movement_only=self._online_inference_movement_only,
            )
            self._online_inference_require_state_after_movement = (
                False
                if self._online_inference_movement_only
                else bool(self._online_inference_require_state_checkbox.value)
            )
            self._online_inference_state_after_movement_prob = (
                0.0
                if self._online_inference_movement_only
                else float(self._online_inference_state_after_prob_input.value)
            )
            payload = {
                "prompt_text": prompt_text,
                "max_new_tokens": self._online_inference_max_new_tokens,
                "temperature": self._online_inference_temperature,
                "do_sample": self._online_inference_do_sample,
                "top_k": self._online_inference_top_k,
                "forbidden_tokens": self._online_inference_forbidden_tokens,
                "require_state_after_movement": self._online_inference_require_state_after_movement,
                "state_after_movement_prob": self._online_inference_state_after_movement_prob,
            }
        except Exception as exc:
            self._online_inference_status = f"Error: {exc}"
            self._online_inference_output_text = f"Error: {exc}"
            self._online_inference_raw_response_text = f"Error: {exc}"
            return
        self._online_inference_busy = True
        threading.Thread(
            target=self._run_online_inference_request,
            args=(payload,),
            daemon=True,
        ).start()

    def _compute_replay_spatial_result(
        self,
        qpos: np.ndarray,
        gripper: float,
        pose_world: dict | None,
    ):
        start_t = time.perf_counter()
        if self._arm_world_result is None:
            result = unavailable_result(
                "Load or calibrate `Arm To World` before replay to align the pusher with the Tblock frame.",
                bbox_min=self._spatial_bbox_min,
                bbox_max=self._spatial_bbox_max,
                resolution_xyz=self._spatial_resolution_xyz,
            )
            return replace(
                result,
                processing_time_ms=(time.perf_counter() - start_t) * 1000.0,
            )
        pose_world = self._parse_tblock_pose_world(pose_world)
        if pose_world is None:
            result = unavailable_result(
                "No recorded Tblock pose for this frame.",
                bbox_min=self._spatial_bbox_min,
                bbox_max=self._spatial_bbox_max,
                resolution_xyz=self._spatial_resolution_xyz,
            )
            return replace(
                result,
                processing_time_ms=(time.perf_counter() - start_t) * 1000.0,
            )
        self._ensure_apriltag_static_model_scene()
        if self._apriltag_static_model is None:
            result = unavailable_result(
                "Static Tblock model is unavailable.",
                bbox_min=self._spatial_bbox_min,
                bbox_max=self._spatial_bbox_max,
                resolution_xyz=self._spatial_resolution_xyz,
            )
            return replace(
                result,
                processing_time_ms=(time.perf_counter() - start_t) * 1000.0,
            )

        cfg = can_qpos_to_urdf_cfg_with_gripper(qpos, gripper)
        eef_position_base, eef_rotation_base, _ = eef_pose_from_urdf_cfg(self._urdf, cfg)
        tip_offset_eef = self._arm_world_result.tip_position_in_eef_m
        pusher_base = eef_position_base + eef_rotation_base @ tip_offset_eef
        pusher_world = point_base_to_world(
            pusher_base,
            self._arm_world_result.T_world_from_base,
        )
        full_rotation = self._rotation_from_wxyz(pose_world["wxyz"])
        # Zero roll/pitch before voxelization; keep only the in-plane yaw.
        yaw_deg = full_rotation.as_euler("xyz", degrees=True)[2]
        rotation = Rotation.from_euler("z", yaw_deg, degrees=True)
        mesh_vertices_world = (
            self._apriltag_static_model.mesh_vertices @ rotation.as_matrix().T
            + np.asarray(pose_world["translation_m"], dtype=np.float64).reshape(1, 3)
        )
        result = compute_spatial_language(
            mesh_vertices_world=mesh_vertices_world,
            mesh_faces=self._apriltag_static_model.mesh_faces,
            pusher_point_world=pusher_world,
            bbox_min=self._spatial_bbox_min,
            bbox_max=self._spatial_bbox_max,
            resolution_xyz=self._spatial_resolution_xyz,
        )
        return replace(
            result,
            processing_time_ms=(time.perf_counter() - start_t) * 1000.0,
        )

    def _update_apriltag_reconstruction(
        self,
        detections: list,
        camera_info: dict | None,
        timestamp_s: float | None = None,
    ):
        self._latest_tblock_pose_world = None
        self._latest_tblock_pose_world_raw = None
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
        filtered_tblock_transform = self._apriltag_tblock_pose_smoother.update(
            reconstruction.T_world_from_object,
            timestamp_s=pose_timestamp_s,
            reproj_error_px=float(reconstruction.object_reproj_error_px),
            visible_tag_count=len(reconstruction.visible_object_tag_ids),
        )
        raw_mesh_vertices_world = reconstruction.mesh_vertices_world
        filtered_mesh_vertices_world = filtered_tblock_transform.apply_points(
            self._apriltag_static_model.mesh_vertices
        )
        filtered_center_world = np.mean(filtered_mesh_vertices_world, axis=0)

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
            filtered_tblock_transform
        )
        self._latest_tblock_pose_world_raw = self._serialize_tblock_pose_world(
            reconstruction.T_world_from_object
        )
        self._apriltag_recon_status = (
            f"World PnP={reconstruction.world_reproj_error_px:.2f}px | "
            f"Object PnP={reconstruction.object_reproj_error_px:.2f}px"
        )
        self._apriltag_points_text = self._format_live_reconstruction_text(
            reconstruction,
            filtered_transform=filtered_tblock_transform,
            filtered_camera_from_world=filtered_camera_from_world,
            filtered_mesh_center_world=filtered_center_world,
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
        self._apriltag_tblock_pose_smoother = self._build_apriltag_camera_world_smoother(
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
        if self._dataset_selected_episodes:
            self._dataset_selected_episodes = [
                episode for episode in self._dataset_selected_episodes if episode in episodes
            ]
            self._update_dataset_selection_ui()

    def _update_dataset_selection_ui(self) -> None:
        if not hasattr(self, "_dataset_selected_md"):
            return
        if not self._dataset_selected_episodes:
            self._dataset_selected_md.content = "**Dataset Episodes:** none"
        else:
            lines = "\n".join(f"- `{episode}`" for episode in self._dataset_selected_episodes)
            self._dataset_selected_md.content = f"**Dataset Episodes ({len(self._dataset_selected_episodes)}):**\n\n{lines}"
        goal_label = self._dataset_global_goal_episode or "per-episode"
        if hasattr(self, "_dataset_goal_md"):
            self._dataset_goal_md.content = f"**Global Goal:** `{goal_label}`"

    def _on_dataset_add_click(self, _event) -> None:
        selected = self._replay_dropdown.value
        if selected == "(none)":
            return
        if selected not in self._dataset_selected_episodes:
            self._dataset_selected_episodes.append(selected)
            self._dataset_selected_episodes.sort()
            self._update_dataset_selection_ui()

    def _on_dataset_remove_click(self, _event) -> None:
        selected = self._replay_dropdown.value
        if selected in self._dataset_selected_episodes:
            self._dataset_selected_episodes.remove(selected)
            self._update_dataset_selection_ui()

    def _on_dataset_add_all_click(self, _event) -> None:
        episodes = [episode for episode in self._list_episodes() if episode != "(none)"]
        self._dataset_selected_episodes = episodes
        self._update_dataset_selection_ui()

    def _on_dataset_clear_click(self, _event) -> None:
        self._dataset_selected_episodes = []
        self._update_dataset_selection_ui()

    def _on_dataset_set_goal_click(self, _event) -> None:
        selected = self._replay_dropdown.value
        if selected == "(none)":
            return
        self._dataset_global_goal_episode = selected
        self._update_dataset_selection_ui()

    def _on_dataset_clear_goal_click(self, _event) -> None:
        self._dataset_global_goal_episode = None
        self._update_dataset_selection_ui()

    def _set_dataset_export_status(self, message: str) -> None:
        if hasattr(self, "_dataset_status_md"):
            self._dataset_status_md.content = f"**Dataset Export:** {message}"

    def _on_dataset_export_click(self, _event) -> None:
        if self._recording or self._replaying:
            self._set_dataset_export_status("stop recording/replay first")
            return
        self._dataset_export_btn.disabled = True
        self._set_dataset_export_status("starting...")
        try:
            output_path = self._export_selected_episodes_dataset()
        except Exception as exc:
            self._set_dataset_export_status(f"failed: {exc}")
            self._dataset_export_btn.disabled = False
            return
        self._set_dataset_export_status(f"saved `{output_path.name}`")
        self._dataset_export_btn.disabled = False

    def _encode_binary_mask(self, mask: np.ndarray) -> str:
        mask = np.asarray(mask, dtype=bool)
        packed = np.packbits(mask.astype(np.uint8).reshape(-1))
        compressed = zlib.compress(packed.tobytes())
        return base64.b64encode(compressed).decode("ascii")

    def _decode_binary_mask(self, encoded: str) -> np.ndarray:
        compressed = base64.b64decode(encoded)
        packed = zlib.decompress(compressed)
        bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
        n_bits = bits.size
        side = int(math.isqrt(n_bits))
        if side * side != n_bits:
            raise ValueError(f"Cannot infer square mask side from {n_bits} bits.")
        return bits.reshape((side, side)).astype(bool)

    def _black_pixel_coords(self, mask: np.ndarray) -> np.ndarray:
        rows, cols = np.where(~np.asarray(mask, dtype=bool))
        if len(rows) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        coords = np.stack([cols, rows], axis=1).astype(np.int32)
        order = np.lexsort((coords[:, 1], coords[:, 0]))
        return coords[order]

    def _latest_dataset_json_path(self) -> Path | None:
        dataset_dir = Path(self._output_dir) / "datasets"
        if not dataset_dir.is_dir():
            return None
        candidates = sorted(dataset_dir.glob("graspgpt_dataset_*.json"))
        candidates = [
            path for path in candidates
            if path.is_file() and not path.name.endswith("_report.json")
        ]
        if not candidates:
            return None
        return candidates[-1]

    def _reload_online_inference_goal(self) -> None:
        dataset_path = self._latest_dataset_json_path()
        if dataset_path is None:
            self._online_inference_goal_coords = np.zeros((0, 2), dtype=np.int32)
            self._online_inference_goal_dataset_path = None
            self._online_inference_goal_source = "No dataset JSON found in output_dir/datasets"
            return
        try:
            payload = json.loads(dataset_path.read_text(encoding="utf-8"))
            trajectories = payload.get("trajectories", [])
            if not trajectories:
                raise ValueError("No trajectories in dataset JSON.")
            goal_encoded = trajectories[0].get("goal")
            if not isinstance(goal_encoded, str) or not goal_encoded:
                raise ValueError("First trajectory has no goal mask.")
            goal_mask = self._decode_binary_mask(goal_encoded)
            goal_coords = self._black_pixel_coords(goal_mask)
            if len(goal_coords) == 0:
                raise ValueError("Goal mask decoded to zero coordinates.")
            self._online_inference_goal_coords = goal_coords
            self._online_inference_goal_dataset_path = dataset_path
            self._online_inference_goal_source = f"`{dataset_path.name}` | {len(goal_coords)} goal cells"
        except Exception as exc:
            self._online_inference_goal_coords = np.zeros((0, 2), dtype=np.int32)
            self._online_inference_goal_dataset_path = dataset_path
            self._online_inference_goal_source = f"`{dataset_path.name}` failed: {exc}"

    def _effective_forbidden_tokens_text(self, user_text: str, *, movement_only: bool) -> str:
        tokens: list[str] = []
        seen: set[str] = set()

        def add_token(token: str) -> None:
            token = str(token).strip()
            if not token or token in seen:
                return
            seen.add(token)
            tokens.append(token)

        raw_text = str(user_text).replace("\n", ",")
        for token in raw_text.split(","):
            add_token(token)

        if movement_only:
            for token in self._MOVEMENT_ONLY_FORBIDDEN_TOKENS:
                add_token(token)

        return ",".join(tokens)

    def _coords_to_encoded_mask(
        self,
        coords_xy: np.ndarray,
        *,
        side: int,
    ) -> str:
        mask = np.ones((side, side), dtype=bool)
        coords_xy = np.asarray(coords_xy, dtype=np.int32).reshape(-1, 2)
        for x_idx, y_idx in coords_xy:
            if 0 <= int(x_idx) < side and 0 <= int(y_idx) < side:
                mask[int(y_idx), int(x_idx)] = False
        return self._encode_binary_mask(mask)

    def _representative_action_coord(self, coords_xy: np.ndarray) -> list[int]:
        coords_xy = np.asarray(coords_xy, dtype=np.int32).reshape(-1, 2)
        if len(coords_xy) == 0:
            return [0, 0]
        center = np.mean(coords_xy.astype(np.float64), axis=0)
        distances = np.sum((coords_xy.astype(np.float64) - center) ** 2, axis=1)
        return coords_xy[int(np.argmin(distances))].astype(int).tolist()

    def _coords_iou(self, coords_a: np.ndarray, coords_b: np.ndarray) -> float:
        coords_a = np.asarray(coords_a, dtype=np.int32).reshape(-1, 2)
        coords_b = np.asarray(coords_b, dtype=np.int32).reshape(-1, 2)
        if len(coords_a) == 0 or len(coords_b) == 0:
            return 0.0
        set_a = {tuple(coord) for coord in coords_a.tolist()}
        set_b = {tuple(coord) for coord in coords_b.tolist()}
        union = len(set_a | set_b)
        if union == 0:
            return 0.0
        return float(len(set_a & set_b) / union)

    def _filter_duplicate_pusher_frames(
        self,
        frames: list[dict],
    ) -> tuple[list[dict], int]:
        filtered_frames: list[dict] = []
        last_kept_pusher_coords: np.ndarray | None = None
        removed_count = 0

        for frame in frames:
            pusher_coords = np.asarray(frame["pusher_coords"], dtype=np.int32).reshape(-1, 2)
            if (
                last_kept_pusher_coords is not None
                and np.array_equal(pusher_coords, last_kept_pusher_coords)
            ):
                removed_count += 1
                continue
            filtered_frames.append(frame)
            last_kept_pusher_coords = pusher_coords.copy()

        return filtered_frames, removed_count

    def _load_episode_metadata(self, episode_dir: Path) -> dict:
        metadata_path = episode_dir / "metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def _episode_to_dataset_frames(
        self,
        episode_dir: Path,
        *,
        use_tblock_pose_filter: bool = True,
        require_pusher_overlap: bool = True,
    ) -> tuple[list[dict], np.ndarray | None, np.ndarray | None, int]:
        metadata = self._load_episode_metadata(episode_dir)
        frame_records = metadata.get("frames", [])
        if len(frame_records) == 0:
            return [], None, None, -1

        timestamps_s = np.asarray(
            [
                frame.get("camera_timestamp", frame.get("timestamp", 0.0))
                for frame in frame_records
            ],
            dtype=np.float64,
        )
        invalid_timestamps = ~np.isfinite(timestamps_s) | (timestamps_s <= 0.0)
        if np.any(invalid_timestamps):
            fallback_dt = 1.0 / max(float(self._fps), 1.0)
            for idx in np.flatnonzero(invalid_timestamps):
                timestamps_s[idx] = timestamps_s[idx - 1] + fallback_dt if idx > 0 else fallback_dt

        raw_tblock_poses = [frame.get("tblock_pose_world") for frame in frame_records]
        if use_tblock_pose_filter and bool(self._replay_filter_checkbox.value):
            tblock_poses, _ = self._filter_replay_tblock_poses(raw_tblock_poses, timestamps_s)
        else:
            tblock_poses = raw_tblock_poses

        valid_frames: list[dict] = []
        last_tbar_coords: np.ndarray | None = None
        last_tbar_coords_full: np.ndarray | None = None
        for frame, pose_world in zip(frame_records, tblock_poses, strict=False):
            spatial_result = self._compute_replay_spatial_result(
                np.asarray(frame["qpos"], dtype=np.float64),
                float(frame["gripper"]),
                pose_world,
            )
            if require_pusher_overlap and not spatial_result.available:
                continue
            pusher_coords = np.asarray(spatial_result.pusher_voxels_2d, dtype=np.int32)
            tbar_coords = np.asarray(spatial_result.tblock_voxels_2d, dtype=np.int32)
            tbar_coords_full = np.asarray(spatial_result.tblock_voxels_2d_full, dtype=np.int32)
            if len(tbar_coords) == 0:
                continue
            if require_pusher_overlap and len(pusher_coords) == 0:
                continue
            valid_frames.append(
                {
                    "frame_index": int(frame.get("frame_index", len(valid_frames))),
                    "pusher_coords": pusher_coords,
                    "tbar_coords": tbar_coords,
                    "tbar_coords_full": tbar_coords_full,
                    "action": self._representative_action_coord(pusher_coords),
                }
            )
            last_tbar_coords = tbar_coords
            last_tbar_coords_full = tbar_coords_full
        last_frame_index = int(frame_records[-1].get("frame_index", len(frame_records) - 1))
        return valid_frames, last_tbar_coords, last_tbar_coords_full, last_frame_index

    def _episode_to_graspgpt_trajectory(
        self,
        episode_dir: Path,
        *,
        goal_coords: np.ndarray | None = None,
        goal_coords_full: np.ndarray | None = None,
    ) -> tuple[dict | None, list[list[int]], dict]:
        valid_frames, last_tbar_coords, last_tbar_coords_full, last_frame_index = self._episode_to_dataset_frames(episode_dir)
        if not valid_frames:
            return None, [], {
                "source_episode": episode_dir.name,
                "exported_frame_count": 0,
                "duplicate_pusher_removed_count": 0,
                "deleted_frame_ranges": [],
                "truncation_trigger_frame": None,
                "truncation_iou_threshold": 0.95,
            }
        if goal_coords is None:
            goal_coords = last_tbar_coords
        if goal_coords_full is None:
            goal_coords_full = last_tbar_coords_full
        if goal_coords is None:
            return None, [], {
                "source_episode": episode_dir.name,
                "exported_frame_count": 0,
                "duplicate_pusher_removed_count": 0,
                "deleted_frame_ranges": [],
                "truncation_trigger_frame": None,
                "truncation_iou_threshold": 0.95,
            }

        deduped_frames, duplicate_pusher_removed_count = self._filter_duplicate_pusher_frames(valid_frames)
        truncation_trigger_frame = None
        deleted_ranges: list[list[int]] = []
        truncated_frames = deduped_frames
        if goal_coords_full is not None:
            for idx, frame in enumerate(deduped_frames):
                if self._coords_iou(frame["tbar_coords_full"], goal_coords_full) > 0.95:
                    truncation_trigger_frame = int(frame["frame_index"])
                    truncated_frames = deduped_frames[: idx + 1]
                    if truncation_trigger_frame < last_frame_index:
                        deleted_ranges.append([truncation_trigger_frame + 1, last_frame_index])
                    break

        side = int(max(self._spatial_resolution_xyz[0], self._spatial_resolution_xyz[1]))
        side = int(math.ceil(side / 4.0) * 4)
        frames_payload = []
        for frame in truncated_frames:
            frames_payload.append(
                {
                    "action": frame["action"],
                    "pusher": self._coords_to_encoded_mask(frame["pusher_coords"], side=side),
                    "tbar": self._coords_to_encoded_mask(frame["tbar_coords"], side=side),
                }
            )

        trajectory = {
            "source_episode": episode_dir.name,
            "goal": self._coords_to_encoded_mask(goal_coords, side=side),
            "frames": frames_payload,
        }
        report_entry = {
            "source_episode": episode_dir.name,
            "exported_frame_count": len(frames_payload),
            "duplicate_pusher_removed_count": duplicate_pusher_removed_count,
            "deleted_frame_ranges": deleted_ranges,
            "truncation_trigger_frame": truncation_trigger_frame,
            "truncation_iou_threshold": 0.95,
        }
        return trajectory, deleted_ranges, report_entry

    def _export_selected_episodes_dataset(self) -> Path:
        if not self._dataset_selected_episodes:
            raise ValueError("No episodes selected.")

        total_selected = len(self._dataset_selected_episodes)
        global_goal_coords: np.ndarray | None = None
        global_goal_coords_full: np.ndarray | None = None
        if self._dataset_global_goal_episode is not None:
            self._set_dataset_export_status(
                f"preparing goal from `{self._dataset_global_goal_episode}`"
            )
            goal_episode_dir = Path(self._output_dir) / self._dataset_global_goal_episode
            if not goal_episode_dir.is_dir():
                raise ValueError(f"Global goal episode not found: {self._dataset_global_goal_episode}")
            _, global_goal_coords, global_goal_coords_full, _ = self._episode_to_dataset_frames(
                goal_episode_dir,
                use_tblock_pose_filter=False,
                require_pusher_overlap=False,
            )
            if global_goal_coords is None or len(global_goal_coords) == 0:
                raise ValueError(
                    f"Global goal episode has no valid final Tblock projection: {self._dataset_global_goal_episode}"
                )

        trajectories = []
        report_entries = []
        for episode_idx, episode_name in enumerate(self._dataset_selected_episodes, start=1):
            self._set_dataset_export_status(
                f"processing {episode_idx}/{total_selected}: `{episode_name}`"
            )
            episode_dir = Path(self._output_dir) / episode_name
            if not episode_dir.is_dir():
                continue
            trajectory, _, report_entry = self._episode_to_graspgpt_trajectory(
                episode_dir,
                goal_coords=global_goal_coords,
                goal_coords_full=global_goal_coords_full,
            )
            report_entries.append(report_entry)
            if trajectory is not None:
                trajectories.append(trajectory)

        if not trajectories:
            raise ValueError("No valid trajectories could be exported.")

        dataset_dir = Path(self._output_dir) / "datasets"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = dataset_dir / f"graspgpt_dataset_{timestamp}.json"
        report_path = dataset_dir / f"graspgpt_dataset_{timestamp}_report.json"
        self._set_dataset_export_status(
            f"writing dataset files for {len(trajectories)} trajectory(s)"
        )
        payload = {
            # Format aligned to GraspGPT PushTDataset:
            # https://github.com/wuminye/GraspGPT/blob/pusht/graspGPT/model/pushT_dataset.py
            "format": "graspgpt_pusht_merged_v1",
            "source": "robodata_Agilex_minye",
            "selected_episodes": list(self._dataset_selected_episodes),
            "global_goal_episode": self._dataset_global_goal_episode,
            "spatial_config": {
                "bbox_min": self._spatial_bbox_min.tolist(),
                "bbox_max": self._spatial_bbox_max.tolist(),
                "resolution_xyz": self._spatial_resolution_xyz.tolist(),
                "tblock_pose_filter_enabled": bool(self._replay_filter_checkbox.value),
                "tblock_projection": "orthographic_xy_boundary_only",
                "pusher_point": "arm_world_calibrated_stick_tip",
            },
            "trajectories": trajectories,
        }
        report_payload = {
            "dataset_file": output_path.name,
            "generated_at": timestamp,
            "global_goal_episode": self._dataset_global_goal_episode,
            "iou_truncation_threshold": 0.95,
            "duplicate_pusher_removed_total": int(
                sum(entry.get("duplicate_pusher_removed_count", 0) for entry in report_entries)
            ),
            "episodes": report_entries,
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
        return output_path

    def _main_loop(self):
        target_dt = 1.0 / self._fps
        while True:
            loop_start = time.perf_counter()
            frame_timings_ms: dict[str, float] = {}

            stage_start = time.perf_counter()
            self._sync_spatial_language_settings()
            frame_timings_ms["settings_sync"] = (time.perf_counter() - stage_start) * 1000.0

            if self._replaying and self._replay_data is not None:
                # Check if stop was requested (from button callback thread)
                if self._replay_stop_requested:
                    self._finish_replay_cleanup()
                    self._update_pipeline_timing_snapshot(
                        mode="Replay",
                        frame_timings_ms=frame_timings_ms,
                        total_ms=(time.perf_counter() - loop_start) * 1000.0,
                    )
                    self._pipeline_timing_md.content = self._format_pipeline_timing_markdown()
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = target_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                # --- Replay mode: use recorded data ---
                rd = self._replay_data
                if self._replay_step_delta != 0:
                    self._replay_idx = int(
                        np.clip(
                            self._replay_idx + self._replay_step_delta,
                            0,
                            rd["num_frames"] - 1,
                        )
                    )
                    self._replay_step_delta = 0
                idx = self._replay_idx

                if idx >= rd["num_frames"]:
                    # Replay finished
                    self._finish_replay_cleanup()
                    self._update_pipeline_timing_snapshot(
                        mode="Replay",
                        frame_timings_ms=frame_timings_ms,
                        total_ms=(time.perf_counter() - loop_start) * 1000.0,
                    )
                    self._pipeline_timing_md.content = self._format_pipeline_timing_markdown()
                    elapsed = time.perf_counter() - loop_start
                    sleep_time = target_dt - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    continue

                # Build display state from recorded data
                stage_start = time.perf_counter()
                display_state = ArmState(
                    qpos=rd["qpos"][idx],
                    qvel=np.zeros(6, dtype=np.float64),
                    gripper=rd["gripper"][idx],
                    timestamp=time.time(),
                )
                color = rd["color"][idx]
                frame_timings_ms["replay_frame"] = (time.perf_counter() - stage_start) * 1000.0

                # Update OpenCV replay window
                stage_start = time.perf_counter()
                color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                cv2.imshow("Replay", color_bgr)
                cv2.waitKey(1)
                frame_timings_ms["replay_window"] = (time.perf_counter() - stage_start) * 1000.0

                stage_start = time.perf_counter()
                preview_color = color
                self._clear_apriltag_live_scene()
                has_tblock_pose = self._update_replay_tblock_visual(
                    rd["tblock_pose_world"][idx]
                )
                self._apriltag_status = "Replay mode: overlay disabled"
                pose_mode = (
                    "filtered"
                    if rd.get("tblock_filter_enabled", False)
                    else "raw"
                )
                if has_tblock_pose:
                    self._apriltag_recon_status = f"Replay mode: {pose_mode} Tblock pose"
                    self._apriltag_points_text = (
                        "Replay mode: "
                        f"{pose_mode} Tblock pose "
                        f"(corrected {rd.get('tblock_filter_corrected', 0)} frames)"
                    )
                else:
                    self._apriltag_recon_status = "Replay mode: no recorded Tblock pose"
                    self._apriltag_points_text = "Replay mode: no recorded Tblock pose"
                frame_timings_ms["replay_tblock_visual"] = (time.perf_counter() - stage_start) * 1000.0

                stage_start = time.perf_counter()
                spatial_result = rd["spatial_language"][idx]
                if spatial_result is None:
                    spatial_result = self._compute_replay_spatial_result(
                        rd["qpos"][idx],
                        rd["gripper"][idx],
                        rd["tblock_pose_world"][idx],
                    )
                    rd["spatial_language"][idx] = spatial_result
                self._update_spatial_gui(spatial_result)
                frame_timings_ms["replay_spatial"] = (time.perf_counter() - stage_start) * 1000.0
                if not self._replay_paused:
                    self._replay_idx += 1
                progress = f"{idx + 1}/{rd['num_frames']}"
                mode_label = "PAUSED" if self._replay_paused else "PLAYING"
                self._status_md.content = f"**Status:** REPLAY {mode_label} | Frame {progress}"
            else:
                # --- Live mode ---
                stage_start = time.perf_counter()
                self._sync_camera_selection()
                self._sync_opencv_zed_mode()
                self._sync_zed_controls()
                self._sync_apriltag_point_list_visibility()
                if self._camera is not None and hasattr(self._camera, "get_camera_info"):
                    self._writer.set_camera_info(self._camera.get_camera_info())
                frame_timings_ms["controls_sync"] = (time.perf_counter() - stage_start) * 1000.0

                stage_start = time.perf_counter()
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
                frame_timings_ms["camera_capture"] = (time.perf_counter() - stage_start) * 1000.0

                camera_info = self._camera.get_camera_info() if self._camera is not None else None
                self._update_camera_info_cache(camera_info)
                stage_start = time.perf_counter()
                detections = self._detect_apriltags(color)
                frame_timings_ms["apriltag_detect"] = (time.perf_counter() - stage_start) * 1000.0

                stage_start = time.perf_counter()
                self._maybe_collect_pre_reconstruct_frame(detections)
                frame_timings_ms["pre_reconstruct"] = (time.perf_counter() - stage_start) * 1000.0

                stage_start = time.perf_counter()
                pose_timestamp_s = camera_timestamp if self._camera is not None else time.time()
                self._update_apriltag_reconstruction(
                    detections,
                    camera_info,
                    timestamp_s=pose_timestamp_s,
                )
                frame_timings_ms["apriltag_reconstruct"] = (time.perf_counter() - stage_start) * 1000.0

                stage_start = time.perf_counter()
                preview_color = self._render_apriltag_overlay(color, detections)
                frame_timings_ms["overlay_render"] = (time.perf_counter() - stage_start) * 1000.0

                stage_start = time.perf_counter()
                self._update_spatial_gui(
                    unavailable_result(
                        "Replay to generate per-frame spatial language.",
                        bbox_min=self._spatial_bbox_min,
                        bbox_max=self._spatial_bbox_max,
                        resolution_xyz=self._spatial_resolution_xyz,
                    )
                )
                frame_timings_ms["spatial_gui"] = (time.perf_counter() - stage_start) * 1000.0

                # Update depth display
                stage_start = time.perf_counter()
                if self._depth_handle is not None:
                    if depth.max() > 0:
                        depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
                        depth_u8 = depth_norm.astype(np.uint8)
                        depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
                        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)
                    else:
                        depth_color = np.zeros((self._frame_h, self._frame_w, 3), dtype=np.uint8)
                    self._depth_handle.image = depth_color
                frame_timings_ms["depth_display"] = (time.perf_counter() - stage_start) * 1000.0

            # --- Common updates (both live and replay) ---
            # Update 3D arm visualization
            stage_start = time.perf_counter()
            cfg = can_qpos_to_urdf_cfg_with_gripper(display_state.qpos, display_state.gripper)
            self._latest_arm_cfg = cfg.copy()
            fk_eef_position_base, fk_eef_rotation_base, fk_eef_wxyz_base = eef_pose_from_urdf_cfg(
                self._urdf,
                cfg,
            )
            direct_pose_valid = (
                display_state.eef_pos_m is not None
                and display_state.eef_euler_deg is not None
                and display_state.pose_timestamp > 0.0
            )
            if direct_pose_valid:
                direct_euler_deg = np.asarray(display_state.eef_euler_deg, dtype=np.float64)
                self._latest_eef_position_base = np.asarray(
                    display_state.eef_pos_m,
                    dtype=np.float64,
                )
                self._latest_eef_rotation_base = Rotation.from_euler(
                    "XYZ",
                    direct_euler_deg,
                    degrees=True,
                ).as_matrix()
                self._latest_eef_wxyz_base = euler_deg_to_wxyz(*direct_euler_deg)
                pose_source_label = f"direct robot pose ({display_state.pose_source or 'unknown'})"
            else:
                self._latest_eef_position_base = fk_eef_position_base
                self._latest_eef_rotation_base = fk_eef_rotation_base
                self._latest_eef_wxyz_base = fk_eef_wxyz_base
                pose_source_label = "URDF FK from joints"
            self._latest_arm_pose_valid = True
            self._record_live_pusher_history()
            self._record_live_spatial_history()
            if self._urdf_vis is not None:
                self._urdf_vis.update_cfg(cfg)
            self._update_world_arm_visual(cfg)
            frame_timings_ms["arm_visual"] = (time.perf_counter() - stage_start) * 1000.0

            # Update camera display
            stage_start = time.perf_counter()
            self._latest_color_frame = color
            gui_preview = self._prepare_gui_preview(preview_color)
            self._color_handle.image = gui_preview
            frame_timings_ms["gui_image_upload"] = (time.perf_counter() - stage_start) * 1000.0

            # Prefer the robot's direct Cartesian pose when it is present; otherwise
            # fall back to the fingertip endpoint center from URDF FK.
            if direct_pose_valid:
                eef_pos_base = self._latest_eef_position_base
            else:
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

            should_update_fast_gui_text = (
                self._pipeline_frame_counter % self._gui_fast_text_update_interval_frames == 0
            )
            should_update_slow_gui_text = (
                self._pipeline_frame_counter % self._gui_slow_text_update_interval_frames == 0
            )

            # Update arm state display
            if should_update_fast_gui_text:
                stage_start = time.perf_counter()
                if self._online_inference_pending_spatial_result is not None:
                    self._update_spatial_gui(self._online_inference_pending_spatial_result)
                    self._online_inference_pending_spatial_result = None
                if self._online_inference_prediction_visual_dirty:
                    self._update_online_inference_prediction_visuals()
                    self._online_inference_prediction_visual_dirty = False
                    if hasattr(self, "_online_inference_vis_handle"):
                        self._online_inference_vis_handle.image = self._online_inference_vis_image
                if should_update_slow_gui_text:
                    self._set_markdown_if_changed(
                        "apriltag_status",
                        self._apriltag_status_md,
                        f"**AprilTag:** {self._apriltag_status}",
                    )
                    self._set_markdown_if_changed(
                        "pre_reconstruct_status",
                        self._pre_reconstruct_status_md,
                        f"**Pre-Reconstruct:** {self._pre_reconstruct_status}",
                    )
                    self._set_markdown_if_changed(
                        "apriltag_points",
                        self._apriltag_points_md,
                        f"**AprilTag 3D Points:**\n\n{self._apriltag_points_text}",
                    )
                if should_update_slow_gui_text and hasattr(self, "_arm_world_status_md"):
                    self._set_markdown_if_changed(
                        "arm_world_status",
                        self._arm_world_status_md,
                        f"**Arm World:** {self._arm_world_status}",
                    )
                if should_update_slow_gui_text:
                    resolution_text = self._format_camera_resolution(
                        color=color,
                        depth=depth,
                    )
                    if resolution_text != self._cached_camera_resolution_text:
                        self._cached_camera_resolution_text = resolution_text
                    self._set_markdown_if_changed(
                        "camera_resolution",
                        self._camera_resolution_md,
                        self._cached_camera_resolution_text,
                    )
                    self._set_markdown_if_changed(
                        "camera_calibration",
                        self._camera_calibration_md,
                        self._cached_camera_calibration_text or "**Calibration:** unavailable",
                    )
                self._set_markdown_if_changed(
                    "eef",
                    self._eef_md,
                    (
                        f"**EEF Position ({frame_label}):**\n\n"
                        f"X: {eef_pos[0]:.4f}  Y: {eef_pos[1]:.4f}  Z: {eef_pos[2]:.4f} m"
                    ),
                )
                self._set_markdown_if_changed(
                    "pusher",
                    self._pusher_md,
                    self._format_pusher_position(),
                )
                self._set_markdown_if_changed(
                    "arm_feedback",
                    self._arm_feedback_md,
                    self._format_arm_feedback(display_state, pose_source_label),
                )
                if hasattr(self, "_arm_lock_btn"):
                    self._arm_lock_btn.disabled = (
                        self._arm_lock_busy or not hasattr(self._arm_reader, "lock_pose")
                    )
                if hasattr(self, "_arm_lock_status_md"):
                    self._set_markdown_if_changed(
                        "arm_lock_status",
                        self._arm_lock_status_md,
                        f"**Arm Lock:** {self._arm_lock_status}",
                    )
                if hasattr(self, "_arm_move_buttons"):
                    move_disabled = (
                        self._arm_move_busy
                        or self._arm_reader is None
                        or not (
                            hasattr(self._arm_reader, "step_eef")
                            or hasattr(self._arm_reader, "move_by_base_delta")
                            or hasattr(self._arm_reader, "move_to_base_position")
                        )
                        or self._arm_world_result is None
                        or not self._latest_arm_pose_valid
                    )
                    for btn in self._arm_move_buttons.values():
                        btn.disabled = move_disabled
                if hasattr(self, "_arm_move_status_md"):
                    self._set_markdown_if_changed(
                        "arm_move_status",
                        self._arm_move_status_md,
                        f"**Pusher Move:** {self._arm_move_status}",
                    )
                qd = np.degrees(display_state.qpos)
                self._set_markdown_if_changed(
                    "qpos",
                    self._qpos_md,
                    (
                        f"**Joint Positions (deg):**\n\n"
                        f"J1: {qd[0]:+7.2f}  J2: {qd[1]:+7.2f}  J3: {qd[2]:+7.2f}\n\n"
                        f"J4: {qd[3]:+7.2f}  J5: {qd[4]:+7.2f}  J6: {qd[5]:+7.2f}"
                    ),
                )
                self._set_markdown_if_changed(
                    "gripper",
                    self._gripper_md,
                    f"**Gripper:** {display_state.gripper*1000:.1f} mm",
                )
                if hasattr(self, "_online_inference_btn"):
                    self._online_inference_btn.disabled = self._online_inference_busy
                if hasattr(self, "_online_inference_current_pusher_coord_handle"):
                    self._online_inference_current_pusher_coord_handle.value = (
                        self._online_inference_current_pusher_coord_text
                    )
                if hasattr(self, "_online_inference_status_md"):
                    self._set_markdown_if_changed(
                        "online_inference_status",
                        self._online_inference_status_md,
                        f"**Online Inference:** {self._online_inference_status}",
                    )
                if should_update_slow_gui_text and hasattr(self, "_online_inference_details_md"):
                    self._set_markdown_if_changed(
                        "online_inference_details",
                        self._online_inference_details_md,
                        self._format_online_inference_details_link(),
                    )
                if should_update_slow_gui_text and hasattr(self, "_online_inference_goal_md"):
                    self._set_markdown_if_changed(
                        "online_inference_goal",
                        self._online_inference_goal_md,
                        f"**Goal Source:** {self._online_inference_goal_source}",
                    )
                if should_update_slow_gui_text and hasattr(self, "_online_inference_mode_md"):
                    mode_text = (
                        "**Decode Mode:** movement-only continuation"
                        if bool(self._online_inference_movement_only_checkbox.value)
                        else "**Decode Mode:** free-form continuation"
                    )
                    self._set_markdown_if_changed(
                        "online_inference_mode",
                        self._online_inference_mode_md,
                        mode_text,
                    )
                if hasattr(self, "_online_inference_require_state_checkbox"):
                    self._online_inference_require_state_checkbox.disabled = bool(
                        self._online_inference_movement_only_checkbox.value
                    )
                if hasattr(self, "_online_inference_state_after_prob_input"):
                    self._online_inference_state_after_prob_input.disabled = bool(
                        self._online_inference_movement_only_checkbox.value
                    )
                if should_update_slow_gui_text and hasattr(self, "_online_inference_prompt_md"):
                    self._set_markdown_if_changed(
                        "online_inference_prompt",
                        self._online_inference_prompt_md,
                        self._format_scroll_box(
                            "Prompt",
                            self._online_inference_prompt_text,
                            height_px=140,
                        ),
                    )
                if should_update_slow_gui_text and hasattr(self, "_spatial_composited_sentence_md"):
                    self._set_markdown_if_changed(
                        "spatial_composited_sentence",
                        self._spatial_composited_sentence_md,
                        self._format_scroll_box(
                            "Composited Sentence",
                            self._spatial_composited_sentence_text,
                            height_px=140,
                        ),
                    )
                if should_update_slow_gui_text and hasattr(self, "_online_inference_output_md"):
                    self._set_markdown_if_changed(
                        "online_inference_output",
                        self._online_inference_output_md,
                        self._format_scroll_box(
                            "Result",
                            self._online_inference_output_text,
                            height_px=140,
                        ),
                    )
                if should_update_slow_gui_text and hasattr(self, "_online_inference_raw_response_md"):
                    self._set_markdown_if_changed(
                        "online_inference_raw_response",
                        self._online_inference_raw_response_md,
                        self._format_scroll_box(
                            "Raw JSON",
                            self._online_inference_raw_response_text,
                            height_px=180,
                        ),
                    )
                frame_timings_ms["gui_text_update"] = (time.perf_counter() - stage_start) * 1000.0

            # Record if active (only in live mode)
            if self._recording and not self._replaying:
                stage_start = time.perf_counter()
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
                    arm_eef_pos_m=display_state.eef_pos_m,
                    arm_eef_euler_deg=display_state.eef_euler_deg,
                    arm_pose_timestamp=(
                        display_state.pose_timestamp
                        if display_state.pose_timestamp > 0.0
                        else None
                    ),
                    arm_pose_source=display_state.pose_source,
                    tblock_pose_world=self._latest_tblock_pose_world_raw,
                )
                n = self._writer.num_frames
                duration = n / self._fps
                if should_update_fast_gui_text:
                    self._status_md.content = self._format_recording_status(n, duration)
                frame_timings_ms["record_write"] = (time.perf_counter() - stage_start) * 1000.0

            total_ms = (time.perf_counter() - loop_start) * 1000.0
            self._update_pipeline_timing_snapshot(
                mode="Replay" if self._replaying and self._replay_data is not None else "Live",
                frame_timings_ms=frame_timings_ms,
                total_ms=total_ms,
            )
            if should_update_slow_gui_text:
                self._pipeline_timing_md.content = self._format_pipeline_timing_markdown()

            elapsed = time.perf_counter() - loop_start
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

    def _on_arm_lock_click(self, _event):
        if self._arm_lock_busy:
            return
        if self._arm_reader is None or not hasattr(self._arm_reader, "lock_pose"):
            self._arm_lock_status = "Unavailable for current arm backend"
            return

        self._arm_lock_busy = True
        self._arm_lock_status = "Locking..."

        def _worker():
            try:
                result = self._arm_reader.lock_pose(speed=50)
                summary = ""
                if isinstance(result, dict):
                    summary = str(result.get("summary") or result.get("status") or "")
                self._arm_lock_status = f"Locked{f' | {summary}' if summary else ''}"
                print(f">> Arm lock requested: {result}")
            except Exception as exc:
                self._arm_lock_status = f"Failed: {type(exc).__name__}: {exc}"
                print(f">> Arm lock failed: {exc}")
            finally:
                self._arm_lock_busy = False

        threading.Thread(target=_worker, daemon=True).start()

    def _on_pusher_nudge_click(self, _event, *, delta: tuple[int, int], label: str):
        if self._arm_move_busy:
            return
        can_move_to_position = hasattr(self._arm_reader, "move_to_base_position")
        can_step_eef = hasattr(self._arm_reader, "step_eef")
        can_move_by_delta = hasattr(self._arm_reader, "move_by_base_delta")
        if self._arm_reader is None or not (
            can_move_to_position or can_step_eef or can_move_by_delta
        ):
            self._arm_move_status = "Unavailable for current arm backend"
            return
        if self._arm_world_result is None:
            self._arm_move_status = "Calibrate Arm To World first"
            return
        if not self._latest_arm_pose_valid:
            self._arm_move_status = "Waiting for arm pose"
            return

        current_coord = self._current_pusher_discrete_coord()
        if current_coord is None:
            self._arm_move_status = "Cannot locate pusher in discrete grid"
            return

        resolution_xy = np.maximum(self._spatial_resolution_xyz[:2], 1)
        target_coord = (
            int(np.clip(current_coord[0] + int(delta[0]), 0, int(resolution_xy[0]) - 1)),
            int(np.clip(current_coord[1] + int(delta[1]), 0, int(resolution_xy[1]) - 1)),
        )
        z_index = int(np.clip(self._arm_move_z_index, 0, max(int(self._spatial_resolution_xyz[2]) - 1, 0)))
        target_points = self._pusher_discrete_coord_to_eef_base_point(
            target_coord,
            z_index=z_index,
        )
        if target_points is None:
            self._arm_move_status = "Cannot convert pusher target to EEF base frame"
            return
        target_world, target_pusher_base, target_eef_base = target_points
        delta_base = None
        if not can_move_to_position and (can_step_eef or can_move_by_delta):
            current_world = self._get_live_pusher_world_point()
            current_base = (
                None
                if current_world is None
                else point_world_to_base(current_world, self._arm_world_result.T_base_from_world)
            )
            delta_base = None if current_base is None else target_pusher_base - current_base
            if delta_base is None:
                self._arm_move_status = "Cannot compute relative robot-base delta"
                return

        self._arm_move_busy = True
        self._arm_move_status = (
            f"{label}: ({current_coord[0]},{current_coord[1]}) -> "
            f"({target_coord[0]},{target_coord[1]}, z={z_index})"
        )

        def _worker():
            try:
                if can_move_to_position:
                    result = self._arm_reader.move_to_base_position(
                        target_eef_base,
                        timesteps=15,
                        speed=1,
                        timeout_s=8.0,
                    )
                elif can_step_eef:
                    result = self._arm_reader.step_eef(
                        delta_base,
                        timesteps=15,
                        speed=1,
                        timeout_s=8.0,
                    )
                elif can_move_by_delta:
                    result = self._arm_reader.move_by_base_delta(
                        delta_base,
                        timesteps=15,
                        speed=1,
                        timeout_s=8.0,
                    )
                status = ""
                if isinstance(result, dict):
                    status = str(result.get("status") or "")
                self._arm_move_status = (
                    f"Moved to ({target_coord[0]},{target_coord[1]}, z={z_index})"
                    f"{f' | {status}' if status else ''}"
                )
                print(
                    ">> Pusher nudge "
                    f"{label}: target discrete=({target_coord[0]},{target_coord[1]},{z_index}), "
                    f"pusher_world={target_world.tolist()}, "
                    f"pusher_base={target_pusher_base.tolist()}, "
                    f"eef_base={target_eef_base.tolist()}, "
                    f"delta_base={None if delta_base is None else delta_base.tolist()}, "
                    f"result={result}"
                )
            except Exception as exc:
                self._arm_move_status = f"Failed: {type(exc).__name__}: {exc}"
                print(f">> Pusher nudge failed: {exc}")
            finally:
                self._arm_move_busy = False

        threading.Thread(target=_worker, daemon=True).start()

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

    def _on_replay_pause_click(self, _event):
        if not self._replaying:
            return
        self._replay_paused = not self._replay_paused
        self._replay_pause_btn.label = "Resume" if self._replay_paused else "Pause"

    def _on_replay_prev_click(self, _event):
        if not self._replaying:
            return
        self._replay_paused = True
        self._replay_pause_btn.label = "Resume"
        self._replay_step_delta = -1

    def _on_replay_next_click(self, _event):
        if not self._replaying:
            return
        self._replay_paused = True
        self._replay_pause_btn.label = "Resume"
        self._replay_step_delta = 1

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

        timestamps_s = np.asarray(
            [
                frame.get("camera_timestamp", frame.get("timestamp", 0.0))
                for frame in frame_records[:num_frames]
            ],
            dtype=np.float64,
        )
        invalid_timestamps = ~np.isfinite(timestamps_s) | (timestamps_s <= 0.0)
        if np.any(invalid_timestamps):
            fallback_dt = 1.0 / max(float(self._fps), 1.0)
            for idx in np.flatnonzero(invalid_timestamps):
                timestamps_s[idx] = (
                    timestamps_s[idx - 1] + fallback_dt if idx > 0 else fallback_dt
                )

        raw_tblock_poses = [
            frame.get("tblock_pose_world")
            for frame in frame_records[:num_frames]
        ]
        use_replay_filter = bool(self._replay_filter_checkbox.value)
        if use_replay_filter:
            replay_tblock_poses, replay_tblock_corrected = self._filter_replay_tblock_poses(
                raw_tblock_poses,
                timestamps_s,
            )
        else:
            replay_tblock_poses = raw_tblock_poses
            replay_tblock_corrected = 0

        self._replay_data = {
            "qpos": np.asarray(
                [frame["qpos"] for frame in frame_records[:num_frames]], dtype=np.float64
            ),
            "gripper": np.asarray(
                [frame["gripper"] for frame in frame_records[:num_frames]], dtype=np.float64
            ),
            "tblock_pose_world": replay_tblock_poses,
            "color": color_frames[:num_frames],
            "num_frames": num_frames,
            "fps": int(round(self._infer_replay_fps(frame_records[:num_frames]))),
            "tblock_filter_enabled": use_replay_filter,
            "tblock_filter_corrected": replay_tblock_corrected,
            "spatial_language": [None] * num_frames,
        }
        self._replay_idx = 0
        self._replaying = True
        self._replay_paused = False
        self._replay_step_delta = 0
        self._replay_filter_enabled = use_replay_filter
        self._replay_filter_checkbox.disabled = True
        self._spatial_resolution_x.disabled = True
        self._spatial_resolution_y.disabled = True
        self._spatial_resolution_z.disabled = True
        self._spatial_bbox_min_x.disabled = True
        self._spatial_bbox_min_y.disabled = True
        self._spatial_bbox_min_z.disabled = True
        self._spatial_bbox_max_x.disabled = True
        self._spatial_bbox_max_y.disabled = True
        self._spatial_bbox_max_z.disabled = True
        self._replay_btn.visible = False
        self._replay_pause_btn.label = "Pause"
        self._replay_pause_btn.visible = True
        self._replay_prev_btn.visible = True
        self._replay_next_btn.visible = True
        self._stop_replay_btn.visible = True
        self._record_btn.disabled = True
        filter_note = (
            f" | Filtered {replay_tblock_corrected} poses"
            if use_replay_filter
            else " | Filter OFF"
        )
        self._status_md.content = (
            f"**Status:** REPLAY | Frame 0/{self._replay_data['num_frames']}{filter_note} | Spatial language on-demand"
        )

    def _finish_replay_cleanup(self):
        """Clean up replay state. Must be called from the main thread."""
        self._replaying = False
        self._replay_stop_requested = False
        self._replay_idx = 0
        self._replay_paused = False
        self._replay_step_delta = 0
        self._remove_replay_tblock_visual()
        self._clear_spatial_voxel_visuals()
        self._clear_online_inference_prediction_visuals()
        # Free large arrays before cv2 cleanup
        del self._replay_data
        self._replay_data = None
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # Flush macOS event queue
        self._replay_filter_checkbox.disabled = False
        self._spatial_resolution_x.disabled = False
        self._spatial_resolution_y.disabled = False
        self._spatial_resolution_z.disabled = False
        self._spatial_bbox_min_x.disabled = False
        self._spatial_bbox_min_y.disabled = False
        self._spatial_bbox_min_z.disabled = False
        self._spatial_bbox_max_x.disabled = False
        self._spatial_bbox_max_y.disabled = False
        self._spatial_bbox_max_z.disabled = False
        self._replay_btn.visible = True
        self._replay_pause_btn.visible = False
        self._replay_prev_btn.visible = False
        self._replay_next_btn.visible = False
        self._stop_replay_btn.visible = False
        self._record_btn.disabled = False
        self._status_md.content = "**Status:** IDLE"
        self._update_spatial_gui(
            unavailable_result(
                "Replay stopped.",
                bbox_min=self._spatial_bbox_min,
                bbox_max=self._spatial_bbox_max,
                resolution_xyz=self._spatial_resolution_xyz,
            )
        )
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

    def _format_arm_feedback(self, state: ArmState, pose_source_label: str) -> str:
        now = time.time()
        joint_text = "no joint feedback"
        if state.timestamp > 0.0:
            age_ms = max(0.0, (now - state.timestamp) * 1000.0)
            joint_text = f"joint/state age {age_ms:.0f} ms"

        pose_text = "no direct pose"
        if state.pose_timestamp > 0.0 and state.eef_pos_m is not None:
            age_ms = max(0.0, (now - state.pose_timestamp) * 1000.0)
            pose_text = f"direct pose age {age_ms:.0f} ms"

        return (
            f"**Arm Feedback:** {pose_source_label}\n\n"
            f"{joint_text} | {pose_text}"
        )

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

    def _camera_calibration_text_from_info(self, camera_info) -> str:
        if not isinstance(camera_info, dict):
            return "**Calibration:** unavailable"

        intrinsics = camera_info.get("intrinsics")
        has_intrinsics = isinstance(intrinsics, dict) and all(
            key in intrinsics for key in ("fx", "fy", "cx", "cy")
        )
        if not has_intrinsics:
            return "**Calibration:** not loaded"

        metadata = camera_info.get("calibration_metadata") or {}
        calibration_path = metadata.get("calibration_path")
        if calibration_path:
            calibration_name = Path(str(calibration_path)).name
            return f"**Calibration:** loaded from `{calibration_name}`"

        backend = str(camera_info.get("backend", "")).lower()
        if backend == "pointgrey":
            return "**Calibration:** loaded"
        return "**Calibration:** device-provided"

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

    def _camera_calibration_text(self) -> str:
        if self._camera is None or not hasattr(self._camera, "get_camera_info"):
            return "**Calibration:** unavailable"

        camera_info = self._camera.get_camera_info()
        return self._camera_calibration_text_from_info(camera_info)

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
        elif source_id.startswith("pointgrey:"):
            backend = "PointGrey Service"
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
