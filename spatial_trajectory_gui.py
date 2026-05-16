#!/usr/bin/env python3
"""Standalone GUI to send voxel trajectories to collect_viser control API."""

from __future__ import annotations

import argparse
import base64
import json
import math
import threading
import tkinter as tk
import time
import zlib
from pathlib import Path
from tkinter import messagebox, ttk
from urllib import error, request

import cv2
import numpy as np

from utils.spatial_language import build_composited_sentence


def decode_binary_mask(encoded: str) -> np.ndarray:
    """Decode GraspGPT-style base64+zlib+packbits mask into a 2D bool array."""
    compressed = base64.b64decode(encoded)
    packed = zlib.decompress(compressed)
    bits = np.unpackbits(np.frombuffer(packed, dtype=np.uint8))
    n_bits = bits.size
    side = int(math.isqrt(n_bits))
    if side * side != n_bits:
        raise ValueError(f"Cannot infer square mask side from {n_bits} bits.")
    return bits.reshape((side, side)).astype(bool)


def black_pixel_coords(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(~np.asarray(mask, dtype=bool))
    if len(rows) == 0:
        return np.zeros((0, 2), dtype=np.int32)
    coords = np.stack([cols, rows], axis=1).astype(np.int32)
    order = np.lexsort((coords[:, 1], coords[:, 0]))
    return coords[order]


class CollectViserTrajectoryGUI:
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

    def __init__(
        self,
        root: tk.Tk,
        *,
        host: str,
        port: int,
        poll_ms: int,
        dataset_json: str | None,
        episode_output_dir: str,
        dataset_output_dir: str,
        inference_server: str,
        inference_max_new_tokens: int,
        inference_temperature: float,
        inference_top_k: int | None,
        inference_do_sample: bool,
        inference_forbidden_tokens: str,
        inference_require_state_after_movement: bool,
        inference_state_after_movement_prob: float,
    ):
        self.root = root
        self.host = host
        self.port = int(port)
        self.poll_ms = int(max(poll_ms, 100))
        self.base_url = f"http://{host}:{port}"
        self.dataset_json = self._resolve_dataset_json_path(dataset_json)
        self.episode_output_dir = Path(episode_output_dir).expanduser().resolve()
        self.episode_output_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_output_dir = Path(dataset_output_dir).expanduser().resolve()
        self.dataset_output_dir.mkdir(parents=True, exist_ok=True)
        self.inference_server_var = tk.StringVar(value=str(inference_server).rstrip("/"))
        self.inference_max_new_tokens_var = tk.IntVar(value=int(inference_max_new_tokens))
        self.inference_temperature_var = tk.DoubleVar(value=float(inference_temperature))
        self.inference_top_k_var = tk.StringVar(
            value="" if inference_top_k is None else str(int(inference_top_k))
        )
        self.inference_do_sample_var = tk.BooleanVar(value=bool(inference_do_sample))
        self.inference_movement_only_var = tk.BooleanVar(value=True)
        self.inference_forbidden_tokens_var = tk.StringVar(value=str(inference_forbidden_tokens))
        self.inference_require_state_after_movement_var = tk.BooleanVar(
            value=bool(inference_require_state_after_movement)
        )
        self.inference_state_after_movement_prob_var = tk.DoubleVar(
            value=float(inference_state_after_movement_prob)
        )
        self.inference_status_text = tk.StringVar(value="Online inference: idle")
        self.inference_prompt_text = tk.StringVar(value="Prompt: waiting")
        self.inference_output_text = tk.StringVar(value="Output: waiting")
        self._online_inference_busy = False
        self._online_inference_prompt = ""
        self._online_inference_output = ""
        self._online_inference_raw_response = ""
        self._online_inference_used_histories: list[dict] = []
        self._online_inference_auto_send = False
        self._auto_online_enabled = False
        self._auto_online_job: str | None = None
        self._auto_online_button: ttk.Button | None = None
        self._spatial_history: list[dict] = []

        self.resolution_x = 64
        self.resolution_y = 64
        self.cell_size = 8
        self.padding = 20
        self.goal_mode = tk.BooleanVar(value=False)
        self.auto_refresh = tk.BooleanVar(value=True)
        self.auto_adjacent_waypoints = tk.BooleanVar(value=False)
        self.record_episode_rgb_var = tk.BooleanVar(value=False)
        self.speed_var = tk.IntVar(value=5)
        self.timesteps_var = tk.IntVar(value=15)
        self.settle_var = tk.DoubleVar(value=0.12)
        self.timeout_var = tk.DoubleVar(value=8.0)
        self.max_traj_step_distance = 8.0
        self.trajectory: list[tuple[int, int]] = []
        self._active_planned_trajectory: list[tuple[int, int]] = []
        self.goal: list[tuple[int, int]] = []
        self.current_pusher: tuple[int, int] | None = None
        self.pusher_coords: list[tuple[int, int]] = []
        self.tblock_coords: list[tuple[int, int]] = []
        self.tblock_coords_full: list[tuple[int, int]] = []
        self.tblock_apriltag_coords_2d: list[tuple[int, int]] = []
        self.tblock_apriltag_points_world: list[dict] = []
        self.status_text = tk.StringVar(value="Waiting for collect_viser...")
        self.run_text = tk.StringVar(value="Run: idle")
        self.pusher_text = tk.StringVar(value="Pusher: unavailable")
        self.episode_text = tk.StringVar(value="Episode: unknown")
        self.paths_text = tk.StringVar(value="")
        self._episode_movement_rows: list[dict] = []
        self._saved_episode_rows: list[dict] = []
        self._saved_replay_active = False
        self._saved_replay_frames: list[dict] = []
        self._saved_replay_index = 0
        self._saved_replay_job: str | None = None
        self._saved_replay_episode_path: Path | None = None
        self._saved_replay_paused = False
        self._saved_replay_delay_ms = 120
        self._pause_replay_button: ttk.Button | None = None
        self._last_popup_error = ""
        self._latest_spatial_config = {
            "bbox_min": [0.0, 0.0, 0.0],
            "bbox_max": [1.0, 1.0, 1.0],
            "resolution_xyz": [self.resolution_x, self.resolution_y, 1],
        }
        self._current_episode_id = self._new_local_episode_id()
        self._current_episode_started_at = time.time()
        self._current_episode_movements: list[dict] = []
        self._current_episode_rgb_frames: list[dict] = []
        self._latest_rgb_frame: np.ndarray | None = None
        self._latest_rgb_timestamp = 0.0
        self._run_busy = False

        self._build_ui()
        self._poll()

    def _build_ui(self) -> None:
        self.root.title("Spatial Trajectory Sender")
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        main.rowconfigure(1, weight=1)
        main.columnconfigure(0, weight=1)

        controls = ttk.Frame(main)
        controls.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        controls.columnconfigure(8, weight=1)

        ttk.Button(
            controls,
            text="Refresh Scene",
            command=lambda: self.fetch_scene(popup_on_error=True),
        ).grid(row=0, column=0, padx=2)
        ttk.Checkbutton(controls, text="Auto Refresh", variable=self.auto_refresh).grid(row=0, column=1, padx=2)
        ttk.Checkbutton(controls, text="Goal Mode", variable=self.goal_mode).grid(row=0, column=2, padx=2)
        ttk.Button(controls, text="Load Goal", command=self.load_goal_from_dataset).grid(row=0, column=3, padx=2)
        ttk.Button(controls, text="Clear Trajectory", command=self.clear_trajectory).grid(row=0, column=4, padx=2)
        ttk.Button(controls, text="Clear Goal", command=self.clear_goal).grid(row=0, column=5, padx=2)
        ttk.Button(controls, text="Send Trajectory", command=self.send_trajectory).grid(row=0, column=6, padx=2)
        ttk.Button(controls, text="Finish Episode", command=self.finish_episode).grid(row=0, column=7, padx=2)
        ttk.Button(controls, text="New Episode", command=self.start_new_episode).grid(row=0, column=8, padx=2)
        ttk.Label(controls, textvariable=self.run_text).grid(row=0, column=9, sticky="e")
        ttk.Label(controls, text="Speed").grid(row=1, column=0, padx=(2, 2), pady=(6, 0), sticky="w")
        ttk.Spinbox(
            controls,
            from_=1,
            to=100,
            increment=1,
            textvariable=self.speed_var,
            width=6,
        ).grid(row=1, column=1, padx=(0, 8), pady=(6, 0), sticky="w")
        ttk.Label(controls, text="Timesteps").grid(row=1, column=2, padx=(2, 2), pady=(6, 0), sticky="w")
        ttk.Spinbox(
            controls,
            from_=1,
            to=300,
            increment=1,
            textvariable=self.timesteps_var,
            width=6,
        ).grid(row=1, column=3, padx=(0, 8), pady=(6, 0), sticky="w")
        ttk.Label(controls, text="Settle (s)").grid(row=1, column=4, padx=(2, 2), pady=(6, 0), sticky="w")
        ttk.Spinbox(
            controls,
            from_=0.0,
            to=5.0,
            increment=0.01,
            textvariable=self.settle_var,
            width=7,
        ).grid(row=1, column=5, padx=(0, 8), pady=(6, 0), sticky="w")
        ttk.Label(controls, text="Timeout (s)").grid(row=1, column=6, padx=(2, 2), pady=(6, 0), sticky="w")
        ttk.Spinbox(
            controls,
            from_=0.1,
            to=30.0,
            increment=0.1,
            textvariable=self.timeout_var,
            width=7,
        ).grid(row=1, column=7, padx=(0, 8), pady=(6, 0), sticky="w")
        ttk.Checkbutton(
            controls,
            text="Adjacent on send",
            variable=self.auto_adjacent_waypoints,
        ).grid(row=1, column=8, columnspan=2, padx=2, pady=(6, 0), sticky="w")
        ttk.Checkbutton(
            controls,
            text="Record RGB per episode",
            variable=self.record_episode_rgb_var,
        ).grid(row=1, column=10, padx=2, pady=(6, 0), sticky="w")
        ttk.Label(controls, text="Inference").grid(row=2, column=0, padx=(2, 2), pady=(6, 0), sticky="w")
        ttk.Entry(controls, textvariable=self.inference_server_var, width=28).grid(
            row=2, column=1, columnspan=3, padx=(0, 8), pady=(6, 0), sticky="ew"
        )
        ttk.Label(controls, text="Tokens").grid(row=2, column=4, padx=(2, 2), pady=(6, 0), sticky="w")
        ttk.Spinbox(
            controls,
            from_=1,
            to=2048,
            increment=1,
            textvariable=self.inference_max_new_tokens_var,
            width=7,
        ).grid(row=2, column=5, padx=(0, 8), pady=(6, 0), sticky="w")
        ttk.Label(controls, text="Temp").grid(row=2, column=6, padx=(2, 2), pady=(6, 0), sticky="w")
        ttk.Spinbox(
            controls,
            from_=0.0,
            to=5.0,
            increment=0.05,
            textvariable=self.inference_temperature_var,
            width=7,
        ).grid(row=2, column=7, padx=(0, 8), pady=(6, 0), sticky="w")
        ttk.Label(controls, text="Top-k").grid(row=2, column=8, padx=(2, 2), pady=(6, 0), sticky="w")
        ttk.Entry(controls, textvariable=self.inference_top_k_var, width=7).grid(
            row=2, column=9, padx=(0, 8), pady=(6, 0), sticky="w"
        )
        ttk.Checkbutton(
            controls,
            text="Sample",
            variable=self.inference_do_sample_var,
        ).grid(row=3, column=1, padx=2, pady=(6, 0), sticky="w")
        ttk.Checkbutton(
            controls,
            text="Movement only",
            variable=self.inference_movement_only_var,
        ).grid(row=3, column=2, padx=2, pady=(6, 0), sticky="w")
        self._auto_online_button = ttk.Button(
            controls,
            text="Start Auto",
            command=self.toggle_auto_online_inference,
        )
        self._auto_online_button.grid(row=3, column=0, padx=2, pady=(6, 0), sticky="ew")
        ttk.Button(
            controls,
            text="Run Online Inference",
            command=self.run_online_inference,
        ).grid(row=3, column=3, columnspan=2, padx=2, pady=(6, 0), sticky="ew")
        ttk.Label(controls, textvariable=self.inference_status_text).grid(
            row=3, column=5, columnspan=5, padx=2, pady=(6, 0), sticky="w"
        )
        ttk.Label(controls, text="Forbidden").grid(row=4, column=0, padx=(2, 2), pady=(6, 0), sticky="w")
        ttk.Entry(controls, textvariable=self.inference_forbidden_tokens_var, width=28).grid(
            row=4, column=1, columnspan=3, padx=(0, 8), pady=(6, 0), sticky="ew"
        )
        ttk.Checkbutton(
            controls,
            text="Require state",
            variable=self.inference_require_state_after_movement_var,
        ).grid(row=4, column=4, padx=2, pady=(6, 0), sticky="w")
        ttk.Label(controls, text="State prob").grid(row=4, column=5, padx=(2, 2), pady=(6, 0), sticky="w")
        ttk.Spinbox(
            controls,
            from_=0.0,
            to=1.0,
            increment=0.05,
            textvariable=self.inference_state_after_movement_prob_var,
            width=7,
        ).grid(row=4, column=6, padx=(0, 8), pady=(6, 0), sticky="w")

        content = ttk.Frame(main)
        content.grid(row=1, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=0)
        content.rowconfigure(0, weight=1)

        canvas_w = self.padding * 2 + self.resolution_x * self.cell_size
        canvas_h = self.padding * 2 + self.resolution_y * self.cell_size
        self.canvas = tk.Canvas(content, width=canvas_w, height=canvas_h, bg="white", highlightthickness=1)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Button-1>", self.on_left_click)
        self.canvas.bind("<Button-3>", self.on_right_click)

        saved_frame = ttk.LabelFrame(content, text="Saved Episodes", padding=6)
        saved_frame.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        saved_frame.columnconfigure(0, weight=1)
        saved_frame.rowconfigure(0, weight=1)
        self._saved_episode_tree = ttk.Treeview(
            saved_frame,
            columns=("episode", "moves", "frames"),
            show="headings",
            height=20,
            selectmode="extended",
        )
        self._saved_episode_tree.heading("episode", text="Episode")
        self._saved_episode_tree.heading("moves", text="Moves")
        self._saved_episode_tree.heading("frames", text="Frames")
        self._saved_episode_tree.column("episode", width=220, anchor="w")
        self._saved_episode_tree.column("moves", width=60, anchor="center")
        self._saved_episode_tree.column("frames", width=60, anchor="center")
        saved_scroll = ttk.Scrollbar(
            saved_frame, orient="vertical", command=self._saved_episode_tree.yview
        )
        self._saved_episode_tree.configure(yscrollcommand=saved_scroll.set)
        self._saved_episode_tree.grid(row=0, column=0, sticky="nsew")
        saved_scroll.grid(row=0, column=1, sticky="ns")
        saved_btns = ttk.Frame(saved_frame)
        saved_btns.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        ttk.Button(saved_btns, text="Replay", command=self.replay_selected_episode).grid(row=0, column=0, padx=2)
        self._pause_replay_button = ttk.Button(
            saved_btns,
            text="Pause",
            command=self.toggle_saved_replay_pause,
        )
        self._pause_replay_button.grid(row=0, column=1, padx=2)
        ttk.Button(saved_btns, text="Stop Replay", command=self.stop_saved_replay).grid(row=0, column=2, padx=2)
        ttk.Button(saved_btns, text="Prev Frame", command=self.previous_saved_replay_frame).grid(
            row=1, column=0, padx=2, pady=(6, 0)
        )
        ttk.Button(saved_btns, text="Next Frame", command=self.next_saved_replay_frame).grid(
            row=1, column=1, padx=2, pady=(6, 0)
        )
        ttk.Button(saved_btns, text="Delete", command=self.delete_selected_episode).grid(
            row=1, column=2, padx=2, pady=(6, 0)
        )
        ttk.Button(saved_btns, text="Refresh", command=self.refresh_saved_episode_list).grid(
            row=1, column=3, padx=2, pady=(6, 0)
        )
        ttk.Button(saved_btns, text="Export Dataset", command=self.export_selected_episodes_dataset).grid(
            row=2, column=0, columnspan=4, padx=2, pady=(6, 0), sticky="ew"
        )

        status = ttk.Label(main, textvariable=self.status_text, anchor="w")
        status.grid(row=2, column=0, sticky="ew", pady=(8, 0))
        pusher_label = ttk.Label(main, textvariable=self.pusher_text, anchor="w")
        pusher_label.grid(row=3, column=0, sticky="ew", pady=(4, 0))
        episode_label = ttk.Label(main, textvariable=self.episode_text, anchor="w")
        episode_label.grid(row=4, column=0, sticky="ew", pady=(4, 0))
        paths_label = ttk.Label(main, textvariable=self.paths_text, anchor="w")
        paths_label.grid(row=6, column=0, sticky="ew", pady=(6, 0))
        movements_frame = ttk.LabelFrame(main, text="Current Episode Movements", padding=6)
        movements_frame.grid(row=5, column=0, sticky="nsew", pady=(8, 0))
        movements_frame.columnconfigure(0, weight=1)
        movements_frame.rowconfigure(0, weight=1)
        self._episode_tree = ttk.Treeview(
            movements_frame,
            columns=("idx", "run_id", "time", "steps", "frames", "goal"),
            show="headings",
            height=8,
        )
        self._episode_tree.heading("idx", text="#")
        self._episode_tree.heading("run_id", text="Run ID")
        self._episode_tree.heading("time", text="Time")
        self._episode_tree.heading("steps", text="Steps")
        self._episode_tree.heading("frames", text="Frames")
        self._episode_tree.heading("goal", text="Goal Cells")
        self._episode_tree.column("idx", width=36, anchor="center")
        self._episode_tree.column("run_id", width=190, anchor="w")
        self._episode_tree.column("time", width=80, anchor="center")
        self._episode_tree.column("steps", width=60, anchor="center")
        self._episode_tree.column("frames", width=60, anchor="center")
        self._episode_tree.column("goal", width=80, anchor="center")
        tree_scroll = ttk.Scrollbar(movements_frame, orient="vertical", command=self._episode_tree.yview)
        self._episode_tree.configure(yscrollcommand=tree_scroll.set)
        self._episode_tree.grid(row=0, column=0, sticky="nsew")
        tree_scroll.grid(row=0, column=1, sticky="ns")
        self.load_goal_from_dataset()
        self.refresh_saved_episode_list()
        self._refresh_local_episode_ui()
        self.paths_text.set(
            f"Episode Output: {self.episode_output_dir} | Dataset Output: {self.dataset_output_dir}"
        )
        self.redraw()

    def _show_error(self, message: str, *, popup: bool = True) -> None:
        self.status_text.set(str(message))
        if not popup:
            return
        text = str(message).strip()
        if not text:
            return
        if text == self._last_popup_error:
            return
        self._last_popup_error = text
        messagebox.showerror("Trajectory GUI Error", text)

    def _show_warning(self, message: str, *, popup: bool = True) -> None:
        self.status_text.set(str(message))
        if not popup:
            return
        text = str(message).strip()
        if not text:
            return
        if text == self._last_popup_error:
            return
        self._last_popup_error = text
        messagebox.showwarning("Trajectory GUI Warning", text)

    def _resolve_dataset_json_path(self, dataset_json: str | None) -> Path | None:
        if dataset_json:
            path = Path(dataset_json).expanduser().resolve()
            return path if path.is_file() else None
        dataset_dir = (Path.cwd() / "data" / "records" / "datasets").resolve()
        if not dataset_dir.is_dir():
            return None
        candidates = sorted(
            path
            for path in dataset_dir.glob("graspgpt_dataset_*.json")
            if path.is_file() and not path.name.endswith("_report.json")
        )
        if not candidates:
            return None
        return candidates[-1]

    def _new_local_episode_id(self) -> str:
        return time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1.0) * 1000):03d}"

    def _save_episode_payload(self, episode_payload: dict) -> Path:
        episode_id = str(episode_payload.get("episode_id", "")).strip()
        if not episode_id:
            raise ValueError("Missing episode_id in episode payload.")
        output_path = self.episode_output_dir / f"spatial_episode_{episode_id}.json"
        output_path.write_text(json.dumps(episode_payload, indent=2), encoding="utf-8")
        return output_path

    def _saved_episode_summary(self, path: Path) -> dict | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        episode_id = str(payload.get("episode_id", path.stem.replace("spatial_episode_", "")))
        movement_count = int(payload.get("movement_count", len(payload.get("movements", []))))
        frame_count = int(payload.get("frame_count", 0))
        return {
            "path": path,
            "episode_id": episode_id,
            "movement_count": movement_count,
            "frame_count": frame_count,
        }

    def refresh_saved_episode_list(self) -> None:
        paths = sorted(
            self.episode_output_dir.glob("spatial_episode_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        rows: list[dict] = []
        for path in paths:
            summary = self._saved_episode_summary(path)
            if summary is not None:
                rows.append(summary)
        self._saved_episode_rows = rows
        for item_id in self._saved_episode_tree.get_children():
            self._saved_episode_tree.delete(item_id)
        for row in rows:
            iid = str(row["path"])
            self._saved_episode_tree.insert(
                "",
                "end",
                iid=iid,
                values=(row["episode_id"], row["movement_count"], row["frame_count"]),
            )

    def _selected_saved_episode_path(self) -> Path | None:
        selected = self._saved_episode_tree.selection()
        if not selected:
            return None
        return Path(selected[0])

    def _selected_saved_episode_paths(self) -> list[Path]:
        selected = list(self._saved_episode_tree.selection())
        return [Path(item) for item in selected]

    def _encode_binary_mask(self, mask: np.ndarray) -> str:
        mask = np.asarray(mask, dtype=bool)
        packed = np.packbits(mask.astype(np.uint8).reshape(-1))
        compressed = zlib.compress(packed.tobytes(), level=9)
        return base64.b64encode(compressed).decode("ascii")

    def _coords_to_encoded_mask(
        self,
        coords: list[tuple[int, int]] | np.ndarray,
        *,
        side: int,
    ) -> str:
        side = int(max(1, side))
        mask = np.ones((side, side), dtype=bool)
        coords_arr = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
        for x_idx, y_idx in coords_arr.tolist():
            if 0 <= int(x_idx) < side and 0 <= int(y_idx) < side:
                mask[int(y_idx), int(x_idx)] = False
        return self._encode_binary_mask(mask)

    def _representative_action_coord(self, coords: list[tuple[int, int]]) -> tuple[int, int]:
        coords_arr = np.asarray(coords, dtype=np.int32).reshape(-1, 2)
        if len(coords_arr) == 0:
            return 0, 0
        mean_xy = np.mean(coords_arr, axis=0)
        distances = np.sum((coords_arr - mean_xy[None, :]) ** 2, axis=1)
        idx = int(np.argmin(distances))
        coord = coords_arr[idx]
        return int(coord[0]), int(coord[1])

    def _frame_missing_object_parts(self, spatial: dict) -> tuple[list[str], list[tuple[int, int]], list[tuple[int, int]]]:
        pusher_coords = self._normalize_coord_list(spatial.get("pusher_coords", []))
        tbar_coords = self._normalize_coord_list(spatial.get("tblock_coords", []))
        missing_parts: list[str] = []
        if len(pusher_coords) == 0:
            missing_parts.append("pusher")
        if len(tbar_coords) == 0:
            missing_parts.append("tblock")
        return missing_parts, pusher_coords, tbar_coords

    def _episode_payload_to_dataset_trajectory(
        self,
        payload: dict,
        *,
        side: int,
    ) -> tuple[dict | None, list[str]]:
        source_episode = str(payload.get("episode_id", "(unknown)"))
        movements = payload.get("movements", [])
        warnings: list[str] = []
        if len(movements) == 0:
            warnings.append(f"{source_episode}: no movements found.")
            return None, warnings
        frames_payload = []
        goal_coords: list[tuple[int, int]] = []
        last_tbar_coords: list[tuple[int, int]] = []

        for movement_idx, movement in enumerate(movements):
            movement_goal = movement.get("goal", [])
            if movement_goal:
                goal_coords = [tuple(map(int, coord[:2])) for coord in movement_goal]

            movement_frames = movement.get("frames", [])
            if len(movement_frames) == 0:
                warnings.append(
                    f"{source_episode}: movement {movement_idx + 1} has no frames."
                )
                continue

            for frame_idx, frame in enumerate(movement_frames):
                spatial = frame.get("spatial", {})
                missing_parts, pusher_coords, tbar_coords = self._frame_missing_object_parts(spatial)
                if missing_parts:
                    warnings.append(
                        f"{source_episode}: empty frame at movement {movement_idx + 1}, "
                        f"frame {frame_idx + 1}; missing {', '.join(missing_parts)}."
                    )
                    continue
                last_tbar_coords = list(tbar_coords)
                target_coord = frame.get("target_coord", [])
                if isinstance(target_coord, list) and len(target_coord) >= 2:
                    action = [int(target_coord[0]), int(target_coord[1])]
                else:
                    action_xy = self._representative_action_coord(pusher_coords)
                    action = [int(action_xy[0]), int(action_xy[1])]
                frames_payload.append(
                    {
                        "action": action,
                        "pusher": self._coords_to_encoded_mask(pusher_coords, side=side),
                        "tbar": self._coords_to_encoded_mask(tbar_coords, side=side),
                    }
                )
                if not goal_coords:
                    spatial_goal = spatial.get("goal_coords", [])
                    if spatial_goal:
                        goal_coords = [tuple(map(int, coord[:2])) for coord in spatial_goal]

        if len(frames_payload) == 0:
            warnings.append(f"{source_episode}: no valid frames found.")
            return None, warnings
        if not goal_coords:
            goal_coords = list(last_tbar_coords)
        if not goal_coords:
            warnings.append(f"{source_episode}: goal is empty.")
            return None, warnings

        return (
            {
                "source_episode": source_episode,
                "goal": self._coords_to_encoded_mask(goal_coords, side=side),
                "frames": frames_payload,
            },
            warnings,
        )

    def export_selected_episodes_dataset(self) -> None:
        selected_paths = self._selected_saved_episode_paths()
        if len(selected_paths) == 0:
            self.status_text.set("Select one or more saved episodes to export.")
            return

        payloads: list[dict] = []
        max_side = 4
        for path in selected_paths:
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._show_error(f"Failed to read {path.name}: {exc}")
                return
            payloads.append(payload)
            spatial_cfg = payload.get("spatial_config", {})
            resolution_xyz = spatial_cfg.get("resolution_xyz", [64, 64, 1])
            if isinstance(resolution_xyz, list) and len(resolution_xyz) >= 2:
                max_side = max(max_side, int(resolution_xyz[0]), int(resolution_xyz[1]))

        side = int(math.ceil(max_side / 4.0) * 4)
        trajectories: list[dict] = []
        warning_lines: list[str] = []
        for path, payload in zip(selected_paths, payloads, strict=False):
            trajectory, warnings = self._episode_payload_to_dataset_trajectory(payload, side=side)
            warning_lines.extend(warnings)
            if trajectory is not None:
                trajectories.append(trajectory)
        empty_frame_warnings = [
            line for line in warning_lines
            if "empty frame" in line.lower()
        ]
        if empty_frame_warnings:
            preview = "\n".join(empty_frame_warnings[:12])
            if len(empty_frame_warnings) > 12:
                preview += f"\n... and {len(empty_frame_warnings) - 12} more empty frame errors."
            self._show_error(
                "Dataset export stopped because some frames are missing pusher or tblock data.\n\n"
                f"{preview}"
            )
            return

        if len(trajectories) == 0:
            message = "No valid trajectories found in selected episodes."
            if warning_lines:
                preview = "\n".join(warning_lines[:8])
                if len(warning_lines) > 8:
                    preview += f"\n... and {len(warning_lines) - 8} more warnings."
                message = f"{message}\n\nWarnings:\n{preview}"
            self._show_warning(message)
            return

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_path = self.dataset_output_dir / f"graspgpt_dataset_{timestamp}.json"
        merged_payload = {
            "format": "graspgpt_pusht_merged_v1",
            "source": "robodata_Agilex_trajectory_gui",
            "selected_episodes": [path.name for path in selected_paths],
            "trajectories": trajectories,
        }
        try:
            output_path.write_text(json.dumps(merged_payload, indent=2), encoding="utf-8")
        except Exception as exc:
            self._show_error(f"Failed to write dataset: {exc}")
            return
        success_message = (
            f"Dataset exported: {output_path} ({len(trajectories)} trajectories). "
            "Empty-frame check passed."
        )
        if warning_lines:
            preview_lines = []
            if empty_frame_warnings:
                preview_lines.append("Empty frames:")
                preview_lines.extend(empty_frame_warnings[:8])
                remaining_empty = len(empty_frame_warnings) - min(len(empty_frame_warnings), 8)
                if remaining_empty > 0:
                    preview_lines.append(f"... and {remaining_empty} more empty frame warnings.")
                other_warnings = [
                    line for line in warning_lines
                    if line not in empty_frame_warnings
                ]
                if other_warnings:
                    preview_lines.append("")
                    preview_lines.append("Other warnings:")
                    preview_lines.extend(other_warnings[:4])
            else:
                preview_lines.extend(warning_lines[:8])
            if len(preview_lines) == 0:
                preview_lines = warning_lines[:8]
            preview = "\n".join(preview_lines)
            remaining_total = len(warning_lines) - sum(
                1 for line in warning_lines
                if line in preview_lines
            )
            if remaining_total > 0 and not empty_frame_warnings:
                preview += f"\n... and {remaining_total} more warnings."
            self._show_warning(f"{success_message}\n\nExport warnings:\n{preview}")
        else:
            self.status_text.set(success_message)

    def _extract_replay_frames(self, payload: dict) -> list[dict]:
        replay_frames: list[dict] = []
        movements = payload.get("movements", [])
        for movement_idx, movement in enumerate(movements):
            frames = movement.get("frames", [])
            for frame_idx, frame in enumerate(frames):
                spatial = frame.get("spatial", {})
                replay_frames.append(
                    {
                        "movement_idx": int(movement_idx),
                        "frame_idx": int(frame_idx),
                        "spatial": spatial,
                    }
                )
        return replay_frames

    def _render_replay_frame(self, replay_frame: dict) -> None:
        spatial = dict(replay_frame.get("spatial", {}))
        resolution = spatial.get("resolution_xyz", [self.resolution_x, self.resolution_y, 1])
        rx = max(1, int(resolution[0]))
        ry = max(1, int(resolution[1]))
        if rx != self.resolution_x or ry != self.resolution_y:
            self.resolution_x = rx
            self.resolution_y = ry
            self._resize_canvas()

        self.tblock_coords = [tuple(coord) for coord in spatial.get("tblock_coords", [])]
        self.tblock_coords_full = [tuple(coord) for coord in spatial.get("tblock_coords_full", [])]
        self.tblock_apriltag_coords_2d = [tuple(coord) for coord in spatial.get("tblock_apriltag_coords_2d", [])]
        self.tblock_apriltag_points_world = [
            dict(item)
            for item in spatial.get("tblock_apriltag_points_world", [])
            if isinstance(item, dict)
        ]
        pusher_coords = [tuple(coord) for coord in spatial.get("pusher_coords", [])]
        self.pusher_coords = pusher_coords
        self.current_pusher = pusher_coords[0] if pusher_coords else None
        self._update_pusher_text()
        self.goal = [tuple(coord) for coord in spatial.get("goal_coords", [])]
        self.trajectory = [tuple(coord) for coord in spatial.get("movement_coords", [])]
        self.redraw()

    def _set_saved_replay_paused(self, paused: bool) -> None:
        self._saved_replay_paused = paused
        if self._pause_replay_button is not None:
            self._pause_replay_button.configure(text="Resume" if paused else "Pause")

    def _cancel_saved_replay_job(self) -> None:
        if self._saved_replay_job is not None:
            self.root.after_cancel(self._saved_replay_job)
            self._saved_replay_job = None

    def _saved_replay_status(self, prefix: str) -> str:
        episode_name = self._saved_replay_episode_path.name if self._saved_replay_episode_path else ""
        total = len(self._saved_replay_frames)
        return f"{prefix} {episode_name} | frame {self._saved_replay_index + 1}/{total}"

    def _render_current_saved_replay_frame(self, prefix: str) -> None:
        if len(self._saved_replay_frames) == 0:
            return
        self._saved_replay_index = max(
            0,
            min(self._saved_replay_index, len(self._saved_replay_frames) - 1),
        )
        self._render_replay_frame(self._saved_replay_frames[self._saved_replay_index])
        self.status_text.set(self._saved_replay_status(prefix))

    def _load_saved_replay_episode(self, episode_path: Path) -> bool:
        try:
            payload = json.loads(episode_path.read_text(encoding="utf-8"))
            replay_frames = self._extract_replay_frames(payload)
        except Exception as exc:
            self._show_error(f"Failed to load replay episode: {exc}")
            return False
        if len(replay_frames) == 0:
            self.status_text.set("Selected episode has no frames to replay.")
            return False
        self._cancel_saved_replay_job()
        self._saved_replay_active = True
        self._saved_replay_frames = replay_frames
        self._saved_replay_index = 0
        self._saved_replay_episode_path = episode_path
        return True

    def _ensure_saved_replay_loaded(self) -> bool:
        if self._saved_replay_active and len(self._saved_replay_frames) > 0:
            return True
        episode_path = self._selected_saved_episode_path()
        if episode_path is None:
            self.status_text.set("Select a saved episode to replay.")
            return False
        return self._load_saved_replay_episode(episode_path)

    def replay_selected_episode(self) -> None:
        if self._auto_online_enabled:
            self._set_auto_online_enabled(False)
        episode_path = self._selected_saved_episode_path()
        if episode_path is None:
            self.status_text.set("Select a saved episode to replay.")
            return
        self.stop_saved_replay(refresh_scene=False)
        if not self._load_saved_replay_episode(episode_path):
            return
        self._set_saved_replay_paused(False)
        self._render_current_saved_replay_frame("Replaying")
        self._schedule_saved_replay_tick()

    def _schedule_saved_replay_tick(self) -> None:
        if self._saved_replay_paused:
            return
        self._cancel_saved_replay_job()
        self._saved_replay_job = self.root.after(
            self._saved_replay_delay_ms,
            self._saved_replay_tick,
        )

    def _saved_replay_tick(self) -> None:
        self._saved_replay_job = None
        if not self._saved_replay_active or self._saved_replay_paused:
            return
        self._saved_replay_index += 1
        if self._saved_replay_index >= len(self._saved_replay_frames):
            self.stop_saved_replay()
            self.status_text.set("Replay finished.")
            return
        self._render_current_saved_replay_frame("Replaying")
        self._schedule_saved_replay_tick()

    def toggle_saved_replay_pause(self) -> None:
        was_loaded = self._saved_replay_active and len(self._saved_replay_frames) > 0
        if not self._ensure_saved_replay_loaded():
            return
        if not was_loaded:
            self._render_current_saved_replay_frame("Replay paused")
        if self._saved_replay_paused:
            self._set_saved_replay_paused(False)
            self.status_text.set(self._saved_replay_status("Replaying"))
            self._schedule_saved_replay_tick()
            return
        self._cancel_saved_replay_job()
        self._set_saved_replay_paused(True)
        self.status_text.set(self._saved_replay_status("Replay paused"))

    def previous_saved_replay_frame(self) -> None:
        if not self._ensure_saved_replay_loaded():
            return
        self._cancel_saved_replay_job()
        self._set_saved_replay_paused(True)
        self._saved_replay_index = max(0, self._saved_replay_index - 1)
        self._render_current_saved_replay_frame("Replay paused")

    def next_saved_replay_frame(self) -> None:
        if not self._ensure_saved_replay_loaded():
            return
        self._cancel_saved_replay_job()
        self._set_saved_replay_paused(True)
        self._saved_replay_index = min(
            len(self._saved_replay_frames) - 1,
            self._saved_replay_index + 1,
        )
        self._render_current_saved_replay_frame("Replay paused")

    def stop_saved_replay(self, refresh_scene: bool = True) -> None:
        self._saved_replay_active = False
        self._saved_replay_frames = []
        self._saved_replay_index = 0
        self._saved_replay_episode_path = None
        self._cancel_saved_replay_job()
        self._set_saved_replay_paused(False)
        if refresh_scene:
            self.fetch_scene()

    def delete_selected_episode(self) -> None:
        episode_path = self._selected_saved_episode_path()
        if episode_path is None:
            self.status_text.set("Select a saved episode to delete.")
            return
        confirm = messagebox.askyesno(
            title="Delete Episode",
            message=f"Delete {episode_path.name}?",
        )
        if not confirm:
            return
        try:
            if (
                self._saved_replay_episode_path is not None
                and episode_path.resolve() == self._saved_replay_episode_path.resolve()
            ):
                self.stop_saved_replay()
            episode_path.unlink()
            video_path = episode_path.with_suffix(".mp4")
            if video_path.is_file():
                video_path.unlink()
            self.refresh_saved_episode_list()
            self.status_text.set(f"Deleted episode: {episode_path.name}")
        except Exception as exc:
            self._show_error(f"Failed to delete episode: {exc}")

    def _format_movement_time(self, timestamp_value) -> str:
        try:
            timestamp = float(timestamp_value)
        except (TypeError, ValueError):
            return "--:--:--"
        if timestamp <= 0.0:
            return "--:--:--"
        return time.strftime("%H:%M:%S", time.localtime(timestamp))

    def _local_episode_movement_rows(self) -> list[dict]:
        rows: list[dict] = []
        for idx, movement in enumerate(self._current_episode_movements):
            rows.append(
                {
                    "index": int(idx),
                    "run_id": str(movement.get("run_id", "")),
                    "timestamp": movement.get("timestamp"),
                    "trajectory_steps": int(len(movement.get("trajectory", []))),
                    "frame_count": int(len(movement.get("frames", []))),
                    "goal_cells": int(len(movement.get("goal", []))),
                }
            )
        return rows

    def _refresh_local_episode_ui(self) -> None:
        rows = self._local_episode_movement_rows()
        frame_count = int(sum(len(m.get("frames", [])) for m in self._current_episode_movements))
        self.episode_text.set(
            f"Episode: {self._current_episode_id} | movements={len(rows)} | frames={frame_count}"
        )
        self._update_episode_movement_table(rows)

    def _update_episode_movement_table(self, movement_rows: list[dict]) -> None:
        if movement_rows == self._episode_movement_rows:
            return
        self._episode_movement_rows = [dict(row) for row in movement_rows]
        for item_id in self._episode_tree.get_children():
            self._episode_tree.delete(item_id)
        for row in movement_rows:
            idx = int(row.get("index", 0))
            run_id = str(row.get("run_id", ""))
            timestamp_txt = self._format_movement_time(row.get("timestamp"))
            steps = int(row.get("trajectory_steps", 0))
            frames = int(row.get("frame_count", 0))
            goal_cells = int(row.get("goal_cells", 0))
            self._episode_tree.insert(
                "",
                "end",
                values=(idx + 1, run_id, timestamp_txt, steps, frames, goal_cells),
            )

    def load_goal_from_dataset(self) -> None:
        if self.dataset_json is None:
            self.status_text.set("No dataset JSON found for goal loading.")
            return
        try:
            payload = json.loads(self.dataset_json.read_text(encoding="utf-8"))
            trajectories = payload.get("trajectories", [])
            if not trajectories:
                raise ValueError("No trajectories in dataset.")
            goal_encoded = trajectories[0].get("goal")
            if not isinstance(goal_encoded, str) or not goal_encoded:
                raise ValueError("First trajectory has no goal mask.")
            goal_mask = decode_binary_mask(goal_encoded)
            goal_coords = black_pixel_coords(goal_mask)
            if len(goal_coords) == 0:
                raise ValueError("Goal mask decoded to zero coordinates.")
            self.goal = [tuple(coord) for coord in goal_coords.tolist()]
            side = int(goal_mask.shape[0])
            resized = (side != self.resolution_x) or (side != self.resolution_y)
            self.resolution_x = side
            self.resolution_y = side
            if resized:
                self._resize_canvas()
            self.status_text.set(
                f"Goal loaded from {self.dataset_json.name}: {len(self.goal)} cells"
            )
            self.redraw()
        except Exception as exc:
            self._show_error(f"Failed to load goal from dataset: {exc}")

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _http_get(self, path: str) -> dict:
        req = request.Request(self._url(path), method="GET")
        with request.urlopen(req, timeout=2.0) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _http_post(self, path: str, payload: dict, *, timeout: float = 5.0) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._url(path),
            data=body,
            method="POST",
            headers={"Content-Type": "application/json; charset=utf-8"},
        )
        with request.urlopen(req, timeout=float(timeout)) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _fetch_rgb_snapshot(self) -> tuple[np.ndarray, float]:
        req = request.Request(self._url("/camera/rgb.jpg"), method="GET")
        with request.urlopen(req, timeout=5.0) as resp:
            encoded = resp.read()
            timestamp = float(resp.headers.get("X-Timestamp", time.time()))
        bgr = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None or bgr.size == 0:
            raise RuntimeError("Failed to decode RGB snapshot.")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), timestamp

    def _refresh_cached_rgb_snapshot(self) -> None:
        frame, timestamp = self._fetch_rgb_snapshot()
        self._latest_rgb_frame = np.array(frame, copy=True)
        self._latest_rgb_timestamp = float(timestamp)

    def _cached_rgb_snapshot(self) -> tuple[np.ndarray, float]:
        if self._latest_rgb_frame is None or self._latest_rgb_frame.size == 0:
            raise RuntimeError("No cached RGB snapshot available.")
        return np.array(self._latest_rgb_frame, copy=True), float(self._latest_rgb_timestamp)

    def _episode_rgb_video_path(self, episode_id: str) -> Path:
        return self.episode_output_dir / f"spatial_episode_{episode_id}.mp4"

    def _episode_rgb_fps(self) -> float:
        timestamps = [
            float(item.get("timestamp", 0.0))
            for item in self._current_episode_rgb_frames
            if float(item.get("timestamp", 0.0)) > 0.0
        ]
        if len(timestamps) < 2:
            return 2.0
        deltas = [
            max(1e-3, float(curr - prev))
            for prev, curr in zip(timestamps[:-1], timestamps[1:], strict=False)
            if float(curr) > float(prev)
        ]
        if not deltas:
            return 2.0
        return float(np.clip(1.0 / float(np.median(deltas)), 1.0, 10.0))

    def _write_episode_rgb_video(self, episode_id: str) -> Path:
        if len(self._current_episode_rgb_frames) == 0:
            raise ValueError("Current episode has no RGB snapshots to encode.")
        first_frame = np.asarray(self._current_episode_rgb_frames[0]["frame"], dtype=np.uint8)
        if first_frame.ndim != 3 or first_frame.shape[2] != 3:
            raise RuntimeError("RGB snapshot has invalid shape for MP4 export.")
        height, width = int(first_frame.shape[0]), int(first_frame.shape[1])
        writer = cv2.VideoWriter(
            str(self._episode_rgb_video_path(episode_id)),
            cv2.VideoWriter_fourcc(*"mp4v"),
            self._episode_rgb_fps(),
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for episode {episode_id}.")

        try:
            for item in self._current_episode_rgb_frames:
                frame = np.asarray(item["frame"], dtype=np.uint8)
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()
        return self._episode_rgb_video_path(episode_id)

    def _normalize_coord_list(self, coords) -> list[tuple[int, int]]:
        normalized: list[tuple[int, int]] = []
        if coords is None:
            return normalized
        for item in coords:
            if item is None:
                continue
            try:
                normalized.append((int(item[0]), int(item[1])))
            except (TypeError, ValueError, IndexError):
                continue
        return normalized

    def _spatial_has_required_objects(self, spatial: dict) -> bool:
        return (
            len(spatial.get("pusher_coords", [])) > 0
            and len(spatial.get("tblock_coords", [])) > 0
        )

    def _normalize_apriltag_points_world(self, points) -> list[dict]:
        normalized_points_world: list[dict] = []
        if not isinstance(points, list):
            return normalized_points_world
        for item in points:
            if not isinstance(item, dict):
                continue
            try:
                tag_id = int(item.get("tag_id"))
                corner_idx = int(item.get("corner_idx"))
                xyz = np.asarray(item.get("xyz_m"), dtype=np.float64).reshape(3)
            except (TypeError, ValueError):
                continue
            coord_xy = self._normalize_coord_list([item.get("coord_xy")])
            normalized_points_world.append(
                {
                    "tag_id": tag_id,
                    "corner_idx": corner_idx,
                    "xyz_m": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                    "coord_xy": (
                        None
                        if not coord_xy
                        else [int(coord_xy[0][0]), int(coord_xy[0][1])]
                    ),
                }
            )
        return normalized_points_world

    def _spatial_has_required_apriltag_points(self, spatial: dict) -> bool:
        return len(self._normalize_apriltag_points_world(spatial.get("tblock_apriltag_points_world", []))) > 0

    def _enrich_spatial_payload(
        self,
        spatial: dict,
        *,
        goal_coords: list[tuple[int, int]],
        movement_coords: list[tuple[int, int]],
    ) -> dict:
        enriched = dict(spatial)
        pusher_coords = self._normalize_coord_list(enriched.get("pusher_coords", []))
        tblock_coords = self._normalize_coord_list(enriched.get("tblock_coords", []))
        tblock_apriltag_coords_2d = self._normalize_coord_list(
            enriched.get("tblock_apriltag_coords_2d", [])
        )
        goal_norm = self._normalize_coord_list(goal_coords)
        movement_norm = self._normalize_coord_list(movement_coords)
        enriched["pusher_coords"] = [[x_idx, y_idx] for x_idx, y_idx in pusher_coords]
        enriched["tblock_coords"] = [[x_idx, y_idx] for x_idx, y_idx in tblock_coords]
        enriched["tblock_apriltag_coords_2d"] = [
            [x_idx, y_idx] for x_idx, y_idx in tblock_apriltag_coords_2d
        ]
        enriched["tblock_apriltag_points_world"] = self._normalize_apriltag_points_world(
            enriched.get("tblock_apriltag_points_world", [])
        )
        enriched["goal_coords"] = [[x_idx, y_idx] for x_idx, y_idx in goal_norm]
        enriched["movement_coords"] = [[x_idx, y_idx] for x_idx, y_idx in movement_norm]
        enriched["sentence"] = build_composited_sentence(
            pusher_coords=pusher_coords,
            tbar_coords=tblock_coords,
            goal_coords=goal_norm,
            movement_coords=movement_norm,
        )
        return enriched

    def _spatial_result_equivalent(self, left: dict, right: dict) -> bool:
        return (
            self._normalize_coord_list(left.get("pusher_coords", []))
            == self._normalize_coord_list(right.get("pusher_coords", []))
            and self._normalize_coord_list(left.get("tblock_coords", []))
            == self._normalize_coord_list(right.get("tblock_coords", []))
        )

    def _spatial_result_same_pusher_position(self, left: dict, right: dict) -> bool:
        left_coord = self._representative_pusher_coord(left)
        right_coord = self._representative_pusher_coord(right)
        return left_coord is not None and left_coord == right_coord

    def _record_spatial_history(self, spatial: dict) -> None:
        snapshot = {
            "pusher_coords": [
                [int(x_idx), int(y_idx)]
                for x_idx, y_idx in self._normalize_coord_list(spatial.get("pusher_coords", []))
            ],
            "tblock_coords": [
                [int(x_idx), int(y_idx)]
                for x_idx, y_idx in self._normalize_coord_list(spatial.get("tblock_coords", []))
            ],
            "tblock_apriltag_coords_2d": [
                [int(x_idx), int(y_idx)]
                for x_idx, y_idx in self._normalize_coord_list(
                    spatial.get("tblock_apriltag_coords_2d", [])
                )
            ],
            "tblock_apriltag_points_world": [
                {
                    "tag_id": int(item.get("tag_id")),
                    "corner_idx": int(item.get("corner_idx")),
                    "xyz_m": [
                        float(np.asarray(item.get("xyz_m"), dtype=np.float64).reshape(3)[0]),
                        float(np.asarray(item.get("xyz_m"), dtype=np.float64).reshape(3)[1]),
                        float(np.asarray(item.get("xyz_m"), dtype=np.float64).reshape(3)[2]),
                    ],
                    "coord_xy": (
                        None
                        if not self._normalize_coord_list([item.get("coord_xy")])
                        else [
                            int(self._normalize_coord_list([item.get("coord_xy")])[0][0]),
                            int(self._normalize_coord_list([item.get("coord_xy")])[0][1]),
                        ]
                    ),
                }
                for item in spatial.get("tblock_apriltag_points_world", [])
                if isinstance(item, dict)
            ],
            "goal_coords": [
                [int(x_idx), int(y_idx)]
                for x_idx, y_idx in self._normalize_coord_list(spatial.get("goal_coords", []))
            ],
            "resolution_xyz": list(spatial.get("resolution_xyz", [self.resolution_x, self.resolution_y, 1])),
        }
        pusher_coord = self._representative_pusher_coord(snapshot)
        if pusher_coord is None:
            return
        snapshot["pusher_coord"] = [int(pusher_coord[0]), int(pusher_coord[1])]
        if self._spatial_history and self._spatial_result_same_pusher_position(
            self._spatial_history[-1],
            snapshot,
        ):
            return
        self._spatial_history.append(snapshot)
        self._spatial_history = self._spatial_history[-30:]
        self._update_pusher_text()

    def _representative_pusher_coord(self, spatial: dict) -> tuple[int, int] | None:
        pusher_coords = self._normalize_coord_list(spatial.get("pusher_coords", []))
        if len(pusher_coords) == 0:
            return None
        return self._representative_action_coord(pusher_coords)

    def _update_pusher_text(self) -> None:
        if len(self.pusher_coords) == 0:
            current_text = "Pusher: unavailable"
        else:
            x_idx, y_idx = self._representative_action_coord(self.pusher_coords)
            current_text = f"Pusher: ({x_idx}, {y_idx})"

        history_coords: list[tuple[int, int]] = []
        for spatial in self._spatial_history[-3:]:
            coord = self._representative_pusher_coord(spatial)
            if coord is not None:
                history_coords.append(coord)
        if not history_coords:
            self.pusher_text.set(f"{current_text} | Movement history: none")
            return

        history_text = " -> ".join(f"({x_idx}, {y_idx})" for x_idx, y_idx in history_coords)
        self.pusher_text.set(f"{current_text} | Movement history: {history_text}")

    def _movement_coords_from_spatial_results(self, spatial_results: list[dict]) -> list[tuple[int, int]]:
        coords: list[tuple[int, int]] = []
        for spatial in spatial_results:
            coord = self._representative_pusher_coord(spatial)
            if coord is None:
                continue
            if coords and coord == coords[-1]:
                continue
            coords.append(coord)
        return coords

    def _compress_spatial_results_by_pusher_runs(self, spatial_results: list[dict]) -> list[dict]:
        compressed: list[dict] = []
        last_coord: tuple[int, int] | None = None
        for spatial in spatial_results:
            if not self._spatial_has_required_objects(spatial):
                continue
            coord = self._representative_pusher_coord(spatial)
            if coord is None:
                continue
            if compressed and coord == last_coord:
                compressed[-1] = spatial
            else:
                compressed.append(spatial)
                last_coord = coord
        return compressed

    def _current_spatial_snapshot(self) -> dict:
        return {
            "pusher_coords": [[int(x_idx), int(y_idx)] for x_idx, y_idx in self.pusher_coords],
            "tblock_coords": [[int(x_idx), int(y_idx)] for x_idx, y_idx in self.tblock_coords],
            "tblock_apriltag_coords_2d": [
                [int(x_idx), int(y_idx)] for x_idx, y_idx in self.tblock_apriltag_coords_2d
            ],
            "tblock_apriltag_points_world": [
                dict(item) for item in self.tblock_apriltag_points_world if isinstance(item, dict)
            ],
            "goal_coords": [[int(x_idx), int(y_idx)] for x_idx, y_idx in self.goal],
            "movement_coords": [[int(x_idx), int(y_idx)] for x_idx, y_idx in self.trajectory],
            "resolution_xyz": [
                int(self.resolution_x),
                int(self.resolution_y),
                int(self._latest_spatial_config.get("resolution_xyz", [1, 1, 1])[2]),
            ],
        }

    def _build_online_inference_prompt_from_spatial(
        self,
        spatial: dict,
        *,
        movement_coords: list[tuple[int, int]] | tuple[tuple[int, int], ...] = (),
    ) -> str:
        prompt_text = build_composited_sentence(
            pusher_coords=self._normalize_coord_list(spatial.get("pusher_coords", [])),
            tbar_coords=self._normalize_coord_list(spatial.get("tblock_coords", [])),
            goal_coords=self._normalize_coord_list(self.goal),
            movement_coords=list(movement_coords),
        )
        if not prompt_text:
            raise RuntimeError("Unable to compose a grammar-compliant prompt.")
        return prompt_text

    def _build_temporal_online_inference_prompt_from_results(
        self,
        spatial_results: list[dict],
    ) -> str:
        if not spatial_results:
            raise RuntimeError("No spatial results available for online inference.")
        available_results = [
            spatial for spatial in spatial_results
            if self._spatial_has_required_objects(spatial)
        ]
        if not available_results:
            raise RuntimeError("Current scene is missing pusher or T-block cells.")

        recent_results = available_results[-3:]
        state_spatial = recent_results[0]
        moveto_coords = self._movement_coords_from_spatial_results(
            recent_results[1:]
        )
        return self._build_online_inference_prompt_from_spatial(
            state_spatial,
            movement_coords=moveto_coords,
        )

    def _build_online_inference_prompt(self) -> str:
        if len(self.goal) == 0:
            self.load_goal_from_dataset()
        if len(self.goal) == 0:
            raise RuntimeError("Goal unavailable. Load or draw a goal before inference.")

        current_spatial = self._current_spatial_snapshot()
        if not self._spatial_has_required_objects(current_spatial):
            raise RuntimeError("Current scene is missing pusher or T-block cells.")
        recent_results = list(self._spatial_history)
        if not recent_results or not self._spatial_result_same_pusher_position(
            recent_results[-1],
            current_spatial,
        ):
            recent_results.append(current_spatial)
        recent_results = recent_results[-3:]
        self._online_inference_used_histories = [
            {
                "pusher_coords": [
                    [int(x_idx), int(y_idx)]
                    for x_idx, y_idx in self._normalize_coord_list(spatial.get("pusher_coords", []))
                ],
                "tblock_coords": [
                    [int(x_idx), int(y_idx)]
                    for x_idx, y_idx in self._normalize_coord_list(spatial.get("tblock_coords", []))
                ],
            }
            for spatial in recent_results
            if self._spatial_has_required_objects(spatial)
        ]
        return self._build_temporal_online_inference_prompt_from_results(recent_results)

    def _parse_spatial_coord_token(self, token: str) -> tuple[int, int] | None:
        token = str(token).strip()
        if not (token.startswith("(") and token.endswith(")")):
            return None
        parts = token[1:-1].split(",", maxsplit=1)
        if len(parts) != 2:
            return None
        try:
            return int(parts[0].strip()), int(parts[1].strip())
        except ValueError:
            return None

    def _parse_spatial_sequence_text(self, text: str) -> dict[str, list[tuple[int, int]]]:
        parsed: dict[str, list[tuple[int, int]]] = {
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
                parsed["moveto"].append(coord)
                mode = None
            elif mode in ("pusher", "tbar", "goal"):
                parsed[mode].append(coord)
        return parsed

    def _future_output_moveto_coords(self, *, prompt_text: str, output_text: str) -> list[tuple[int, int]]:
        prompt_moveto = self._parse_spatial_sequence_text(prompt_text)["moveto"]
        output_moveto = self._parse_spatial_sequence_text(output_text)["moveto"]
        if len(prompt_moveto) > 0 and output_moveto[: len(prompt_moveto)] == prompt_moveto:
            output_moveto = output_moveto[len(prompt_moveto):]
        parsed = [
            (int(x_idx), int(y_idx))
            for x_idx, y_idx in output_moveto
            if 0 <= int(x_idx) < self.resolution_x and 0 <= int(y_idx) < self.resolution_y
        ]
        return self._sample_autoinference_waypoints(parsed)

    def _nearest_tblock_distance(self, coord: tuple[int, int]) -> float:
        if not self.tblock_coords:
            return math.inf
        x_idx, y_idx = int(coord[0]), int(coord[1])
        return min(
            math.hypot(float(x_idx - tblock_x), float(y_idx - tblock_y))
            for tblock_x, tblock_y in self.tblock_coords
        )

    def _sample_autoinference_waypoints(
        self,
        waypoints: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if len(waypoints) <= 3:
            return list(waypoints)

        prefix = waypoints[:-3]
        tail = waypoints[-3:]
        sampled: list[tuple[int, int]] = []
        for start_idx in range(0, len(prefix), 8):
            group = prefix[start_idx:start_idx + 8]
            if (
                len(group) > 0
                and all(self._nearest_tblock_distance(coord) > 3.0 for coord in group)
            ):
                sampled.append(group[0])
            else:
                sampled.extend(group)
        sampled.extend(tail)
        return sampled

    def _effective_forbidden_tokens_text(self, user_text: str, *, movement_only: bool) -> str:
        tokens: list[str] = []
        seen: set[str] = set()

        def add_token(token: str) -> None:
            token = str(token).strip()
            if not token or token in seen:
                return
            seen.add(token)
            tokens.append(token)

        for token in str(user_text).replace("\n", ",").split(","):
            add_token(token)
        if movement_only:
            for token in self._MOVEMENT_ONLY_FORBIDDEN_TOKENS:
                add_token(token)
        return ",".join(tokens)

    def _set_inference_text_preview(self, prompt_text: str, output_text: str) -> None:
        prompt_preview = " ".join(str(prompt_text).split())
        output_preview = " ".join(str(output_text).split())
        if len(prompt_preview) > 180:
            prompt_preview = prompt_preview[:177] + "..."
        if len(output_preview) > 180:
            output_preview = output_preview[:177] + "..."
        self.inference_prompt_text.set(f"Prompt: {prompt_preview if prompt_preview else 'empty'}")
        self.inference_output_text.set(f"Output: {output_preview if output_preview else 'empty'}")

    def _inference_http_get(self, base_url: str, path: str) -> dict:
        req = request.Request(
            f"{base_url}{path}",
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="GET",
        )
        try:
            with request.urlopen(req, timeout=5.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {text}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Failed to reach server: {exc}") from exc

    def _inference_http_post(self, base_url: str, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            f"{base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=60.0) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"HTTP {exc.code} {exc.reason}: {text}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Failed to reach server: {exc}") from exc

    def _cancel_auto_online_job(self) -> None:
        if self._auto_online_job is not None:
            self.root.after_cancel(self._auto_online_job)
            self._auto_online_job = None

    def _set_auto_online_enabled(self, enabled: bool) -> None:
        self._auto_online_enabled = bool(enabled)
        if self._auto_online_button is not None:
            self._auto_online_button.configure(text="Stop Auto" if enabled else "Start Auto")
        if enabled:
            self.inference_status_text.set("Auto online inference: enabled")
            self._schedule_auto_online_tick(delay_ms=100)
        else:
            self._cancel_auto_online_job()
            self._online_inference_auto_send = False
            self.inference_status_text.set("Auto online inference: stopped")

    def toggle_auto_online_inference(self) -> None:
        self._set_auto_online_enabled(not self._auto_online_enabled)

    def _schedule_auto_online_tick(self, *, delay_ms: int = 1500) -> None:
        if not self._auto_online_enabled:
            return
        self._cancel_auto_online_job()
        self._auto_online_job = self.root.after(max(100, int(delay_ms)), self._auto_online_tick)

    def _auto_online_tick(self) -> None:
        self._auto_online_job = None
        if not self._auto_online_enabled:
            return
        if self._online_inference_busy or self._run_busy or self._saved_replay_active:
            self._schedule_auto_online_tick(delay_ms=1000)
            return
        self.run_online_inference(auto_send=True)

    def _continue_auto_online_after_idle(self, *, delay_ms: int = 1500) -> None:
        if self._auto_online_enabled:
            self._schedule_auto_online_tick(delay_ms=delay_ms)

    def run_online_inference(self, *, auto_send: bool = False) -> None:
        if self._online_inference_busy:
            return
        if auto_send and self._run_busy:
            self._continue_auto_online_after_idle(delay_ms=1000)
            return
        server_url = self.inference_server_var.get().strip().rstrip("/")
        if not server_url:
            self.inference_status_text.set("Online inference: server URL is empty")
            self._continue_auto_online_after_idle(delay_ms=3000)
            return
        try:
            prompt_text = self._build_online_inference_prompt()
            top_k_text = self.inference_top_k_var.get().strip()
            movement_only = bool(self.inference_movement_only_var.get())
            payload = {
                "prompt_text": prompt_text,
                "max_new_tokens": max(1, int(self.inference_max_new_tokens_var.get())),
                "temperature": float(self.inference_temperature_var.get()),
                "do_sample": bool(self.inference_do_sample_var.get()),
                "top_k": int(top_k_text) if top_k_text else None,
                "forbidden_tokens": self._effective_forbidden_tokens_text(
                    self.inference_forbidden_tokens_var.get(),
                    movement_only=movement_only,
                ),
                "require_state_after_movement": (
                    False
                    if movement_only
                    else bool(self.inference_require_state_after_movement_var.get())
                ),
                "state_after_movement_prob": (
                    0.0
                    if movement_only
                    else float(self.inference_state_after_movement_prob_var.get())
                ),
            }
        except Exception as exc:
            self.inference_status_text.set(f"Online inference: error: {exc}")
            self._continue_auto_online_after_idle(delay_ms=3000)
            return

        self._online_inference_busy = True
        self._online_inference_auto_send = bool(auto_send)
        self._online_inference_prompt = prompt_text
        self._online_inference_output = ""
        self._online_inference_raw_response = ""
        self._set_inference_text_preview(prompt_text, "")
        self.inference_status_text.set(
            "Auto online inference: working..."
            if auto_send
            else "Online inference: working..."
        )

        def _worker() -> None:
            try:
                status_payload = self._inference_http_get(server_url, "/api/status")
                if not status_payload.get("status", {}).get("ready", False):
                    raise RuntimeError("Server is not ready. Load a model on the server first.")
                result = self._inference_http_post(server_url, "/api/infer", payload)
                output_text = result.get("result", {}).get("output_text", "") or "(empty response)"
                raw_response = json.dumps(result, indent=2, ensure_ascii=False)
                self.root.after(
                    0,
                    lambda output_text=output_text, raw_response=raw_response:
                    self._on_online_inference_success(output_text, raw_response),
                )
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._on_online_inference_failure(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_online_inference_success(self, output_text: str, raw_response: str) -> None:
        self._online_inference_busy = False
        self._online_inference_output = str(output_text)
        self._online_inference_raw_response = str(raw_response)
        auto_send = self._online_inference_auto_send
        self._online_inference_auto_send = False
        predicted = self._future_output_moveto_coords(
            prompt_text=self._online_inference_prompt,
            output_text=self._online_inference_output,
        )
        self._set_inference_text_preview(self._online_inference_prompt, self._online_inference_output)
        if predicted:
            self.trajectory = predicted
            self._active_planned_trajectory.clear()
            history_count = len(self._online_inference_used_histories)
            self.inference_status_text.set(
                "Online inference: completed | "
                f"loaded {len(predicted)} predicted waypoint(s), visualizing {history_count} history frame(s)"
            )
            self.status_text.set("Predicted trajectory loaded. Edit or send when ready.")
            self.redraw()
            if auto_send and self._auto_online_enabled:
                self.status_text.set("Auto online inference loaded trajectory; sending...")
                self.root.after(100, self.send_trajectory)
        else:
            self.inference_status_text.set("Online inference: completed | no moveto waypoints parsed")
            self.redraw()
            if auto_send:
                self._continue_auto_online_after_idle(delay_ms=2500)

    def _on_online_inference_failure(self, exc: Exception) -> None:
        self._online_inference_busy = False
        auto_send = self._online_inference_auto_send
        self._online_inference_auto_send = False
        self._online_inference_output = f"Error: {exc}"
        self._online_inference_raw_response = f"Error: {exc}"
        self._set_inference_text_preview(self._online_inference_prompt, self._online_inference_output)
        self.inference_status_text.set(f"Online inference: error: {exc}")
        if auto_send:
            self._continue_auto_online_after_idle(delay_ms=3000)

    def _wait_for_valid_scene_frame(
        self,
        *,
        step_index: int,
        target_coord: tuple[int, int],
        move_payload: dict,
        goal_coords: list[tuple[int, int]],
        movement_coords: list[tuple[int, int]],
        wait_timeout_s: float,
    ) -> tuple[dict, int]:
        start_time = time.time()
        attempts = 0
        last_status = ""
        while True:
            attempts += 1
            try:
                data = self._http_get("/scene")
                if not data.get("ok", False):
                    raise RuntimeError(data.get("error", "scene request failed"))
                scene = data.get("scene", {})
                spatial = dict(scene.get("spatial", {}))
                last_status = str(spatial.get("status", ""))
                visual_status = (
                    f"Moving waypoint {step_index + 1}: waiting for scene "
                    f"(poll {attempts})"
                )
                self.root.after(
                    0,
                    lambda spatial=spatial, goal_coords=list(goal_coords),
                    movement_coords=list(movement_coords), visual_status=visual_status:
                    self._apply_spatial_to_view(
                        spatial,
                        goal_coords=goal_coords,
                        movement_coords=movement_coords,
                        status_text=visual_status,
                        refresh_episode=False,
                    ),
                )
                if (
                    self._spatial_has_required_objects(spatial)
                    and self._spatial_has_required_apriltag_points(spatial)
                ):
                    try:
                        self._refresh_cached_rgb_snapshot()
                    except Exception:
                        pass
                    enriched_spatial = self._enrich_spatial_payload(
                        spatial,
                        goal_coords=goal_coords,
                        movement_coords=movement_coords,
                    )
                    return (
                        {
                            "step_index": int(step_index),
                            "timestamp": float(time.time()),
                            "target_coord": [int(target_coord[0]), int(target_coord[1])],
                            "move": move_payload,
                            "spatial": enriched_spatial,
                            "scene_wait_attempts": int(attempts),
                        },
                        attempts,
                    )
            except (error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                last_status = f"{type(exc).__name__}: {exc}"
            if time.time() - start_time > wait_timeout_s:
                raise TimeoutError(
                    "Timed out waiting for non-empty pusher/tblock/AprilTag scene data "
                    f"at waypoint {step_index + 1}. Last scene status: {last_status}"
                )
            time.sleep(0.05)

    def _resize_canvas(self) -> None:
        canvas_w = self.padding * 2 + self.resolution_x * self.cell_size
        canvas_h = self.padding * 2 + self.resolution_y * self.cell_size
        self.canvas.config(width=canvas_w, height=canvas_h)

    def _canvas_to_coord(self, event_x: int, event_y: int) -> tuple[int, int] | None:
        local_x = event_x - self.padding
        local_y = event_y - self.padding
        if local_x < 0 or local_y < 0:
            return None
        x_idx = local_x // self.cell_size
        y_canvas = local_y // self.cell_size
        if x_idx < 0 or y_canvas < 0:
            return None
        if x_idx >= self.resolution_x or y_canvas >= self.resolution_y:
            return None
        y_idx = self.resolution_y - 1 - y_canvas
        return int(x_idx), int(y_idx)

    def _coord_to_canvas_rect(self, coord: tuple[int, int]) -> tuple[int, int, int, int]:
        x_idx, y_idx = int(coord[0]), int(coord[1])
        left = self.padding + x_idx * self.cell_size
        top = self.padding + (self.resolution_y - 1 - y_idx) * self.cell_size
        return left, top, left + self.cell_size, top + self.cell_size

    def _coord_centers(self, coords: list[tuple[int, int]]) -> list[tuple[float, float]]:
        centers: list[tuple[float, float]] = []
        for coord in coords:
            left, top, right, bottom = self._coord_to_canvas_rect(coord)
            centers.append(((left + right) / 2.0, (top + bottom) / 2.0))
        return centers

    def _adjacent_line_waypoints(
        self,
        prev: tuple[int, int],
        target: tuple[int, int],
    ) -> list[tuple[int, int]]:
        dx = int(target[0] - prev[0])
        dy = int(target[1] - prev[1])
        steps = max(abs(dx), abs(dy))
        if steps == 0:
            return []

        waypoints: list[tuple[int, int]] = []
        for step_index in range(1, steps + 1):
            alpha = step_index / steps
            x_idx = int(round(prev[0] + dx * alpha))
            y_idx = int(round(prev[1] + dy * alpha))
            waypoint = (
                int(np.clip(x_idx, 0, self.resolution_x - 1)),
                int(np.clip(y_idx, 0, self.resolution_y - 1)),
            )
            if not waypoints or waypoint != waypoints[-1]:
                waypoints.append(waypoint)
        return waypoints

    def _expand_trajectory_to_adjacent_waypoints(
        self,
        control_points: list[tuple[int, int]],
    ) -> list[tuple[int, int]]:
        if len(control_points) == 0:
            return []
        if self.current_pusher is None:
            raise RuntimeError(
                "Current pusher position is unavailable; cannot expand the first segment "
                "into adjacent waypoints."
            )

        expanded: list[tuple[int, int]] = []
        anchor = self.current_pusher
        for target in control_points:
            segment = self._adjacent_line_waypoints(anchor, target)
            if segment:
                expanded.extend(segment)
                anchor = segment[-1]
        return expanded

    def _clamp_point_along_line(
        self,
        prev: tuple[int, int],
        target: tuple[int, int],
    ) -> tuple[int, int]:
        max_dist = float(self.max_traj_step_distance)
        dx = float(target[0] - prev[0])
        dy = float(target[1] - prev[1])
        dist = math.hypot(dx, dy)
        if dist <= max_dist or dist <= 1e-9:
            return target

        ux = dx / dist
        uy = dy / dist
        best = prev
        best_score = -1.0
        # Pick the farthest integer voxel along the ray that stays within max_dist.
        for radius in np.linspace(max_dist, 0.0, num=33):
            cx = int(round(prev[0] + ux * float(radius)))
            cy = int(round(prev[1] + uy * float(radius)))
            cx = int(np.clip(cx, 0, self.resolution_x - 1))
            cy = int(np.clip(cy, 0, self.resolution_y - 1))
            d = math.hypot(float(cx - prev[0]), float(cy - prev[1]))
            if d <= max_dist + 1e-9 and d > best_score:
                best = (cx, cy)
                best_score = d
        return best

    def _draw_pusher_movement_history_overlay(self) -> None:
        history_frames = self._spatial_history[-3:]
        if not history_frames:
            return
        pusher_track: list[tuple[int, int]] = []
        colors = [
            {
                "pusher": "#b64b22",
                "label": "#5f4b32",
            },
            {
                "pusher": "#ffb15c",
                "label": "#7b6b91",
            },
            {
                "pusher": "#7a58c9",
                "label": "#4d3f7f",
            },
        ]
        for history_idx, spatial in enumerate(history_frames):
            palette = colors[min(history_idx, len(colors) - 1)]
            line_width = 3 if history_idx == 0 else 2
            dash = (6, 3) if history_idx == 0 else (2, 2)
            pusher_coords = self._normalize_coord_list(spatial.get("pusher_coords", []))
            pusher_anchor = self._representative_action_coord(pusher_coords) if pusher_coords else None
            if pusher_anchor is not None:
                pusher_track.append(pusher_anchor)
            for coord in pusher_coords:
                self.canvas.create_rectangle(
                    *self._coord_to_canvas_rect(coord),
                    outline=palette["pusher"],
                    width=max(1, line_width - 1),
                    dash=dash,
                )
            if pusher_anchor is not None:
                center = self._coord_centers([pusher_anchor])[0]
                radius = max(self.cell_size * 0.32, 2.0)
                self.canvas.create_oval(
                    center[0] - radius,
                    center[1] - radius,
                    center[0] + radius,
                    center[1] + radius,
                    outline=palette["pusher"],
                    width=line_width,
                )
                self.canvas.create_text(
                    center[0],
                    center[1] - self.cell_size * 1.0,
                    text=f"P{history_idx + 1}",
                    fill=palette["label"],
                    font=("TkDefaultFont", 8, "bold"),
                )
        pusher_centers = self._coord_centers(pusher_track)
        for start, end in zip(pusher_centers[:-1], pusher_centers[1:], strict=False):
            self.canvas.create_line(
                start[0],
                start[1],
                end[0],
                end[1],
                fill="#d45f33",
                width=3,
                arrow=tk.LAST,
            )

    def redraw(self) -> None:
        self.canvas.delete("all")
        for x_idx in range(self.resolution_x + 1):
            x = self.padding + x_idx * self.cell_size
            self.canvas.create_line(x, self.padding, x, self.padding + self.resolution_y * self.cell_size, fill="#dddddd")
        for y_idx in range(self.resolution_y + 1):
            y = self.padding + y_idx * self.cell_size
            self.canvas.create_line(self.padding, y, self.padding + self.resolution_x * self.cell_size, y, fill="#dddddd")

        for coord in self.tblock_coords_full:
            self.canvas.create_rectangle(
                *self._coord_to_canvas_rect(coord),
                outline="#2f6fb3",
                width=1,
                dash=(2, 2),
            )
        for center in self._coord_centers(self.tblock_apriltag_coords_2d):
            radius = max(self.cell_size * 0.22, 2.0)
            self.canvas.create_oval(
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
                outline="#d43f3a",
                width=2,
            )
            self.canvas.create_line(
                center[0] - radius,
                center[1] - radius,
                center[0] + radius,
                center[1] + radius,
                fill="#d43f3a",
                width=2,
            )
            self.canvas.create_line(
                center[0] - radius,
                center[1] + radius,
                center[0] + radius,
                center[1] - radius,
                fill="#d43f3a",
                width=2,
            )
        for coord in self.tblock_coords:
            self.canvas.create_rectangle(*self._coord_to_canvas_rect(coord), fill="#63a7ff", outline="")
        for coord in self.goal:
            self.canvas.create_rectangle(*self._coord_to_canvas_rect(coord), fill="#6fd07a", outline="")
        if self.current_pusher is not None:
            self.canvas.create_rectangle(*self._coord_to_canvas_rect(self.current_pusher), fill="#ff8f5a", outline="")

        planned_centers = self._coord_centers(self._active_planned_trajectory)
        for start, end in zip(planned_centers[:-1], planned_centers[1:], strict=False):
            self.canvas.create_line(
                start[0],
                start[1],
                end[0],
                end[1],
                fill="#c9a8eb",
                width=2,
                dash=(4, 3),
            )
        for center in planned_centers:
            self.canvas.create_oval(
                center[0] - self.cell_size * 0.13,
                center[1] - self.cell_size * 0.13,
                center[0] + self.cell_size * 0.13,
                center[1] + self.cell_size * 0.13,
                outline="#c9a8eb",
                width=2,
            )

        centers = self._coord_centers(self.trajectory)
        for start, end in zip(centers[:-1], centers[1:], strict=False):
            self.canvas.create_line(start[0], start[1], end[0], end[1], fill="#9b53d8", width=2)
        for center in centers:
            self.canvas.create_oval(
                center[0] - self.cell_size * 0.2,
                center[1] - self.cell_size * 0.2,
                center[0] + self.cell_size * 0.2,
                center[1] + self.cell_size * 0.2,
                fill="#9b53d8",
                outline="",
            )

    def on_left_click(self, event) -> None:
        coord = self._canvas_to_coord(event.x, event.y)
        if coord is None:
            return
        if self.goal_mode.get():
            if coord not in self.goal:
                self.goal.append(coord)
            self.status_text.set(f"Goal voxel added: ({coord[0]}, {coord[1]})")
        else:
            anchor = None
            if len(self.trajectory) > 0:
                anchor = self.trajectory[-1]
            else:
                anchor = self.current_pusher
            if anchor is not None:
                prev = anchor
                clamped = self._clamp_point_along_line(prev, coord)
                if clamped != coord:
                    coord = clamped
                    if coord == prev:
                        self._show_warning(
                            "Trajectory point too far; no reachable new voxel on that line.",
                            popup=False,
                        )
                        return
                    label = "control point" if self.auto_adjacent_waypoints.get() else "point"
                    self._show_warning(
                        f"Clicked {label} was farther than max step; "
                        f"added farthest reachable voxel ({coord[0]}, {coord[1]}) instead.",
                        popup=False,
                    )
            elif len(self.trajectory) == 0:
                self._show_warning(
                    "Current pusher position is unavailable; cannot place the first trajectory point.",
                    popup=False,
                )
                return
            self.trajectory.append(coord)
            if self.auto_adjacent_waypoints.get():
                self.status_text.set(f"Trajectory control voxel added: ({coord[0]}, {coord[1]})")
            else:
                self.status_text.set(f"Trajectory voxel added: ({coord[0]}, {coord[1]})")
        self.redraw()

    def on_right_click(self, _event) -> None:
        if self.goal_mode.get():
            if self.goal:
                removed = self.goal.pop()
                self.status_text.set(f"Goal voxel removed: ({removed[0]}, {removed[1]})")
        else:
            if self.trajectory:
                removed = self.trajectory.pop()
                self.status_text.set(f"Trajectory voxel removed: ({removed[0]}, {removed[1]})")
        self.redraw()

    def clear_trajectory(self) -> None:
        current_spatial = self._current_spatial_snapshot()
        if self._spatial_has_required_objects(current_spatial):
            self._record_spatial_history(current_spatial)
        self.trajectory.clear()
        self._active_planned_trajectory.clear()
        self.status_text.set("Trajectory cleared.")
        self.redraw()

    def clear_goal(self) -> None:
        self.goal.clear()
        self.status_text.set("Goal cleared.")
        self.redraw()

    def _apply_spatial_to_view(
        self,
        spatial: dict,
        *,
        goal_coords: list[tuple[int, int]] | None = None,
        movement_coords: list[tuple[int, int]] | None = None,
        status_text: str | None = None,
        refresh_episode: bool = True,
    ) -> None:
        resolution = spatial.get("resolution_xyz", [self.resolution_x, self.resolution_y, 1])
        rx = max(1, int(resolution[0]))
        ry = max(1, int(resolution[1]))
        resized = (rx != self.resolution_x) or (ry != self.resolution_y)
        self.resolution_x = rx
        self.resolution_y = ry
        if resized:
            self._resize_canvas()
        bbox_min = spatial.get("bbox_min", self._latest_spatial_config["bbox_min"])
        bbox_max = spatial.get("bbox_max", self._latest_spatial_config["bbox_max"])
        rz = int(resolution[2]) if len(resolution) >= 3 else int(self._latest_spatial_config["resolution_xyz"][2])
        self._latest_spatial_config = {
            "bbox_min": [float(v) for v in bbox_min[:3]],
            "bbox_max": [float(v) for v in bbox_max[:3]],
            "resolution_xyz": [int(rx), int(ry), int(max(rz, 1))],
        }

        self.tblock_coords = self._normalize_coord_list(spatial.get("tblock_coords", []))
        self.tblock_coords_full = self._normalize_coord_list(spatial.get("tblock_coords_full", []))
        self.tblock_apriltag_coords_2d = self._normalize_coord_list(
            spatial.get("tblock_apriltag_coords_2d", [])
        )
        self.tblock_apriltag_points_world = [
            dict(item)
            for item in spatial.get("tblock_apriltag_points_world", [])
            if isinstance(item, dict)
        ]
        pusher_coords = self._normalize_coord_list(spatial.get("pusher_coords", []))
        self.pusher_coords = pusher_coords
        self.current_pusher = pusher_coords[0] if pusher_coords else None
        self._update_pusher_text()
        if goal_coords is not None:
            self.goal = self._normalize_coord_list(goal_coords)
        if movement_coords is not None:
            self.trajectory = self._normalize_coord_list(movement_coords)
        if status_text is not None:
            self.status_text.set(status_text)
        elif not self._run_busy:
            status = spatial.get("status", "")
            self.status_text.set(status if status else "Scene updated.")
        if self._spatial_has_required_objects(spatial):
            self._record_spatial_history(spatial)
        if refresh_episode:
            self._refresh_local_episode_ui()
        self.redraw()

    def fetch_scene(self, *, popup_on_error: bool = False) -> None:
        try:
            data = self._http_get("/scene")
            if not data.get("ok", False):
                raise RuntimeError(data.get("error", "scene request failed"))
            scene = data["scene"]
            spatial = scene.get("spatial", {})
            self._apply_spatial_to_view(spatial)
            try:
                self._refresh_cached_rgb_snapshot()
            except Exception:
                pass
        except (error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
            self._show_error(f"Scene fetch failed: {exc}", popup=popup_on_error)

    def _build_local_episode_payload(self) -> dict:
        if len(self._current_episode_movements) == 0:
            raise ValueError("Current episode has no movement data to save.")
        normalized_movements = [self._normalize_movement_for_episode_save(m) for m in self._current_episode_movements]
        frame_count = int(sum(len(m.get("frames", [])) for m in normalized_movements))
        return {
            "format": "robodata_spatial_episode_v1",
            "episode_id": self._current_episode_id,
            "started_at": self._current_episode_started_at,
            "finished_at": time.time(),
            "spatial_config": dict(self._latest_spatial_config),
            "spatial_history": [
                {
                    "pusher_coords": [
                        [int(x_idx), int(y_idx)]
                        for x_idx, y_idx in self._normalize_coord_list(history.get("pusher_coords", []))
                    ],
                    "tblock_coords": [
                        [int(x_idx), int(y_idx)]
                        for x_idx, y_idx in self._normalize_coord_list(history.get("tblock_coords", []))
                    ],
                    "tblock_apriltag_coords_2d": [
                        [int(x_idx), int(y_idx)]
                        for x_idx, y_idx in self._normalize_coord_list(
                            history.get("tblock_apriltag_coords_2d", [])
                        )
                    ],
                    "tblock_apriltag_points_world": [
                        dict(item)
                        for item in history.get("tblock_apriltag_points_world", [])
                        if isinstance(item, dict)
                    ],
                    "goal_coords": [
                        [int(x_idx), int(y_idx)]
                        for x_idx, y_idx in self._normalize_coord_list(history.get("goal_coords", []))
                    ],
                    "pusher_coord": [
                        int(value)
                        for value in history.get("pusher_coord", [])
                    ],
                    "resolution_xyz": list(history.get("resolution_xyz", [self.resolution_x, self.resolution_y, 1])),
                }
                for history in self._spatial_history
            ],
            "movement_count": len(normalized_movements),
            "frame_count": frame_count,
            "rgb_recording_enabled": bool(self.record_episode_rgb_var.get()),
            "rgb_snapshot_count": int(len(self._current_episode_rgb_frames)),
            "movements": normalized_movements,
        }

    def _normalize_frame_for_episode_save(self, frame: dict) -> dict:
        normalized = dict(frame)
        spatial = dict(normalized.get("spatial", {}))
        spatial["tblock_apriltag_coords_2d"] = [
            [int(x_idx), int(y_idx)]
            for x_idx, y_idx in self._normalize_coord_list(spatial.get("tblock_apriltag_coords_2d", []))
        ]
        spatial["tblock_apriltag_points_world"] = self._normalize_apriltag_points_world(
            spatial.get("tblock_apriltag_points_world", [])
        )
        normalized["spatial"] = spatial
        return normalized

    def _normalize_movement_for_episode_save(self, movement: dict) -> dict:
        normalized = dict(movement)
        normalized["frames"] = [
            self._normalize_frame_for_episode_save(frame)
            for frame in movement.get("frames", [])
            if isinstance(frame, dict)
        ]
        return normalized

    def _reset_local_episode(self) -> None:
        self._current_episode_id = self._new_local_episode_id()
        self._current_episode_started_at = time.time()
        self._current_episode_movements = []
        self._current_episode_rgb_frames = []
        self._spatial_history = []
        self._online_inference_used_histories = []
        self._refresh_local_episode_ui()

    def finish_episode(self) -> None:
        if self._auto_online_enabled:
            self._set_auto_online_enabled(False)
        try:
            episode_payload = self._build_local_episode_payload()
            video_path: Path | None = None
            if len(self._current_episode_rgb_frames) > 0:
                episode_id = str(episode_payload.get("episode_id", ""))
                video_path = self._write_episode_rgb_video(episode_id)
                episode_payload["rgb_video_file"] = video_path.name
                episode_payload["rgb_video_fps"] = self._episode_rgb_fps()
            else:
                episode_payload["rgb_video_file"] = None
            output_path = self._save_episode_payload(episode_payload)
            self.refresh_saved_episode_list()
            self._reset_local_episode()
            self.trajectory.clear()
            self._active_planned_trajectory.clear()
            self.redraw()
            if video_path is not None:
                self.status_text.set(
                    f"Episode saved: {output_path} | MP4 saved: {video_path}. Started a new episode."
                )
            else:
                self.status_text.set(f"Episode saved: {output_path}. Started a new episode.")
            self.fetch_scene(popup_on_error=False)
        except (RuntimeError, ValueError) as exc:
            self._show_error(f"Failed to finish episode: {exc}")
        except OSError as exc:
            self._show_error(f"Failed to write episode file: {exc}")

    def start_new_episode(self) -> None:
        confirm = messagebox.askyesno(
            title="Start New Episode",
            message=(
                "Start a new episode now?\n\n"
                "Current buffered movements will be discarded unless you click "
                "'Finish Episode' first."
            ),
        )
        if not confirm:
            return
        if self._auto_online_enabled:
            self._set_auto_online_enabled(False)
        self._reset_local_episode()
        self.trajectory.clear()
        self._active_planned_trajectory.clear()
        self.redraw()
        self.status_text.set("Started a new episode. Current buffer cleared.")
        self.fetch_scene(popup_on_error=False)

    def send_trajectory(self) -> None:
        if self._run_busy:
            self._show_warning("A trajectory is already running. Please wait.")
            return
        if not self.trajectory:
            self.status_text.set("Select at least one voxel for the trajectory.")
            return
        try:
            speed = max(1, int(self.speed_var.get()))
            timesteps = max(1, int(self.timesteps_var.get()))
            settle_s = max(0.0, float(self.settle_var.get()))
            timeout_s = max(0.1, float(self.timeout_var.get()))
        except (tk.TclError, ValueError) as exc:
            self._show_error(f"Invalid motion parameters: {exc}")
            return
        self.speed_var.set(speed)
        self.timesteps_var.set(timesteps)
        self.settle_var.set(settle_s)
        self.timeout_var.set(timeout_s)
        control_trajectory = [(int(x), int(y)) for x, y in self.trajectory]
        record_rgb = bool(self.record_episode_rgb_var.get())
        try:
            if self.auto_adjacent_waypoints.get():
                trajectory = self._expand_trajectory_to_adjacent_waypoints(control_trajectory)
                if len(trajectory) == 0:
                    self.status_text.set("Selected control cells do not add any adjacent movement.")
                    return
            else:
                trajectory = list(control_trajectory)
        except RuntimeError as exc:
            self._show_error(str(exc))
            return
        goal = [(int(x), int(y)) for x, y in self.goal]
        num_steps = len(trajectory)
        self._run_busy = True
        self._active_planned_trajectory = list(trajectory)
        self.redraw()
        if self.auto_adjacent_waypoints.get():
            self.run_text.set(f"Run: expanded {len(control_trajectory)} click(s) -> {num_steps} step(s)")
        else:
            self.run_text.set(f"Run: running | step 0/{num_steps}")

        def _worker() -> None:
            try:
                run_id = time.strftime("%Y%m%d_%H%M%S") + f"_{int((time.time() % 1.0) * 1000):03d}"
                frames: list[dict] = []
                warnings: list[str] = []
                rgb_snapshots: list[dict] = []
                waypoint_run_ids: list[str] = []
                scene_wait_timeout_s = max(float(timeout_s), 30.0)
                for step_index, target_coord in enumerate(trajectory):
                    self.root.after(
                        0,
                        lambda step_index=step_index: self.run_text.set(
                            f"Run: moving | step {step_index + 1}/{num_steps}"
                        ),
                    )
                    step_payload = {
                        "trajectory": [[int(target_coord[0]), int(target_coord[1])]],
                        "goal": [[int(x), int(y)] for x, y in goal],
                        "timesteps": timesteps,
                        "speed": speed,
                        "settle_s": settle_s,
                        "timeout_s": timeout_s,
                    }
                    data = self._http_post(
                        "/trajectory/execute",
                        step_payload,
                        timeout=max(float(timeout_s) + 30.0, 60.0),
                    )
                    if not data.get("ok", False):
                        raise RuntimeError(data.get("error", "execute request failed"))
                    waypoint_run_ids.append(str(data.get("run_id", "")))
                    step_movement = data.get("movement")
                    if not isinstance(step_movement, dict):
                        raise RuntimeError("Missing movement payload from collect_viser.")
                    move_payload = {}
                    step_frames = step_movement.get("frames", [])
                    if step_frames:
                        move_payload = dict(step_frames[0].get("move", {}))
                    movement_prefix = trajectory[: step_index + 1]
                    self.root.after(
                        0,
                        lambda step_index=step_index: self.run_text.set(
                            f"Run: waiting scene | step {step_index + 1}/{num_steps}"
                        ),
                    )
                    frame_payload, attempts = self._wait_for_valid_scene_frame(
                        step_index=step_index,
                        target_coord=target_coord,
                        move_payload=move_payload,
                        goal_coords=goal,
                        movement_coords=movement_prefix,
                        wait_timeout_s=scene_wait_timeout_s,
                    )
                    if attempts > 1:
                        warnings.append(
                            f"Waypoint {step_index + 1}/{num_steps} waited {attempts} scene polls "
                            "for non-empty pusher/tblock data."
                        )
                    frames.append(frame_payload)
                    if record_rgb:
                        try:
                            rgb_frame, rgb_timestamp = self._cached_rgb_snapshot()
                            rgb_snapshots.append(
                                {
                                    "frame": rgb_frame,
                                    "timestamp": float(rgb_timestamp),
                                    "step_index": int(step_index),
                                }
                            )
                        except Exception as exc:
                            warnings.append(
                                f"Waypoint {step_index + 1}/{num_steps} RGB snapshot failed: {exc}"
                            )
                movement = {
                    "run_id": run_id,
                    "timestamp": float(time.time()),
                    "trajectory": [[int(x_idx), int(y_idx)] for x_idx, y_idx in trajectory],
                    "control_trajectory": [
                        [int(x_idx), int(y_idx)] for x_idx, y_idx in control_trajectory
                    ],
                    "goal": [[int(x_idx), int(y_idx)] for x_idx, y_idx in goal],
                    "frames": frames,
                    "warnings": warnings,
                    "rgb_snapshot_count": int(len(rgb_snapshots)),
                    "rgb_snapshot_timestamps": [
                        float(item["timestamp"]) for item in rgb_snapshots
                    ],
                    "waypoint_run_ids": waypoint_run_ids,
                    "send_mode": (
                        "adjacent_waypoints_on_send"
                        if self.auto_adjacent_waypoints.get()
                        else "waypoint_by_waypoint"
                    ),
                }
                self.root.after(
                    0,
                    lambda rgb_snapshots=rgb_snapshots: self._on_send_trajectory_success(
                        movement=movement,
                        speed=speed,
                        timesteps=timesteps,
                        rgb_snapshots=rgb_snapshots,
                    ),
                )
            except Exception as exc:
                self.root.after(0, lambda exc=exc: self._on_send_trajectory_failure(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_send_trajectory_success(
        self,
        *,
        movement: dict,
        speed: int,
        timesteps: int,
        rgb_snapshots: list[dict],
    ) -> None:
        self._run_busy = False
        self.run_text.set("Run: completed")
        self._current_episode_movements.append(movement)
        self._current_episode_rgb_frames.extend(
            {
                "frame": np.array(item["frame"], copy=True),
                "timestamp": float(item["timestamp"]),
                "step_index": int(item["step_index"]),
            }
            for item in rgb_snapshots
        )
        self._refresh_local_episode_ui()
        run_id = str(movement.get("run_id", ""))
        self.trajectory.clear()
        self._active_planned_trajectory.clear()
        self.redraw()
        warnings = movement.get("warnings", [])
        warning_suffix = f" | scene waits={len(warnings)}" if warnings else ""
        self.status_text.set(
            f"Trajectory done. run_id={run_id} | speed={speed} timesteps={timesteps}{warning_suffix}"
        )
        self.fetch_scene(popup_on_error=False)
        self._continue_auto_online_after_idle(delay_ms=1500)

    def _on_send_trajectory_failure(self, exc: Exception) -> None:
        self._run_busy = False
        self._active_planned_trajectory.clear()
        self.run_text.set("Run: failed")
        if self._auto_online_enabled:
            self._set_auto_online_enabled(False)
        self._show_error(f"Failed to send trajectory: {exc}")

    def _poll(self) -> None:
        if self.auto_refresh.get() and not self._saved_replay_active:
            self.fetch_scene(popup_on_error=False)
        self.root.after(self.poll_ms, self._poll)


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone GUI trajectory sender for collect_viser")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="collect_viser control API host")
    parser.add_argument("--port", type=int, default=8765, help="collect_viser control API port")
    parser.add_argument("--poll-ms", type=int, default=500, help="Polling interval in milliseconds")
    parser.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help=(
            "Path to graspgpt_dataset_*.json. If omitted, uses the latest file "
            "under ./data/records/datasets."
        ),
    )
    parser.add_argument(
        "--episode-output-dir",
        type=str,
        default="./data/records/trajectory_runs",
        help="Directory where Finish Episode writes spatial_episode_*.json",
    )
    parser.add_argument(
        "--dataset-output-dir",
        type=str,
        default="./data/records/datasets",
        help="Directory where Export Dataset writes graspgpt_dataset_*.json",
    )
    parser.add_argument(
        "--inference-server",
        type=str,
        default="http://127.0.0.1:8000",
        help="Base URL for the online inference server",
    )
    parser.add_argument(
        "--inference-max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to request from the inference server",
    )
    parser.add_argument(
        "--inference-temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for online inference",
    )
    parser.add_argument(
        "--inference-top-k",
        type=int,
        default=None,
        help="Optional top-k sampling limit for online inference",
    )
    parser.add_argument(
        "--inference-no-sample",
        action="store_true",
        help="Disable sampling for online inference",
    )
    parser.add_argument(
        "--inference-forbidden-tokens",
        type=str,
        default="",
        help="Comma-separated token strings to block during online inference",
    )
    parser.add_argument(
        "--inference-require-state-after-movement",
        action="store_true",
        help="Require the model to emit a state token after movement tokens",
    )
    parser.add_argument(
        "--inference-state-after-movement-prob",
        type=float,
        default=0.0,
        help="Probability bias for requiring a state token after movement tokens",
    )
    args = parser.parse_args()

    root = tk.Tk()
    CollectViserTrajectoryGUI(
        root,
        host=args.host,
        port=args.port,
        poll_ms=args.poll_ms,
        dataset_json=args.dataset_json,
        episode_output_dir=args.episode_output_dir,
        dataset_output_dir=args.dataset_output_dir,
        inference_server=args.inference_server,
        inference_max_new_tokens=args.inference_max_new_tokens,
        inference_temperature=args.inference_temperature,
        inference_top_k=args.inference_top_k,
        inference_do_sample=not args.inference_no_sample,
        inference_forbidden_tokens=args.inference_forbidden_tokens,
        inference_require_state_after_movement=args.inference_require_state_after_movement,
        inference_state_after_movement_prob=args.inference_state_after_movement_prob,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
