#!/usr/bin/env python3
"""
PIPER recording script with explicit per-episode start control.

Episode flow:
  1. Script connects hardware and waits.
  2. Press ENTER — 3-second countdown runs, then beep + banner signals recording start.
  3. During recording:
       RIGHT ARROW  — end episode and save it
       LEFT ARROW   — discard episode (re-record same number)
       ESCAPE       — save current episode and stop entirely
  4. After the episode ends you return to step 2.

Usage:
    python lerobot_data_collection/record_piper.py [options]

    # via env vars (same interface as record.sh):
    TASK="Grab cube" NUM_EPISODES=20 python lerobot_data_collection/record_piper.py
"""

import argparse
import logging
import os
import sys
import threading
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — must come before any project/plugin import
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

# Running the script directly adds lerobot_data_collection/ to sys.path[0].
# That directory contains lerobot_camera_zed2i/ (the install root, no __init__.py),
# which shadows the properly installed package.  Remove it first.
if _HERE in sys.path:
    sys.path.remove(_HERE)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Register plugins before any lerobot import that touches the registry
# ---------------------------------------------------------------------------
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

register_third_party_plugins()

from lerobot.cameras.configs import CameraConfig  # noqa: F401, E402
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401, E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.pipeline_features import (  # noqa: E402
    aggregate_pipeline_dataset_features,
    create_initial_features,
)
from lerobot.datasets.utils import combine_feature_dicts  # noqa: E402
from lerobot.datasets.video_utils import VideoEncodingManager  # noqa: E402
from lerobot.processor import make_default_processors  # noqa: E402
from lerobot.robots import make_robot_from_config  # noqa: E402
from lerobot.scripts.lerobot_record import record_loop  # noqa: E402
from lerobot.teleoperators import make_teleoperator_from_config  # noqa: E402
from lerobot.utils.control_utils import init_keyboard_listener, is_headless  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data  # noqa: E402
from lerobot_camera_zed2i import ZED2iCameraConfig  # noqa: E402
from lerobot_robot_piper_follower import PiperFollowerConfig  # noqa: E402
from lerobot_teleoperator_piper_leader import PiperLeaderConfig  # noqa: E402

logger = logging.getLogger(__name__)

# ANSI colour helpers (gracefully disabled on non-ANSI terminals)
_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_RED   = "\033[91m"
_CYAN  = "\033[96m"
_RST   = "\033[0m"
_BELL  = "\a"


# Joint names — must match what PiperFollower/PiperLeader report
_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
_ACTION_PATHS  = [f"action.{j}.pos" for j in _JOINTS] + ["action.gripper.pos"]
_OBS_JOINT_PATHS = [f"observation.{j}.pos" for j in _JOINTS] + ["observation.gripper.pos"]


# ---------------------------------------------------------------------------
# Rerun helpers
# ---------------------------------------------------------------------------

def _setup_rerun_blueprint() -> None:
    """Cameras + status on top row; action and observation time series below."""
    import rerun as rr
    import rerun.blueprint as rrb

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(name="RealSense",  contents=["observation.realsense"]),
                rrb.Spatial2DView(name="ZED 2i",     contents=["observation.zed2i"]),
                rrb.TextDocumentView(name="Status",  contents=["status"]),
                column_shares=[4, 4, 1],
            ),
            rrb.TimeSeriesView(name="Actions",      contents=_ACTION_PATHS),
            rrb.TimeSeriesView(name="Observations", contents=_OBS_JOINT_PATHS),
            row_shares=[3, 1, 1],
        ),
        auto_layout=False,
        auto_views=False,
    )
    rr.send_blueprint(blueprint)


def _log_status(is_recording: bool, episode_num: int = 0, num_episodes: int = 0) -> None:
    """Log a markdown status card to the rerun 'status' entity."""
    import rerun as rr
    if is_recording:
        md = (
            f"# 🔴 RECORDING\n\n"
            f"Episode **{episode_num}** / {num_episodes}\n\n"
            f"`RIGHT →` save &nbsp;|&nbsp; `LEFT ←` discard &nbsp;|&nbsp; `ESC` stop"
        )
    else:
        md = (
            f"# 🟢 IDLE\n\n"
            f"Episode **{episode_num}** / {num_episodes}\n\n"
            f"Press **ENTER** to start"
        )
    rr.log("status", rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN))


def _get_realsense_params(cam) -> dict:
    """Extract intrinsics from a connected lerobot RealSenseCamera."""
    try:
        import pyrealsense2 as rs
        stream = cam.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = stream.get_intrinsics()
        fov  = rs.rs2_fov(intr)
        serial = getattr(cam, "serial_number", None) or getattr(cam, "camera_index", None)
        return {
            "type":               "intelrealsense",
            "serial_number":      str(serial) if serial is not None else None,
            "width":              intr.width,
            "height":             intr.height,
            "fps":                getattr(cam, "fps", None),
            "fx":                 intr.fx,
            "fy":                 intr.fy,
            "cx":                 intr.ppx,
            "cy":                 intr.ppy,
            "h_fov_deg":          fov[0],
            "v_fov_deg":          fov[1],
            "distortion_model":   str(intr.model),
            "distortion_coeffs":  list(intr.coeffs),
        }
    except Exception as exc:
        return {"type": "intelrealsense", "error": str(exc)}


def _dump_cameras_info(robot, dataset_root: Path) -> None:
    """Collect intrinsics from all robot cameras and write meta/cameras_info.json."""
    import json

    info: dict = {}
    for cam_key, cam in robot.cameras.items():
        if hasattr(cam, "rs_profile") and cam.rs_profile is not None:
            info[cam_key] = _get_realsense_params(cam)
        elif hasattr(cam, "get_camera_params"):
            info[cam_key] = cam.get_camera_params()
        else:
            info[cam_key] = {
                "width":  getattr(cam, "width",  None),
                "height": getattr(cam, "height", None),
            }

    out_path = dataset_root / "meta" / "cameras_info.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"Camera info saved → {out_path}")


def _preview_loop(robot, teleop, fps: int, stop_event: threading.Event) -> None:
    """Read sensors and log to rerun continuously without saving to dataset."""
    dt = 1.0 / fps
    while not stop_event.is_set():
        t0 = time.time()
        try:
            obs = robot.get_observation()
            action = teleop.get_action()
            log_rerun_data(observation=obs, action=action)
        except Exception:
            pass
        wait = dt - (time.time() - t0)
        if wait > 0:
            stop_event.wait(wait)   # interruptible sleep


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _beep(n: int = 1) -> None:
    """Write n terminal bell characters."""
    sys.stdout.write(_BELL * n)
    sys.stdout.flush()


def _banner(text: str, colour: str = _GREEN) -> None:
    width = 55
    bar = "=" * width
    pad = max(0, width - 2 - len(text)) // 2
    print(f"\n{colour}{_BOLD}{bar}")
    print(f"  {' ' * pad}{text}")
    print(f"{bar}{_RST}")


def _wait_for_start(episode_num: int, num_episodes: int) -> bool:
    """Print prompt and block on ENTER. Returns False when user quits."""
    _banner(f"Episode {episode_num} / {num_episodes}", _CYAN)
    print(f"  {_BOLD}Press ENTER{_RST} to arm recording.")
    print(f"  Type {_BOLD}'q' + ENTER{_RST} to quit early.")
    print()
    try:
        return input("> ").strip().lower() != "q"
    except (EOFError, KeyboardInterrupt):
        return False


def _countdown(seconds: int = 3) -> None:
    """Print a live 3-second countdown before recording begins."""
    for i in range(seconds, 0, -1):
        sys.stdout.write(f"\r  Starting in {_BOLD}{i}{_RST}...  ")
        sys.stdout.flush()
        time.sleep(1)
    sys.stdout.write("\r" + " " * 30 + "\r")
    sys.stdout.flush()


def _recording_start_notification(episode_num: int) -> None:
    """Banner + beep shown exactly when recording begins."""
    _banner(f"● REC  Episode {episode_num}", _RED)
    print(f"  {_BOLD}RIGHT →{_RST}  save & end episode")
    print(f"  {_BOLD}LEFT  ←{_RST}  discard & re-record")
    print(f"  {_BOLD}ESC{_RST}      stop all recording")
    print()
    _beep(2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PIPER lerobot recording with countdown and start notification"
    )
    parser.add_argument("--repo-id",
        default=os.environ.get("REPO_ID", "your_hf_username/piper-record"))
    parser.add_argument("--task",
        default=os.environ.get("TASK", "Pick and place the cube"))
    parser.add_argument("--num-episodes", type=int,
        default=int(os.environ.get("NUM_EPISODES", "10")))
    parser.add_argument("--fps", type=int,
        default=int(os.environ.get("FPS", "15")))
    parser.add_argument("--episode-time-s", type=float,
        default=float(os.environ.get("EPISODE_TIME_S", "60")))
    parser.add_argument("--root",
        default=os.environ.get("HF_LEROBOT_HOME",
                               os.environ.get("DATASET_ROOT", "/mnt/disk2")),
        help="Parent directory; dataset stored at root/repo_id.")
    parser.add_argument("--realsense-serial",
        default=os.environ.get("REALSENSE_SERIAL", "332322070769"))
    parser.add_argument("--no-display", dest="display",
        action="store_false", default=True)
    parser.add_argument("--vcodec", default="libsvtav1")
    args = parser.parse_args()

    init_logging()

    bridge_python = os.environ.get(
        "ZED_BRIDGE_PYTHON",
        "/home/zwa0839/miniconda3/envs/zed_bridge/bin/python3.10",
    )

    # --- Hardware configs -------------------------------------------------
    robot_cfg = PiperFollowerConfig(
        id="piper_slave",
        cameras={
            "realsense": RealSenseCameraConfig(
                serial_number_or_name=args.realsense_serial,
                fps=args.fps,
                width=1280,
                height=720,
            ),
            "zed2i": ZED2iCameraConfig(
                fps=args.fps,
                width=1280,
                height=720,
                bridge_python=bridge_python,
            ),
        },
    )
    teleop_cfg = PiperLeaderConfig(id="piper_master")

    robot  = make_robot_from_config(robot_cfg)
    teleop = make_teleoperator_from_config(teleop_cfg)

    (
        teleop_action_processor,
        robot_action_processor,
        robot_observation_processor,
    ) = make_default_processors()

    # --- Dataset features -------------------------------------------------
    dataset_features = combine_feature_dicts(
        aggregate_pipeline_dataset_features(
            pipeline=teleop_action_processor,
            initial_features=create_initial_features(action=robot.action_features),
            use_videos=True,
        ),
        aggregate_pipeline_dataset_features(
            pipeline=robot_observation_processor,
            initial_features=create_initial_features(observation=robot.observation_features),
            use_videos=True,
        ),
    )

    dataset_root = Path(args.root) / args.repo_id
    dataset = LeRobotDataset.create(
        args.repo_id,
        args.fps,
        root=dataset_root,
        robot_type=robot.name,
        features=dataset_features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=4 * len(robot.cameras),
        batch_encoding_size=1,
        vcodec=args.vcodec,
        streaming_encoding=True,
        encoder_queue_maxsize=30,
        encoder_threads=2,
    )

    if args.display:
        init_rerun(session_name="piper_recording")
        _setup_rerun_blueprint()

    robot.connect()
    teleop.connect()
    _dump_cameras_info(robot, dataset_root)

    recorded_episodes = 0
    listener = None

    try:
        with VideoEncodingManager(dataset):
            while recorded_episodes < args.num_episodes:
                current_ep = recorded_episodes + 1

                # 1. Show idle status; stream cameras/joints into rerun while waiting
                if args.display:
                    _log_status(False, episode_num=current_ep, num_episodes=args.num_episodes)
                    stop_preview = threading.Event()
                    preview_thread = threading.Thread(
                        target=_preview_loop,
                        args=(robot, teleop, args.fps, stop_preview),
                        daemon=True,
                    )
                    preview_thread.start()

                if not _wait_for_start(current_ep, args.num_episodes):
                    if args.display:
                        stop_preview.set()
                        preview_thread.join(timeout=2.0)
                    break

                if args.display:
                    stop_preview.set()
                    preview_thread.join(timeout=2.0)

                # 2. Countdown, then notify + switch status to RECORDING
                _countdown()
                if args.display:
                    _log_status(True, episode_num=current_ep, num_episodes=args.num_episodes)
                _recording_start_notification(current_ep)

                # 3. Record
                listener, events = init_keyboard_listener()

                record_loop(
                    robot=robot,
                    events=events,
                    fps=args.fps,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    teleop=teleop,
                    dataset=dataset,
                    control_time_s=args.episode_time_s,
                    single_task=args.task,
                    display_data=args.display,
                )

                if not is_headless() and listener:
                    listener.stop()
                    listener = None

                # 4. Save or discard
                if events["rerecord_episode"]:
                    _banner("Episode discarded — re-recording", _CYAN)
                    dataset.clear_episode_buffer()
                    continue

                dataset.save_episode()
                recorded_episodes += 1
                _banner(f"Episode {recorded_episodes} saved  ✓", _GREEN)
                _beep(1)

                if events["stop_recording"]:
                    break

    finally:
        if not is_headless() and listener:
            listener.stop()

        dataset.finalize()

        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()

        _banner(f"Done — {recorded_episodes} episode(s) recorded", _CYAN)
        print(f"  Dataset: {dataset_root}\n")


if __name__ == "__main__":
    main()
