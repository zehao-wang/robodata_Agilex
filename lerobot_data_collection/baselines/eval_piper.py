#!/usr/bin/env python3
"""
Run policy inference on the physical PIPER arm.

Loads a trained checkpoint, connects the robot, and runs a closed-loop
inference episode until the user presses ESC or max_steps is reached.
The arm is safely stopped and disconnected on exit.

Usage:
    python lerobot_data_collection/baselines/eval_piper.py \\
        --checkpoint outputs/train/dp_piper/checkpoints/100000 \\
        --repo-id zehao_lerobot_dset/piper-record

    # No display (headless):
    python ... --no-display
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))  # robodata_Agilex/

# Avoid shadowing installed plugins with the source directories on sys.path.
_LEROBOT_DC = os.path.join(_PROJECT_ROOT, "lerobot_data_collection")
for _p in [_LEROBOT_DC, _HERE]:
    if _p in sys.path:
        sys.path.remove(_p)

if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# Register third-party plugins (ZED 2i, PiperFollower, PiperLeader) before
# any lerobot import that touches the plugin registry.
from lerobot.utils.import_utils import register_third_party_plugins  # noqa: E402

register_third_party_plugins()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: E402
from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402
from lerobot.datasets.utils import build_dataset_frame  # noqa: E402
from lerobot.policies.pretrained import PreTrainedPolicy  # noqa: E402
from lerobot.policies.factory import make_pre_post_processors  # noqa: E402
from lerobot.processor import make_default_processors  # noqa: E402
from lerobot.robots import make_robot_from_config  # noqa: E402
from lerobot.utils.control_utils import init_keyboard_listener, is_headless, predict_action  # noqa: E402
from lerobot.utils.torch_utils import get_safe_torch_device  # noqa: E402
from lerobot.utils.utils import init_logging  # noqa: E402
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data  # noqa: E402
from lerobot_camera_zed2i import ZED2iCameraConfig  # noqa: E402
from lerobot_robot_piper_follower import PiperFollowerConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Unit conversion (must match arm_reader.py / replay_piper.py)
# ---------------------------------------------------------------------------
_RAW_TO_RAD = math.pi / 180.0 / 1000.0
_RAD_TO_RAW = 1.0 / _RAW_TO_RAD      # radians  → 0.001 deg
_RAW_TO_METER = 1e-6
_METER_TO_RAW = 1.0 / _RAW_TO_METER  # metres   → 0.001 mm

_GRIPPER_EFFORT = 1000   # 0.001 N/m, range 0–5000

# Dataset keys used as observations
_OBS_STATE_KEY = "observation.state"
_CAM_KEYS = ["observation.images.realsense", "observation.images.zed2i"]


# ---------------------------------------------------------------------------
# Rerun blueprint (same layout as record_piper.py)
# ---------------------------------------------------------------------------

def _setup_blueprint() -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    _action_paths = [f"action.joint_{i+1}.pos" for i in range(6)] + ["action.gripper.pos"]
    _state_paths  = [f"observation.joint_{i+1}.pos" for i in range(6)] + ["observation.gripper.pos"]

    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(name="RealSense", contents=["observation.realsense"]),
                rrb.Spatial2DView(name="ZED 2i",    contents=["observation.zed2i"]),
                rrb.TextDocumentView(name="Status", contents=["status"]),
                column_shares=[4, 4, 1],
            ),
            rrb.TimeSeriesView(name="Actions",      contents=_action_paths),
            rrb.TimeSeriesView(name="Observations", contents=_state_paths),
            row_shares=[3, 1, 1],
        ),
        auto_layout=False,
        auto_views=False,
    )
    rr.send_blueprint(blueprint)


def _log_status(step: int, max_steps: int) -> None:
    import rerun as rr
    md = (
        f"# 🟢 EVALUATING\n\n"
        f"Step **{step}** / {max_steps}\n\n"
        f"`ESC` to stop"
    )
    rr.log("status", rr.TextDocument(md, media_type=rr.MediaType.MARKDOWN))


# ---------------------------------------------------------------------------
# Arm helpers
# ---------------------------------------------------------------------------

def _action_to_raw(action: np.ndarray):
    """Convert 7-dim dataset action (rad×6 + m×1) → piper SDK raw ints."""
    joints  = [int(action[i] * _RAD_TO_RAW) for i in range(6)]
    gripper = int(action[6] * _METER_TO_RAW)
    return joints, gripper


def _send_action(piper, joints: list[int], gripper: int, speed: int) -> None:
    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)   # joint control mode
    piper.JointCtrl(*joints)
    piper.GripperCtrl(gripper, _GRIPPER_EFFORT, 0x01, 0x00)


def _connect_and_enable_arm(channel: str) -> object:
    from robot.arm_controller import create_piper, enable_arm, wait_for_feedback

    print(f"[eval] Connecting arm on {channel}...")
    piper = create_piper("socketcan", channel)
    time.sleep(1.0)

    feedback = wait_for_feedback(piper, timeout=3.0)
    if feedback:
        print(f"[eval] EEF: x={feedback['x']:.1f} y={feedback['y']:.1f} z={feedback['z']:.1f} mm")
    else:
        print("[eval] WARNING: no EEF feedback received.")

    if not enable_arm(piper, timeout=5.0):
        raise RuntimeError("Failed to enable arm.")
    print("[eval] Arm enabled.")
    return piper


def _safe_stop_arm(piper) -> None:
    """Stop motion and disable arm without crashing."""
    try:
        piper.MotionCtrl_2(0x01, 0x01, 0, 0x00)  # speed = 0 → hold
    except Exception:
        pass
    try:
        piper.DisablePiper()
    except Exception:
        pass
    try:
        piper.DisconnectPort()
    except Exception:
        pass
    print("[eval] Arm stopped and disconnected.")


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_eval(
    checkpoint: Path,
    repo_id: str,
    dataset_root: Path,
    task: str,
    fps: int,
    max_steps: int,
    speed: int,
    channel: str,
    display: bool,
    realsense_serial: str,
    bridge_python: str,
) -> None:
    # --- Load dataset for features (episodes=[0] → minimal I/O) ---
    dataset = LeRobotDataset(repo_id, root=dataset_root, episodes=[0])
    ds_features = dataset.features

    # --- Load policy ---
    print(f"[eval] Loading policy from {checkpoint} ...")
    policy = PreTrainedPolicy.from_pretrained(str(checkpoint))
    device  = get_safe_torch_device(policy.config.device)
    policy  = policy.to(device)
    policy.eval()

    # Training uses random crop augmentation; eval always uses center crop.
    policy.config.crop_is_random = False

    # --- Processors ---
    (
        _teleop_action_processor,
        _robot_action_processor,
        robot_observation_processor,
    ) = make_default_processors()

    # Build policy-side pre/post processors from the (now-patched) config.
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(checkpoint),
    )

    # --- Hardware ---
    robot_cfg = PiperFollowerConfig(
        id="piper_slave",
        cameras={
            "realsense": RealSenseCameraConfig(
                serial_number_or_name=realsense_serial,
                fps=fps,
                width=1280,
                height=720,
            ),
            "zed2i": ZED2iCameraConfig(
                fps=fps,
                width=1280,
                height=720,
                bridge_python=bridge_python,
            ),
        },
    )
    robot = make_robot_from_config(robot_cfg)
    robot.connect()
    print("[eval] Robot connected.")

    piper = _connect_and_enable_arm(channel)

    # --- Rerun ---
    if display:
        init_rerun(session_name="piper_eval")
        _setup_blueprint()

    listener, events = init_keyboard_listener()

    policy.reset()
    preprocessor.reset()
    postprocessor.reset()

    dt = 1.0 / fps
    step = 0
    print(f"[eval] Starting inference — max {max_steps} steps, {fps} Hz. Press ESC to stop.")

    try:
        while step < max_steps:
            t_start = time.perf_counter()

            if events.get("exit_early"):
                print("[eval] ESC pressed — stopping.")
                break

            # Observation
            obs_raw = robot.get_observation()
            obs_processed = robot_observation_processor(obs_raw)

            # Build dataset-format frame expected by policy
            obs_frame = build_dataset_frame(ds_features, obs_processed, prefix="observation")

            # Inference
            action_tensor = predict_action(
                observation=obs_frame,
                policy=policy,
                device=device,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=getattr(policy.config, "use_amp", False),
                task=task,
                robot_type=robot.robot_type,
            )

            # action_tensor may be [1, action_dim] or [action_dim] depending on postprocessor
            action_np = action_tensor.squeeze().cpu().numpy()  # → (action_dim,)
            joints, gripper = _action_to_raw(action_np)
            _send_action(piper, joints, gripper, speed)

            step += 1

            if display:
                import rerun as rr
                rr.set_time("step", sequence=step)
                _log_status(step, max_steps)
                # Build named action dict for rerun (same key format as obs_processed)
                joint_names = [f"joint_{i+1}.pos" for i in range(6)] + ["gripper.pos"]
                action_named = {name: float(action_np[i]) for i, name in enumerate(joint_names)}
                log_rerun_data(observation=obs_processed, action=action_named)

            elapsed = time.perf_counter() - t_start
            wait = dt - elapsed
            if wait > 0:
                time.sleep(wait)
            elif wait < -0.005:
                import logging
                logging.warning(
                    f"[eval] Loop overrun by {-wait*1000:.1f} ms at step {step}. "
                    "Policy inference may be too slow for target FPS."
                )

    finally:
        if not is_headless() and listener:
            listener.stop()

        _safe_stop_arm(piper)

        if robot.is_connected:
            robot.disconnect()

        print(f"[eval] Done — {step} steps executed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained policy inference on the physical PIPER arm."
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to a checkpoint directory (e.g. outputs/train/dp_piper/checkpoints/100000).",
    )
    parser.add_argument(
        "--repo-id",
        default=os.environ.get("REPO_ID", "zehao_lerobot_dset/piper-record"),
        help="Dataset repo-id (for feature schema).",
    )
    parser.add_argument(
        "--root",
        default=os.environ.get("DATASET_ROOT", "/mnt/disk2"),
        help="Parent directory; dataset at root/repo_id.",
    )
    parser.add_argument(
        "--task",
        default=os.environ.get("TASK", "Pick and place the cube"),
        help="Task description passed to the policy.",
    )
    parser.add_argument("--fps",       type=int, default=int(os.environ.get("FPS", "15")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "300")),
                        help="Stop after this many inference steps (default: 300 = 20 s at 15 Hz).")
    parser.add_argument("--speed",     type=int, default=int(os.environ.get("SPEED", "50")),
                        help="Arm motion speed 1-100 (default: 50).")
    parser.add_argument("--channel",   default="can0", help="CAN channel (default: can0).")
    parser.add_argument("--no-display", dest="display", action="store_false", default=True)
    parser.add_argument(
        "--realsense-serial",
        default=os.environ.get("REALSENSE_SERIAL", "332322070769"),
    )
    args = parser.parse_args()

    init_logging()

    bridge_python = os.environ.get(
        "ZED_BRIDGE_PYTHON",
        "/home/zwa0839/miniconda3/envs/zed_bridge/bin/python3.10",
    )

    dataset_root = Path(args.root) / args.repo_id

    run_eval(
        checkpoint=Path(args.checkpoint),
        repo_id=args.repo_id,
        dataset_root=dataset_root,
        task=args.task,
        fps=args.fps,
        max_steps=args.max_steps,
        speed=args.speed,
        channel=args.channel,
        display=args.display,
        realsense_serial=args.realsense_serial,
        bridge_python=bridge_python,
    )


if __name__ == "__main__":
    main()
