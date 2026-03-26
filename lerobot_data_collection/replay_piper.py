#!/usr/bin/env python3
"""
Visualize (and optionally replay on hardware) a recorded PIPER episode.

Layout:
  Top row  — RealSense camera | ZED 2i camera
  Bottom   — Actions time series | Observations time series

Usage:
    # Visualization only (no hardware)
    python lerobot_data_collection/replay_piper.py --episode 9

    # Visualize + replay on physical arm
    python lerobot_data_collection/replay_piper.py --episode 9 --enable-physical-arm
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path so `robot.*` is importable.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from torch.utils.data import DataLoader

# Dataset feature keys (from meta/info.json)
_CAMERA_KEYS = ["observation.images.realsense", "observation.images.zed2i"]
_N_JOINTS    = 7   # joint_1..6 + gripper
_ACTION_PATHS = [f"action/{i}" for i in range(_N_JOINTS)]
_STATE_PATHS  = [f"state/{i}"  for i in range(_N_JOINTS)]

# Unit conversion (must match arm_reader.py)
_RAW_TO_RAD   = math.pi / 180.0 / 1000.0
_RAD_TO_RAW   = 1.0 / _RAW_TO_RAD          # radians  → 0.001 deg
_RAW_TO_METER = 1e-6
_METER_TO_RAW = 1.0 / _RAW_TO_METER        # metres   → 0.001 mm

# AgileX PIPER hardware gripper range (piper_sdk piper_param_manager.py)
_GRIPPER_MAX_M   = 0.068                              # 68 mm
_GRIPPER_MAX_RAW = int(_GRIPPER_MAX_M * _METER_TO_RAW)  # 68_000

# Gripper torque limit (0.001 N/m, range 0-5000).
# 2000 = 2 N/m — hardware-reported max for firm grasp (piper_control wrapper).
_GRIPPER_EFFORT = 2000

# When measured gripper effort (0.001 N/m) reaches this value, lock the gripper
# width until the next open command.  200 = 0.2 N/m.
_GRIPPER_CONTACT_EFFORT = 200

# Sticky hold state: -1 = not holding; >= 0 = locked hold angle (raw 0.001 mm).
_gripper_hold_angle: int = -1


# ---------------------------------------------------------------------------
# Rerun blueprint
# ---------------------------------------------------------------------------

def _setup_blueprint() -> None:
    blueprint = rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial2DView(name="RealSense", contents=_CAMERA_KEYS[:1]),
                rrb.Spatial2DView(name="ZED 2i",    contents=_CAMERA_KEYS[1:]),
                column_shares=[1, 1],
            ),
            rrb.TimeSeriesView(name="Actions",      contents=_ACTION_PATHS),
            rrb.TimeSeriesView(name="Observations", contents=_STATE_PATHS),
            row_shares=[3, 1, 1],
        ),
        auto_layout=False,
        auto_views=False,
    )
    rr.send_blueprint(blueprint)


# ---------------------------------------------------------------------------
# Arm control helpers
# ---------------------------------------------------------------------------

def _read_binarized_gripper(dataset_root: Path) -> bool:
    """Read binarized_gripper flag from meta/info.json (False if absent)."""
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return False
    with open(info_path) as f:
        return json.load(f).get("binarized_gripper", False)


def _action_to_raw(action: np.ndarray, binarized_gripper: bool):
    """Convert dataset action (rad×6 + gripper×1) to piper SDK raw ints.

    binarized_gripper=True  — gripper is 0.0/1.0; scale to hardware range.
        0.0 → 0 raw  (closed)
        1.0 → _GRIPPER_MAX_RAW = 70_000 raw  (fully open, 70 mm)

    binarized_gripper=False — gripper is a continuous width in metres.
    """
    joints  = [int(action[i] * _RAD_TO_RAW) for i in range(6)]
    g       = float(action[6])
    gripper = int(g * _GRIPPER_MAX_RAW) if binarized_gripper else int(g * _METER_TO_RAW)
    return joints, gripper


def _send_action(piper, joints: list[int], gripper: int, speed: int) -> None:
    """Send one joint + gripper command to the physical arm."""
    global _gripper_hold_angle

    piper.MotionCtrl_2(0x01, 0x01, speed, 0x00)   # joint control mode
    piper.JointCtrl(*joints)

    # Gripper compliance: on open command, clear the hold and execute normally.
    # On close command, once effort >= _GRIPPER_CONTACT_EFFORT, lock the current
    # width and keep it until the next open command.
    is_close = gripper < _GRIPPER_MAX_RAW // 2

    if not is_close:
        _gripper_hold_angle = -1
        target_angle = gripper
    elif _gripper_hold_angle >= 0:
        target_angle = _gripper_hold_angle
    else:
        fb = piper.GetArmGripperMsgs()
        if abs(fb.gripper_state.grippers_effort) >= _GRIPPER_CONTACT_EFFORT:
            _gripper_hold_angle = fb.gripper_state.grippers_angle
            target_angle = _gripper_hold_angle
        else:
            target_angle = gripper

    # status_code=0x03: Enable + clear error so a stall fault never permanently
    # disables the gripper across cycles.
    piper.GripperCtrl(target_angle, _GRIPPER_EFFORT, 0x03, 0x00)


def _connect_arm(interface: str, channel: str) -> object:
    """Connect, enable, and return piper handle. Raises on failure."""
    from robot.arm_controller import create_piper, enable_arm, wait_for_feedback

    print(f"[replay] Connecting arm via {interface}/{channel}...")
    piper = create_piper(interface, channel)

    time.sleep(1.0)
    feedback = wait_for_feedback(piper, timeout=3.0)
    if feedback:
        print(f"[replay] Arm EEF: x={feedback['x']:.1f} y={feedback['y']:.1f} "
              f"z={feedback['z']:.1f} mm")
    else:
        print("[replay] WARNING: no EEF feedback received.")

    print("[replay] Enabling arm...")
    if not enable_arm(piper, timeout=5.0):
        raise RuntimeError("Failed to enable arm.")
    print("[replay] Arm enabled.")
    return piper


def _move_to_first_frame(piper, first_action: np.ndarray, speed: int,
                         binarized_gripper: bool = False) -> None:
    """Slowly move to the first recorded position before starting replay."""
    from robot.arm_controller import move_to_joint_waypoint

    joints, gripper = _action_to_raw(first_action, binarized_gripper)
    print("[replay] Moving to start position...")
    ok = move_to_joint_waypoint(piper, joints, speed=max(10, speed // 3), timeout=15.0)
    if not ok:
        raise RuntimeError("Failed to reach start position. Aborting replay.")
    piper.GripperCtrl(gripper, _GRIPPER_EFFORT, 0x01, 0x00)
    time.sleep(0.5)
    print("[replay] At start position.")


# ---------------------------------------------------------------------------
# Main replay / visualize loop
# ---------------------------------------------------------------------------

def run_episode(
    repo_id: str,
    root: Path,
    episode: int,
    physical_arm: bool = False,
    interface: str = "socketcan",
    channel: str = "can0",
    speed: int = 50,
) -> None:
    binarized_gripper = _read_binarized_gripper(root)

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset    = LeRobotDataset(repo_id, episodes=[episode], root=root)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    fps        = dataset.fps

    # --- Init rerun ---
    session_name = f"{repo_id.replace('/', '_')}_ep{episode}"
    rr.init(session_name)
    rr.spawn(memory_limit=os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%"))
    _setup_blueprint()

    # --- Connect arm if requested ---
    piper = None
    if physical_arm:
        piper = _connect_arm(interface, channel)
        first_action = dataset[0]["action"].numpy()
        _move_to_first_frame(piper, first_action, speed, binarized_gripper=binarized_gripper)
        print("[replay] Starting replay in 2 s...")
        time.sleep(2.0)

    first_index = dataset[0]["index"].item()
    dt = 1.0 / fps

    try:
        for batch in dataloader:
            for i in range(len(batch["index"])):
                t_start = time.time()

                frame_idx = batch["index"][i].item() - first_index
                timestamp = batch["timestamp"][i].item()
                rr.set_time("frame_index", sequence=frame_idx)
                rr.set_time("timestamp",   timestamp=timestamp)

                # Camera images — (C, H, W) float32 → (H, W, C) uint8
                for key in _CAMERA_KEYS:
                    if key not in batch:
                        continue
                    img = batch[key][i]
                    arr = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
                    rr.log(key, rr.Image(arr))

                # Actions and observations
                for dim_idx, val in enumerate(batch["action"][i]):
                    rr.log(f"action/{dim_idx}", rr.Scalars(val.item()))
                for dim_idx, val in enumerate(batch["observation.state"][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))

                # Physical arm command
                if piper is not None:
                    action = batch["action"][i].numpy()
                    joints, gripper = _action_to_raw(action, binarized_gripper)
                    _send_action(piper, joints, gripper, speed)

                # Maintain dataset FPS
                elapsed = time.time() - t_start
                wait = dt - elapsed
                if wait > 0:
                    time.sleep(wait)

    finally:
        if piper is not None:
            try:
                piper.DisconnectPort()
            except Exception:
                pass
            print("[replay] Arm disconnected.")

    print(f"[replay] Done — episode {episode}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize (and optionally replay on hardware) a PIPER episode"
    )
    parser.add_argument("--repo-id",
        default=os.environ.get("REPO_ID", "zehao_lerobot_dset/piper-record"))
    parser.add_argument("--episode", type=int,
        default=int(os.environ.get("EPISODE", "0")))
    parser.add_argument("--root",
        default=os.environ.get("DATASET_ROOT", "/mnt/disk2"),
        help="Parent directory; dataset at root/repo_id.")
    parser.add_argument("--enable-physical-arm", action="store_true",
        help="Also replay actions on the physical arm via CAN.")
    parser.add_argument("--interface", default="socketcan",
        choices=["socketcan", "gs_usb"],
        help="CAN interface type (default: socketcan).")
    parser.add_argument("--channel", default="can0",
        help="CAN channel (default: can0).")
    parser.add_argument("--speed", type=int, default=50,
        help="Arm motion speed 1-100 (default: 50).")
    args = parser.parse_args()

    dataset_root = Path(args.root) / args.repo_id
    run_episode(
        repo_id=args.repo_id,
        root=dataset_root,
        episode=args.episode,
        physical_arm=args.enable_physical_arm,
        interface=args.interface,
        channel=args.channel,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
