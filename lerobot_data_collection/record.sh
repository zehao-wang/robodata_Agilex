#!/bin/bash
# Record a dataset with PIPER master-slave arms + RealSense D435i + ZED 2i.
#
# Prerequisites:
#   1. bash setup_can.sh          (run once per boot to bring up can0)
#   2. bash lerobot_data_collection/install_plugins.sh  (run once to install plugins)
#
# Usage:
#   bash lerobot_data_collection/record.sh
#
# Key env vars:
#   HF_USER           HuggingFace username
#   ZED_BRIDGE_PYTHON Path to Python with pyzed + numpy<2.0
#   COUNTDOWN_S       Seconds to count down before recording starts (default 3)
#
# Episode controls:
#   ENTER       — arm and start recording
#   RIGHT ARROW — end episode and save
#   LEFT ARROW  — discard episode (re-record same number)
#   ESC         — save current episode and stop all recording

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Configuration — edit these before recording
# ---------------------------------------------------------------------------

HF_USER="${HF_USER:-zehao_lerobot_dset}"
TASK="${TASK:-Pick and place the cube}"
NUM_EPISODES="${NUM_EPISODES:-10}"
FPS="${FPS:-15}"
EPISODE_TIME_S="${EPISODE_TIME_S:-60}"
REPO_ID="${REPO_ID:-${HF_USER}/piper-record}"

# Dataset root on the large disk.
# Datasets are stored at $DATASET_ROOT/<repo_id>.
DATASET_ROOT="${DATASET_ROOT:-/mnt/disk2}"

# ZED 2i bridge Python (conda env with pyzed + numpy<2.0)
export ZED_BRIDGE_PYTHON="${ZED_BRIDGE_PYTHON:-/home/zwa0839/miniconda3/envs/zed_bridge/bin/python3.10}"

# RealSense D435i serial number (from tests/load_realsense_cam.py)
REALSENSE_SERIAL="${REALSENSE_SERIAL:-332322070769}"

# ---------------------------------------------------------------------------
# Run custom record CLI with countdown + start notification
# ---------------------------------------------------------------------------
python "${SCRIPT_DIR}/record_piper.py" \
    --repo-id="${REPO_ID}" \
    --task="${TASK}" \
    --num-episodes="${NUM_EPISODES}" \
    --fps="${FPS}" \
    --episode-time-s="${EPISODE_TIME_S}" \
    --root="${DATASET_ROOT}" \
    --realsense-serial="${REALSENSE_SERIAL}"
