#!/bin/bash
# Run trained Diffusion Policy inference on the physical PIPER arm.
#
# Usage:
#   bash lerobot_data_collection/baselines/run_eval.sh
#
# Key env vars:
#   CHECKPOINT         Path to checkpoint dir        (default: outputs/train/dp_piper/checkpoints/100000)
#   REPO_ID            Dataset repo id               (default: zehao_lerobot_dset/piper-record)
#   DATASET_ROOT       Parent dir of dataset         (default: /mnt/disk2)
#   TASK               Task description              (default: Pick and place the cube)
#   FPS                Inference rate (Hz)           (default: 15)
#   MAX_STEPS          Steps before auto-stop        (default: 300  ≈ 20 s at 15 Hz)
#   SPEED              Arm motion speed 1-100        (default: 50)
#   CHANNEL            CAN channel                   (default: can0)
#   REALSENSE_SERIAL   RealSense D435i serial number (default: 332322070769)
#   ZED_BRIDGE_PYTHON  Python with pyzed + numpy<2.0 (default: zed_bridge conda env)
#   NO_DISPLAY         Set to 1 to disable rerun viz (default: 0)
#
# Prerequisites:
#   bash setup_can.sh    # bring up CAN interface

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_USER="${HF_USER:-zehao_lerobot_dset}"
REPO_ID="${REPO_ID:-${HF_USER}/piper-record}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/disk2}"
CHECKPOINT="${CHECKPOINT:-outputs/train/dp_piper/checkpoints/100000}"
TASK="${TASK:-Pick and place the cube}"
FPS="${FPS:-15}"
MAX_STEPS="${MAX_STEPS:-300}"
SPEED="${SPEED:-50}"
CHANNEL="${CHANNEL:-can0}"
REALSENSE_SERIAL="${REALSENSE_SERIAL:-332322070769}"

export HF_LEROBOT_HOME="${DATASET_ROOT}"
export ZED_BRIDGE_PYTHON="${ZED_BRIDGE_PYTHON:-/home/zwa0839/miniconda3/envs/zed_bridge/bin/python3.10}"

EXTRA_ARGS=""
if [ "${NO_DISPLAY:-0}" = "1" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --no-display"
fi

python "${SCRIPT_DIR}/eval_piper.py" \
    --checkpoint="${CHECKPOINT}" \
    --repo-id="${REPO_ID}" \
    --root="${DATASET_ROOT}" \
    --task="${TASK}" \
    --fps="${FPS}" \
    --max-steps="${MAX_STEPS}" \
    --speed="${SPEED}" \
    --channel="${CHANNEL}" \
    --realsense-serial="${REALSENSE_SERIAL}" \
    ${EXTRA_ARGS}
