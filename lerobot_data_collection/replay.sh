#!/bin/bash
# Visualize (and optionally replay on hardware) a recorded PIPER episode.
#
# Usage:
#   bash lerobot_data_collection/replay.sh                          # episode 0, viz only
#   EPISODE=3 bash lerobot_data_collection/replay.sh                # episode 3, viz only
#   EPISODE=3 PHYSICAL_ARM=1 bash lerobot_data_collection/replay.sh # replay on arm
#
# Env vars:
#   EPISODE            Episode index (default: 0)
#   PHYSICAL_ARM       Set to 1 to also command the physical arm
#   SPEED              Arm motion speed 1-100 (default: 50)
#   REPO_ID            Dataset repo id
#   DATASET_ROOT       Parent directory of the dataset (default: /mnt/disk2)
#
# Prerequisites for physical arm:
#   bash setup_can.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HF_USER="${HF_USER:-zehao_lerobot_dset}"
REPO_ID="${REPO_ID:-${HF_USER}/piper-record}"
EPISODE="${EPISODE:-0}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/disk2}"
SPEED="${SPEED:-50}"

EXTRA_ARGS=""
if [ "${PHYSICAL_ARM:-0}" = "1" ]; then
    EXTRA_ARGS="--enable-physical-arm --speed=${SPEED}"
fi

python "${SCRIPT_DIR}/replay_piper.py" \
    --repo-id="${REPO_ID}" \
    --episode="${EPISODE}" \
    --root="${DATASET_ROOT}" \
    ${EXTRA_ARGS}
