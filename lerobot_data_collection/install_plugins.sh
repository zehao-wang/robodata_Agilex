#!/bin/bash
# Install the three lerobot plugin packages into the lerobot conda env.
# Run once (or after any code changes to the plugins).
#
# Usage:
#   bash lerobot_data_collection/install_plugins.sh

set -e

CONDA_ENV="lerobot"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Installing lerobot plugins into conda env: $CONDA_ENV ==="

conda run -n "$CONDA_ENV" pip install -e "$SCRIPT_DIR/lerobot_camera_zed2i"
conda run -n "$CONDA_ENV" pip install -e "$SCRIPT_DIR/lerobot_robot_piper_follower"
conda run -n "$CONDA_ENV" pip install -e "$SCRIPT_DIR/lerobot_teleoperator_piper_leader"

echo "=== Done. Verify with:"
echo "  conda run -n $CONDA_ENV pip list | grep lerobot_"
