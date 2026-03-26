#!/bin/bash
# Train Diffusion Policy on the PIPER dataset with lerobot.
#
# Usage:
#   bash lerobot_data_collection/baselines/run_dp_train.sh
#
# Key env vars:
#   REPO_ID       Dataset repo id            (default: zehao_lerobot_dset/piper-record)
#   DATASET_ROOT  Parent dir of the dataset  (default: /mnt/disk2)
#   OUTPUT_DIR    Where to save checkpoints  (default: outputs/train/dp_piper)
#   STEPS         Training steps             (default: 100000)
#   BATCH_SIZE    Batch size                 (default: 64)
#   LR            Learning rate              (default: 1e-4)
#   WANDB         Set to 1 to enable W&B     (default: 0)
#   NUM_WORKERS   DataLoader workers         (default: 16)
#
# Gripper action (binary vs continuous) is determined by how the dataset was
# recorded (BINARIZED_GRIPPER flag in record.sh). Training reads whatever is
# in the dataset — no special handling needed here.

set -e

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HF_USER="${HF_USER:-zehao_lerobot_dset}"
REPO_ID="${REPO_ID:-${HF_USER}/piper-record}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/disk2}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/train/dp_piper}"
STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
LR="${LR:-1e-4}"
WANDB="${WANDB:-0}"
NUM_WORKERS="${NUM_WORKERS:-16}"

# Image resize: decode 1280×720 source videos at this resolution, then apply
# random crop augmentation (90% crop → 216×288). Matches official lerobot DP
# real-robot configuration (real_pusht_image / pusht_real).
RESIZE_H=240
RESIZE_W=320
CROP_RATIO=0.90   # → crop to 216×288

export HF_LEROBOT_HOME="${DATASET_ROOT}"

WANDB_ARGS="--policy.push_to_hub=false --wandb.enable=false"
if [ "${WANDB}" = "1" ]; then
    export WANDB_API_KEY=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
    WANDB_ARGS="--policy.push_to_hub=false --wandb.enable=true --wandb.project=piper_dp"
fi

# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
# input_features / output_features are intentionally omitted: lerobot infers
# them automatically from the dataset's meta/info.json.
lerobot-train \
    --policy.type=diffusion \
    \
    --dataset.repo_id="${REPO_ID}" \
    --dataset.root="${DATASET_ROOT}/${REPO_ID}" \
    --dataset.video_backend=torchcodec \
    \
    --policy.resize_shape="[${RESIZE_H}, ${RESIZE_W}]" \
    --policy.crop_ratio="${CROP_RATIO}" \
    --policy.crop_is_random=true \
    --policy.use_separate_rgb_encoder_per_camera=true \
    \
    --policy.n_obs_steps=2 \
    --policy.horizon=16 \
    --policy.n_action_steps=8 \
    \
    --policy.noise_scheduler_type=DDPM \
    --policy.num_train_timesteps=100 \
    \
    --policy.optimizer_lr="${LR}" \
    --policy.device=cuda \
    \
    --batch_size="${BATCH_SIZE}" \
    --steps="${STEPS}" \
    --num_workers="${NUM_WORKERS}" \
    --output_dir="${OUTPUT_DIR}" \
    --save_checkpoint=true \
    --save_freq=5000 \
    --log_freq=100 \
    ${WANDB_ARGS}
