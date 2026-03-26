# LeRobot Data Collection for AgileX PIPER

Records datasets in [lerobot 0.5.0](https://huggingface.co/docs/lerobot) format using a custom recording CLI (`record_piper.py`). Three plugin packages register the PIPER arms and ZED 2i camera with lerobot's discovery system.

## Quick Start

```bash
# One-time: install the three lerobot plugins into the lerobot conda env
bash lerobot_data_collection/install_plugins.sh

# Once per boot: bring up CAN interface
bash setup_can.sh

# Record episodes
bash lerobot_data_collection/record.sh
```

Override defaults with env vars:

```bash
TASK="Grab the black cube" NUM_EPISODES=25 REPO_ID="myuser/piper-grab" \
    bash lerobot_data_collection/record.sh
```

## Recording

`record.sh` calls `record_piper.py`, a custom CLI that adds per-episode start control and rerun visualization.

### Episode flow

1. Script connects hardware and opens rerun viewer.
2. Press **ENTER** to arm an episode — a 3-second countdown runs.
3. A beep + red banner signals recording start.
4. During recording:
   - **RIGHT →** — save episode and continue
   - **LEFT ←** — discard episode (re-record same number)
   - **ESC** — save current episode and stop
5. Return to step 2 for the next episode.

### Rerun layout

| Panel | Content |
|---|---|
| Top-left | RealSense D435i live feed |
| Top-center | ZED 2i live feed |
| Top-right | Status indicator (🟢 IDLE / 🔴 RECORDING) |
| Middle | Actions time series (7 channels) |
| Bottom | Observations time series (7 channels) |

Camera and joint data are streamed live even before recording starts (idle preview).

### Key env vars

| Var | Default | Description |
|---|---|---|
| `TASK` | `Pick and place the cube` | Task description stored in dataset |
| `NUM_EPISODES` | `10` | Number of episodes to record |
| `FPS` | `15` | Capture frame rate |
| `EPISODE_TIME_S` | `60` | Max recording time per episode (seconds) |
| `REPO_ID` | `{HF_USER}/piper-record` | Dataset identifier |
| `DATASET_ROOT` | `/mnt/disk2` | Parent directory; dataset stored at `DATASET_ROOT/REPO_ID` |
| `REALSENSE_SERIAL` | `332322070769` | RealSense D435i serial number |
| `ZED_BRIDGE_PYTHON` | `…/envs/zed_bridge/bin/python3.10` | Python with pyzed + numpy<2.0 |

## Replay & Visualization

```bash
# Visualize episode 0 in rerun (no hardware needed)
bash lerobot_data_collection/replay.sh

# Visualize a specific episode
EPISODE=3 bash lerobot_data_collection/replay.sh

# Visualize + replay on physical arm
EPISODE=3 PHYSICAL_ARM=1 bash lerobot_data_collection/replay.sh

# Slower speed for safety
EPISODE=3 PHYSICAL_ARM=1 SPEED=20 bash lerobot_data_collection/replay.sh
```

The replay viewer uses the same rerun layout as recording (cameras top, time series bottom).

When `PHYSICAL_ARM=1`:
- The arm moves slowly to the first recorded position before playback.
- Actions are replayed at the dataset FPS (15 Hz) via `JointCtrl` + `GripperCtrl`.
- Requires `bash setup_can.sh` first.

## Camera Info

After each recording session, camera intrinsics are saved to:

```
{DATASET_ROOT}/{REPO_ID}/meta/cameras_info.json
```

Includes fx, fy, cx, cy, FOV, distortion coefficients, resolution, and (for ZED 2i) stereo baseline.

## Plugin Architecture

Three pip-installable packages are registered with lerobot's `register_third_party_plugins()` discovery mechanism:

| Package | Type key | Class | Role |
|---|---|---|---|
| `lerobot_robot_piper_follower` | `piper_follower` | `PiperFollower` | Reads slave arm joints + cameras |
| `lerobot_teleoperator_piper_leader` | `piper_slave_echo` | `PiperLeader` | Echoes slave arm joints as actions |
| `lerobot_camera_zed2i` | `zed2i` | `ZED2iCamera` | Wraps `camera/zed2.py` subprocess bridge |

### Master-slave design

The hardware master-slave link forwards master arm motion to the slave arm automatically — no software command forwarding needed.

- **Robot** (`PiperFollower`): reads slave arm joint positions (CAN IDs `0x2A5`–`0x2A8`) + camera images. `send_action()` is a no-op.
- **Teleop** (`PiperLeader` / type `piper_slave_echo`): opens a second socketcan socket on `can0` and reads the same slave CAN feedback frames, returning them as dataset actions.

The master arm is **not** read. Master and slave have different kinematics — master joint angles are not in the slave's joint space. `action[t] = observation.state[t]` (both from slave).

### Dataset keys

| Key | Description | Unit |
|---|---|---|
| `observation.state` | Slave arm joint positions (7-dim) | radians × 6, metres × 1 |
| `action` | Slave arm joint targets (7-dim) | radians × 6, metres × 1 |
| `observation.images.realsense` | RealSense D435i color frame (1280×720) | image |
| `observation.images.zed2i` | ZED 2i color frame (1280×720) | image |

### ZED 2i camera

The ZED 2i plugin delegates to `camera/zed2.py` which runs a subprocess bridge under a separate Python interpreter (pyzed requires numpy<2.0). Set the bridge interpreter via:

```bash
export ZED_BRIDGE_PYTHON=/home/zwa0839/miniconda3/envs/zed_bridge/bin/python3.10
```

## File Structure

```
lerobot_data_collection/
├── install_plugins.sh                        # pip install -e all three packages
├── record.sh                                 # recording entry point
├── record_piper.py                           # custom recording CLI
├── replay.sh                                 # replay/visualization entry point
├── replay_piper.py                           # replay CLI (rerun + optional arm control)
├── lerobot_robot_piper_follower/
│   ├── pyproject.toml
│   └── lerobot_robot_piper_follower/
│       ├── config_piper_follower.py          # PiperFollowerConfig (type: piper_follower)
│       └── piper_follower.py                 # PiperFollower(Robot)
├── lerobot_teleoperator_piper_leader/
│   ├── pyproject.toml
│   └── lerobot_teleoperator_piper_leader/
│       ├── config_piper_leader.py            # PiperLeaderConfig (type: piper_slave_echo)
│       └── piper_leader.py                   # PiperLeader(Teleoperator)
└── lerobot_camera_zed2i/
    ├── pyproject.toml
    └── lerobot_camera_zed2i/
        ├── config_zed2i.py                   # ZED2iCameraConfig (type: zed2i)
        └── zed2i.py                          # ZED2iCamera(Camera)
```
