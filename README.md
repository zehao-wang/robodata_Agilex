# Agilex PIPER Data Collection & Control System

![Python](https://img.shields.io/badge/python-3.11-blue?logo=python&logoColor=white)

Data collection and arm control system for the Agilex PIPER dual-arm setup (teaching + operation arms) via CAN bus. Captures joint angles, gripper width, and D435i RGBD video in HDF5 format compatible with ACT / Diffusion Policy frameworks.

## Hardware

- Agilex PIPER dual arms (master/teaching + slave/operation), CAN bus connected
- CAN USB adapter (candleLight firmware, gs_usb protocol)
- Intel RealSense D435i depth camera (USB-C)

## Install

Clone pyroki (IK solver) from source first (Python 3.12+; 3.10–3.11 also supported,
see [pyroki docs](https://chungmin99.github.io/pyroki/)):

```bash
git clone https://github.com/chungmin99/pyroki.git
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## World Frame Calibration

Define a custom world coordinate frame by pointing the arm at 4 corners of a physical rectangle. This frame is then used in data collection and arm control.

```bash
# With physical arm
python calibrate_world.py

# Demo mode (no hardware)
python calibrate_world.py --demo
```

**Procedure:**
1. Move the arm to the origin corner (P1) and click "Record P1 (Origin)"
2. Move to the +X direction corner (P2) and click "Record P2 (+X)"
3. Move to the opposite corner (P3) and click "Record P3 (Opposite)"
4. Move to the +Y direction corner (P4) and click "Record P4 (+Y)"
5. Click "Save Calibration"

The config is saved to `data/world_config.json` and automatically loaded by `collect_viser.py` and `control_arm.py`.

## Data Collection

```bash
# With hardware (RGB only, default)
python collect_viser.py --output_dir ./data --fps 15

# RGB + Depth
python collect_viser.py --output_dir ./data --fps 15 --streams rgbd

# macOS (gs_usb)
python collect_viser.py --output_dir ./data

# Linux (socketcan)
python collect_viser.py --output_dir ./data --can-interface socketcan --can-channel can0

# Demo mode (no hardware)
python collect_viser.py --demo

# With world frame calibration (loaded automatically if file exists)
python collect_viser.py --world-config ./data/world_config.json
```

Opens a viser web GUI at http://localhost:8080 with 3D arm visualization, camera feeds, and recording controls.

**Sidebar layout:**
1. Task Name / Instruction inputs (top)
2. Camera folder (color feed; depth only shown with `--streams rgbd` or `--streams depth`)
3. Arm State folder (EEF position, joint angles, gripper)
4. Recording folder (Start/Stop button — turns red while recording)
5. Replay folder (select and replay recorded episodes)

**Replay:** Select an episode from the dropdown and click "Replay". The arm visualization animates with recorded poses, and an OpenCV window shows the recorded video. Click "Stop Replay" or wait for it to finish — live arm input resumes automatically.

## Arm Control

```bash
# With physical arm
python control_arm.py

# GUI preview only
python control_arm.py --demo

# With world frame (input fields show world-frame coordinates)
python control_arm.py --world-config ./data/world_config.json
```

Interactive drag control with IK solving. When world frame is loaded, the sidebar input fields display world-frame coordinates while the 3D handle and IK solver work in the robot base frame.

## Data Format

Each episode is saved as an HDF5 file:

```
episode_0000.hdf5
├── observations/
│   ├── qpos          (N, 6)           float64   # Joint angles (rad)
│   ├── qvel          (N, 6)           float64   # Joint velocities
│   ├── gripper       (N, 1)           float64   # Gripper width (m)
│   ├── eef_pos       (N, 3)           float64   # EEF position (m), world or base frame
│   └── images/
│       ├── color     (N, 480, 640, 3) uint8     # RGB
│       └── depth     (N, 480, 640)    uint16    # Depth (mm)
├── action/
│   ├── qpos          (N, 6)           float64
│   └── gripper       (N, 1)           float64
├── timestamps        (N,)             float64   # Unix timestamps
├── T_base_from_world (4, 4)           float64   # World→base transform (if calibrated)
├── T_world_from_base (4, 4)           float64   # Base→world transform (if calibrated)
└── attrs:
    ├── num_frames              int
    ├── fps                     int
    ├── duration_s              float
    ├── task_name               str
    ├── instruction             str
    └── world_frame_calibrated  bool
```

## Troubleshooting

### RealSense 相机帧超时 (`Frame didn't arrive within 5000`)

D435i 需要 USB 3.0 带宽。如果日志显示 `USB: 2.1`，说明接在了 USB 2.0 口上，带宽不够会导致丢帧/超时。

- **优先**：换到 USB 3.0 端口（蓝色内芯），日志应显示 `USB: 3.2`
- **USB 2.0 降级方案**：降低帧率和分辨率
  ```bash
  python collect_viser.py --fps 15 --width 424 --height 240
  ```
- USB 2.0 下 RGBD 模式（`--streams rgbd`）基本不可用，建议只用 `--streams rgb`

### 相机启动后无 RGB 画面

RealSense 刚插入后 pipeline 需要初始化时间，可能几秒到几十秒无画面。

1. 等待约 1 分钟
2. 如果仍然没有画面，拔掉 USB，等待几秒后重新插入
3. 确认 `realsense-viewer`（Intel 官方工具）能正常显示画面后再启动采集

### macOS 权限

`gs_usb` CAN 适配器需要 root 权限：
```bash
sudo /path/to/python collect_viser.py --fps 15
```

## Project Structure

```
robodata_Agilex/
├── collect_viser.py          # Data collection entry point (viser GUI)
├── control_arm.py            # Arm control entry point (viser GUI)
├── calibrate_world.py        # World frame calibration
├── can_sniff.py              # CAN bus diagnostic tool
├── requirements.txt
├── setup_can.sh              # Linux CAN interface init
├── robot/
│   ├── arm_reader.py         # CAN protocol reader (single arm)
│   ├── dual_arm_reader.py    # Dual arm reader (master+slave)
│   └── arm_controller.py     # Arm control via piper_sdk
├── camera/
│   └── realsense.py          # D435i RGBD capture
├── storage/
│   └── hdf5_writer.py        # HDF5 episode writer
├── gui/
│   ├── viser_collector.py    # Viser data collection app
│   └── arm_control_app.py    # Viser arm control app
├── solver/
│   └── pyroki_ik.py          # IK solver (pyroki)
├── utils/
│   ├── arm_visualizer.py     # FK + matplotlib arm rendering
│   ├── urdf_loader.py        # URDF loading + joint mapping
│   └── world_frame.py        # World frame calibration & transforms
├── assets/
│   └── piper_description/    # URDF + meshes
└── data/
    └── world_config.json     # World frame calibration (auto-generated)
```
