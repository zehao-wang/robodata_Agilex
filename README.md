# Agilex PIPER 数据采集系统

为 Agilex PIPER 双臂（示教 + 操作端）构建的数据采集系统。同时采集操作端的关节角度、夹爪宽度和 D435i RGBD 视频流，以 HDF5 格式存储，兼容 ACT / Diffusion Policy 等主流模仿学习框架。

## 硬件要求

- Agilex PIPER 双臂（示教端 + 操作端），通过 CAN 总线连接
- CAN USB 适配器（candleLight 固件，gs_usb 协议）
- Intel RealSense D435i 深度相机（USB-C）

## 安装

```bash
pip install -r requirements.txt
```

## 使用

### macOS（gs_usb 模式）

插入 CAN USB 适配器和 D435i 相机后直接运行：

```bash
python collect.py --output_dir ./data
```

### Linux（socketcan 模式）

```bash
# 初始化 CAN 接口
bash setup_can.sh can0 1000000

# 运行采集
python collect.py --output_dir ./data --can-interface socketcan --can-channel can0
```

### 操作方式

- **空格键** - 开始/停止录制一个 episode
- **q** - 退出程序

每次按空格开始录制，再按空格停止并自动保存为 `episode_XXXX.hdf5`。

### 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--output_dir` | `./data` | 保存目录 |
| `--can-interface` | `gs_usb` | CAN 接口类型 (`gs_usb` / `socketcan`) |
| `--can-channel` | `can0` | CAN 通道名（socketcan 模式） |
| `--bitrate` | `1000000` | CAN 波特率 |
| `--width` | `640` | 图像宽度 |
| `--height` | `480` | 图像高度 |
| `--fps` | `30` | 采集帧率 |

## 数据格式

每个 episode 一个 HDF5 文件：

```
episode_0000.hdf5
├── observations/
│   ├── qpos          (N, 6)          float64   # 关节角度 (rad)
│   ├── qvel          (N, 6)          float64   # 关节速度
│   ├── gripper       (N, 1)          float64   # 夹爪宽度 (m)
│   └── images/
│       ├── color     (N, 480, 640, 3) uint8    # RGB 图像
│       └── depth     (N, 480, 640)    uint16   # 深度图 (mm)
├── action/
│   ├── qpos          (N, 6)          float64   # 同 observations（示教模式）
│   └── gripper       (N, 1)          float64
├── timestamps        (N,)            float64   # Unix 时间戳
└── attrs:
    ├── task_name     str                        # 任务名称
    └── instruction   str                        # 自然语言指令
```

## 验证数据

```python
import h5py

f = h5py.File("data/episode_0000.hdf5", "r")
print("Keys:", list(f.keys()))
print("Frames:", f.attrs["num_frames"])
print("Duration:", f.attrs["duration_s"], "s")
print("Task:", f.attrs["task_name"])
print("Instruction:", f.attrs["instruction"])
print("qpos shape:", f["observations/qpos"].shape)
print("color shape:", f["observations/images/color"].shape)
f.close()
```

## 项目结构

```
robodata_Agilex/
├── collect.py             # 主采集脚本（入口）
├── requirements.txt       # Python 依赖
├── setup_can.sh           # Linux CAN 接口初始化
├── robot/
│   └── arm_reader.py      # PIPER CAN 协议解析
├── camera/
│   └── realsense.py       # D435i RGBD 采集
├── storage/
│   └── hdf5_writer.py     # HDF5 数据写入
└── utils/
    ├── keyboard.py         # 键盘事件监听（备用）
    └── annotation_dialog.py # 录制标注对话框（tkinter）
```
