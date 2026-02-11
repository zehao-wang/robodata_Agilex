# Changelog

## 2025-02-11 — Recording UI Improvements + Replay Feature

### New Features

- **Recording button color toggle**: Button turns red during recording, blue when idle, making recording state visually obvious.
- **Sidebar layout**: Task Name and Instruction inputs moved to the top of the sidebar (above Camera/Arm State folders) for easier access.
- **RGB-only mode**: Depth display panel is hidden when `--streams rgb` (default). Only shown when `--streams depth` or `--streams rgbd`.
- **Episode replay**: New "Replay" folder in sidebar with episode dropdown and Replay/Stop buttons. Loads recorded HDF5 episode and plays back:
  - Arm poses animate in the 3D viser visualization
  - Recorded color frames shown in an OpenCV window
  - Sidebar arm state updates with recorded joint angles
  - After replay ends, automatically resumes live arm input and closes the OpenCV window

### Bug Fixes

- **Replay-to-live transition delay on macOS**: After replay ended, the app would freeze for several seconds before resuming live mode. Root causes:
  1. `cv2.destroyAllWindows()` was called from viser's button callback thread. On macOS, OpenCV GUI operations must run on the main thread — calling from a background thread causes a long block.
  2. Freeing the large in-memory color array (`N x H x W x 3` uint8) triggered a GC pause simultaneously with the OpenCV cleanup.
  - **Fix**: The "Stop Replay" button now sets a flag (`_replay_stop_requested`). The main loop detects this flag and performs cleanup on the main thread: explicitly `del`s the replay data first, then calls `cv2.destroyAllWindows()` + `cv2.waitKey(1)` to flush the macOS event queue. Transition is now near-instant.

- **Button label not updating**: viser v1.0+ uses `.label` (not `.name`) to change button text. Fixed `_start_recording` / `_stop_recording` to use `self._record_btn.label`.

- **RealSense D435i frame timeout (`Frame didn't arrive within 5000`)**: 摄像头连接在 USB 2.0 端口（日志显示 `USB: 2.1`）时带宽不足，高分辨率/高帧率下会频繁超时。解决方案：使用 USB 3.0 端口（日志应显示 `USB: 3.2`），或降低帧率 `--fps 15` / 降低分辨率 `--width 424 --height 240`。

### Changes to `collect_viser.py`

- Now passes `streams` and `output_dir` arguments to `ViserDataCollectorApp` so the app can conditionally show/hide depth and locate episode files for replay.

### Known Issues & Troubleshooting

- **USB 2.0 带宽不足**: D435i 在 USB 2.0 下只能跑低帧率（~15fps RGB）。RGBD 模式几乎无法使用。务必接 USB 3.0 口（蓝色内芯）。
- **相机启动时无 RGB 画面**: RealSense 刚插入后 pipeline 可能需要一段时间初始化。等待约 1 分钟，如果仍然没有画面，拔掉 USB 等待几秒后重新插入。
- **macOS 权限**: 使用 `gs_usb` CAN 适配器需要 `sudo` 运行 `collect_viser.py`。
