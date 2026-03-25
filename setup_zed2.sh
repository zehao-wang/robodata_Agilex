#!/usr/bin/env bash
set -e

# ── RealSense ────────────────────────────────────────────────────────────────
echo "[1/3] Installing RealSense SDK..."
sudo mkdir -p /etc/apt/keyrings
curl -sSf https://librealsense.intel.com/Debian/librealsense.pgp \
    | sudo tee /etc/apt/keyrings/librealsense.pgp > /dev/null
echo "deb [signed-by=/etc/apt/keyrings/librealsense.pgp] https://librealsense.intel.com/Debian/apt-repo jammy main" \
    | sudo tee /etc/apt/sources.list.d/librealsense.list
sudo apt-get update
sudo apt-get install -y librealsense2-dkms librealsense2-utils librealsense2-dev
pip install pyrealsense2

# ── CAN adapter udev rule (gs_usb / candleLight) ─────────────────────────────
echo "[2/3] Setting up CAN adapter USB permissions..."
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="1d50", ATTRS{idProduct}=="606f", MODE="0666"' \
    | sudo tee /etc/udev/rules.d/99-gs-usb.rules
sudo udevadm control --reload-rules
sudo udevadm trigger
echo "  -> udev rule written. Re-plug the CAN adapter if it is already connected."

# ── Verify ───────────────────────────────────────────────────────────────────
echo "[3/4] Verifying all camera dependencies..."
python3 -c "
import pyzed.sl as sl;      print('ZED    OK')
import pyrealsense2 as rs;  print('RS     OK:', rs.__version__)
import cv2;                 print('OpenCV OK:', cv2.__version__)
"

echo "[4/4] Setup complete. Run with:"
echo "  python collect_viser.py --demo                    # GUI only, no hardware"
echo "  python collect_viser.py --camera zed2 --streams rgbd"
echo "  python collect_viser.py --camera realsense --streams rgbd"
