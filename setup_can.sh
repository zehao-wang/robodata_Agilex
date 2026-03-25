#!/usr/bin/env bash
# CAN interface initialization for Linux (socketcan + gs_usb candleLight adapter)

set -e

CAN_INTERFACE="${1:-can0}"
BITRATE="${2:-1000000}"

echo "Setting up CAN interface: $CAN_INTERFACE at ${BITRATE} bps"

# --- Step 1: load gs_usb kernel module if not already loaded ---
if ! lsmod | grep -q "^gs_usb"; then
    echo "Loading gs_usb kernel module..."
    sudo modprobe gs_usb
fi

# --- Step 2: bind gs_usb driver to candleLight adapter if can0 doesn't exist ---
if ! ip link show "$CAN_INTERFACE" &>/dev/null; then
    echo "Binding gs_usb driver to candleLight adapter..."
    SYSFS_IFACE=$(grep -rl "1d50" /sys/bus/usb/devices/*/idVendor 2>/dev/null \
        | sed 's|/idVendor||' \
        | head -1)
    if [ -z "$SYSFS_IFACE" ]; then
        echo "ERROR: candleLight USB adapter (1d50:606f) not found. Is it plugged in?"
        exit 1
    fi
    DEVICE=$(basename "$SYSFS_IFACE")
    BIND_IFACE="${DEVICE}:1.0"
    if [ ! -e "/sys/bus/usb/drivers/gs_usb/${BIND_IFACE}" ]; then
        echo "Binding $BIND_IFACE to gs_usb..."
        sudo sh -c "echo '${BIND_IFACE}' > /sys/bus/usb/drivers/gs_usb/bind"
        sleep 0.5
    fi
fi

# --- Step 3: verify can0 now exists ---
if ! ip link show "$CAN_INTERFACE" &>/dev/null; then
    echo "ERROR: $CAN_INTERFACE still not found after binding. Check dmesg for errors."
    exit 1
fi

# --- Step 4: configure bitrate and bring up ---
sudo ip link set "$CAN_INTERFACE" down 2>/dev/null || true
sudo ip link set "$CAN_INTERFACE" type can bitrate "$BITRATE"
sudo ip link set "$CAN_INTERFACE" up

echo "CAN interface $CAN_INTERFACE is up."
ip -details link show "$CAN_INTERFACE"
