#!/usr/bin/env bash
# CAN interface initialization for Linux (socketcan)
# Not needed on macOS (uses gs_usb via python-can directly)

set -e

CAN_INTERFACE="${1:-can0}"
BITRATE="${2:-1000000}"

echo "Setting up CAN interface: $CAN_INTERFACE at ${BITRATE} bps"

# Bring down if already up
sudo ip link set "$CAN_INTERFACE" down 2>/dev/null || true

# Set bitrate and bring up
sudo ip link set "$CAN_INTERFACE" type can bitrate "$BITRATE"
sudo ip link set "$CAN_INTERFACE" up

echo "CAN interface $CAN_INTERFACE is up."
ip -details link show "$CAN_INTERFACE"
