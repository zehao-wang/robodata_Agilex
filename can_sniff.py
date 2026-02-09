#!/usr/bin/env python3
"""Sniff CAN bus to see which arms and signals are active.

Usage:
    python can_sniff.py              # 5 second scan, summary only
    python can_sniff.py -t 10        # 10 second scan
    python can_sniff.py --raw        # print every frame (hex)
    python can_sniff.py --decode     # print every frame fully decoded
    python can_sniff.py --decode -t 1  # quick 1s decoded dump
"""

import argparse
import platform
import struct
import time
from collections import defaultdict

# ---------------------------------------------------------------------------
# Known PIPER CAN ID map
# ---------------------------------------------------------------------------

KNOWN_IDS = {}

def _register_feedback(base, arm_name):
    mapping = {
        0x01: "arm status",
        0x02: "end pose 1 (X, Y)",
        0x03: "end pose 2 (Z, RX)",
        0x04: "end pose 3 (RY, RZ)",
        0x05: "joint 1-2",
        0x06: "joint 3-4",
        0x07: "joint 5-6",
        0x08: "gripper",
    }
    for off, desc in mapping.items():
        KNOWN_IDS[base + off] = (arm_name, desc)

_register_feedback(0x2A0, "master")
_register_feedback(0x2B0, "slave")
_register_feedback(0x2C0, "arm_3")

# control IDs
CTRL_MAP = {
    0x150: "motion ctrl 1", 0x151: "motion ctrl 2",
    0x152: "cartesian X,Y", 0x153: "cartesian Z,RX",
    0x154: "cartesian RY,RZ",
    0x155: "joint ctrl 1-2", 0x156: "joint ctrl 3-4",
    0x157: "joint ctrl 5-6", 0x158: "circular pattern",
    0x159: "gripper ctrl",
}
for cid, desc in list(CTRL_MAP.items()):
    KNOWN_IDS[cid] = ("master_ctrl", desc)
    KNOWN_IDS[cid + 0x10] = ("slave_ctrl", desc)
    KNOWN_IDS[cid + 0x20] = ("arm3_ctrl", desc)

# high/low speed driver feedback (per offset group)
for base_hi, base_lo, arm in [(0x250, 0x260, "master"),
                                (0x350, 0x360, "slave"),
                                (0x450, 0x460, "arm_3")]:
    for i in range(1, 7):
        KNOWN_IDS[base_hi + i] = (arm, f"motor {i} high-spd fb")
        KNOWN_IDS[base_lo + i] = (arm, f"motor {i} low-spd fb")

# config IDs
KNOWN_IDS[0x470] = ("config", "master-slave mode config")
KNOWN_IDS[0x471] = ("config", "motor enable/disable")
KNOWN_IDS[0x475] = ("config", "joint config")
KNOWN_IDS[0x476] = ("config", "instruction response config")
KNOWN_IDS[0x477] = ("config", "param enquiry")
KNOWN_IDS[0x47B] = ("config", "crash protection feedback")
KNOWN_IDS[0x47D] = ("config", "gripper teaching param")
KNOWN_IDS[0x47E] = ("config", "gripper teaching param fb")
KNOWN_IDS[0x121] = ("config", "light ctrl")

# velocity/acceleration feedback
for i in range(1, 7):
    KNOWN_IDS[0x480 + i] = ("master", f"joint {i} vel/acc fb")

RAW_TO_DEG = 0.001
RAW_TO_MM = 0.001

# Torque conversion factors (current -> effort)
EFFORT_FACTOR_J123 = 1.18125  # joints 1-3
EFFORT_FACTOR_J456 = 0.95844  # joints 4-6

# ---------------------------------------------------------------------------
# Arm status enums
# ---------------------------------------------------------------------------

CTRL_MODE = {
    0: "standby", 1: "CAN ctrl", 2: "teaching", 3: "ethernet",
    4: "wifi", 5: "remote", 6: "linkage teaching", 7: "offline traj",
}
ARM_STATUS = {
    0: "normal", 1: "e-stop", 2: "no solution", 3: "singularity",
    4: "angle limit", 5: "joint comm err", 6: "brake not released",
    7: "collision", 8: "teaching overspeed", 9: "joint abnormal",
    0xA: "other err", 0xB: "teaching record", 0xC: "teaching exec",
    0xD: "teaching pause", 0xE: "NTC over-temp", 0xF: "resistor over-temp",
}
MOVE_MODE = {0: "MOVE_P", 1: "MOVE_J", 2: "MOVE_L", 3: "MOVE_C", 4: "MOVE_M", 5: "MOVE_CPV"}
TEACH_STATUS = {
    0: "off", 1: "record start", 2: "record end", 3: "execute",
    4: "pause", 5: "continue", 6: "terminate", 7: "goto start",
}

# ---------------------------------------------------------------------------
# Full frame decoder
# ---------------------------------------------------------------------------

def decode_frame(cid, data):
    """Decode a CAN frame into human-readable field dict."""
    _, desc = KNOWN_IDS.get(cid, ("", ""))
    d = bytes(data)
    if len(d) < 8:
        d = d + b'\x00' * (8 - len(d))

    # --- Joint feedback ---
    if desc in ("joint 1-2", "joint 3-4", "joint 5-6"):
        v0 = struct.unpack(">i", d[0:4])[0]
        v1 = struct.unpack(">i", d[4:8])[0]
        pair = desc.split()[-1]  # "1-2" / "3-4" / "5-6"
        j_a, j_b = pair.split("-")
        return f"J{j_a}={v0 * RAW_TO_DEG:+.3f}° J{j_b}={v1 * RAW_TO_DEG:+.3f}°"

    # --- End pose ---
    if "end pose" in desc:
        v0 = struct.unpack(">i", d[0:4])[0]
        v1 = struct.unpack(">i", d[4:8])[0]
        if "(X, Y)" in desc:
            return f"X={v0 * RAW_TO_MM:+.1f}mm Y={v1 * RAW_TO_MM:+.1f}mm"
        elif "(Z, RX)" in desc:
            return f"Z={v0 * RAW_TO_MM:+.1f}mm RX={v1 * RAW_TO_DEG:+.2f}°"
        elif "(RY, RZ)" in desc:
            return f"RY={v0 * RAW_TO_DEG:+.2f}° RZ={v1 * RAW_TO_DEG:+.2f}°"

    # --- Gripper feedback ---
    if desc == "gripper":
        stroke = struct.unpack(">i", d[0:4])[0]
        torque = struct.unpack(">h", d[4:6])[0]
        status = d[6]
        enabled = "EN" if (status >> 6) & 1 else "DIS"
        homed = "homed" if (status >> 7) & 1 else "not-homed"
        flags = []
        if status & 0x01: flags.append("low-V")
        if status & 0x02: flags.append("motor-OT")
        if status & 0x04: flags.append("over-I")
        if status & 0x08: flags.append("drv-OT")
        if status & 0x10: flags.append("sensor-err")
        if status & 0x20: flags.append("drv-err")
        flag_str = " " + ",".join(flags) if flags else ""
        return (f"stroke={stroke * RAW_TO_MM:.2f}mm "
                f"torque={torque * 0.001:.3f}Nm [{enabled},{homed}]{flag_str}")

    # --- Arm status ---
    if desc == "arm status":
        ctrl = CTRL_MODE.get(d[0], f"0x{d[0]:02X}")
        status = ARM_STATUS.get(d[1], f"0x{d[1]:02X}")
        mode = MOVE_MODE.get(d[2], f"0x{d[2]:02X}")
        teach = TEACH_STATUS.get(d[3], f"0x{d[3]:02X}")
        motion = "reached" if d[4] == 0 else "moving"
        traj_pt = d[5]
        angle_err = d[6]
        comm_err = d[7]
        parts = [f"ctrl={ctrl}", f"status={status}", f"mode={mode}",
                 f"teach={teach}", motion]
        if traj_pt: parts.append(f"traj#{traj_pt}")
        if angle_err:
            lim = [f"J{i+1}" for i in range(6) if angle_err & (1 << i)]
            parts.append(f"angle_limit=[{','.join(lim)}]")
        if comm_err:
            ce = [f"J{i+1}" for i in range(6) if comm_err & (1 << i)]
            parts.append(f"comm_err=[{','.join(ce)}]")
        return " | ".join(parts)

    # --- High-speed motor feedback ---
    if "high-spd fb" in desc:
        speed = struct.unpack(">h", d[0:2])[0]   # 0.001 rad/s
        current = struct.unpack(">h", d[2:4])[0]  # 0.001 A
        position = struct.unpack(">i", d[4:8])[0]
        # Determine joint number for effort calculation
        motor_num = int(desc.split()[1])
        factor = EFFORT_FACTOR_J123 if motor_num <= 3 else EFFORT_FACTOR_J456
        effort = current * factor  # 0.001 Nm
        return (f"spd={speed * 0.001:+.3f}rad/s "
                f"cur={current * 0.001:+.3f}A "
                f"effort={effort * 0.001:+.4f}Nm "
                f"pos={position}")

    # --- Low-speed motor feedback ---
    if "low-spd fb" in desc:
        voltage = struct.unpack(">H", d[0:2])[0]   # 0.1V
        drv_temp = struct.unpack(">h", d[2:4])[0]   # 1°C
        mot_temp = struct.unpack("b", d[4:5])[0]     # 1°C
        status = d[5]
        bus_cur = struct.unpack(">H", d[6:8])[0]     # 0.001A
        enabled = "EN" if (status >> 6) & 1 else "DIS"
        flags = []
        if status & 0x01: flags.append("low-V")
        if status & 0x02: flags.append("motor-OT")
        if status & 0x04: flags.append("over-I")
        if status & 0x08: flags.append("drv-OT")
        if status & 0x10: flags.append("collision")
        if status & 0x20: flags.append("drv-err")
        if status & 0x80: flags.append("stall")
        flag_str = " " + ",".join(flags) if flags else ""
        return (f"bus={voltage * 0.1:.1f}V "
                f"drv_T={drv_temp}°C mot_T={mot_temp}°C "
                f"bus_I={bus_cur * 0.001:.3f}A [{enabled}]{flag_str}")

    # --- Motor enable/disable ---
    if desc == "motor enable/disable":
        motor = d[0]
        action = {1: "disable", 2: "enable"}.get(d[1], f"0x{d[1]:02X}")
        motor_str = f"motor {motor}" if motor < 0xFF else "ALL"
        return f"{motor_str} → {action}"

    # --- Master-slave config ---
    if desc == "master-slave mode config":
        linkage = {0: "invalid", 0xFA: "teaching-input(master)", 0xFC: "motion-output(slave)"}.get(d[0], f"0x{d[0]:02X}")
        fb_off = f"+0x{d[1]:02X}" if d[1] else "none"
        ctrl_off = f"+0x{d[2]:02X}" if d[2] else "none"
        link_off = f"+0x{d[3]:02X}" if d[3] else "none"
        return f"linkage={linkage} fb_offset={fb_off} ctrl_offset={ctrl_off} link_offset={link_off}"

    # --- Joint vel/acc feedback ---
    if "vel/acc fb" in desc:
        lin_vel = struct.unpack(">H", d[0:2])[0]
        ang_vel = struct.unpack(">H", d[2:4])[0]
        lin_acc = struct.unpack(">H", d[4:6])[0]
        ang_acc = struct.unpack(">H", d[6:8])[0]
        return (f"lin_vel={lin_vel * 0.001:.3f}m/s "
                f"ang_vel={ang_vel * 0.001:.3f}rad/s "
                f"lin_acc={lin_acc * 0.001:.3f}m/s² "
                f"ang_acc={ang_acc * 0.001:.3f}rad/s²")

    # --- Fallback: raw hex ---
    return d.hex(" ")


# ---------------------------------------------------------------------------
# CAN bus open
# ---------------------------------------------------------------------------

def _patch_macos():
    if "darwin" not in platform.system().lower():
        return
    try:
        from gs_usb.gs_usb import GsUsb
    except ImportError:
        return
    _orig = GsUsb.start
    def _patched(self, flags=None):
        orig_reset = self.gs_usb.reset
        self.gs_usb.reset = lambda: None
        try:
            _orig(self, flags) if flags is not None else _orig(self)
        finally:
            self.gs_usb.reset = orig_reset
    GsUsb.start = _patched


def open_can(interface, channel, bitrate):
    import can
    if interface == "gs_usb":
        _patch_macos()
        import usb.core
        dev = usb.core.find(idVendor=0x1D50, idProduct=0x606F)
        if dev is None:
            raise RuntimeError("CAN adapter not found. Check USB connection.")
        return can.Bus(interface="gs_usb", channel=dev.product,
                       bus=dev.bus, address=dev.address, bitrate=bitrate)
    else:
        return can.Bus(interface="socketcan", channel=channel, bitrate=bitrate)


# ---------------------------------------------------------------------------
# Sniff
# ---------------------------------------------------------------------------

def sniff(bus, duration, raw, decode):
    stats = defaultdict(lambda: {"count": 0, "first": None, "last": None, "sample": None})
    t0 = time.time()
    total = 0

    print(f"Sniffing CAN bus for {duration}s ...\n")

    try:
        while time.time() - t0 < duration:
            msg = bus.recv(timeout=0.1)
            if msg is None:
                continue
            total += 1
            cid = msg.arbitration_id
            now = time.time()

            entry = stats[cid]
            entry["count"] += 1
            if entry["first"] is None:
                entry["first"] = now
            entry["last"] = now
            entry["sample"] = bytes(msg.data)

            if decode:
                arm, desc = KNOWN_IDS.get(cid, ("???", "unknown"))
                decoded = decode_frame(cid, msg.data)
                print(f"  0x{cid:03X} [{arm:12s}] {desc:28s} | {decoded}")
            elif raw:
                arm, desc = KNOWN_IDS.get(cid, ("???", "unknown"))
                hex_data = msg.data.hex(" ")
                print(f"  0x{cid:03X}  [{arm:12s}] {desc:24s}  {hex_data}")

    except KeyboardInterrupt:
        print("\n(interrupted)")

    elapsed = time.time() - t0
    return stats, total, elapsed


def print_report(stats, total, elapsed):
    print("\n" + "=" * 80)
    print(f"  Scan result: {total} frames in {elapsed:.1f}s")
    print("=" * 80)

    if not stats:
        print("  No CAN frames received. Check connection / bitrate.")
        return

    # Group by arm
    arms = defaultdict(list)
    for cid in sorted(stats):
        arm, desc = KNOWN_IDS.get(cid, ("unknown", f"unknown (0x{cid:03X})"))
        arms[arm].append((cid, desc, stats[cid]))

    for arm in ["master", "slave", "arm_3",
                "master_ctrl", "slave_ctrl", "arm3_ctrl",
                "config", "unknown"]:
        if arm not in arms:
            continue
        entries = arms[arm]
        label = arm.upper().replace("_", " ")
        print(f"\n  [{label}]")
        for cid, desc, st in entries:
            hz = st["count"] / max(st["last"] - st["first"], 0.001) if st["count"] > 1 else 0
            decoded = decode_frame(cid, st["sample"])
            print(f"    0x{cid:03X}  {desc:28s}  {st['count']:6d} frames  ~{hz:5.0f} Hz")
            print(f"           last: {decoded}")

    # Summary
    print("\n" + "-" * 80)
    alive = []
    for arm_name in ["master", "slave", "arm_3"]:
        if arm_name in arms:
            n = sum(st["count"] for _, _, st in arms[arm_name])
            alive.append(f"{arm_name} ({n} frames)")
    if alive:
        print(f"  Active arms: {', '.join(alive)}")
    else:
        print("  No known arm feedback detected.")
    print("-" * 80)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sniff PIPER CAN bus signals")
    parser.add_argument("-t", "--time", type=float, default=5.0,
                        help="Scan duration in seconds (default: 5)")
    parser.add_argument("--interface", default="gs_usb",
                        choices=["gs_usb", "socketcan"])
    parser.add_argument("--channel", default="can0")
    parser.add_argument("--bitrate", type=int, default=1_000_000)
    parser.add_argument("--raw", action="store_true",
                        help="Print every CAN frame as raw hex")
    parser.add_argument("--decode", action="store_true",
                        help="Print every CAN frame fully decoded")
    args = parser.parse_args()

    bus = open_can(args.interface, args.channel, args.bitrate)
    try:
        stats, total, elapsed = sniff(bus, args.time, args.raw, args.decode)
        print_report(stats, total, elapsed)
    finally:
        bus.shutdown()


if __name__ == "__main__":
    main()
