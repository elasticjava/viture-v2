#!/usr/bin/env python3
"""Viture Protokoll V2 direkt ueber hidraw - ohne Vendor-SDK.

Rahmenformat (alles little-endian), aus usbmon-Mitschnitt gegen SDK-Log verifiziert:
    [0..1] 0x0010     Praeambel
    [2..3] MsgID
    [4..5] PayloadLen
    [6..7] Checksum = Summe der Payload-Bytes
    [8..]  Payload

    0x0301 IMU-Steuerung, Payload [mode, frequency]; [0,0] stoppt
    0x2301 ACK auf 0x0301, Payload[0] = Status (0 = ok)
    0x7308 IMU-Pose, Payload 24 B: u32 seq, u32 timestamp, f32 qw qx qy qz
"""
import math
import os
import select
import struct
import sys
import time

DEV = "/dev/hidraw0"
MSG_IMU_CTRL = 0x0301
MSG_IMU_ACK = 0x2301
MSG_IMU_POSE = 0x7308

MODE_RAW, MODE_POSE = 0, 1
FREQ = {60: 0, 90: 1, 120: 2, 240: 3, 500: 4, 1000: 5}


def frame(msg_id: int, payload: bytes) -> bytes:
    return struct.pack("<HHHH", 0x0010, msg_id, len(payload), sum(payload) & 0xFFFF) + payload


def parse(buf: bytes):
    if len(buf) < 8:
        return None
    pre, msg_id, plen, cks = struct.unpack_from("<HHHH", buf, 0)
    if pre != 0x0010 or 8 + plen > len(buf):
        return None
    payload = buf[8:8 + plen]
    return msg_id, payload, (sum(payload) & 0xFFFF) == cks


def quat_to_euler(w, x, y, z):
    """Roll/Pitch/Yaw in Grad (ZYX-Konvention)."""
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(s)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return tuple(math.degrees(v) for v in (roll, pitch, yaw))


def main():
    mode = MODE_RAW if "--raw" in sys.argv else MODE_POSE
    hz = 120
    fd = os.open(DEV, os.O_RDWR | os.O_NONBLOCK)
    try:
        cmd = frame(MSG_IMU_CTRL, bytes([mode, FREQ[hz]]))
        print(f"TX  {cmd.hex(' ')}   (IMU an, mode={mode}, {hz} Hz)")
        os.write(fd, b"\x00" + cmd)

        counts, first, last_seq = {}, {}, None
        t0 = time.time()
        n_pose = 0
        while time.time() - t0 < 5.0:
            r, _, _ = select.select([fd], [], [], 0.5)
            if not r:
                continue
            got = parse(os.read(fd, 128))
            if not got:
                continue
            msg_id, payload, ok = got
            counts[msg_id] = counts.get(msg_id, 0) + 1
            if msg_id not in first:
                first[msg_id] = (payload, ok)
                print(f"RX  0x{msg_id:04X} len={len(payload):3d} crc={'ok' if ok else 'BAD'}  {payload[:24].hex(' ')}")
            if msg_id == MSG_IMU_POSE and len(payload) >= 24:
                seq, ts, qw, qx, qy, qz = struct.unpack("<IIffff", payload[:24])
                n_pose += 1
                if n_pose <= 3 or n_pose % 100 == 0:
                    r_, p_, y_ = quat_to_euler(qw, qx, qy, qz)
                    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
                    print(f"    seq={seq:<6d} ts={ts:<10d} |q|={norm:.4f}  "
                          f"roll={r_:7.2f} pitch={p_:7.2f} yaw={y_:7.2f}")
                last_seq = seq

        dt = time.time() - t0
        print(f"\n--- {dt:.1f}s: " + ", ".join(f"0x{k:04X}={v}" for k, v in sorted(counts.items())))
        if n_pose:
            print(f"--- effektive Rate: {n_pose / dt:.0f} Hz, letzte seq={last_seq}")
        else:
            print("--- keine Pose-Pakete")
    finally:
        try:
            os.write(fd, b"\x00" + frame(MSG_IMU_CTRL, bytes([0, 0])))
        except OSError:
            pass
        os.close(fd)


if __name__ == "__main__":
    main()
