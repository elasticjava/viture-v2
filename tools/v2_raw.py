#!/usr/bin/env python3
"""RAW-Nutzdaten (MsgID 0x7309, 56 B) vollstaendig auslesen und Layout bestimmen."""
import math
import os
import select
import struct
import time

DEV = "/dev/hidraw0"
STREAM_POSE, STREAM_RAW = 0x01, 0x02   # Wire-Bitmaske, nicht das Header-Enum


def frame(msg_id, payload):
    return struct.pack("<HHHH", 0x0010, msg_id, len(payload), sum(payload) & 0xFFFF) + payload


fd = os.open(DEV, os.O_RDWR | os.O_NONBLOCK)
try:
    os.write(fd, b"\x00" + frame(0x0301, bytes([STREAM_RAW, 2])))
    samples, t0 = [], time.time()
    while time.time() - t0 < 4.0 and len(samples) < 400:
        r, _, _ = select.select([fd], [], [], 0.5)
        if not r:
            continue
        buf = os.read(fd, 128)
        pre, msg, plen, cks = struct.unpack_from("<HHHH", buf, 0)
        if pre != 0x0010 or msg != 0x7309:
            continue
        samples.append(buf[8:8 + plen])

    print(f"{len(samples)} RAW-Pakete, Nutzdaten je {len(samples[0]) if samples else 0} B\n")
    for s in samples[:2]:
        print("payload:", s.hex(' '))

    if samples:
        s = samples[0]
        print("\n--- f32-Interpretation je Startoffset ---")
        for off in (8, 10, 12):
            vals = struct.unpack_from(f"<{(len(s) - off) // 4}f", s, off)
            print(f"off {off:2d}: " + " ".join(f"{v:11.5f}" if abs(v) < 1e6 else f"{v:11.3e}" for v in vals[:12]))

        print("\n--- Betraege moeglicher 3er-Gruppen (Suche nach 1 g) ---")
        for off in (8, 10, 12):
            n = (len(s) - off) // 4
            vals = struct.unpack_from(f"<{n}f", s, off)
            for i in range(0, min(n, 12) - 2, 3):
                g = vals[i:i + 3]
                if all(abs(v) < 1e3 for v in g):
                    print(f"off {off:2d} idx {i}: |{g[0]:.4f},{g[1]:.4f},{g[2]:.4f}| = "
                          f"{math.sqrt(sum(v * v for v in g)):.4f}")

        print("\n--- Kopffelder ---")
        for s2 in samples[:6]:
            f1, ts = struct.unpack_from("<II", s2, 0)
            u16 = struct.unpack_from("<H", s2, 8)[0]
            print(f"  f1={f1:<8d} ts={ts:<10d} u16@8={u16}")
finally:
    try:
        os.write(fd, b"\x00" + frame(0x0301, bytes([0, 0])))
    except OSError:
        pass
    os.close(fd)
