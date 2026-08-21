# VITURE Protocol V2 (Gen2) over USB HID

Determined on **VITURE Pro 2 XR Glasses**, `VID 0x35CA / PID 0x1301`, firmware
`30.0.00.002_20260804`, `bcdDevice 0x0200`.

Method: `usbmon` capture of the official SDK (`libglasses.so`) matched against the
SDK's own frame logging, which prints MsgID, payload length and checksum for every
command it builds. That yields a labelled dataset rather than guesswork. Every finding
below was then re-verified by an independent implementation that talks to the device
without any vendor library.

Reverse engineered for interoperability with hardware we own.

## Transport

| Property | Value |
|---|---|
| USB | 2.01, full speed, 1 configuration, **1 interface** |
| Interface | class 03 (HID), subclass 00, protocol 00 |
| Endpoints | `0x01` interrupt OUT, `0x81` interrupt IN, 64 B each, 1 ms interval |
| HID report descriptor | usage page `0xFF00`, usage `0x01`, one opaque 64-byte IN and OUT report, **no report IDs** |

Commands are written to the OUT endpoint at their actual frame length (short packets are
fine). Responses arrive zero-padded to 64 bytes. Over `hidraw` a leading report-ID byte
`0x00` must be prepended; over usbfs the frame goes on the wire unchanged.

## Frame format

All fields little-endian.

```
Offset  Size  Field
0       2     Preamble, constant 0x0010
2       2     MsgID
4       2     PayloadLen
6       2     Checksum = sum of all payload bytes (mod 2^16)
8       n     Payload
```

Total length is `8 + PayloadLen`. The checksum is a plain byte sum, **not** a CRC —
verified against a command (`0x01 + 0x02 = 0x0003`) and a data packet (sum 2204 =
`0x089C`).

**Reply convention:** the response to a request carries `MsgID + 0x2000` and starts with
a status byte, `0x00` meaning success.

## Messages

### `0x0301` — IMU control (host → glasses)

Payload: 2 bytes `[stream, rate]`.

`stream` is a **bitmask**, and it does not match the SDK header enum:

| Value | Meaning |
|---|---|
| `0x00` | streams off |
| `0x01` | pose stream (fused quaternion) |
| `0x02` | raw stream (gyro + accelerometer) |

The SDK header defines `VITURE_IMU_MODE_RAW = 0` and `..._POSE = 1`; on the wire those
are `2` and `1`. The API values are translated, not passed through. This is the single
most likely thing to trip up an independent implementation.

`rate`: `0` = 60 Hz, `1` = 90 Hz, `2` = 120 Hz, `3` = 240 Hz, `4` = 500 Hz, `5` = 1000 Hz.
The Pro 2 supports raw up to 1000 Hz but pose only up to 240 Hz (queried through the
SDK's `is_product_support_imu_frequency`).

```
10 00 01 03 02 00 03 00 01 02    pose stream, 120 Hz
10 00 01 03 02 00 04 00 02 02    raw stream, 120 Hz
10 00 01 03 02 00 00 00 00 00    both off
```

### `0x2301` — ACK for `0x0301`

Payload: 1 status byte, `0x00` = success.

### `0x7308` — pose event (glasses → host)

Payload: 24 bytes.

```
0   u32   unknown (always < 2^16, jumps around — not a counter)
4   u32   timestamp
8   f32   qw
12  f32   qx
16  f32   qy
20  f32   qz
```

The quaternion is normalised (measured `|q| = 1.0000`). **Roll/pitch/yaw are never
transmitted** — the Euler angles the SDK hands to its callbacks are computed host-side.

### `0x7309` — raw event (glasses → host)

Payload: 56 bytes.

```
0   u32   unknown (as above)
4   u32   timestamp, increments by 1 per sample
8   u16   unknown, ~186/187, drifts slightly
10  f32   gyro X   [rad/s]
14  f32   gyro Y
18  f32   gyro Z
22  f32   accel X  [g]
26  f32   accel Y
30  f32   accel Z
34  f32   constant 117.30   — presumably calibration/scale
38  f32   constant  60.75
42  f32   constant 254.85
46  ...   remainder varies, not plausible as f32
```

Mind the offset difference: pose floats start at 8, raw floats at 10 because of the extra
u16.

Cross-check: accelerometer magnitude `|(0.0024, -0.3221, -0.9287)| = 0.983 g`, and
`atan2(0.322, 0.929) = 19.1°` matches the pitch reported by the pose stream at the same
moment.

## Queries

A query is a frame with `PayloadLen = 0`, i.e. 8 bytes. The reply carries `MsgID + 0x2000`
and begins with a status byte followed by the value.

| Function | MsgID | Reply | Payload | Measured on a Pro 2 |
|---|---|---|---|---|
| Serial number | `0x3002` | `0x5002` | status + 15 ASCII | **plaintext, not a hash** |
| Firmware version | `0x3003` | `0x5003` | status + 20 ASCII | `30.0.00.002_20260804` |
| Brightness | `0x3122` | `0x5122` | status + u8 | 3 |
| Duty cycle | `0x3125` | `0x5125` | status + u8 (percent) | 98 |
| Display mode | `0x3141` | `0x5141` | status + u8 | `0x31` |
| Volume | `0x3201` | `0x5201` | status + u8 | 5 |
| Wear status | `0x3321` | `0x5321` | status + u8 | 0 = not worn |

```
OUT  10 00 22 31 00 00 00 00
IN   10 00 22 51 02 00 03 00 00 03      status 0, value 3
```

MsgIDs are grouped: `0x30xx` device information, `0x31xx` display, `0x32xx` audio,
`0x33xx` sensors, `0x03xx` IMU control, `0x73xx` IMU events. Setters presumably live in
the same groups but have not been mapped, since probing them changes device state.

### Two reasons to talk to the device directly

The **serial number travels in plaintext**. The SDK computes the SHA-256 host-side, even
though its own header states the raw number is never exposed.

`xr_device_provider_get_wear_status()` returns `0` for success but **does not write the
output parameter** — a sentinel placed in the buffer survived the call unchanged. The wire
carries the value correctly. Reading the device directly is more accurate than the vendor
SDK here.

## Display modes

Constants from `viture_protocol_public.h`, confirmed through `0x3141`:

| Value | Mode |
|---|---|
| `0x31` | 1920×1080 @ 60 Hz |
| `0x32` | 3840×1080 @ 60 Hz (SBS) |
| `0x33` | 1920×1080 @ 90 Hz |
| `0x34` | 1920×1080 @ 120 Hz |
| `0x35` | 3840×1080 @ 90 Hz (SBS) |

The 1200p variants (`0x41`–`0x45`) belong to the Luma models.

## No handshake required

`xr_device_provider_initialize()` and `start()` send **nothing** on the interrupt
endpoint. A full SDK session captured on the wire consists of two OUT commands (stream on,
stream off), two ACKs, and the data packets. A client only has to open the device and send
a single 10-byte command.

## Not supported on the Pro 2

`native_get_*` and `get_film_mode` return `-4` (`NOT_SUPPORTED`) and emit **no command at
all**. That independently confirms: no on-glasses DOF, no electrochromic film. The
marketing claim that the Pro 2 "has no head tracking" means exactly this — the IMU is
present and streams at up to 1000 Hz, but the fusion happens on the host.

## Still open

The meaning of the leading u32 in event packets, and of the u16 at offset 8 in raw
packets. Setter MsgIDs. Mapping them is mechanical: call the corresponding SDK function,
capture, and match against the SDK's `ProtocolBuilderV2: Built command - MsgID: ...` log
line.
