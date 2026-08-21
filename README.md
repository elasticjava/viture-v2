# viture-v2

A dependency-free Rust driver for **VITURE Gen2 XR glasses** — no vendor SDK, no `libc`,
no crates at all.

The official VITURE SDK is a closed-source binary. It exists for x86-64 and aarch64 Linux,
for Windows, and for Android, and everything built on top of it inherits that dependency:
you have to request developer access, vendor the `.so`, and accept a glibc binary that
will not load under Android's Bionic. This crate removes that dependency for Gen2 devices
by speaking the protocol directly.

The wire protocol is documented in [PROTOCOL.md](PROTOCOL.md). As far as GitHub code
search can tell, it has not been published anywhere else.

## Status

Developed and measured against a **VITURE Pro 2 XR** (`35CA:1301`). Other Gen2 devices
speak the same protocol but are untested — the product ID is the only thing that needs
changing.

| Feature | State |
|---|---|
| Pose stream (fused quaternion) | works, measured 119 Hz |
| Raw stream (gyro + accelerometer) | works, measured up to 795 Hz |
| Firmware version, serial number | works |
| Brightness, volume, duty cycle, display mode | works (read) |
| Wear status | works — more reliable than the vendor SDK, see PROTOCOL.md |
| Setters (brightness, display mode, …) | not mapped yet |
| Sensor fusion / prediction | not implemented |

## Design

Three interchangeable transports behind one trait, so the protocol core is written once:

- **`uring`** — real `io_uring`. A ring of pre-armed read buffers stays queued in the
  kernel; completions are reaped in batches. Waiting uses `io_uring_enter` with
  `IORING_ENTER_EXT_ARG`, so the timeout rides along without burning an SQE.
- **`usbfs`** — `USBDEVFS_SUBMITURB` / `USBDEVFS_REAPURBNDELAY`. Conceptually identical to
  io_uring — a submission and a completion queue, expressed as ioctls. **This is the
  Android path**, because Android's app sandbox blocks `io_uring_setup` via seccomp and an
  unprivileged process cannot lift its own filter.
- **`hidraw`** — blocking `ppoll` + `read`. Two syscalls per event, and the thread sleeps
  in the kernel in between.

There is no polling and no sleeping anywhere in the I/O path. Between the reader thread
and the consumer sits a lock-free SPSC ring buffer with head and tail on separate cache
lines; the consumer drains in batches and the producer never blocks.

The syscall layer is written directly in inline assembly for x86-64 and aarch64, which is
why the crate needs neither `libc` nor a build script. That also makes cross-compiling
trivial — there is no C anywhere.

## Measurements

Pro 2 XR, raw stream at 1000 Hz requested (~770–795 Hz delivered over a USB/IP link):

| Transport | Wakeups | CPU | per event |
|---|---|---|---|
| io_uring, wake per sample | every sample | 97.2 ms | 24.0 µs |
| io_uring, wake per 8 samples | batched | **16.5 ms** | **3.6 µs** |
| io_uring, wake per 32 samples | batched | 9.3 ms | 2.0 µs |
| hidraw, wake per sample | every sample | 91.5 ms | 19.8 µs |
| usbfs, wake per sample | every sample | 143.9 ms | 30.1 µs |
| usbfs, wake per 8 samples | batched | **26.7 ms** | **5.6 µs** |

The interesting result: io_uring halves the syscall count against `hidraw` and is still
marginally *slower*. The cost is not in the syscall, it is in **waking the consumer**.
Batching the notification cuts CPU time by 5–10× at identical throughput, on every
transport.

Practical consequence for anything rendering head-tracked content: **drain once per frame,
not once per sample.** At 795 Hz and a 120 Hz display that is about six samples per
wake-up — exactly the regime the numbers above describe, and it costs no latency that the
frame boundary did not already impose.

## Usage

```rust
use viture_v2::{Device, Event, Rate, Streams};
use viture_v2::hidraw::{find_fd, Hidraw};

let mut dev = Device::new(Hidraw::new(find_fd(0x35CA, 0x1301)?));

let mut buf = [0u8; 64];
println!("{}", dev.firmware_version(&mut buf)?);
println!("worn: {}", dev.worn()?);

dev.set_imu(Streams::POSE, Rate::Hz120)?;
loop {
    if let Some(Event::Pose(p)) = dev.next_event(u64::MAX)? {
        let [roll, pitch, yaw] = p.euler_deg();
        println!("{roll:7.2} {pitch:7.2} {yaw:7.2}");
    }
}
```

On Android the only difference is where the descriptor comes from:

```rust
// termux-usb -r -e ./prog /dev/bus/usb/BBB/DDD  passes the fd as argv[1]
let fd: i32 = std::env::args().nth(1).unwrap().parse()?;
let mut dev = Device::new(viture_v2::usbfs::Usbfs::new(fd, 0)?);
```

### Command line

```
viture-v2 info                        device information
viture-v2 pose  [secs] [backend] [Hz] pose stream
viture-v2 raw   [secs] [backend] [Hz] raw stream
```

`backend` is `uring`, `usbfs` or `hidraw`. `VITURE_COALESCE=N` sets how many samples
accumulate before the consumer is woken.

Reading `hidraw` needs permission — either run as root or install a udev rule:

```
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="35ca", MODE="0660", GROUP="plugdev"
```

## Building

```
cargo build --release
cargo test
cargo build --release --target aarch64-unknown-linux-musl   # static, runs on Android
```

## tools/

The instruments the protocol was reverse engineered with. `v2_hidraw.py` and `v2_raw.py`
talk the protocol directly and were the first confirmation. `sdk_probe.c` and `sdk_map.c`
drive the official SDK through `dlopen` to produce labelled captures — they need the
VITURE SDK, which is **not** included here and is not redistributable.

## License

MIT
