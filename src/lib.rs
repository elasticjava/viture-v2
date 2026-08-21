//! VITURE Gen2 ("protocol V2") over USB HID — without the vendor SDK.
//!
//! Determined on a VITURE Pro 2 XR (`35CA:1301`) by capturing the official
//! SDK's traffic and matching it against the SDK's own frame logging.
//!
//! Frame layout (little-endian):
//! ```text
//! 0  u16  preamble 0x0010
//! 2  u16  MsgID
//! 4  u16  payload length
//! 6  u16  checksum = byte sum of the payload
//! 8  ..   payload
//! ```
//! The reply carries the request's MsgID plus [`RESPONSE_OFFSET`] and starts
//! with a status byte (0 = success).
//!
//! The hot path does not allocate: frames live on fixed-size stack buffers and
//! events are read out of slices with `from_le_bytes`.

use core::fmt;

pub mod hidraw;
pub mod pointer;
pub mod ring;
pub mod sys;
#[cfg(any(target_os = "linux", target_os = "android"))]
pub mod uring;
pub mod usbfs;
pub mod xr;

/// Fixed report size of the device.
pub const FRAME_MAX: usize = 64;
/// Header size preceding the payload.
pub const HEADER_LEN: usize = 8;
/// Constant preamble of every frame.
pub const PREAMBLE: u16 = 0x0010;
/// Reply MsgID = request MsgID plus this offset.
pub const RESPONSE_OFFSET: u16 = 0x2000;

/// Known message IDs.
pub mod msg {
    /// IMU control, payload `[streams, rate]`.
    pub const IMU_CTRL: u16 = 0x0301;
    /// Serial number (plaintext ASCII).
    pub const SERIAL: u16 = 0x3002;
    /// Firmware version (ASCII).
    pub const VERSION: u16 = 0x3003;
    /// Brightness level.
    pub const BRIGHTNESS: u16 = 0x3122;
    /// Display duty cycle in percent.
    pub const DUTY_CYCLE: u16 = 0x3125;
    /// Display mode, see [`super::DisplayMode`].
    pub const DISPLAY_MODE: u16 = 0x3141;
    /// Volume level.
    pub const VOLUME: u16 = 0x3201;
    /// Wear status, 0 = not worn.
    pub const WEAR_STATUS: u16 = 0x3321;

    /// Event: fused orientation (quaternion).
    pub const EVT_POSE: u16 = 0x7308;
    /// Event: raw data (gyroscope + accelerometer).
    pub const EVT_RAW: u16 = 0x7309;
}

/// Which IMU streams should run. This is a bitmask on the wire — note that the
/// SDK header enum uses different values (`RAW = 0`, `POSE = 1`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Streams(pub u8);

impl Streams {
    pub const OFF: Streams = Streams(0x00);
    pub const POSE: Streams = Streams(0x01);
    pub const RAW: Streams = Streams(0x02);

    #[inline]
    pub const fn with(self, other: Streams) -> Streams {
        Streams(self.0 | other.0)
    }
}

/// Reporting rate. The Pro 2 supports raw up to 1000 Hz, pose only to 240 Hz.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum Rate {
    Hz60 = 0,
    Hz90 = 1,
    Hz120 = 2,
    Hz240 = 3,
    Hz500 = 4,
    Hz1000 = 5,
}

/// Display modes from the SDK header, confirmed through `0x3141`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DisplayMode {
    P1080_60 = 0x31,
    Sbs3840x1080_60 = 0x32,
    P1080_90 = 0x33,
    P1080_120 = 0x34,
    Sbs3840x1080_90 = 0x35,
    Unknown = 0,
}

impl DisplayMode {
    #[inline]
    pub fn from_raw(v: u8) -> Self {
        match v {
            0x31 => Self::P1080_60,
            0x32 => Self::Sbs3840x1080_60,
            0x33 => Self::P1080_90,
            0x34 => Self::P1080_120,
            0x35 => Self::Sbs3840x1080_90,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug)]
pub enum Error {
    Io(std::io::Error),
    /// Frame was shorter than the header or carried the wrong preamble.
    Malformed,
    /// Checksum did not match.
    Checksum,
    /// Device reported a non-zero status byte.
    Status(u8),
    /// No matching reply arrived within the time window.
    Timeout,
    /// Payload did not fit the destination buffer.
    Overflow,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "I/O: {e}"),
            Error::Malformed => f.write_str("malformed frame"),
            Error::Checksum => f.write_str("checksum mismatch"),
            Error::Status(s) => write!(f, "device reports status {s}"),
            Error::Timeout => f.write_str("timed out"),
            Error::Overflow => f.write_str("payload too large for buffer"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

pub type Result<T> = core::result::Result<T, Error>;

/// Builds a frame into `buf` and returns the part that was filled.
#[inline]
pub fn build<'a>(buf: &'a mut [u8; FRAME_MAX], msg_id: u16, payload: &[u8]) -> Result<&'a [u8]> {
    let end = HEADER_LEN + payload.len();
    if end > FRAME_MAX {
        return Err(Error::Overflow);
    }
    let sum = payload.iter().fold(0u16, |a, &b| a.wrapping_add(b as u16));
    buf[0..2].copy_from_slice(&PREAMBLE.to_le_bytes());
    buf[2..4].copy_from_slice(&msg_id.to_le_bytes());
    buf[4..6].copy_from_slice(&(payload.len() as u16).to_le_bytes());
    buf[6..8].copy_from_slice(&sum.to_le_bytes());
    buf[HEADER_LEN..end].copy_from_slice(payload);
    Ok(&buf[..end])
}

/// A received frame; borrows from the receive buffer.
#[derive(Clone, Copy)]
pub struct Frame<'a> {
    pub msg_id: u16,
    pub payload: &'a [u8],
}

/// Splits a received report apart, checking preamble, length and checksum.
#[inline]
pub fn parse(buf: &[u8]) -> Result<Frame<'_>> {
    if buf.len() < HEADER_LEN {
        return Err(Error::Malformed);
    }
    let pre = u16::from_le_bytes([buf[0], buf[1]]);
    if pre != PREAMBLE {
        return Err(Error::Malformed);
    }
    let msg_id = u16::from_le_bytes([buf[2], buf[3]]);
    let len = u16::from_le_bytes([buf[4], buf[5]]) as usize;
    let cks = u16::from_le_bytes([buf[6], buf[7]]);
    let end = HEADER_LEN + len;
    if end > buf.len() {
        return Err(Error::Malformed);
    }
    let payload = &buf[HEADER_LEN..end];
    let sum = payload.iter().fold(0u16, |a, &b| a.wrapping_add(b as u16));
    if sum != cks {
        return Err(Error::Checksum);
    }
    Ok(Frame { msg_id, payload })
}

/// Fused orientation. `tick` counts inside the device, `q` is `[w, x, y, z]`.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub tick: u32,
    pub q: [f32; 4],
}

impl Pose {
    /// Payload of [`msg::EVT_POSE`]: `u32 unknown, u32 tick, 4x f32`.
    #[inline]
    pub fn parse(p: &[u8]) -> Result<Pose> {
        if p.len() < 24 {
            return Err(Error::Malformed);
        }
        Ok(Pose {
            tick: u32::from_le_bytes([p[4], p[5], p[6], p[7]]),
            q: [f32_at(p, 8), f32_at(p, 12), f32_at(p, 16), f32_at(p, 20)],
        })
    }

    /// Roll/pitch/yaw in degrees (ZYX convention).
    #[inline]
    pub fn euler_deg(&self) -> [f32; 3] {
        let [w, x, y, z] = self.q;
        let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
        let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        [roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees()]
    }

    /// Orientation relative to a reference pose, i.e. how far the head has
    /// turned since the last recentre.
    #[inline]
    pub fn relative_to(&self, reference: &Pose) -> Pose {
        Pose {
            tick: self.tick,
            q: quat_mul(quat_conj(reference.q), self.q),
        }
    }
}

/// Quaternion conjugate.
#[inline]
pub fn quat_conj(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

/// Hamilton product, `[w, x, y, z]` order.
#[inline]
pub fn quat_mul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// Raw IMU data. Gyroscope in rad/s, acceleration in g.
#[derive(Clone, Copy, Debug)]
pub struct Raw {
    pub tick: u32,
    pub gyro: [f32; 3],
    pub accel: [f32; 3],
}

impl Raw {
    /// Payload of [`msg::EVT_RAW`]: like pose, but with a u16 before the
    /// floats, hence offset 10 instead of 8.
    #[inline]
    pub fn parse(p: &[u8]) -> Result<Raw> {
        if p.len() < 34 {
            return Err(Error::Malformed);
        }
        Ok(Raw {
            tick: u32::from_le_bytes([p[4], p[5], p[6], p[7]]),
            gyro: [f32_at(p, 10), f32_at(p, 14), f32_at(p, 18)],
            accel: [f32_at(p, 22), f32_at(p, 26), f32_at(p, 30)],
        })
    }
}

#[inline(always)]
fn f32_at(p: &[u8], off: usize) -> f32 {
    f32::from_le_bytes([p[off], p[off + 1], p[off + 2], p[off + 3]])
}

/// A received event, if it is one of the known kinds.
#[derive(Clone, Copy, Debug)]
pub enum Event {
    Pose(Pose),
    Raw(Raw),
    /// Anything else — the MsgID is kept for further mapping.
    Other(u16),
}

/// Transport for 64-byte reports.
///
/// `recv` **waits** up to `timeout_ns` inside the kernel — nothing is polled
/// and nothing sleeps. `0` means "collect whatever is already there",
/// [`u64::MAX`] means "wait indefinitely". A return of `0` means timeout.
pub trait Transport {
    fn send(&mut self, frame: &[u8]) -> Result<()>;
    fn recv(&mut self, buf: &mut [u8; FRAME_MAX], timeout_ns: u64) -> Result<usize>;
}

/// Default patience for command replies.
pub const REPLY_TIMEOUT_NS: u64 = 250_000_000;

/// A device on top of any transport.
pub struct Device<T: Transport> {
    transport: T,
    rx: [u8; FRAME_MAX],
    tx: [u8; FRAME_MAX],
}

impl<T: Transport> Device<T> {
    pub fn new(transport: T) -> Self {
        Device {
            transport,
            rx: [0; FRAME_MAX],
            tx: [0; FRAME_MAX],
        }
    }

    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Sends a frame and waits for its reply. IMU events arriving in between
    /// are skipped.
    ///
    /// Writes the payload **without** the leading status byte into `out`.
    pub fn request(
        &mut self,
        msg_id: u16,
        payload: &[u8],
        out: &mut [u8],
        timeout_ns: u64,
    ) -> Result<usize> {
        let frame = build(&mut self.tx, msg_id, payload)?;
        self.transport.send(frame)?;

        let want = msg_id.wrapping_add(RESPONSE_OFFSET);
        let start = std::time::Instant::now();
        loop {
            let left = timeout_ns.saturating_sub(start.elapsed().as_nanos() as u64);
            if left == 0 {
                return Err(Error::Timeout);
            }
            let n = self.transport.recv(&mut self.rx, left)?;
            if n == 0 {
                continue; // slice expired, remaining time recomputed above
            }
            let f = match parse(&self.rx[..n]) {
                Ok(f) => f,
                Err(_) => continue,
            };
            if f.msg_id != want {
                continue;
            }
            let (status, body) = f.payload.split_first().ok_or(Error::Malformed)?;
            if *status != 0 {
                return Err(Error::Status(*status));
            }
            if body.len() > out.len() {
                return Err(Error::Overflow);
            }
            out[..body.len()].copy_from_slice(body);
            return Ok(body.len());
        }
    }

    /// Query whose reply is a single value byte.
    pub fn get_u8(&mut self, msg_id: u16) -> Result<u8> {
        let mut out = [0u8; 4];
        let n = self.request(msg_id, &[], &mut out, REPLY_TIMEOUT_NS)?;
        if n < 1 {
            return Err(Error::Malformed);
        }
        Ok(out[n - 1])
    }

    /// Query whose reply is ASCII (version, serial number).
    pub fn get_ascii<'b>(&mut self, msg_id: u16, out: &'b mut [u8]) -> Result<&'b str> {
        let n = self.request(msg_id, &[], out, REPLY_TIMEOUT_NS)?;
        let s = &out[..n];
        let s = s.split(|&b| b == 0).next().unwrap_or(s);
        core::str::from_utf8(s).map_err(|_| Error::Malformed)
    }

    pub fn firmware_version<'b>(&mut self, out: &'b mut [u8]) -> Result<&'b str> {
        self.get_ascii(msg::VERSION, out)
    }

    pub fn serial<'b>(&mut self, out: &'b mut [u8]) -> Result<&'b str> {
        self.get_ascii(msg::SERIAL, out)
    }

    pub fn brightness(&mut self) -> Result<u8> {
        self.get_u8(msg::BRIGHTNESS)
    }

    pub fn volume(&mut self) -> Result<u8> {
        self.get_u8(msg::VOLUME)
    }

    pub fn duty_cycle(&mut self) -> Result<u8> {
        self.get_u8(msg::DUTY_CYCLE)
    }

    /// `true` if the glasses are being worn. Returns a value even where the
    /// vendor SDK leaves its output parameter untouched.
    pub fn worn(&mut self) -> Result<bool> {
        Ok(self.get_u8(msg::WEAR_STATUS)? != 0)
    }

    pub fn display_mode(&mut self) -> Result<DisplayMode> {
        Ok(DisplayMode::from_raw(self.get_u8(msg::DISPLAY_MODE)?))
    }

    /// Starts or stops the IMU streams. One command, no handshake.
    pub fn set_imu(&mut self, streams: Streams, rate: Rate) -> Result<()> {
        let mut out = [0u8; 4];
        self.request(
            msg::IMU_CTRL,
            &[streams.0, rate as u8],
            &mut out,
            REPLY_TIMEOUT_NS,
        )?;
        Ok(())
    }

    /// Waits up to `timeout_ns` for the next event. `Ok(None)` means timeout —
    /// the waiting happens in the kernel, not in a loop.
    #[inline]
    pub fn next_event(&mut self, timeout_ns: u64) -> Result<Option<Event>> {
        let n = self.transport.recv(&mut self.rx, timeout_ns)?;
        if n == 0 {
            return Ok(None);
        }
        let f = parse(&self.rx[..n])?;
        Ok(Some(match f.msg_id {
            msg::EVT_POSE => Event::Pose(Pose::parse(f.payload)?),
            msg::EVT_RAW => Event::Raw(Raw::parse(f.payload)?),
            other => Event::Other(other),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Against the real capture: open_imu(POSE, 120 Hz).
    #[test]
    fn builds_imu_command_like_the_sdk() {
        let mut buf = [0u8; FRAME_MAX];
        let f = build(
            &mut buf,
            msg::IMU_CTRL,
            &[Streams::POSE.0, Rate::Hz120 as u8],
        )
        .unwrap();
        assert_eq!(
            f,
            &[0x10, 0x00, 0x01, 0x03, 0x02, 0x00, 0x03, 0x00, 0x01, 0x02]
        );
    }

    #[test]
    fn builds_query_without_payload() {
        let mut buf = [0u8; FRAME_MAX];
        let f = build(&mut buf, msg::BRIGHTNESS, &[]).unwrap();
        assert_eq!(f, &[0x10, 0x00, 0x22, 0x31, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn parses_brightness_reply() {
        let bytes = [0x10, 0x00, 0x22, 0x51, 0x02, 0x00, 0x03, 0x00, 0x00, 0x03];
        let f = parse(&bytes).unwrap();
        assert_eq!(f.msg_id, msg::BRIGHTNESS + RESPONSE_OFFSET);
        assert_eq!(f.payload, &[0x00, 0x03]);
    }

    #[test]
    fn rejects_bad_checksum() {
        let bytes = [0x10, 0x00, 0x22, 0x51, 0x02, 0x00, 0xFF, 0x00, 0x00, 0x03];
        assert!(matches!(parse(&bytes), Err(Error::Checksum)));
    }

    /// Pose packet from the capture; the quaternion must be normalised.
    #[test]
    fn parses_pose_event() {
        let mut bytes = vec![0x10, 0x00, 0x08, 0x73, 0x18, 0x00, 0x00, 0x00];
        let payload: [u8; 24] = [
            0x26, 0x01, 0x00, 0x00, 0x59, 0xd6, 0x21, 0x00, 0xdd, 0xe3, 0x2b, 0x3e, 0x52, 0x5d,
            0x7c, 0x3f, 0xdd, 0x45, 0x9a, 0x3b, 0xe8, 0x27, 0x52, 0x3a,
        ];
        let sum = payload.iter().fold(0u16, |a, &b| a.wrapping_add(b as u16));
        bytes[6..8].copy_from_slice(&sum.to_le_bytes());
        bytes.extend_from_slice(&payload);

        let f = parse(&bytes).unwrap();
        let pose = Pose::parse(f.payload).unwrap();
        assert_eq!(pose.tick, 0x0021_d659);
        let n = pose.q.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-3, "|q| = {n}");
    }

    /// Relative orientation against itself must be the identity rotation.
    #[test]
    fn relative_to_self_is_identity() {
        let p = Pose {
            tick: 0,
            q: [0.1153, 0.9922, -0.0417, -0.0051],
        };
        let r = p.relative_to(&p);
        assert!((r.q[0] - 1.0).abs() < 1e-3, "w = {}", r.q[0]);
        for c in &r.q[1..] {
            assert!(c.abs() < 1e-3, "axis component {c}");
        }
    }
}
