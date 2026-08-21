//! VITURE Gen2 („Protokoll V2") über USB-HID — ohne Vendor-SDK.
//!
//! Ermittelt an VITURE Pro 2 XR (`35CA:1301`) durch Mitschnitt des offiziellen
//! SDK gegen dessen eigene Frame-Protokollierung.
//!
//! Rahmen (little-endian):
//! ```text
//! 0  u16  Präambel 0x0010
//! 2  u16  MsgID
//! 4  u16  PayloadLen
//! 6  u16  Checksum = Bytesumme der Payload
//! 8  ..   Payload
//! ```
//! Antwort-MsgID = Anfrage-MsgID + [`RESPONSE_OFFSET`]. Antworten beginnen mit
//! einem Statusbyte (0 = ok).
//!
//! Der Hot Path allokiert nicht: Frames liegen auf Stack-Puffern fester Größe,
//! Ereignisse werden per `from_le_bytes` aus Slices gelesen.

use core::fmt;

pub mod hidraw;
pub mod ring;
pub mod sys;
pub mod usbfs;
#[cfg(target_os = "linux")]
pub mod uring;

/// Feste Report-Größe des Geräts.
pub const FRAME_MAX: usize = 64;
/// Kopfgröße vor der Payload.
pub const HEADER_LEN: usize = 8;
/// Konstante Präambel jedes Frames.
pub const PREAMBLE: u16 = 0x0010;
/// Antwort-MsgID = Anfrage + dieser Wert.
pub const RESPONSE_OFFSET: u16 = 0x2000;

/// Bekannte Nachrichten-IDs.
pub mod msg {
    /// IMU-Steuerung, Payload `[streams, rate]`.
    pub const IMU_CTRL: u16 = 0x0301;
    /// Seriennummer (Klartext-ASCII).
    pub const SERIAL: u16 = 0x3002;
    /// Firmware-Version (ASCII).
    pub const VERSION: u16 = 0x3003;
    /// Helligkeitsstufe.
    pub const BRIGHTNESS: u16 = 0x3122;
    /// Duty-Cycle der Anzeige in Prozent.
    pub const DUTY_CYCLE: u16 = 0x3125;
    /// Anzeigemodus, siehe [`super::DisplayMode`].
    pub const DISPLAY_MODE: u16 = 0x3141;
    /// Lautstärkestufe.
    pub const VOLUME: u16 = 0x3201;
    /// Trage-Status, 0 = nicht getragen.
    pub const WEAR_STATUS: u16 = 0x3321;

    /// Ereignis: fusionierte Lage (Quaternion).
    pub const EVT_POSE: u16 = 0x7308;
    /// Ereignis: Rohdaten (Gyro + Beschleunigung).
    pub const EVT_RAW: u16 = 0x7309;
}

/// Welche IMU-Ströme laufen sollen. Bitmaske auf dem Draht — Achtung, das
/// SDK-Header-Enum benutzt andere Werte (`RAW = 0`, `POSE = 1`).
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

/// Meldefrequenz. Die Pro 2 unterstützt Roh bis 1000 Hz, Pose nur bis 240 Hz.
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

/// Anzeigemodi laut SDK-Header, bestätigt durch `0x3141`.
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
    /// Frame war kürzer als der Kopf oder hatte die falsche Präambel.
    Malformed,
    /// Checksumme stimmte nicht.
    Checksum,
    /// Gerät meldete ein Statusbyte ungleich null.
    Status(u8),
    /// Innerhalb des Zeitfensters kam keine passende Antwort.
    Timeout,
    /// Payload passte nicht in den Zielpuffer.
    Overflow,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Io(e) => write!(f, "E/A: {e}"),
            Error::Malformed => f.write_str("ungültiger Rahmen"),
            Error::Checksum => f.write_str("Checksumme falsch"),
            Error::Status(s) => write!(f, "Gerät meldet Status {s}"),
            Error::Timeout => f.write_str("Zeitüberschreitung"),
            Error::Overflow => f.write_str("Payload zu groß für Puffer"),
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

/// Baut einen Frame in `buf` und liefert den belegten Teil zurück.
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

/// Ein empfangener Frame; borgt aus dem Empfangspuffer.
#[derive(Clone, Copy)]
pub struct Frame<'a> {
    pub msg_id: u16,
    pub payload: &'a [u8],
}

/// Zerlegt einen empfangenen Report. Prüft Präambel, Länge und Checksumme.
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

/// Fusionierte Lage. `tick` zählt geräteintern, `q` ist `[w, x, y, z]`.
#[derive(Clone, Copy, Debug)]
pub struct Pose {
    pub tick: u32,
    pub q: [f32; 4],
}

impl Pose {
    /// Payload von [`msg::EVT_POSE`]: `u32 unbekannt, u32 tick, 4× f32`.
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

    /// Roll/Pitch/Yaw in Grad (ZYX).
    #[inline]
    pub fn euler_deg(&self) -> [f32; 3] {
        let [w, x, y, z] = self.q;
        let roll = (2.0 * (w * x + y * z)).atan2(1.0 - 2.0 * (x * x + y * y));
        let pitch = (2.0 * (w * y - z * x)).clamp(-1.0, 1.0).asin();
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        [roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees()]
    }
}

/// Rohdaten der IMU. Gyro in rad/s, Beschleunigung in g.
#[derive(Clone, Copy, Debug)]
pub struct Raw {
    pub tick: u32,
    pub gyro: [f32; 3],
    pub accel: [f32; 3],
}

impl Raw {
    /// Payload von [`msg::EVT_RAW`]: wie Pose, aber mit u16 vor den Floats,
    /// daher Versatz 10 statt 8.
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

/// Ein empfangenes Ereignis, sofern es eines der bekannten ist.
#[derive(Clone, Copy, Debug)]
pub enum Event {
    Pose(Pose),
    Raw(Raw),
    /// Alles andere — MsgID zur weiteren Kartierung.
    Other(u16),
}

/// Transport für 64-Byte-Reports.
///
/// `recv` **wartet** bis zu `timeout_ns` im Kernel — es wird nirgends gepollt
/// und nirgends geschlafen. `0` heißt „nur abholen, was schon da ist",
/// [`u64::MAX`] heißt „unbegrenzt warten". Rückgabe `0` bedeutet
/// Zeitüberschreitung.
pub trait Transport {
    fn send(&mut self, frame: &[u8]) -> Result<()>;
    fn recv(&mut self, buf: &mut [u8; FRAME_MAX], timeout_ns: u64) -> Result<usize>;
}

/// Vorgabe für Antworten auf Kommandos.
pub const REPLY_TIMEOUT_NS: u64 = 250_000_000;

/// Gerät auf einem beliebigen Transport.
pub struct Device<T: Transport> {
    transport: T,
    rx: [u8; FRAME_MAX],
    tx: [u8; FRAME_MAX],
}

impl<T: Transport> Device<T> {
    pub fn new(transport: T) -> Self {
        Device { transport, rx: [0; FRAME_MAX], tx: [0; FRAME_MAX] }
    }

    pub fn transport_mut(&mut self) -> &mut T {
        &mut self.transport
    }

    /// Sendet einen Frame und wartet auf die zugehörige Antwort. IMU-Ereignisse,
    /// die zwischendurch eintreffen, werden übersprungen.
    ///
    /// Gibt die Payload **ohne** das führende Statusbyte in `out` zurück.
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
                continue; // Zeitscheibe abgelaufen, Restzeit oben neu berechnet
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

    /// Getter, dessen Antwort aus genau einem Wertbyte besteht.
    pub fn get_u8(&mut self, msg_id: u16) -> Result<u8> {
        let mut out = [0u8; 4];
        let n = self.request(msg_id, &[], &mut out, REPLY_TIMEOUT_NS)?;
        if n < 1 {
            return Err(Error::Malformed);
        }
        Ok(out[n - 1])
    }

    /// Getter, dessen Antwort ASCII ist (Version, Seriennummer).
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

    /// `true`, wenn die Brille getragen wird. Liefert auch dann einen Wert,
    /// wenn das Vendor-SDK den Ausgabeparameter nicht befüllt.
    pub fn worn(&mut self) -> Result<bool> {
        Ok(self.get_u8(msg::WEAR_STATUS)? != 0)
    }

    pub fn display_mode(&mut self) -> Result<DisplayMode> {
        Ok(DisplayMode::from_raw(self.get_u8(msg::DISPLAY_MODE)?))
    }

    /// Startet oder stoppt die IMU-Ströme. Ein einziges Kommando, kein Handshake.
    pub fn set_imu(&mut self, streams: Streams, rate: Rate) -> Result<()> {
        let mut out = [0u8; 4];
        self.request(msg::IMU_CTRL, &[streams.0, rate as u8], &mut out, REPLY_TIMEOUT_NS)?;
        Ok(())
    }

    /// Wartet bis zu `timeout_ns` auf das nächste Ereignis. `Ok(None)` heißt
    /// Zeitüberschreitung — gewartet wird im Kernel, nicht in einer Schleife.
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

    /// Gegen den echten Mitschnitt: open_imu(POSE, 120 Hz).
    #[test]
    fn baut_imu_kommando_wie_das_sdk() {
        let mut buf = [0u8; FRAME_MAX];
        let f = build(&mut buf, msg::IMU_CTRL, &[Streams::POSE.0, Rate::Hz120 as u8]).unwrap();
        assert_eq!(f, &[0x10, 0x00, 0x01, 0x03, 0x02, 0x00, 0x03, 0x00, 0x01, 0x02]);
    }

    #[test]
    fn baut_getter_ohne_payload() {
        let mut buf = [0u8; FRAME_MAX];
        let f = build(&mut buf, msg::BRIGHTNESS, &[]).unwrap();
        assert_eq!(f, &[0x10, 0x00, 0x22, 0x31, 0x00, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn zerlegt_helligkeitsantwort() {
        let bytes = [0x10, 0x00, 0x22, 0x51, 0x02, 0x00, 0x03, 0x00, 0x00, 0x03];
        let f = parse(&bytes).unwrap();
        assert_eq!(f.msg_id, msg::BRIGHTNESS + RESPONSE_OFFSET);
        assert_eq!(f.payload, &[0x00, 0x03]);
    }

    #[test]
    fn erkennt_falsche_checksumme() {
        let bytes = [0x10, 0x00, 0x22, 0x51, 0x02, 0x00, 0xFF, 0x00, 0x00, 0x03];
        assert!(matches!(parse(&bytes), Err(Error::Checksum)));
    }

    /// Pose-Paket aus dem Mitschnitt; Quaternion muss normiert sein.
    #[test]
    fn zerlegt_pose_ereignis() {
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
}
