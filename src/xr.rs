//! The moving parts of a head-tracked desktop, behind a C ABI.
//!
//! Everything here is the part where Rust earns its place: reading the glasses
//! at up to 1000 Hz, predicting where the head will be when the frame actually
//! reaches the panel, and turning head plus hand into a cursor. The Android
//! framework side — `Presentation`, `VirtualDisplay`, launching apps, the
//! taskbar — stays in Kotlin, because those are framework calls, not compute.
//!
//! The contract is deliberately narrow so the JNI shim around it is trivial:
//!
//! ```c
//! XrTracker *t = xr_open(fd, 240);
//! xr_set_phone_quat(t, w, x, y, z);   // from Android SensorManager
//! xr_recentre(t);
//! XrState s; xr_state(t, &s);         // once per frame
//! xr_close(t);
//! ```
//!
//! `xr_state` is the only call in the render loop. It reads atomics, does a
//! prediction step and returns — no locks, no allocation, no syscall.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::pointer::{rotate, Pointer};

use crate::usbfs::Usbfs;
use crate::{quat_conj, quat_mul, Device, Event, Rate, Streams, Transport};


/// Snapshot handed to the renderer once per frame.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct XrState {
    /// Head orientation relative to the last recentre, `[w, x, y, z]`.
    pub head_q: [f32; 4],
    /// Head orientation extrapolated by the look-ahead.
    pub predicted_q: [f32; 4],
    /// Cursor in normalised screen coordinates, centre `(0, 0)`.
    pub cursor_x: f32,
    pub cursor_y: f32,
    /// Non-zero when the cursor is on screen.
    pub cursor_valid: u32,
    /// Samples seen so far — proof that both sources are live.
    pub head_samples: u64,
    pub phone_samples: u64,
}

/// Lock-free storage for the hot values. `f32` lives as its bit pattern.
#[derive(Default)]
struct Hot {
    head: [AtomicU32; 4],
    fresh: AtomicBool,
    gyro: [AtomicU32; 3],
    phone: [AtomicU32; 4],
    head_samples: AtomicU64,
    phone_samples: AtomicU64,
}

impl Hot {
    fn store_quat(slot: &[AtomicU32; 4], q: [f32; 4]) {
        for (a, v) in slot.iter().zip(q) {
            a.store(v.to_bits(), Ordering::Relaxed);
        }
    }

    fn load_quat(slot: &[AtomicU32; 4]) -> [f32; 4] {
        let mut q = [0f32; 4];
        for (i, a) in slot.iter().enumerate() {
            q[i] = f32::from_bits(a.load(Ordering::Relaxed));
        }
        q
    }

    fn load_vec3(slot: &[AtomicU32; 3]) -> [f32; 3] {
        let mut v = [0f32; 3];
        for (i, a) in slot.iter().enumerate() {
            v[i] = f32::from_bits(a.load(Ordering::Relaxed));
        }
        v
    }
}

/// Cold configuration, touched on recentre and on setting knobs.
struct Cold {
    pointer: Pointer,
    /// How far ahead to extrapolate, in seconds.
    lookahead_s: f32,
}

/// Device facts read once while the command path is still free, before the
/// reader thread takes ownership of the transport. They change rarely; a
/// renderer that wants live values can reopen.
#[derive(Clone, Default)]
pub struct DeviceInfo {
    pub firmware: String,
    pub brightness: i32,
    pub volume: i32,
    pub display_mode: i32,
    pub worn: bool,
}

pub struct Tracker {
    hot: Arc<Hot>,
    cold: Mutex<Cold>,
    stop: Arc<AtomicBool>,
    reader: Option<JoinHandle<()>>,
    info: DeviceInfo,
}

impl Tracker {
    /// Takes an open usbfs descriptor — from `termux-usb` on the command line
    /// or from `UsbDeviceConnection.getFileDescriptor()` inside an app.
    ///
    /// Runs pose and raw simultaneously: pose carries absolute orientation,
    /// raw carries the angular rate that the prediction needs. The wire format
    /// allows both because the stream field is a bitmask.
    pub fn open(fd: i32, rate: Rate) -> crate::Result<Tracker> {
        let mut dev = Device::new(Usbfs::new(fd, 0)?);

        // Read the static facts while nothing else is using the command path.
        let mut buf = [0u8; 64];
        let info = DeviceInfo {
            firmware: dev.firmware_version(&mut buf).unwrap_or("unknown").to_owned(),
            brightness: dev.brightness().map(i32::from).unwrap_or(-1),
            volume: dev.volume().map(i32::from).unwrap_or(-1),
            display_mode: dev.get_u8(crate::msg::DISPLAY_MODE).map(i32::from).unwrap_or(-1),
            worn: dev.worn().unwrap_or(false),
        };

        // Both streams at once relies on the wire field being a bitmask. That
        // is inferred, not documented, so fall back to pose only if the device
        // refuses. Without raw the prediction simply has no rate to work with
        // and becomes a no-op, which is safe.
        if dev.set_imu(Streams::POSE.with(Streams::RAW), rate).is_err() {
            dev.set_imu(Streams::POSE, rate)?;
        }

        let hot = Arc::new(Hot::default());
        Hot::store_quat(&hot.head, [1.0, 0.0, 0.0, 0.0]);
        Hot::store_quat(&hot.phone, [1.0, 0.0, 0.0, 0.0]);

        let stop = Arc::new(AtomicBool::new(false));
        let reader = {
            let hot = Arc::clone(&hot);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || read_loop(dev, hot, stop))
        };

        Ok(Tracker {
            hot,
            cold: Mutex::new(Cold { pointer: Pointer::default(), lookahead_s: 0.020 }),
            stop,
            reader: Some(reader),
            info,
        })
    }

    pub fn info(&self) -> &DeviceInfo {
        &self.info
    }

    /// Absolute head orientation as the vendor SDK reports it:
    /// `[roll, pitch, yaw, qw, qx, qy, qz]`, Euler angles in degrees.
    /// This is what a host that does its own recentring wants.
    pub fn pose7(&self) -> [f32; 7] {
        let q = Hot::load_quat(&self.hot.head);
        let [roll, pitch, yaw] = crate::Pose { tick: 0, q }.euler_deg();
        [roll, pitch, yaw, q[0], q[1], q[2], q[3]]
    }

    /// True once per new pose sample; clears the flag.
    pub fn pose_fresh(&self) -> bool {
        self.hot.fresh.swap(false, Ordering::Acquire)
    }

    /// Feeds the phone orientation, e.g. from `TYPE_GAME_ROTATION_VECTOR`.
    pub fn set_phone_quat(&self, q: [f32; 4]) {
        Hot::store_quat(&self.hot.phone, q);
        self.hot.phone_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Declares the current head and phone orientation to be the centre.
    pub fn recentre(&self) {
        let head = Hot::load_quat(&self.hot.head);
        let phone = Hot::load_quat(&self.hot.phone);
        if let Ok(mut c) = self.cold.lock() {
            c.pointer.recentre(head, phone);
        }
    }

    pub fn set_lookahead_s(&self, seconds: f32) {
        if let Ok(mut c) = self.cold.lock() {
            c.lookahead_s = seconds.clamp(0.0, 0.1);
        }
    }

    pub fn set_distance(&self, distance: f32) {
        if let Ok(mut c) = self.cold.lock() {
            c.pointer.distance = distance.max(0.1);
        }
    }

    /// One snapshot for the current frame.
    pub fn state(&self) -> XrState {
        let head = Hot::load_quat(&self.hot.head);
        let gyro = Hot::load_vec3(&self.hot.gyro);
        let phone = Hot::load_quat(&self.hot.phone);

        let (head_ref, lookahead, cursor) = match self.cold.lock() {
            Ok(c) => (c.pointer.head_ref, c.lookahead_s, c.pointer.cursor(head, phone)),
            Err(_) => ([1.0, 0.0, 0.0, 0.0], 0.0, None),
        };

        let relative = quat_mul(quat_conj(head_ref), head);
        let predicted = integrate(relative, gyro, lookahead);

        XrState {
            head_q: relative,
            predicted_q: predicted,
            cursor_x: cursor.map(|c| c.0).unwrap_or(0.0),
            cursor_y: cursor.map(|c| c.1).unwrap_or(0.0),
            cursor_valid: cursor.is_some() as u32,
            head_samples: self.hot.head_samples.load(Ordering::Relaxed),
            phone_samples: self.hot.phone_samples.load(Ordering::Relaxed),
        }
    }

    /// Where the head is looking, as a direction vector — useful for picking
    /// which window the gaze falls on.
    pub fn gaze_direction(&self) -> [f32; 3] {
        rotate(self.state().predicted_q, [0.0, 0.0, -1.0])
    }
}

impl Drop for Tracker {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(h) = self.reader.take() {
            let _ = h.join();
        }
    }
}

fn read_loop<T: Transport>(mut dev: Device<T>, hot: Arc<Hot>, stop: Arc<AtomicBool>) {
    while !stop.load(Ordering::Relaxed) {
        match dev.next_event(50_000_000) {
            Ok(Some(Event::Pose(p))) => {
                Hot::store_quat(&hot.head, p.q);
                hot.head_samples.fetch_add(1, Ordering::Relaxed);
                hot.fresh.store(true, Ordering::Release);
            }
            Ok(Some(Event::Raw(r))) => {
                for (a, v) in hot.gyro.iter().zip(r.gyro) {
                    a.store(v.to_bits(), Ordering::Relaxed);
                }
            }
            Ok(Some(Event::Other(_))) | Ok(None) => {}
            Err(_) => break,
        }
    }
    let _ = dev.set_imu(Streams::OFF, Rate::Hz120);
}

/// Extrapolates an orientation forward by `dt` seconds at angular rate
/// `omega` (rad/s, body frame): `q' = q * exp(0.5 * omega * dt)`.
///
/// Small-angle friendly and exact enough for the 10–30 ms a display pipeline
/// costs; the alternative is a cursor that lags behind every head turn.
#[inline]
pub fn integrate(q: [f32; 4], omega: [f32; 3], dt: f32) -> [f32; 4] {
    let theta = (omega[0] * omega[0] + omega[1] * omega[1] + omega[2] * omega[2]).sqrt() * dt;
    if theta < 1e-6 {
        return q;
    }
    let half = theta * 0.5;
    let s = half.sin() / (theta / dt);
    let delta = [half.cos(), omega[0] * s, omega[1] * s, omega[2] * s];
    let out = quat_mul(q, delta);
    let n = (out[0] * out[0] + out[1] * out[1] + out[2] * out[2] + out[3] * out[3]).sqrt();
    [out[0] / n, out[1] / n, out[2] / n, out[3] / n]
}

// ---- C ABI -----------------------------------------------------------------

/// Opens the tracker. `rate_hz` is one of 60/90/120/240; anything else is
/// rounded down to 120. Returns null on failure.
///
/// # Safety
/// `fd` must be an open usbfs descriptor that stays valid until `xr_close`.
#[no_mangle]
pub unsafe extern "C" fn xr_open(fd: i32, rate_hz: u32) -> *mut Tracker {
    let rate = match rate_hz {
        60 => Rate::Hz60,
        90 => Rate::Hz90,
        240 => Rate::Hz240,
        _ => Rate::Hz120,
    };
    match Tracker::open(fd, rate) {
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(_) => core::ptr::null_mut(),
    }
}

/// # Safety
/// `t` must come from [`xr_open`] and must not be used afterwards.
#[no_mangle]
pub unsafe extern "C" fn xr_close(t: *mut Tracker) {
    if !t.is_null() {
        drop(Box::from_raw(t));
    }
}

/// # Safety
/// `t` must be a live tracker.
#[no_mangle]
pub unsafe extern "C" fn xr_set_phone_quat(t: *mut Tracker, w: f32, x: f32, y: f32, z: f32) {
    if let Some(t) = t.as_ref() {
        t.set_phone_quat([w, x, y, z]);
    }
}

/// # Safety
/// `t` must be a live tracker.
#[no_mangle]
pub unsafe extern "C" fn xr_recentre(t: *mut Tracker) {
    if let Some(t) = t.as_ref() {
        t.recentre();
    }
}

/// # Safety
/// `t` must be a live tracker.
#[no_mangle]
pub unsafe extern "C" fn xr_set_lookahead_ms(t: *mut Tracker, ms: f32) {
    if let Some(t) = t.as_ref() {
        t.set_lookahead_s(ms / 1000.0);
    }
}

/// # Safety
/// `t` must be a live tracker.
#[no_mangle]
pub unsafe extern "C" fn xr_set_distance(t: *mut Tracker, distance: f32) {
    if let Some(t) = t.as_ref() {
        t.set_distance(distance);
    }
}

/// Fills `out` with the current snapshot. Returns 0 on success.
///
/// # Safety
/// `t` must be a live tracker and `out` must point at a writable [`XrState`].
#[no_mangle]
pub unsafe extern "C" fn xr_state(t: *mut Tracker, out: *mut XrState) -> i32 {
    match (t.as_ref(), out.as_mut()) {
        (Some(t), Some(out)) => {
            *out = t.state();
            0
        }
        _ => -1,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_rate_leaves_orientation_alone() {
        let q = [1.0, 0.0, 0.0, 0.0];
        assert_eq!(integrate(q, [0.0, 0.0, 0.0], 0.02), q);
    }

    /// A constant rate about Z for dt must rotate by exactly omega * dt.
    #[test]
    fn constant_rate_rotates_by_omega_dt() {
        let omega = [0.0, 0.0, 1.0]; // 1 rad/s
        let dt = 0.5; // -> 0.5 rad
        let q = integrate([1.0, 0.0, 0.0, 0.0], omega, dt);
        let angle = 2.0 * q[0].acos();
        assert!((angle - 0.5).abs() < 1e-4, "angle = {angle}");
        assert!((q[3] - (0.25f32).sin()).abs() < 1e-4, "z = {}", q[3]);
    }

    #[test]
    fn prediction_stays_normalised() {
        let q = integrate([1.0, 0.0, 0.0, 0.0], [3.0, -2.0, 1.5], 0.03);
        let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!((n - 1.0).abs() < 1e-5, "|q| = {n}");
    }

    #[test]
    fn state_is_plain_old_data() {
        // The JNI shim copies this straight through, so no padding surprises.
        assert_eq!(core::mem::size_of::<XrState>(), 4 * 4 + 4 * 4 + 4 + 4 + 4 + 4 + 8 + 8);
    }
}

/// Fills `out` with `[roll, pitch, yaw, qw, qx, qy, qz]`, matching what the
/// vendor SDK hands to its pose callback.
///
/// # Safety
/// `t` must be live, `out` must hold seven floats.
#[no_mangle]
pub unsafe extern "C" fn xr_pose7(t: *mut Tracker, out: *mut f32) -> i32 {
    match t.as_ref() {
        Some(t) if !out.is_null() => {
            core::ptr::copy_nonoverlapping(t.pose7().as_ptr(), out, 7);
            0
        }
        _ => -1,
    }
}

/// Non-zero once per new pose sample.
///
/// # Safety
/// `t` must be live.
#[no_mangle]
pub unsafe extern "C" fn xr_pose_fresh(t: *mut Tracker) -> i32 {
    t.as_ref().map(|t| t.pose_fresh() as i32).unwrap_or(0)
}

/// # Safety
/// `t` must be live.
#[no_mangle]
pub unsafe extern "C" fn xr_brightness(t: *mut Tracker) -> i32 {
    t.as_ref().map(|t| t.info.brightness).unwrap_or(-1)
}

/// # Safety
/// `t` must be live.
#[no_mangle]
pub unsafe extern "C" fn xr_volume(t: *mut Tracker) -> i32 {
    t.as_ref().map(|t| t.info.volume).unwrap_or(-1)
}

/// # Safety
/// `t` must be live.
#[no_mangle]
pub unsafe extern "C" fn xr_display_mode(t: *mut Tracker) -> i32 {
    t.as_ref().map(|t| t.info.display_mode).unwrap_or(-1)
}

/// Copies the firmware string into `out` as NUL-terminated ASCII.
///
/// # Safety
/// `t` must be live, `out` must hold `cap` bytes.
#[no_mangle]
pub unsafe extern "C" fn xr_firmware(t: *mut Tracker, out: *mut u8, cap: usize) -> i32 {
    let Some(t) = t.as_ref() else { return -1 };
    if out.is_null() || cap == 0 {
        return -1;
    }
    let bytes = t.info.firmware.as_bytes();
    let n = bytes.len().min(cap - 1);
    core::ptr::copy_nonoverlapping(bytes.as_ptr(), out, n);
    *out.add(n) = 0;
    n as i32
}
