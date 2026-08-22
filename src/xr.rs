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
//! `xr_state` is the only call in the render loop: it reads the hot values from
//! atomics, does a prediction step and returns. No allocation and no syscall.
//! It does take one uncontended mutex for the recentre reference and the
//! look-ahead — those are written only when the user recentres or turns a knob,
//! so the render loop never waits on it in practice.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::pointer::{rotate, Pointer};

use crate::usbfs::Usbfs;
use crate::{quat_mul, Device, Event, Rate, Streams, Transport};

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
    /// Set while the reader thread is alive, cleared when it leaves.
    reader_alive: AtomicBool,
    /// Errno of the error that ended the reader, or 0.
    reader_errno: AtomicU32,
    /// A display mode waiting to be sent, tagged with [`PENDING_SET`].
    pending_mode: AtomicU32,
    /// Events seen that were neither pose nor raw — a mapping gap would show up
    /// here rather than as silence.
    other_events: AtomicU64,
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
    /// Reference heading as a unit `(cos, sin)` about world Y.
    ///
    /// Recentring cancels **heading only**. Pitch and roll stay as the device
    /// reports them, which keeps them gravity-anchored. A full-orientation
    /// recentre folds the reference pitch into the yaw axis, and the scene then
    /// tilts as you pan — the workspace ends up visibly angled. This mirrors the
    /// semantics the Kotlin side used before the maths moved down here.
    ref_yaw: (f32, f32),
}

/// Marks [`Hot::pending_mode`] as carrying a request rather than being idle.
const PENDING_SET: u32 = 0x100;

/// How much of each newly measured angular rate to keep, per sample at 120 Hz.
///
/// Differentiating positions amplifies their noise, and prediction turns rate
/// noise straight into a restless image. A third per sample settles within a
/// couple of frames — fast enough that a head turn is not damped, slow enough
/// that a stationary head does not shimmer.
const RATE_SMOOTHING: f32 = 0.33;

/// Body-frame angular velocity, in radians per second, between two orientations
/// `dt` apart.
///
/// The rotation from one to the other is `conj(a) * b`; for a small rotation its
/// vector part is half the axis-angle, so the rate is twice that over `dt`. The
/// sign is normalised to the shorter arc, since `q` and `-q` are the same
/// orientation and the difference between them is half a turn.
#[inline]
fn angular_rate(a: [f32; 4], b: [f32; 4], dt: f32) -> [f32; 3] {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    let mut d = [
        aw * bw + ax * bx + ay * by + az * bz,
        aw * bx - ax * bw - ay * bz + az * by,
        aw * by + ax * bz - ay * bw - az * bx,
        aw * bz - ax * by + ay * bx - az * bw,
    ];
    if d[0] < 0.0 {
        d = [-d[0], -d[1], -d[2], -d[3]];
    }
    let k = 2.0 / dt;
    [d[1] * k, d[2] * k, d[3] * k]
}

/// Cancels the reference heading from an orientation: `conj(yaw_twist) * q`,
/// with `yaw_twist = (cos, 0, sin, 0)`.
#[inline]
fn recentre_yaw(q: [f32; 4], (c, s): (f32, f32)) -> [f32; 4] {
    let [w, x, y, z] = q;
    [c * w + s * y, c * x - s * z, c * y - s * w, c * z + s * x]
}

/// The heading part of an orientation, as a unit `(cos, sin)` about world Y.
#[inline]
fn yaw_twist(q: [f32; 4]) -> (f32, f32) {
    let n = (q[0] * q[0] + q[2] * q[2]).sqrt();
    if n > 1e-6 {
        (q[0] / n, q[2] / n)
    } else {
        (1.0, 0.0)
    }
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
            firmware: dev
                .firmware_version(&mut buf)
                .unwrap_or("unknown")
                .to_owned(),
            brightness: dev.brightness().map(i32::from).unwrap_or(-1),
            volume: dev.volume().map(i32::from).unwrap_or(-1),
            display_mode: dev
                .get_u8(crate::msg::DISPLAY_MODE)
                .map(i32::from)
                .unwrap_or(-1),
            worn: dev.worn().unwrap_or(false),
        };

        // Pose only by default. Running both streams at once relies on the wire
        // field being a bitmask, which is inferred rather than documented — and
        // the device accepts `3` with a success ACK and then sends nothing at
        // all, so the error path never fires and tracking simply stays silent.
        // That cost an afternoon; do not re-enable it without measuring.
        //
        // Set VITURE_STREAMS=raw or =both to experiment. Without raw the
        // prediction has no angular rate and degrades to a no-op, which is safe.
        let streams = match std::env::var("VITURE_STREAMS").as_deref() {
            Ok("raw") => Streams::RAW,
            Ok("both") => Streams::POSE.with(Streams::RAW),
            _ => Streams::POSE,
        };
        dev.set_imu(streams, rate)?;

        let hot = Arc::new(Hot::default());
        Hot::store_quat(&hot.head, [1.0, 0.0, 0.0, 0.0]);
        Hot::store_quat(&hot.phone, [1.0, 0.0, 0.0, 0.0]);

        let stop = Arc::new(AtomicBool::new(false));
        let raw_stream = streams.has(Streams::RAW);
        let reader = {
            let hot = Arc::clone(&hot);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || read_loop(dev, hot, stop, raw_stream))
        };

        Ok(Tracker {
            hot,
            cold: Mutex::new(Cold {
                pointer: Pointer::default(),
                lookahead_s: 0.020,
                ref_yaw: (1.0, 0.0),
            }),
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

    /// Queues a display-mode change for the reader thread to send.
    ///
    /// The reader owns the transport, so the command cannot simply be issued
    /// here. It is handed over as a pending request and goes out between two
    /// reads, which is also the only point where the reply can be observed.
    pub fn request_display_mode(&self, mode: u8) -> i32 {
        self.hot
            .pending_mode
            .store(mode as u32 | PENDING_SET, Ordering::Release);
        0
    }

    /// The angular rate the reader last observed, in radians per second, body
    /// frame. Measured from the raw stream where it runs, differentiated from
    /// the pose stream otherwise.
    pub fn angular_rate(&self) -> [f32; 3] {
        Hot::load_vec3(&self.hot.gyro)
    }

    /// Extrapolates an orientation forward by `dt` seconds at the current
    /// angular rate.
    ///
    /// This is what closes the gap between where the head was when a pose was
    /// sampled and where it will be when the frame reaches the panel — a pose is
    /// a few milliseconds old before it is read, and the frame built from it is
    /// two or three refreshes from being seen. Left uncorrected, that shows up
    /// as the whole world sliding a little behind every head turn.
    ///
    /// The caller passes the orientation rather than having it read here so that
    /// a renderer can predict the frame it is about to draw from the pose it has
    /// already recentred. Body-frame rate is unaffected by recentring, which
    /// left-multiplies by a fixed world rotation, so the same rate applies.
    pub fn predict(&self, q: [f32; 4], dt: f32) -> [f32; 4] {
        integrate(q, self.angular_rate(), dt)
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
    ///
    /// The head reference is heading-only; the phone reference is the full
    /// orientation, because the pointer is expressed in the head frame anyway.
    pub fn recentre(&self) {
        let head = Hot::load_quat(&self.hot.head);
        let phone = Hot::load_quat(&self.hot.phone);
        if let Ok(mut c) = self.cold.lock() {
            c.ref_yaw = yaw_twist(head);
            // The head handed to the pointer is already recentred, so its own
            // head reference stays the identity.
            c.pointer.recentre([1.0, 0.0, 0.0, 0.0], phone);
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

        let (relative, lookahead, cursor) = match self.cold.lock() {
            Ok(c) => {
                // Heading-only recentring: pitch and roll stay gravity-anchored,
                // so panning cannot tilt the workspace.
                let relative = recentre_yaw(head, c.ref_yaw);
                (relative, c.lookahead_s, c.pointer.cursor(relative, phone))
            }
            Err(_) => (head, 0.0, None),
        };

        // Extrapolating the recentred orientation rather than the raw one keeps
        // the prediction in the frame the renderer actually draws in.
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

fn read_loop<T: Transport>(
    mut dev: Device<T>,
    hot: Arc<Hot>,
    stop: Arc<AtomicBool>,
    raw_stream: bool,
) {
    hot.reader_alive.store(true, Ordering::Release);
    // Previous pose and when it arrived, for estimating angular rate.
    let mut previous: Option<([f32; 4], std::time::Instant)> = None;
    while !stop.load(Ordering::Relaxed) {
        // Send a queued display-mode change between reads — the only place the
        // transport is free.
        let pending = hot.pending_mode.swap(0, Ordering::Acquire);
        if pending & PENDING_SET != 0 {
            let _ = dev.set_display_mode_raw((pending & 0xFF) as u8);
        }
        match dev.next_event(50_000_000) {
            Ok(Some(Event::Pose(p))) => {
                Hot::store_quat(&hot.head, p.q);
                hot.head_samples.fetch_add(1, Ordering::Relaxed);
                hot.fresh.store(true, Ordering::Release);

                // Angular rate, differentiated from the pose stream.
                //
                // The device can report rates directly, but only on the raw
                // stream, and asking for pose and raw together makes it ACK the
                // request and then send nothing at all — see `Tracker::open`. So
                // the rate is recovered from the poses instead, which arrive at
                // 120 Hz and are all that is needed to predict a few tens of
                // milliseconds ahead.
                //
                // Skipped when the raw stream is running, because then the
                // device's own measurements are better than a difference.
                let now = std::time::Instant::now();
                if !raw_stream {
                    if let Some((before, then)) = previous {
                        let dt = now.duration_since(then).as_secs_f32();
                        // Below a tenth of a millisecond the division amplifies
                        // quantisation into noise; above a tenth of a second the
                        // stream has stalled and the old rate is meaningless.
                        if (1e-4..0.1).contains(&dt) {
                            let measured = angular_rate(before, p.q, dt);
                            for (a, v) in hot.gyro.iter().zip(measured) {
                                let last = f32::from_bits(a.load(Ordering::Relaxed));
                                // A light low-pass. Differentiating amplifies the
                                // jitter in each sample, and prediction turns rate
                                // noise into a visibly restless image; this trades
                                // a sample of lag for a steady one.
                                let smoothed = last + (v - last) * RATE_SMOOTHING;
                                a.store(smoothed.to_bits(), Ordering::Relaxed);
                            }
                        }
                    }
                    previous = Some((p.q, now));
                }
            }
            Ok(Some(Event::Raw(r))) => {
                for (a, v) in hot.gyro.iter().zip(r.gyro) {
                    a.store(v.to_bits(), Ordering::Relaxed);
                }
            }
            Ok(Some(Event::Other(_))) => {
                hot.other_events.fetch_add(1, Ordering::Relaxed);
            }
            Ok(None) => {}
            Err(e) => {
                // Record why the reader gave up. Silence here was impossible to
                // diagnose from the outside.
                let code = match &e {
                    crate::Error::Io(io) => io.raw_os_error().unwrap_or(-1),
                    _ => -2,
                };
                hot.reader_errno.store(code as u32, Ordering::Relaxed);
                break;
            }
        }
    }
    hot.reader_alive.store(false, Ordering::Release);
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
/// Extrapolates `[w, x, y, z]` forward by `dt_seconds` at the tracker's current
/// angular rate, writing the result to `out`.
///
/// # Safety
/// `out` must point to four writable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_predict(
    t: *mut Tracker,
    w: f32,
    x: f32,
    y: f32,
    z: f32,
    dt_seconds: f32,
    out: *mut f32,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let Some(t) = t.as_ref() else { return -1 };
    let q = t.predict([w, x, y, z], dt_seconds);
    std::ptr::copy_nonoverlapping(q.as_ptr(), out, 4);
    0
}

/// The tracker's current angular rate, radians per second, body frame.
///
/// # Safety
/// `out` must point to three writable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_angular_rate(t: *mut Tracker, out: *mut f32) -> i32 {
    if out.is_null() {
        return -1;
    }
    let Some(t) = t.as_ref() else { return -1 };
    let r = t.angular_rate();
    std::ptr::copy_nonoverlapping(r.as_ptr(), out, 3);
    0
}

/// True once per new pose sample, then false until the next one.
///
/// # Safety
/// `t` must be a tracker from [`xr_open`], or null.
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

/// Diagnostics for the JNI side: `[head_samples, phone_samples, reader_alive,
/// reader_errno, other_events]`. Silence in the pose stream is otherwise
/// impossible to tell apart from a dead reader thread.
///
/// # Safety
/// `t` must be live and `out` must hold five `i64`s.
#[no_mangle]
pub unsafe extern "C" fn xr_diag(t: *mut Tracker, out: *mut i64) -> i32 {
    let Some(t) = t.as_ref() else { return -1 };
    if out.is_null() {
        return -1;
    }
    let values = [
        t.hot.head_samples.load(Ordering::Relaxed) as i64,
        t.hot.phone_samples.load(Ordering::Relaxed) as i64,
        t.hot.reader_alive.load(Ordering::Acquire) as i64,
        t.hot.reader_errno.load(Ordering::Relaxed) as i32 as i64,
        t.hot.other_events.load(Ordering::Relaxed) as i64,
    ];
    core::ptr::copy_nonoverlapping(values.as_ptr(), out, values.len());
    0
}

/// Switches the panel's display mode. `0x31` = 1920×1080 2D, `0x32` = 3840×1080
/// side-by-side 3D.
///
/// The device owns the command path from its reader thread, so this reopens a
/// short-lived command channel rather than fighting it. Returns 0 on success.
///
/// # Safety
/// `t` must be live.
#[no_mangle]
pub unsafe extern "C" fn xr_set_display_mode(t: *mut Tracker, mode: u8) -> i32 {
    match t.as_ref() {
        Some(t) => t.request_display_mode(mode),
        None => -1,
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
    fn angular_rate_recovers_a_known_turn() {
        // Half a second of yawing at 1 rad/s, sampled 8 ms apart, must read back
        // as 1 rad/s about Y. This is the measurement prediction rests on when
        // the device will not stream its own rates.
        let dt = 1.0 / 120.0;
        let at = |t: f32| {
            let half = 0.5 * t; // 1 rad/s
            [half.cos(), 0.0, half.sin(), 0.0]
        };
        let rate = angular_rate(at(0.5), at(0.5 + dt), dt);
        assert!(rate[0].abs() < 1e-3, "x {}", rate[0]);
        assert!((rate[1] - 1.0).abs() < 1e-3, "y {}", rate[1]);
        assert!(rate[2].abs() < 1e-3, "z {}", rate[2]);
    }

    #[test]
    fn angular_rate_ignores_the_sign_of_the_quaternion() {
        // q and -q are the same orientation. Reading the difference between them
        // literally gives half a turn in one sample, which would send the
        // prediction across the room.
        let dt = 1.0 / 120.0;
        let a = [0.9999, 0.0, 0.0139, 0.0];
        let b = [-0.9998, 0.0, -0.0208, 0.0];
        let rate = angular_rate(a, b, dt);
        let speed = (rate[0] * rate[0] + rate[1] * rate[1] + rate[2] * rate[2]).sqrt();
        assert!(speed < 5.0, "flipped sign read as {speed} rad/s");
    }

    #[test]
    fn a_still_head_has_no_rate() {
        let q = [
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
            std::f32::consts::FRAC_1_SQRT_2,
            0.0,
        ];
        let rate = angular_rate(q, q, 1.0 / 120.0);
        assert!(rate.iter().all(|v| v.abs() < 1e-5), "{rate:?}");
    }

    #[test]
    fn prediction_leads_a_steady_turn() {
        // Extrapolating by dt from a pose must land where the head actually is
        // dt later. This is the whole point: without it the image trails the
        // head by the length of the display pipeline.
        let dt = 0.030;
        let at = |t: f32| {
            let half = 0.5 * t;
            [half.cos(), 0.0, half.sin(), 0.0]
        };
        let predicted = integrate(at(0.0), [0.0, 1.0, 0.0], dt);
        let actual = at(dt);
        for (p, a) in predicted.iter().zip(actual) {
            assert!(
                (p - a).abs() < 1e-4,
                "predicted {predicted:?} actual {actual:?}"
            );
        }
    }

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

    /// Recentring must cancel heading and leave pitch and roll alone — the
    /// property the Kotlin side relied on before this moved into the driver.
    #[test]
    fn recentre_cancels_heading_but_keeps_pitch() {
        // 30 degrees yaw about Y, combined with 20 degrees pitch about X.
        let ya = 30f32.to_radians() / 2.0;
        let pa = 20f32.to_radians() / 2.0;
        let yaw = [ya.cos(), 0.0, ya.sin(), 0.0];
        let pitch = [pa.cos(), pa.sin(), 0.0, 0.0];
        let head = quat_mul(yaw, pitch);

        let recentred = recentre_yaw(head, yaw_twist(head));
        // Mind the naming clash: `euler_deg` follows the aerospace convention
        // where the Y rotation is called "pitch" and Z is "yaw", while the
        // glasses frame uses Y as the heading (up) axis. Cancelling the heading
        // about Y therefore has to leave the X rotation untouched — and that is
        // what `euler_deg` reports as roll.
        let [roll, heading, _] = crate::Pose {
            tick: 0,
            q: recentred,
        }
        .euler_deg();

        assert!(heading.abs() < 0.5, "heading not cancelled: {heading}");
        assert!(
            (roll.abs() - 20.0).abs() < 0.5,
            "the X rotation was changed: {roll}"
        );
    }

    /// Pure heading must recentre to the identity.
    #[test]
    fn pure_heading_recentres_to_identity() {
        let a = 47f32.to_radians() / 2.0;
        let head = [a.cos(), 0.0, a.sin(), 0.0];
        let r = recentre_yaw(head, yaw_twist(head));
        assert!((r[0] - 1.0).abs() < 1e-4, "w = {}", r[0]);
        assert!(r[2].abs() < 1e-4, "y = {}", r[2]);
    }

    #[test]
    fn state_is_plain_old_data() {
        // The JNI shim copies this straight through, so no padding surprises.
        assert_eq!(
            core::mem::size_of::<XrState>(),
            4 * 4 + 4 * 4 + 4 + 4 + 4 + 4 + 8 + 8
        );
    }
}
