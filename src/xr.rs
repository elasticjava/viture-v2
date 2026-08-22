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
    /// Version counters for [`Hot::head`] and [`Hot::phone`]. Odd while a write
    /// is in progress; see [`Hot::store_quat`].
    head_seq: AtomicU32,
    phone_seq: AtomicU32,
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
    /// Publishes several words so that no reader can see half of the set.
    ///
    /// The write side of a sequence lock. The counter goes odd before the
    /// stores and even after; a reader that saw an odd counter, or a different
    /// one either side of its read, tries again.
    ///
    /// Writers do not exclude each other here — the callers that use it are
    /// either the single reader thread or the rare configuration calls, and
    /// those are serialised by [`Tracker::writer`].
    #[inline]
    fn publish(seq: &AtomicU32, write: impl FnOnce()) {
        let version = seq.load(Ordering::Relaxed);
        // Odd: a write is in progress. Release, so the stores cannot be seen
        // before it.
        seq.store(version.wrapping_add(1), Ordering::Release);
        write();
        // Even again, and Release so the stores are visible before it is.
        seq.store(version.wrapping_add(2), Ordering::Release);
    }

    /// Reads a set of words that was whole at some instant.
    ///
    /// The read side. Never blocks and never waits on a writer: it retries a
    /// bounded number of times and then takes what is there, because on a path
    /// the renderer walks every frame a slightly stale value beats a stalled
    /// frame. The writer holds the lock for a handful of stores, so losing
    /// twice running is already unlikely and losing eight times is not a thing
    /// that happens.
    #[inline]
    fn consistent<T>(seq: &AtomicU32, read: impl Fn() -> T) -> T {
        for _ in 0..LOAD_RETRIES {
            let before = seq.load(Ordering::Acquire);
            if before & 1 != 0 {
                // A write is in progress. A pause hint, not a yield and not a
                // sleep: the writer is a few stores from finishing.
                core::hint::spin_loop();
                continue;
            }
            let value = read();
            if seq.load(Ordering::Acquire) == before {
                return value;
            }
        }
        read()
    }

    /// Publishes a quaternion so that no reader can see half of it.
    ///
    /// Four atomics written one at a time are not one atomic write, and this
    /// used to be exactly that: the reader thread stored w, x, y, z separately
    /// at 119 Hz while the render thread read them separately once a frame. A
    /// reader landing in the middle got w from one pose and x from the next,
    /// which is not a rotation — the components differ most during fast head
    /// movement, so the error was largest exactly when it was most visible.
    ///
    /// A sequence lock fixes it without a mutex, which matters on a path the
    /// renderer touches every frame and the reader touches every 8 ms. The
    /// counter goes odd before the write and even after; a reader that sees an
    /// odd counter, or a different one afterwards, tries again.
    fn store_quat(seq: &AtomicU32, slot: &[AtomicU32; 4], q: [f32; 4]) {
        Hot::publish(seq, || {
            for (a, v) in slot.iter().zip(q) {
                a.store(v.to_bits(), Ordering::Relaxed);
            }
        });
    }

    /// Reads a quaternion that was whole at some instant.
    ///
    /// Retries a bounded number of times rather than spinning: the writer holds
    /// the lock for four stores, so a reader is unlucky to lose once and
    /// essentially cannot lose repeatedly. Giving up returns the components as
    /// they are — which is what the old code always did — because a stale-ish
    /// pose beats blocking a frame.
    fn load_quat(seq: &AtomicU32, slot: &[AtomicU32; 4]) -> [f32; 4] {
        Hot::consistent(seq, || {
            let mut q = [0f32; 4];
            for (i, a) in slot.iter().enumerate() {
                q[i] = f32::from_bits(a.load(Ordering::Relaxed));
            }
            q
        })
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
/// The rarely-written settings, read on every frame.
///
/// These used to live behind a mutex, and `state()` took it once per frame on
/// the render thread. Uncontended that is cheap, and cheap is not the point: a
/// lock means the renderer can be made to *wait* for whichever thread holds it,
/// and on a phone whose scheduler can park a thread mid-critical-section that
/// is a dropped frame arriving through no fault of the renderer's.
///
/// They are written by hand three times a session and read a hundred and twenty
/// times a second, so they are published through the same sequence lock the
/// poses use. Readers never block. Writers still exclude each other — through
/// [`Tracker::writer`], which is never touched on the frame path.
struct Cold {
    /// Bumped around every write; see [`Hot::publish`].
    seq: AtomicU32,
    /// `[head_ref, phone_ref]` of the pointer, then its distance.
    pointer_head_ref: [AtomicU32; 4],
    pointer_phone_ref: [AtomicU32; 4],
    pointer_distance: AtomicU32,
    /// How far ahead to extrapolate, in seconds.
    lookahead_s: AtomicU32,
    /// Reference heading as a unit `(cos, sin)` about world Y.
    ///
    /// Recentring cancels **heading only**. Pitch and roll stay as the device
    /// reports them, which keeps them gravity-anchored. A full-orientation
    /// recentre folds the reference pitch into the yaw axis, and the scene then
    /// tilts as you pan — the workspace ends up visibly angled.
    ref_yaw: [AtomicU32; 2],
}

impl Default for Cold {
    /// Zero is not the identity here, and assuming it was cost a debugging
    /// session: a reference heading of `(0, 0)` is not "no rotation" but a
    /// degenerate one, and recentring against it produces a quaternion of
    /// nothing. The identity heading is `(cos, sin) = (1, 0)`.
    fn default() -> Cold {
        let bits = |v: f32| AtomicU32::new(v.to_bits());
        let identity = || [bits(1.0), bits(0.0), bits(0.0), bits(0.0)];
        let pointer = Pointer::default();
        Cold {
            seq: AtomicU32::new(0),
            pointer_head_ref: identity(),
            pointer_phone_ref: identity(),
            pointer_distance: bits(pointer.distance),
            lookahead_s: bits(0.0),
            ref_yaw: [bits(1.0), bits(0.0)],
        }
    }
}

impl Cold {
    /// Everything the frame path needs, read without blocking.
    fn snapshot(&self, forward: [f32; 3]) -> (Pointer, f32, (f32, f32)) {
        Hot::consistent(&self.seq, || {
            let quat = |slot: &[AtomicU32; 4]| {
                let mut q = [0f32; 4];
                for (i, a) in slot.iter().enumerate() {
                    q[i] = f32::from_bits(a.load(Ordering::Relaxed));
                }
                q
            };
            (
                Pointer {
                    head_ref: quat(&self.pointer_head_ref),
                    phone_ref: quat(&self.pointer_phone_ref),
                    distance: f32::from_bits(self.pointer_distance.load(Ordering::Relaxed)),
                    forward,
                },
                f32::from_bits(self.lookahead_s.load(Ordering::Relaxed)),
                (
                    f32::from_bits(self.ref_yaw[0].load(Ordering::Relaxed)),
                    f32::from_bits(self.ref_yaw[1].load(Ordering::Relaxed)),
                ),
            )
        })
    }
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
/// Public because it is the whole basis of prediction on this device — the raw
/// stream that would report rate directly cannot run alongside poses — and
/// because a caller integrating this stack wants to check it against a known
/// movement.
///
/// The rotation from one to the other is `conj(a) * b`; for a small rotation its
/// vector part is half the axis-angle, so the rate is twice that over `dt`. The
/// sign is normalised to the shorter arc, since `q` and `-q` are the same
/// orientation and the difference between them is half a turn.
#[inline]
pub fn angular_rate(a: [f32; 4], b: [f32; 4], dt: f32) -> [f32; 3] {
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
    cold: Cold,
    /// Serialises the rare configuration writers against each other. Never
    /// taken on the frame path — see [`Cold`].
    writer: Mutex<()>,
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
        Tracker::with_transport(Device::new(Usbfs::new(fd, 0)?), rate)
    }

    /// The same, over any transport.
    ///
    /// Everything above the wire — the reader loop, recentring, rate estimation,
    /// prediction — is the part that can be wrong in ways nobody notices until
    /// the glasses are on. Taking the transport as an argument is what lets it
    /// be driven from a scripted head movement instead. See [`crate::sim`].
    pub fn with_transport<T>(mut dev: Device<T>, rate: Rate) -> crate::Result<Tracker>
    where
        T: Transport + Send + 'static,
    {
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
        // Starting the stream is the one step here that must succeed, and a
        // single flaky read used to be enough to fail it — on a cable that
        // would have worked perfectly a moment later. Retried, because the
        // difference between "this device is not there" and "that read went
        // wrong" is not visible in one attempt.
        let mut attempt = 0;
        loop {
            match dev.set_imu(streams, rate) {
                Ok(()) => break,
                Err(e) if is_fatal(&e) => return Err(e),
                Err(e) => {
                    attempt += 1;
                    if attempt >= OPEN_ATTEMPTS {
                        return Err(e);
                    }
                }
            }
        }

        let hot = Arc::new(Hot::default());
        Hot::store_quat(&hot.head_seq, &hot.head, [1.0, 0.0, 0.0, 0.0]);
        Hot::store_quat(&hot.phone_seq, &hot.phone, [1.0, 0.0, 0.0, 0.0]);

        let stop = Arc::new(AtomicBool::new(false));
        let raw_stream = streams.has(Streams::RAW);
        let reader = {
            let hot = Arc::clone(&hot);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || read_loop(dev, hot, stop, raw_stream))
        };

        Ok(Tracker {
            hot,
            cold: Cold::default(),
            writer: Mutex::new(()),
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
        let q = Hot::load_quat(&self.hot.head_seq, &self.hot.head);
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
        Hot::store_quat(&self.hot.phone_seq, &self.hot.phone, q);
        self.hot.phone_samples.fetch_add(1, Ordering::Relaxed);
    }

    /// Declares the current head and phone orientation to be the centre.
    ///
    /// The head reference is heading-only; the phone reference is the full
    /// orientation, because the pointer is expressed in the head frame anyway.
    pub fn recentre(&self) {
        let head = Hot::load_quat(&self.hot.head_seq, &self.hot.head);
        let phone = Hot::load_quat(&self.hot.phone_seq, &self.hot.phone);
        let _writing = self.writer.lock();
        let (cos, sin) = yaw_twist(head);
        Hot::publish(&self.cold.seq, || {
            self.cold.ref_yaw[0].store(cos.to_bits(), Ordering::Relaxed);
            self.cold.ref_yaw[1].store(sin.to_bits(), Ordering::Relaxed);
            // The head handed to the pointer is already recentred, so its own
            // head reference stays the identity.
            for (a, v) in self
                .cold
                .pointer_head_ref
                .iter()
                .zip([1.0f32, 0.0, 0.0, 0.0])
            {
                a.store(v.to_bits(), Ordering::Relaxed);
            }
            for (a, v) in self.cold.pointer_phone_ref.iter().zip(phone) {
                a.store(v.to_bits(), Ordering::Relaxed);
            }
        });
    }

    pub fn set_lookahead_s(&self, seconds: f32) {
        let _writing = self.writer.lock();
        let clamped = seconds.clamp(0.0, 0.1);
        Hot::publish(&self.cold.seq, || {
            self.cold
                .lookahead_s
                .store(clamped.to_bits(), Ordering::Relaxed);
        });
    }

    pub fn set_distance(&self, distance: f32) {
        let _writing = self.writer.lock();
        let clamped = distance.max(0.1);
        Hot::publish(&self.cold.seq, || {
            self.cold
                .pointer_distance
                .store(clamped.to_bits(), Ordering::Relaxed);
        });
    }

    /// One snapshot for the current frame.
    pub fn state(&self) -> XrState {
        let head = Hot::load_quat(&self.hot.head_seq, &self.hot.head);
        let gyro = Hot::load_vec3(&self.hot.gyro);
        let phone = Hot::load_quat(&self.hot.phone_seq, &self.hot.phone);

        // No lock on this path. `state()` is called once per rendered frame and
        // at the pose rate besides; a mutex here means the renderer can be made
        // to wait for another thread, which is a dropped frame it did nothing
        // to deserve.
        let (pointer, lookahead, ref_yaw) = self.cold.snapshot(POINTER_FORWARD);
        // Heading-only recentring: pitch and roll stay gravity-anchored, so
        // panning cannot tilt the workspace.
        let relative = recentre_yaw(head, ref_yaw);
        let cursor = pointer.cursor(relative, phone);

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

/// Which phone axis points forward for the pointer: the top edge, which is what
/// pointing the phone like a wand feels like.
///
/// A constant rather than a stored field, because nothing changes it and a
/// value nobody writes does not need publishing.
const POINTER_FORWARD: [f32; 3] = [0.0, 1.0, 0.0];

/// How many times a reader retries a torn quaternion before taking what is
/// there. The writer holds the lock for four stores, so losing twice in a row
/// is already improbable and losing eight times is not a thing that happens.
const LOAD_RETRIES: u32 = 8;

/// How many times to ask the device to start streaming before giving up.
///
/// Each attempt is a command and a reply with its own timeout, so this is
/// seconds at worst and only on a device that is genuinely not answering. The
/// alternative — one attempt — meant a cable that hiccupped at the wrong moment
/// looked exactly like no glasses at all.
const OPEN_ATTEMPTS: u32 = 5;

/// How many reads may fail in a row before the stream is called dead.
///
/// At a 50 ms poll timeout this is a couple of seconds of nothing working —
/// far longer than any glitch and far shorter than a person's patience. Low
/// enough that a genuinely dead device does not spin, high enough that a cable
/// being flexed does not end the session.
const MAX_CONSECUTIVE_ERRORS: u32 = 40;

/// Whether an error means the device has gone, as opposed to a read having gone
/// wrong.
///
/// Worth drawing precisely: wrong in one direction this ends a working session,
/// wrong in the other it spins forever on a device that is not there.
///
/// Gone — the descriptor no longer refers to anything. Unplugged, or the kernel
/// tore the interface down; nothing that follows can succeed.
///
/// Not gone — a frame that did not parse, a checksum that did not match, a
/// timeout, a transfer the bus gave up on. All survivable, several routine on a
/// bus shared with a video stream.
fn is_fatal(error: &crate::Error) -> bool {
    match error {
        crate::Error::Io(io) => matches!(
            io.raw_os_error(),
            // ENOENT, EBADF, ENODEV, EPIPE, ESHUTDOWN. usbfs reports ENODEV on
            // an unplug, which is the one that matters.
            Some(2) | Some(9) | Some(19) | Some(32) | Some(108)
        ),
        _ => false,
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
    // Errors since the last read that worked. Cleared by any success.
    let mut consecutive_errors: u32 = 0;
    while !stop.load(Ordering::Relaxed) {
        // Send a queued display-mode change between reads — the only place the
        // transport is free.
        let pending = hot.pending_mode.swap(0, Ordering::Acquire);
        if pending & PENDING_SET != 0 {
            let _ = dev.set_display_mode_raw((pending & 0xFF) as u8);
        }
        match dev.next_event(50_000_000) {
            Ok(Some(Event::Pose(p))) => {
                consecutive_errors = 0;
                Hot::store_quat(&hot.head_seq, &hot.head, p.q);
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
                consecutive_errors = 0;
                for (a, v) in hot.gyro.iter().zip(r.gyro) {
                    a.store(v.to_bits(), Ordering::Relaxed);
                }
            }
            Ok(Some(Event::Other(_))) => {
                consecutive_errors = 0;
                hot.other_events.fetch_add(1, Ordering::Relaxed);
            }
            Ok(None) => {
                consecutive_errors = 0;
            }
            Err(e) => {
                // Record why, whether or not it proves fatal. Silence here was
                // impossible to diagnose from the outside.
                let code = match &e {
                    crate::Error::Io(io) => io.raw_os_error().unwrap_or(-1),
                    _ => -2,
                };
                hot.reader_errno.store(code as u32, Ordering::Relaxed);

                // Whether to stop. This used to stop on anything at all, which
                // meant one flaky read ended head tracking for the whole
                // session — and a cable in a pocket produces those. The
                // watchdogs upstairs exist because of it.
                //
                // A device that has gone is worth stopping for; a read that
                // went wrong is not. What separates them is not the error alone
                // but whether anything works afterwards, so transient errors
                // are counted and the count is cleared by the next success.
                // Persistent failure still ends the loop, on evidence rather
                // than on suspicion.
                if is_fatal(&e) {
                    break;
                }
                consecutive_errors += 1;
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS {
                    break;
                }
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

/// Opens a tracker with no glasses on the other end.
///
/// The whole stack above the wire runs against it: the reader thread, the ring,
/// recentring, rate estimation, prediction, and every command the host sends.
/// What it answers with is what the hardware was measured answering — see
/// [`crate::sim::measured`] — so a session driven this way is a session, not a
/// demonstration.
///
/// This exists because the glasses are one cable and one battery, and work on
/// everything above them should not stop when either runs out. It is also the
/// only way to test the parts that need the head to do something specific: a
/// person cannot turn their head at exactly 45 degrees a second for two
/// seconds, and [`crate::sim::Turning`] can.
///
/// `motion` picks what the simulated head does: 0 still, 1 turning steadily,
/// 2 turning and then stopping. `rate_hz` is accepted for symmetry with
/// [`xr_open`] and, as on the device, is what the host *asks* for rather than
/// what arrives — the simulation delivers at the measured rate.
///
/// Returns null if the simulated device refuses to start, which it does not,
/// but the caller's error path should be the same either way.
#[no_mangle]
#[cfg(feature = "sim")]
pub extern "C" fn xr_open_simulated(rate_hz: u32, motion: u32) -> *mut Tracker {
    use crate::sim::{Simulated, Still, TurnThenStop, Turning};
    const TURN_RATE: f32 = 45.0 * core::f32::consts::PI / 180.0;
    let rate = match rate_hz {
        60 => Rate::Hz60,
        90 => Rate::Hz90,
        240 => Rate::Hz240,
        _ => Rate::Hz120,
    };
    let trajectory: Box<dyn crate::sim::Trajectory> = match motion {
        // 45 degrees a second is a brisk but ordinary look-around, and it is
        // the motion that shows lag.
        1 => Box::new(Turning {
            axis: [0.0, 1.0, 0.0],
            rate: TURN_RATE,
        }),
        2 => Box::new(TurnThenStop {
            axis: [0.0, 1.0, 0.0],
            rate: TURN_RATE,
            until: 2.0,
        }),
        _ => Box::new(Still::default()),
    };
    // Paced: the caller is a renderer reading a clock, and an unpaced stream
    // would hand it a thousand poses before the first frame.
    let device = crate::Device::new(Simulated::paced(trajectory));
    match Tracker::with_transport(device, rate) {
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(_) => core::ptr::null_mut(),
    }
}

/// Whether this build can open a tracker without hardware.
///
/// The caller asks rather than assuming, so a release built without the
/// simulation reports it instead of returning null from a function that looks
/// like it should have worked.
#[no_mangle]
pub extern "C" fn xr_has_simulation() -> i32 {
    i32::from(cfg!(feature = "sim"))
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
