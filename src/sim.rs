//! A simulated pair of glasses.
//!
//! Everything above the wire — frame parsing, the reader thread, the ring
//! buffer, recentring, angular-rate estimation, prediction — is exercised by
//! whatever the transport hands up, and the transport is a trait. So the whole
//! stack can be driven from a scripted head movement with no hardware attached,
//! and the tests can assert things a person wearing the glasses cannot: that the
//! pose after two seconds of turning is *exactly* where it should be, that
//! prediction leads by the amount asked for, that a stalled stream stops the
//! prediction rather than sailing off.
//!
//! It is a simulation of the device, not of the driver. Real frames are built
//! with [`crate::build`] and parsed by [`crate::parse`], so a mistake in the
//! header, the checksum or the payload layout fails here exactly as it would on
//! a desk. What is *not* simulated is USB itself: no lost packets, no partial
//! reads, no EBUSY. Those belong to the transports, and each has its own.
//!
//! Available in test builds and behind the `sim` feature, so the harness can
//! also be driven from an example or a benchmark.

use crate::{build, msg, Error, Rate, Result, Streams, Transport, FRAME_MAX, RESPONSE_OFFSET};

/// Where the head is at a given moment.
pub trait Trajectory: Send {
    /// Orientation as `[w, x, y, z]` at `t` seconds since the stream started.
    fn at(&self, t: f32) -> [f32; 4];
}

/// A head that does not move.
pub struct Still(pub [f32; 4]);

impl Default for Still {
    fn default() -> Self {
        Still([1.0, 0.0, 0.0, 0.0])
    }
}

impl Trajectory for Still {
    fn at(&self, _t: f32) -> [f32; 4] {
        self.0
    }
}

/// A head turning at a constant rate about an axis — the motion the whole
/// pipeline is judged on, because it is the one that shows lag.
pub struct Turning {
    /// Radians per second.
    pub rate: f32,
    /// Unit axis in the body frame; `[0, 1, 0]` is a yaw.
    pub axis: [f32; 3],
}

impl Trajectory for Turning {
    fn at(&self, t: f32) -> [f32; 4] {
        let half = 0.5 * self.rate * t;
        let (s, c) = half.sin_cos();
        [c, self.axis[0] * s, self.axis[1] * s, self.axis[2] * s]
    }
}

/// A head that turns and then stops, for testing what prediction does at the
/// end of a movement — where over-prediction shows up as a spring-back.
pub struct TurnThenStop {
    pub rate: f32,
    pub axis: [f32; 3],
    /// When the movement ends, in seconds.
    pub until: f32,
}

impl Trajectory for TurnThenStop {
    fn at(&self, t: f32) -> [f32; 4] {
        Turning {
            rate: self.rate,
            axis: self.axis,
        }
        .at(t.min(self.until))
    }
}

/// The device's own state, as the driver can observe it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceState {
    pub streams: Streams,
    pub rate: Rate,
    pub display_mode: u8,
    pub brightness: u8,
    pub volume: u8,
    pub worn: bool,
}

impl Default for DeviceState {
    fn default() -> Self {
        DeviceState {
            streams: Streams::OFF,
            rate: Rate::Hz120,
            // 1920x1080 at 60 Hz, which is what a Pro 2 comes up in.
            display_mode: 0x31,
            brightness: 5,
            volume: 60,
            worn: true,
        }
    }
}

/// A transport that answers like the glasses do.
///
/// Time is a counter rather than a clock: `recv` advances it by one sampling
/// interval each time it produces an event. A test therefore runs in
/// microseconds and is exactly reproducible, which a wall clock would not be.
pub struct Simulated {
    trajectory: Box<dyn Trajectory>,
    state: DeviceState,
    /// Samples emitted so far; the clock.
    tick: u32,
    /// Queued replies, in order, each a complete frame.
    replies: Vec<Vec<u8>>,
    /// Events to withhold, simulating a stream that has stopped.
    stalled: bool,
    /// Whether to spend real time between samples.
    paced: bool,
    /// When the first paced sample went out, so simulated and real time agree.
    started: Option<std::time::Instant>,
    /// Every command the driver sent, for asserting on.
    pub sent: Vec<(u16, Vec<u8>)>,
}

impl Simulated {
    pub fn new(trajectory: Box<dyn Trajectory>) -> Simulated {
        Simulated {
            trajectory,
            state: DeviceState::default(),
            tick: 0,
            replies: Vec::new(),
            stalled: false,
            paced: false,
            started: None,
            sent: Vec::new(),
        }
    }

    /// A trajectory sampled in real time, for code that measures the clock.
    pub fn paced(trajectory: Box<dyn Trajectory>) -> Simulated {
        let mut sim = Simulated::new(trajectory);
        sim.pace(true);
        sim
    }

    /// A still head, which is what most command-path tests want.
    pub fn still() -> Simulated {
        Simulated::new(Box::new(Still::default()))
    }

    pub fn state(&self) -> DeviceState {
        self.state
    }

    /// Makes each sample take the wall time it would on the device.
    ///
    /// Off by default, because a test that only checks orientations should not
    /// spend a second doing it. On when the code under test measures elapsed
    /// time — the angular-rate estimate divides by the interval between two
    /// poses, and an interval of nothing is discarded as noise. Simulated time
    /// cannot stand in for that, because the measurement is of the real clock.
    pub fn pace(&mut self, paced: bool) {
        self.paced = paced;
    }

    /// Stops producing events without disconnecting — a stream that has died
    /// while the device still answers commands, which is the failure the
    /// watchdogs exist for.
    pub fn stall(&mut self, stalled: bool) {
        self.stalled = stalled;
    }

    /// Seconds represented by one sample at the configured rate.
    pub fn interval(&self) -> f32 {
        1.0 / self.state.rate.hz() as f32
    }

    /// How far the clock has advanced.
    ///
    /// Counted in samples when running free, and read from the real clock when
    /// paced. The difference matters: paced mode exists because the code under
    /// test measures elapsed time, and if the trajectory advanced by a fixed
    /// step while the driver measured a longer real interval — which is what a
    /// loaded machine produces — the rate it computed would come out too low
    /// through no fault of its own. Taking both from the same clock makes an
    /// overrun harmless.
    pub fn elapsed(&self) -> f32 {
        match self.started {
            Some(start) => start.elapsed().as_secs_f32(),
            None => self.tick as f32 * self.interval(),
        }
    }

    fn reply(&mut self, msg_id: u16, payload: &[u8]) {
        let mut body = Vec::with_capacity(payload.len() + 1);
        body.push(0); // status: success
        body.extend_from_slice(payload);
        let mut buf = [0u8; FRAME_MAX];
        let frame = build(&mut buf, msg_id + RESPONSE_OFFSET, &body).expect("reply fits");
        self.replies.push(frame.to_vec());
    }

    /// Builds the pose event the device would send for the current tick.
    fn pose_event(&self) -> Vec<u8> {
        let q = self.trajectory.at(self.elapsed());
        // Payload of EVT_POSE: u32 unknown, u32 tick, then four floats.
        let mut payload = Vec::with_capacity(24);
        payload.extend_from_slice(&0u32.to_le_bytes());
        payload.extend_from_slice(&self.tick.to_le_bytes());
        for v in q {
            payload.extend_from_slice(&v.to_le_bytes());
        }
        let mut buf = [0u8; FRAME_MAX];
        build(&mut buf, msg::EVT_POSE, &payload)
            .expect("pose fits")
            .to_vec()
    }
}

impl Transport for Simulated {
    fn send(&mut self, frame: &[u8]) -> Result<()> {
        let parsed = crate::parse(frame)?;
        self.sent.push((parsed.msg_id, parsed.payload.to_vec()));

        match parsed.msg_id {
            msg::IMU_CTRL => {
                // Payload is the stream bitmask then the rate.
                if parsed.payload.len() >= 2 {
                    self.state.streams = Streams(parsed.payload[0]);
                    self.state.rate = match parsed.payload[1] {
                        0 => Rate::Hz60,
                        1 => Rate::Hz90,
                        3 => Rate::Hz240,
                        _ => Rate::Hz120,
                    };
                }
                self.reply(msg::IMU_CTRL, &[]);
            }
            msg::SET_DISPLAY_MODE => {
                if let Some(&mode) = parsed.payload.first() {
                    self.state.display_mode = mode;
                }
                self.reply(msg::SET_DISPLAY_MODE, &[]);
            }
            msg::DISPLAY_MODE => {
                let mode = self.state.display_mode;
                self.reply(msg::DISPLAY_MODE, &[mode]);
            }
            msg::BRIGHTNESS => {
                let v = self.state.brightness;
                self.reply(msg::BRIGHTNESS, &[v]);
            }
            msg::VOLUME => {
                let v = self.state.volume;
                self.reply(msg::VOLUME, &[v]);
            }
            msg::WEAR_STATUS => {
                let worn = self.state.worn as u8;
                self.reply(msg::WEAR_STATUS, &[worn]);
            }
            msg::VERSION => {
                self.reply(msg::VERSION, b"SIM-1.0.0");
            }
            msg::SERIAL => {
                self.reply(msg::SERIAL, b"SIMULATED");
            }
            other => {
                // Unknown commands are acknowledged rather than ignored: a
                // device that never replies would hang the driver, and that is
                // the transport's job to test, not this one's.
                self.reply(other, &[]);
            }
        }
        Ok(())
    }

    fn recv(&mut self, buf: &mut [u8; FRAME_MAX], _timeout_ns: u64) -> Result<usize> {
        // Replies come first and jump the queue, exactly as they do on the wire:
        // a command sent between two samples is answered before the next one.
        if !self.replies.is_empty() {
            let reply = self.replies.remove(0);
            buf[..reply.len()].copy_from_slice(&reply);
            return Ok(reply.len());
        }

        if self.stalled || !self.state.streams.has(Streams::POSE) {
            // Nothing to say. A real transport would block until the timeout;
            // returning zero is the same outcome without the wait.
            return Ok(0);
        }

        if self.paced {
            if self.started.is_none() {
                self.started = Some(std::time::Instant::now());
            }
            std::thread::sleep(std::time::Duration::from_secs_f32(self.interval()));
        }
        let event = self.pose_event();
        self.tick += 1;
        buf[..event.len()].copy_from_slice(&event);
        Ok(event.len())
    }
}

/// Convenience for tests that want the driver's view rather than the device's.
///
/// Returns the orientations the driver would have seen after `samples` events,
/// which is the ground truth to compare a tracker's output against.
pub fn expected_poses(trajectory: &dyn Trajectory, rate: Rate, samples: u32) -> Vec<[f32; 4]> {
    let interval = 1.0 / rate.hz() as f32;
    (0..samples)
        .map(|i| trajectory.at(i as f32 * interval))
        .collect()
}

/// Angle between two orientations, in radians — the measure that matters, since
/// `q` and `-q` are the same rotation and a component-wise comparison is not.
///
/// Computed from the difference quaternion rather than as `2·acos(dot)`. That
/// form loses all its precision exactly where these tests need it: near zero,
/// `acos` turns a rounding error of `ε` in the dot product into `sqrt(2ε)` in
/// the answer, which at `f32` is 0.04° of pure noise — enough to fail an
/// assertion about a head that is not moving at all. Taking the arctangent of
/// the vector part against the scalar part is well conditioned there.
pub fn angle_between(a: [f32; 4], b: [f32; 4]) -> f32 {
    let [aw, ax, ay, az] = a;
    let [bw, bx, by, bz] = b;
    // conj(a) * b
    let w = aw * bw + ax * bx + ay * by + az * bz;
    let x = aw * bx - ax * bw - ay * bz + az * by;
    let y = aw * by + ax * bz - ay * bw - az * bx;
    let z = aw * bz - ax * by + ay * bx - az * bw;
    let vector = (x * x + y * y + z * z).sqrt();
    2.0 * vector.atan2(w.abs())
}

/// Errors the simulator can be asked to produce, so the driver's failure paths
/// are exercised too.
pub struct Failing;

impl Transport for Failing {
    fn send(&mut self, _frame: &[u8]) -> Result<()> {
        Err(Error::Io(std::io::Error::from_raw_os_error(19))) // ENODEV
    }

    fn recv(&mut self, _buf: &mut [u8; FRAME_MAX], _timeout_ns: u64) -> Result<usize> {
        Err(Error::Io(std::io::Error::from_raw_os_error(19)))
    }
}
