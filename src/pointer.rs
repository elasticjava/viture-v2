//! Head-relative pointing: glasses IMU plus phone IMU.
//!
//! The glasses image is head-locked — it follows wherever you look. So a
//! pointer driven by the phone alone drifts off target the moment you turn your
//! head. The correct quantity is the phone's orientation **relative to the
//! head**:
//!
//! ```text
//! q_pointer = conj(q_head) * q_phone
//! ```
//!
//! Rotating the phone's forward axis by that and intersecting it with a plane
//! at distance `d` gives a cursor position on the virtual screen. `d` is the
//! sensitivity knob: it corresponds to how far away the screen feels.
//!
//! The phone's orientation comes from Android's `TYPE_GAME_ROTATION_VECTOR`,
//! read through `termux-sensor` from Termux:API. That sensor fuses gyroscope
//! and accelerometer without the magnetometer, so it is immune to the magnetic
//! interference a docked phone sits in — at the price of slow yaw drift, which
//! is what the recentre control is for.

use std::io::{BufRead, BufReader};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

use crate::quat_mul;

/// A quaternion `[w, x, y, z]` plus the moment it was seen.
#[derive(Clone, Copy, Debug)]
pub struct Orientation {
    pub q: [f32; 4],
    pub seq: u64,
}

impl Orientation {
    pub const IDENTITY: Orientation = Orientation { q: [1.0, 0.0, 0.0, 0.0], seq: 0 };
}

/// Reads the phone's own rotation vector via `termux-sensor`.
///
/// Termux:API prints one pretty-printed JSON object per sample, so the parser
/// only has to find `"values"` and collect the numbers that follow. That is
/// robust against the exact key name, which differs between devices.
pub struct PhoneSensor {
    child: Child,
    latest: Arc<AtomicU64>,
    quat: Arc<[AtomicU64; 4]>,
    stop: Arc<AtomicBool>,
}

impl PhoneSensor {
    /// Starts the sensor stream. `delay_ms` is a hint; Android may deliver
    /// slower, and apps targeting API 31+ are capped at 200 Hz anyway.
    pub fn start(delay_ms: u32) -> std::io::Result<PhoneSensor> {
        let mut child = Command::new("termux-sensor")
            .args(["-s", "Game Rotation Vector", "-d", &delay_ms.to_string()])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let stdout = child.stdout.take().expect("piped");

        let latest = Arc::new(AtomicU64::new(0));
        let quat: Arc<[AtomicU64; 4]> = Arc::new([
            AtomicU64::new(f32::to_bits(1.0) as u64),
            AtomicU64::new(0),
            AtomicU64::new(0),
            AtomicU64::new(0),
        ]);
        let stop = Arc::new(AtomicBool::new(false));

        {
            let latest = Arc::clone(&latest);
            let quat = Arc::clone(&quat);
            let stop = Arc::clone(&stop);
            std::thread::spawn(move || parse_loop(stdout, latest, quat, stop));
        }

        Ok(PhoneSensor { child, latest, quat, stop })
    }

    /// Most recent orientation, or the identity if nothing has arrived yet.
    pub fn read(&self) -> Orientation {
        let seq = self.latest.load(Ordering::Acquire);
        let mut q = [0f32; 4];
        for (i, slot) in self.quat.iter().enumerate() {
            q[i] = f32::from_bits(slot.load(Ordering::Relaxed) as u32);
        }
        Orientation { q, seq }
    }

    /// How many samples have arrived so far.
    pub fn samples(&self) -> u64 {
        self.latest.load(Ordering::Relaxed)
    }
}

impl Drop for PhoneSensor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        let _ = self.child.kill();
        let _ = self.child.wait();
        // Release the sensor so Termux:API stops holding a wakelock on it.
        let _ = Command::new("termux-sensor").arg("-c").stdout(Stdio::null()).status();
    }
}

fn parse_loop(
    stdout: ChildStdout,
    latest: Arc<AtomicU64>,
    quat: Arc<[AtomicU64; 4]>,
    stop: Arc<AtomicBool>,
) {
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    let mut collecting = false;
    let mut values: Vec<f32> = Vec::with_capacity(5);
    let mut seq = 0u64;

    while !stop.load(Ordering::Relaxed) {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => break,
            Ok(_) => {}
            Err(_) => break,
        }

        if line.contains("\"values\"") {
            collecting = true;
            values.clear();
            continue;
        }
        if !collecting {
            continue;
        }
        if line.contains(']') {
            collecting = false;
            if values.len() >= 3 {
                // Android hands out [x, y, z] and usually w as the fourth
                // element. Where it is missing, reconstruct it: the rotation
                // vector is a unit quaternion.
                let (x, y, z) = (values[0], values[1], values[2]);
                let w = if values.len() >= 4 {
                    values[3]
                } else {
                    (1.0 - (x * x + y * y + z * z)).max(0.0).sqrt()
                };
                for (slot, v) in quat.iter().zip([w, x, y, z]) {
                    slot.store(v.to_bits() as u64, Ordering::Relaxed);
                }
                seq += 1;
                latest.store(seq, Ordering::Release);
            }
            continue;
        }
        if let Some(v) = line.trim().trim_end_matches(',').parse::<f32>().ok() {
            values.push(v);
        }
    }
}

/// Turns two orientations into a cursor position.
pub struct Pointer {
    /// Head orientation at the last recentre.
    pub head_ref: [f32; 4],
    /// Phone orientation at the last recentre.
    pub phone_ref: [f32; 4],
    /// Virtual screen distance; larger means less sensitive.
    pub distance: f32,
    /// Which phone axis points forward. `[0, 1, 0]` is the top edge, which is
    /// what pointing the phone like a wand feels like.
    pub forward: [f32; 3],
}

impl Default for Pointer {
    fn default() -> Self {
        Pointer {
            head_ref: [1.0, 0.0, 0.0, 0.0],
            phone_ref: [1.0, 0.0, 0.0, 0.0],
            distance: 2.0,
            forward: [0.0, 1.0, 0.0],
        }
    }
}

impl Pointer {
    /// Declares the current orientations to be "pointing at the centre".
    pub fn recentre(&mut self, head: [f32; 4], phone: [f32; 4]) {
        self.head_ref = head;
        self.phone_ref = phone;
    }

    /// Cursor position in normalised screen coordinates, `(0, 0)` at the
    /// centre, roughly `±1` at the edges. `None` when the phone points away
    /// from the screen.
    pub fn cursor(&self, head: [f32; 4], phone: [f32; 4]) -> Option<(f32, f32)> {
        // Both orientations relative to their reference, then the phone
        // expressed in the head's frame.
        let head_rel = quat_mul(crate::quat_conj(self.head_ref), head);
        let phone_rel = quat_mul(crate::quat_conj(self.phone_ref), phone);
        let rel = quat_mul(crate::quat_conj(head_rel), phone_rel);

        let dir = rotate(rel, self.forward);
        // The forward axis has to keep pointing into the screen half-space.
        if dir[1] <= 0.05 {
            return None;
        }
        Some((dir[0] / dir[1] * self.distance, dir[2] / dir[1] * self.distance))
    }
}

/// Rotates a vector by a quaternion: `v' = q v q*`.
#[inline]
pub fn rotate(q: [f32; 4], v: [f32; 3]) -> [f32; 3] {
    let [w, x, y, z] = q;
    let t = [
        2.0 * (y * v[2] - z * v[1]),
        2.0 * (z * v[0] - x * v[2]),
        2.0 * (x * v[1] - y * v[0]),
    ];
    [
        v[0] + w * t[0] + (y * t[2] - z * t[1]),
        v[1] + w * t[1] + (z * t[0] - x * t[2]),
        v[2] + w * t[2] + (x * t[1] - y * t[0]),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_rotation_keeps_vector() {
        let v = rotate([1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0]);
        assert!((v[1] - 1.0).abs() < 1e-6);
    }

    /// A quarter turn about Z maps the forward axis onto -X.
    #[test]
    fn quarter_turn_about_z() {
        let s = (0.5f32).sqrt();
        let v = rotate([s, 0.0, 0.0, s], [0.0, 1.0, 0.0]);
        assert!((v[0] + 1.0).abs() < 1e-5, "x = {}", v[0]);
        assert!(v[1].abs() < 1e-5, "y = {}", v[1]);
    }

    /// Head and phone turning together must leave the cursor where it was —
    /// that is the whole point of the head-relative construction.
    #[test]
    fn turning_the_head_along_does_not_move_the_cursor() {
        let mut p = Pointer::default();
        p.recentre([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
        let before = p.cursor([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]).unwrap();

        // 20 degrees about Z, applied to both.
        let a = 20f32.to_radians() / 2.0;
        let turn = [a.cos(), 0.0, 0.0, a.sin()];
        let after = p.cursor(turn, turn).unwrap();

        assert!((before.0 - after.0).abs() < 1e-4, "x drifted: {before:?} -> {after:?}");
        assert!((before.1 - after.1).abs() < 1e-4, "y drifted: {before:?} -> {after:?}");
    }

    /// Turning only the phone must move the cursor.
    #[test]
    fn turning_only_the_phone_moves_the_cursor() {
        let mut p = Pointer::default();
        p.recentre([1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]);
        let a = 10f32.to_radians() / 2.0;
        let turn = [a.cos(), 0.0, 0.0, a.sin()];
        let moved = p.cursor([1.0, 0.0, 0.0, 0.0], turn).unwrap();
        assert!(moved.0.abs() > 0.2, "cursor barely moved: {moved:?}");
    }
}
