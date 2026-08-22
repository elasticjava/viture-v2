//! The tracker as the application uses it, over a simulated pair of glasses.
//!
//! `simulated_glasses.rs` drives the device and the maths. This drives the
//! layer above both: the reader thread, the shared state it publishes,
//! recentring, and the predicted orientation a renderer actually draws from.
//!
//! These tests spend real time. They have to: the angular-rate estimate divides
//! by the interval between two poses, measured on the wall clock, so a simulator
//! that answered instantly would produce intervals of nothing and the estimate
//! would be discarded as noise. The simulated device is therefore paced, and the
//! tests are kept short enough to stay in fractions of a second.

use std::time::{Duration, Instant};

use viture_v2::sim::{angle_between, Simulated, Still, Turning};
use viture_v2::xr::Tracker;
use viture_v2::{Device, Rate};

const RATE: Rate = Rate::Hz120;

fn degrees(radians: f32) -> f32 {
    radians.to_degrees()
}

fn tracker(trajectory: Box<dyn viture_v2::sim::Trajectory>) -> Tracker {
    Tracker::with_transport(Device::new(Simulated::paced(trajectory)), RATE)
        .expect("the simulated device should open")
}

/// Waits until the tracker has published at least `count` samples, or gives up.
fn wait_for_samples(tracker: &Tracker, count: u64) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while tracker.state().head_samples < count {
        if Instant::now() > deadline {
            panic!(
                "only {} samples arrived; expected {count}",
                tracker.state().head_samples,
            );
        }
        std::thread::sleep(Duration::from_millis(2));
    }
}

#[test]
fn opening_reads_the_static_facts_and_starts_the_stream() {
    let tracker = tracker(Box::new(Still::default()));
    let info = tracker.info();
    assert_eq!(info.firmware, "SIM-1.0.0");
    assert_eq!(info.brightness, 5);
    assert_eq!(info.volume, 60);
    assert_eq!(info.display_mode, 0x31);
    assert!(info.worn);

    // And poses start arriving without anyone asking again.
    wait_for_samples(&tracker, 5);
}

#[test]
fn a_still_head_publishes_a_still_orientation() {
    let tracker = tracker(Box::new(Still::default()));
    wait_for_samples(&tracker, 20);

    let state = tracker.state();
    assert!(
        degrees(angle_between(state.head_q, [1.0, 0.0, 0.0, 0.0])) < 0.01,
        "drifted to {:?}",
        state.head_q,
    );
    // Nothing is moving, so nothing should be predicted to move either — rate
    // noise on a stationary head is felt as a restless image.
    assert!(
        degrees(angle_between(state.predicted_q, state.head_q)) < 0.01,
        "a still head was predicted to move",
    );
}

#[test]
fn freshness_is_reported_once_per_sample() {
    let tracker = tracker(Box::new(Still::default()));
    wait_for_samples(&tracker, 3);

    // The flag is a latch: true once per new pose, then false until the next.
    assert!(tracker.pose_fresh(), "no pose was reported fresh");
    assert!(!tracker.pose_fresh(), "the flag did not clear");
}

#[test]
fn a_turning_head_moves_the_published_orientation() {
    let rate = 1.0f32;
    let tracker = tracker(Box::new(Turning {
        rate,
        axis: [0.0, 1.0, 0.0],
    }));
    wait_for_samples(&tracker, 60);

    let turned = degrees(angle_between(tracker.state().head_q, [1.0, 0.0, 0.0, 0.0]));
    // Sixty samples at 120 Hz is half a second, so about 28.6°. The bound is
    // wide because the reader thread is scheduled by the operating system and
    // may be a sample or two behind; the point is that it moved and by roughly
    // the right amount.
    assert!(
        (10.0..60.0).contains(&turned),
        "turned {turned}° after half a second at 1 rad/s",
    );
}

#[test]
fn the_angular_rate_is_measured_from_the_stream() {
    // The device will not send rates alongside poses, so this is differentiated
    // from them — and it is what prediction rests on.
    let expected = 1.5f32;
    let tracker = tracker(Box::new(Turning {
        rate: expected,
        axis: [0.0, 1.0, 0.0],
    }));
    wait_for_samples(&tracker, 60);

    let measured = tracker.angular_rate();
    // Wide, because the interval is the operating system's scheduling rather
    // than an exact 8.3 ms. A tight bound here would be a flaky test, and the
    // exact arithmetic is checked in the unit tests.
    assert!(
        (0.5..3.0).contains(&measured[1]),
        "measured {:?} rad/s for a {expected} rad/s turn",
        measured,
    );
    assert!(
        measured[0].abs() < 0.5 && measured[2].abs() < 0.5,
        "off-axis rate {measured:?}",
    );
}

#[test]
fn prediction_leads_the_published_orientation_while_turning() {
    let tracker = tracker(Box::new(Turning {
        rate: 1.5,
        axis: [0.0, 1.0, 0.0],
    }));
    tracker.set_lookahead_s(0.030);
    wait_for_samples(&tracker, 60);

    let state = tracker.state();
    let lead = degrees(angle_between(state.predicted_q, state.head_q));
    // Thirty milliseconds at 1.5 rad/s is 2.6°. Anything near zero means the
    // rate never reached the prediction, which is exactly the bug that made
    // this whole path a no-op before.
    assert!(lead > 0.5, "prediction led by only {lead}°");
    assert!(
        lead < 8.0,
        "prediction ran {lead}° ahead, which would overshoot"
    );
}

#[test]
fn prediction_leads_in_the_direction_of_travel() {
    // Leading the wrong way would look like double the lag, and an angle alone
    // cannot tell the two apart.
    let tracker = tracker(Box::new(Turning {
        rate: 1.5,
        axis: [0.0, 1.0, 0.0],
    }));
    tracker.set_lookahead_s(0.030);
    wait_for_samples(&tracker, 60);

    let state = tracker.state();
    // For a yaw about +Y both quaternions have the form (cos, 0, sin, 0), and
    // the predicted one must be further round.
    assert!(
        state.predicted_q[2].abs() > state.head_q[2].abs(),
        "predicted {:?} is not ahead of {:?}",
        state.predicted_q,
        state.head_q,
    );
}

#[test]
fn recentring_cancels_the_heading_that_was_reached() {
    let tracker = tracker(Box::new(Turning {
        rate: 1.5,
        axis: [0.0, 1.0, 0.0],
    }));
    wait_for_samples(&tracker, 40);

    let before = degrees(angle_between(tracker.state().head_q, [1.0, 0.0, 0.0, 0.0]));
    assert!(before > 5.0, "the head had not turned far enough to test");

    tracker.recentre();
    // Read straight away: the head keeps turning, so waiting would let it move
    // away from the reference again.
    let after = degrees(angle_between(tracker.state().head_q, [1.0, 0.0, 0.0, 0.0]));
    assert!(
        after < before / 2.0,
        "recentring left {after}° of the original {before}°",
    );
}

#[test]
fn recentring_leaves_pitch_alone() {
    // Heading-only recentring is what keeps the horizon level. Cancelling the
    // whole orientation folds the reference pitch into the yaw axis, and the
    // workspace then tilts as you pan.
    let pitch = 20f32.to_radians();
    let (s, c) = (pitch / 2.0).sin_cos();
    let tracker = tracker(Box::new(Still([c, s, 0.0, 0.0])));
    wait_for_samples(&tracker, 10);

    tracker.recentre();
    let after = tracker.state().head_q;
    let residual = degrees(angle_between(after, [1.0, 0.0, 0.0, 0.0]));
    assert!(
        (15.0..25.0).contains(&residual),
        "pitch became {residual}°, expected to keep its 20°",
    );
    // And it is still a pitch, not a heading.
    assert!(after[2].abs() < 0.02, "heading appeared: {after:?}");
}

#[test]
fn a_display_mode_change_reaches_the_device_through_the_reader() {
    // The reader owns the transport once it starts, so a mode change is queued
    // and sent between two reads. That hand-off is easy to get wrong and
    // invisible when it is.
    let tracker = tracker(Box::new(Still::default()));
    wait_for_samples(&tracker, 5);

    tracker.request_display_mode(0x32);

    // What can be observed from here is that the stream survives the round trip.
    // The cached device facts are read once at open time and do not update, and
    // the transport is the reader's — so the assertion is that poses keep
    // coming, which is what breaks when the hand-off is wrong.
    let before = tracker.state().head_samples;
    std::thread::sleep(Duration::from_millis(100));
    assert!(
        tracker.state().head_samples > before,
        "the stream stopped after a display-mode change",
    );
}

#[test]
fn dropping_the_tracker_stops_the_reader() {
    // The reader is joined on drop. A tracker that leaked its thread would keep
    // the transport open, and the next open would fail on hardware.
    let tracker = tracker(Box::new(Still::default()));
    wait_for_samples(&tracker, 5);
    let started = Instant::now();
    drop(tracker);
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "dropping the tracker took {:?}",
        started.elapsed(),
    );
}
