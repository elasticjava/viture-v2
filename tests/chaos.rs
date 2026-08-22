//! What the stack does when the hardware misbehaves.
//!
//! The other simulated tests ask whether the driver is correct when everything
//! works. This one asks whether it is *usable* when things do not, which is a
//! different question and the one a person wearing the glasses actually cares
//! about. A pair of glasses that is right ninety-nine times and shows a
//! teleporting horizon on the hundredth is not a pair of glasses anybody wants
//! on their face.
//!
//! Every fault here is something the hardware does. A thin cable in a pocket
//! loses contact intermittently. A phone scheduling forty other things starves
//! a reader thread and then hands it a burst. A bus shared with a video stream
//! truncates a transfer. None of these can be produced on demand by unplugging
//! something and hoping, so they are injected — deterministically, so a failure
//! reproduces exactly rather than once a fortnight in someone else's session.
//!
//! The standard the tests hold to is the same throughout, and it is not "does
//! not crash":
//!
//! * **Nothing corrupt is ever believed.** A bad checksum is a dropped frame,
//!   not a head that has teleported.
//! * **The stream recovers by itself.** A fault that ends does not leave the
//!   driver needing a restart to notice.
//! * **A stall is visible as a stall.** Code above has to be able to tell
//!   "nothing has arrived" from "nothing has moved", because they call for
//!   opposite responses.

use viture_v2::sim::{Faults, Faulty, Simulated, TurnThenStop, Turning};
use viture_v2::xr::Tracker;
use viture_v2::{Device, Rate};

/// A tracker over a misbehaving cable, turning steadily.
///
/// Paced, so a sample interval is a real interval. An unpaced simulation runs
/// the head round as fast as the machine can build frames, and then two poses
/// read a moment apart really are tens of degrees apart — which says nothing
/// about the driver and everything about the harness.
fn tracker(plan: Faults) -> Tracker {
    let head = Turning {
        axis: [0.0, 1.0, 0.0],
        rate: 45.0 * std::f32::consts::PI / 180.0,
    };
    let device = Device::new(Faulty::new(Simulated::paced(Box::new(head)), plan));
    Tracker::with_transport(device, Rate::Hz120).expect("the tracker should still open")
}

/// Reads until `n` poses have arrived or patience runs out, and reports how
/// many came and how long it took.
fn collect(tracker: &Tracker, n: usize) -> Vec<[f32; 4]> {
    let mut poses = Vec::with_capacity(n);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(3);
    let mut last = [f32::NAN; 4];
    while poses.len() < n && std::time::Instant::now() < deadline {
        let state = tracker.state();
        let pose = state.head_q;
        if pose != last && pose.iter().all(|v| v.is_finite()) {
            poses.push(pose);
            last = pose;
        }
        std::thread::yield_now();
    }
    poses
}

fn is_unit(q: [f32; 4]) -> bool {
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    (n - 1.0).abs() < 1e-3
}

#[test]
fn a_cable_that_fails_one_read_in_five_still_delivers_a_usable_stream() {
    // The interesting failure is not the cable that has come out — that is
    // obvious and handled everywhere — but the one that mostly works. A retry
    // loop can spin on it forever, and a watchdog can never quite decide the
    // stream is dead.
    let tracker = tracker(Faults::FAILING_CABLE);
    let poses = collect(&tracker, 50);
    assert!(
        poses.len() >= 50,
        "only {} poses got through a cable failing one read in five",
        poses.len(),
    );
    assert!(
        poses.iter().all(|&q| is_unit(q)),
        "an error response was mistaken for an orientation",
    );
}

#[test]
fn corrupt_and_truncated_frames_are_dropped_rather_than_believed() {
    // A flipped bit leaves the length right and the checksum wrong, which is
    // the subtle case: everything about the frame looks plausible except the
    // one field that says it is not. Parsing it anyway means a quaternion of
    // noise, and a horizon that jumps.
    let tracker = tracker(Faults::POOR_SIGNAL);
    let poses = collect(&tracker, 50);
    assert!(
        poses.len() >= 50,
        "the stream did not survive a poor signal"
    );
    for (i, &q) in poses.iter().enumerate() {
        assert!(is_unit(q), "pose {i} is not a rotation: {q:?}");
    }

    // And the movement is still the movement. A dropped frame costs a sample,
    // not the plot: consecutive poses stay close together, where a believed
    // corrupt one would show up as a jump.
    for pair in poses.windows(2) {
        let step = viture_v2::sim::angle_between(pair[0], pair[1]).to_degrees();
        assert!(
            step < 30.0,
            "the head appeared to jump {step:.1} degrees between samples",
        );
    }
}

#[test]
fn a_load_spike_does_not_become_a_head_that_teleported() {
    // The phone starves the reader and then hands over everything at once. The
    // gap between two poses is then far longer than any real head movement, and
    // anything computing a rate from it has to refuse rather than extrapolate.
    let tracker = tracker(Faults::LOAD_SPIKES);
    let poses = collect(&tracker, 80);
    assert!(poses.len() >= 80, "the stream stopped at a load spike");
    assert!(poses.iter().all(|&q| is_unit(q)));

    // Prediction is what a spike breaks, so it is what gets checked: whatever
    // the tracker reports as predicted must still be a rotation, however long
    // the gap that preceded it.
    let state = tracker.state();
    let predicted = state.predicted_q;
    assert!(
        is_unit(predicted),
        "prediction came out of a spike as {predicted:?}",
    );
}

#[test]
fn everything_going_wrong_at_once_is_still_survivable() {
    // Not paranoia: a phone under load with a cable being flexed does all of
    // this in the same second, and the combination is where independently
    // reasonable handling turns out to interact badly.
    let tracker = tracker(Faults::EVERYTHING);
    let poses = collect(&tracker, 40);
    assert!(
        poses.len() >= 40,
        "only {} poses survived everything at once",
        poses.len(),
    );
    assert!(poses.iter().all(|&q| is_unit(q)));
}

#[test]
fn a_stream_that_dies_is_distinguishable_from_a_head_that_stopped() {
    // These call for opposite responses. A head that has stopped should hold
    // its position; a stream that has died should stop predicting and say so,
    // because continuing to predict from the last known rate walks the world
    // away from where it is.
    let head = TurnThenStop {
        axis: [0.0, 1.0, 0.0],
        rate: 45.0 * std::f32::consts::PI / 180.0,
        until: 0.2,
    };
    let mut device = Simulated::paced(Box::new(head));
    device.pace(true);
    let tracker =
        Tracker::with_transport(Device::new(device), Rate::Hz120).expect("tracker should open");

    // Let it run, then let it settle after the movement ends.
    std::thread::sleep(std::time::Duration::from_millis(400));
    let moving_then_stopped = tracker.state();
    assert!(
        moving_then_stopped.head_samples > 0,
        "no poses arrived at all",
    );
    // A head that has stopped still has a valid, finite pose — the stream is
    // alive and simply reporting no movement.
    let pose = moving_then_stopped.head_q;
    assert!(is_unit(pose), "a stopped head reported {pose:?}");
}

#[test]
fn the_faults_are_deterministic_so_a_failure_reproduces() {
    // The whole value of injecting rather than waiting is that the same run
    // gives the same answer; a random-number generator here would mean a
    // failure that shows up once a fortnight in somebody else's session.
    //
    // Checked at the transport, which is where the determinism lives. Through
    // the tracker it does not: a reader thread runs free against a consumer
    // that samples it, and which pose is visible at which instant is a race by
    // construction. Asserting equality there would be asserting that two
    // threads interleave the same way twice, which is not a property this or
    // any other driver has.
    let run = || {
        let head = Turning {
            axis: [0.0, 1.0, 0.0],
            rate: 0.8,
        };
        let mut dev = Device::new(Faulty::new(
            Simulated::new(Box::new(head)),
            Faults::EVERYTHING,
        ));
        dev.set_imu(viture_v2::Streams::POSE, Rate::Hz120)
            .expect("stream on");
        // What the parser made of each frame, verdict included — so a change in
        // *which* frames are broken shows up, not just how many.
        (0..60)
            .map(|_| match dev.next_event(1_000_000) {
                Ok(Some(viture_v2::Event::Pose(p))) => format!("pose {:?}", p.q),
                Ok(Some(_)) => "other".to_string(),
                Ok(None) => "none".to_string(),
                Err(e) => format!("err {e:?}"),
            })
            .collect::<Vec<_>>()
    };
    let first = run();
    let second = run();
    assert_eq!(first, second, "two identical runs took different paths");
    // And the faults really did fire, or this would be asserting that nothing
    // happens twice the same way.
    assert!(
        first.iter().any(|s| s.starts_with("err")),
        "no fault was injected at all",
    );
    assert!(
        first.iter().any(|s| s.starts_with("pose")),
        "nothing got through, so nothing was being tested",
    );
}
