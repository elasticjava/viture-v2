//! The whole driver, driven by a simulated pair of glasses.
//!
//! The unit tests check pieces: a quaternion here, a bit reader there. These
//! check the thing that actually ships — command path, frame parsing, reader
//! loop, recentring, rate estimation and prediction, in one stack, against a
//! head movement whose ground truth is known exactly.
//!
//! Everything runs in simulated time, so a two-second head turn takes
//! microseconds and gives the same answer every run.

use viture_v2::sim::{angle_between, Failing, Simulated, Still, TurnThenStop, Turning};
use viture_v2::xr::Tracker;
use viture_v2::{msg, Device, Event, Rate, Streams};

/// Degrees, for assertions that are easier to judge than radians.
fn degrees(radians: f32) -> f32 {
    radians.to_degrees()
}

/// Runs the device forward, returning the poses the driver saw.
fn collect_poses(device: &mut Device<Simulated>, samples: usize) -> Vec<[f32; 4]> {
    let mut poses = Vec::with_capacity(samples);
    while poses.len() < samples {
        match device.next_event(1_000_000) {
            Ok(Some(Event::Pose(p))) => poses.push(p.q),
            Ok(Some(_)) | Ok(None) => {}
            Err(e) => panic!("event failed: {e}"),
        }
    }
    poses
}

// -- The command path ---------------------------------------------------------

#[test]
fn starting_the_imu_reaches_the_device() {
    let mut device = Device::new(Simulated::still());
    device
        .set_imu(Streams::POSE, Rate::Hz120)
        .expect("the device should acknowledge");

    let state = device.transport().state();
    assert_eq!(state.streams, Streams::POSE);
    assert_eq!(state.rate, Rate::Hz120);

    // And it went out as one IMU_CTRL frame, not as something the device merely
    // tolerated.
    let log = device.transport().log();
    let sent = &*log.lock().unwrap();
    assert_eq!(sent.len(), 1);
    assert_eq!(sent[0].0, msg::IMU_CTRL);
}

#[test]
fn no_events_arrive_until_the_stream_is_started() {
    // The device is silent until asked. A driver that assumed otherwise would
    // look fine here and hang on real hardware.
    let mut device = Device::new(Simulated::still());
    assert!(matches!(device.next_event(1_000_000), Ok(None)));

    device.set_imu(Streams::POSE, Rate::Hz120).unwrap();
    assert!(matches!(
        device.next_event(1_000_000),
        Ok(Some(Event::Pose(_)))
    ));
}

#[test]
fn a_display_mode_change_is_reflected_by_the_device() {
    let mut device = Device::new(Simulated::still());
    device.set_display_mode_raw(0x32).expect("acknowledged");
    assert_eq!(device.transport().state().display_mode, 0x32);
}

#[test]
fn the_static_facts_read_back() {
    let mut device = Device::new(Simulated::still());
    let mut buf = [0u8; 64];
    assert_eq!(device.firmware_version(&mut buf).unwrap(), "SIM-1.0.0");
    assert_eq!(device.brightness().unwrap(), 5);
    assert_eq!(device.volume().unwrap(), 60);
    assert!(device.worn().unwrap());
}

#[test]
fn a_dead_device_reports_an_error_rather_than_hanging() {
    let mut device = Device::new(Failing);
    assert!(device.set_imu(Streams::POSE, Rate::Hz120).is_err());
    assert!(device.next_event(1_000_000).is_err());
}

// -- The event path -----------------------------------------------------------

#[test]
fn a_still_head_produces_the_same_orientation_every_sample() {
    let mut device = Device::new(Simulated::new(Box::new(Still::default())));
    device.set_imu(Streams::POSE, Rate::Hz120).unwrap();

    for q in collect_poses(&mut device, 240) {
        assert!(
            degrees(angle_between(q, [1.0, 0.0, 0.0, 0.0])) < 1e-3,
            "a still head drifted to {q:?}",
        );
    }
}

#[test]
fn a_turning_head_arrives_where_the_trajectory_says() {
    // One radian per second about Y for two seconds at 120 Hz.
    let rate = 1.0f32;
    let mut device = Device::new(Simulated::new(Box::new(Turning {
        rate,
        axis: [0.0, 1.0, 0.0],
    })));
    device.set_imu(Streams::POSE, Rate::Hz120).unwrap();

    let poses = collect_poses(&mut device, 240);
    // The last sample is at t = 239/120 s, not 2 s: the first is at zero.
    let t = 239.0 / 120.0;
    let half = 0.5 * rate * t;
    let expected = [half.cos(), 0.0, half.sin(), 0.0];
    let error = degrees(angle_between(*poses.last().unwrap(), expected));
    assert!(error < 0.01, "off by {error}°");
}

#[test]
fn the_wire_format_survives_a_round_trip_of_every_sample() {
    // The simulator builds frames with the real builder and the driver reads
    // them with the real parser, so a header, checksum or payload mistake fails
    // here. Sweeping the whole rotation covers every sign and magnitude the
    // float encoding has to carry.
    // A properly normalised diagonal axis: 0.577 is three decimal places of
    // 1/sqrt(3), and the sixth one is enough to fail a tight norm check on the
    // simulator's own output rather than on the driver's parsing.
    let d = 1.0 / 3f32.sqrt();
    let mut device = Device::new(Simulated::new(Box::new(Turning {
        rate: std::f32::consts::TAU,
        axis: [d, d, d],
    })));
    device.set_imu(Streams::POSE, Rate::Hz120).unwrap();

    for q in collect_poses(&mut device, 120) {
        let norm = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "denormalised quaternion {q:?}");
    }
}

#[test]
fn a_stalled_stream_goes_quiet_without_erroring() {
    // A stream that stops while the device still answers is the failure the
    // watchdogs exist for; it must look like silence, not like a broken link.
    let mut device = Device::new(Simulated::new(Box::new(Still::default())));
    device.set_imu(Streams::POSE, Rate::Hz120).unwrap();
    assert!(matches!(device.next_event(1_000), Ok(Some(Event::Pose(_)))));

    device.transport_mut().stall(true);
    for _ in 0..10 {
        assert!(matches!(device.next_event(1_000), Ok(None)));
    }

    device.transport_mut().stall(false);
    assert!(matches!(device.next_event(1_000), Ok(Some(Event::Pose(_)))));
}

#[test]
fn a_command_sent_mid_stream_is_answered_before_the_next_sample() {
    // The reader sends queued commands between reads, and a reply that arrived
    // behind a burst of poses would be read as an event. This is the ordering
    // that stops that.
    let mut device = Device::new(Simulated::new(Box::new(Still::default())));
    device.set_imu(Streams::POSE, Rate::Hz120).unwrap();
    let _ = device.next_event(1_000);

    device.set_display_mode_raw(0x32).expect("acknowledged");
    assert_eq!(device.transport().state().display_mode, 0x32);
}

// -- Rate estimation and prediction, over the tracker ------------------------

/// Feeds the simulated stream through the same maths the tracker uses, without
/// the thread: the reader's job is to call these in order, and the order is what
/// is being checked.
mod pipeline {
    use super::*;
    use viture_v2::sim::Simulated;

    /// Runs `samples` poses and returns them with their simulated timestamps.
    fn run(
        trajectory: Box<dyn viture_v2::sim::Trajectory>,
        samples: usize,
    ) -> Vec<(f32, [f32; 4])> {
        let mut device = Device::new(Simulated::new(trajectory));
        device.set_imu(Streams::POSE, Rate::Hz120).unwrap();
        let interval = 1.0 / 120.0;
        let mut out = Vec::with_capacity(samples);
        let mut index = 0usize;
        while out.len() < samples {
            if let Ok(Some(Event::Pose(p))) = device.next_event(1_000) {
                out.push((index as f32 * interval, p.q));
                index += 1;
            }
        }
        out
    }

    #[test]
    fn differentiating_the_stream_recovers_the_turn_rate() {
        // This is what makes prediction possible at all: the device will not
        // stream angular rate alongside poses, so the rate is a difference.
        let expected = 1.5f32;
        let poses = run(
            Box::new(Turning {
                rate: expected,
                axis: [0.0, 1.0, 0.0],
            }),
            120,
        );
        let dt = 1.0 / 120.0;
        for window in poses.windows(2).skip(10) {
            let rate = viture_v2::xr::angular_rate(window[0].1, window[1].1, dt);
            assert!(
                (rate[1] - expected).abs() < 0.01,
                "read {} rad/s, expected {expected}",
                rate[1],
            );
            assert!(
                rate[0].abs() < 0.01 && rate[2].abs() < 0.01,
                "off-axis {rate:?}"
            );
        }
    }

    #[test]
    fn prediction_lands_on_the_head_rather_than_behind_it() {
        // The point of the whole exercise. Extrapolating a sample forward by the
        // display pipeline's latency must arrive where the head will actually be.
        let rate = 1.5f32;
        let lookahead = 0.030f32;
        let poses = run(
            Box::new(Turning {
                rate,
                axis: [0.0, 1.0, 0.0],
            }),
            120,
        );
        let dt = 1.0 / 120.0;

        let mut uncorrected_worst = 0.0f32;
        let mut predicted_worst = 0.0f32;
        for window in poses.windows(2).skip(10) {
            let (t, q) = window[1];
            let measured = viture_v2::xr::angular_rate(window[0].1, q, dt);
            let predicted = viture_v2::xr::integrate(q, measured, lookahead);

            let half = 0.5 * rate * (t + lookahead);
            let truth = [half.cos(), 0.0, half.sin(), 0.0];

            uncorrected_worst = uncorrected_worst.max(degrees(angle_between(q, truth)));
            predicted_worst = predicted_worst.max(degrees(angle_between(predicted, truth)));
        }

        // Thirty milliseconds at 1.5 rad/s is 2.6° of lag — plainly visible.
        assert!(
            uncorrected_worst > 2.0,
            "the test is not exercising lag: worst uncorrected error {uncorrected_worst}°",
        );
        assert!(
            predicted_worst < 0.05,
            "prediction left {predicted_worst}° of error",
        );
    }

    #[test]
    fn prediction_overshoots_when_the_head_stops_and_recovers() {
        // The honest cost of predicting: at the end of a movement the rate is
        // still the old one for a sample or two, so the view runs past and comes
        // back. It has to be small and it has to settle.
        let rate = 1.5f32;
        let lookahead = 0.030f32;
        let poses = run(
            Box::new(TurnThenStop {
                rate,
                axis: [0.0, 1.0, 0.0],
                until: 0.5,
            }),
            120,
        );
        let dt = 1.0 / 120.0;

        let mut overshoot_after_stop = 0.0f32;
        let mut settled = 0.0f32;
        for window in poses.windows(2) {
            let (t, q) = window[1];
            if t < 0.5 {
                continue;
            }
            let measured = viture_v2::xr::angular_rate(window[0].1, q, dt);
            let predicted = viture_v2::xr::integrate(q, measured, lookahead);
            let error = degrees(angle_between(predicted, q));
            if t < 0.52 {
                overshoot_after_stop = overshoot_after_stop.max(error);
            } else {
                settled = settled.max(error);
            }
        }
        // A single sample of stale rate is 30 ms of 1.5 rad/s: under three
        // degrees, and gone by the next sample.
        assert!(
            overshoot_after_stop < 3.0,
            "overshoot {overshoot_after_stop}°"
        );
        assert!(settled < 0.01, "still moving {settled}° after settling");
    }

    #[test]
    fn a_still_head_is_not_predicted_anywhere() {
        // Rate noise turns straight into a restless image, so a stationary head
        // must produce an exactly stationary prediction.
        let poses = run(Box::new(Still::default()), 60);
        let dt = 1.0 / 120.0;
        for window in poses.windows(2) {
            let rate = viture_v2::xr::angular_rate(window[0].1, window[1].1, dt);
            let predicted = viture_v2::xr::integrate(window[1].1, rate, 0.030);
            assert!(
                degrees(angle_between(predicted, window[1].1)) < 1e-3,
                "a still head was predicted to move",
            );
        }
    }
}

/// The panel is left the way the session found it.
///
/// Side-by-side is a rendering arrangement, not a display setting: the panel
/// cuts every frame in half and sends one to each eye, which is correct only
/// while something is deliberately drawing a stereo pair. When the session ends
/// nothing is, and every ordinary picture on that display arrives halved — the
/// wearer sees the left half of a desktop in one eye and the right half in the
/// other, with no way from inside the glasses to work out why.
///
/// It happened during testing on real hardware, and the shape of it is what
/// puts this in the driver rather than in a caller: the panel was switched, the
/// thing that was going to draw stereo then failed to start, and the teardown
/// that would have fixed it belonged to the thing that never ran. There are
/// several ways for a session to end, and asking each of them to remember is
/// asking for the one that forgets.
#[test]
fn the_panel_is_left_in_the_mode_the_session_found_it_in() {
    use viture_v2::sim::measured;

    let mut device = Simulated::still();
    // A session that begins with the panel already splitting frames.
    device.set_display_mode(measured::MODE_SIDE_BY_SIDE);

    let log = device.log();
    let tracker = Tracker::with_transport(Device::new(device), Rate::Hz120).expect("tracker");
    // Something asks for 2D mid-session, as the renderer does when it is not
    // drawing a stereo pair.
    tracker.request_display_mode(measured::MODE_2D);
    std::thread::sleep(std::time::Duration::from_millis(120));
    drop(tracker);
    let sent = log.lock().unwrap().clone();

    // The last word on the panel must put back what was there, not leave the
    // mode whoever spoke last happened to want.
    let modes: Vec<u8> = sent
        .iter()
        .filter(|(id, _)| *id == viture_v2::msg::SET_DISPLAY_MODE)
        .filter_map(|(_, payload)| payload.first().copied())
        .collect();
    assert!(
        modes.contains(&measured::MODE_2D),
        "the mid-session request never reached the panel: {modes:02X?}",
    );
    assert_eq!(
        modes.last().copied(),
        Some(measured::MODE_SIDE_BY_SIDE),
        "the session ended without putting the panel back: {modes:02X?}",
    );
}

/// And a session that changes nothing says nothing.
///
/// Switching the mode renegotiates the video link, which from inside the
/// glasses is a black flash. Doing that on the way out of a session that never
/// touched the panel would be a flash for no reason at all.
#[test]
fn a_session_that_left_the_panel_alone_does_not_speak_to_it_on_the_way_out() {
    let device = Simulated::still();
    let log = device.log();
    let tracker = Tracker::with_transport(Device::new(device), Rate::Hz120).expect("tracker");
    std::thread::sleep(std::time::Duration::from_millis(60));
    drop(tracker);
    let modes: Vec<u8> = log
        .lock()
        .unwrap()
        .iter()
        .filter(|(id, _)| *id == viture_v2::msg::SET_DISPLAY_MODE)
        .filter_map(|(_, payload)| payload.first().copied())
        .collect();
    assert!(
        modes.is_empty(),
        "the panel was spoken to for no reason: {modes:02X?}"
    );
}
