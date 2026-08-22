//! Holds the simulated glasses to what the real ones were measured doing.
//!
//! The simulation is only worth having if it is a simulation *of something*.
//! `fixtures/measured_device.json` is a record of the hardware — the pose rate
//! counted rather than claimed, the display modes it advertises in each panel
//! mode, the power state it was all measured in — and the constants the driver
//! ships are supposed to be that record.
//!
//! Two copies of a number drift. This is what stops them: change the fixture
//! after a re-measurement and the driver fails here until it is brought along,
//! which is the right way round — the hardware is the authority and the code is
//! the copy.

use std::path::PathBuf;

use viture_v2::sim::measured;

fn fixture() -> serde_json::Value {
    let root = std::env::var_os("XR_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"));
    let path = root.join("measured_device.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("the measurement should be valid JSON")
}

#[test]
fn the_pose_rate_is_the_one_that_was_counted() {
    // 118.9, not the 120 the device is asked for. The gap is small and it is
    // not nothing: prediction multiplies the sample interval by a lookahead,
    // and one percent of that is about a millisecond of lead that would go
    // missing without ever looking wrong.
    let measured_hz = fixture()["glasses"]["poseRateHz"].as_f64().unwrap() as f32;
    assert!(
        (measured::POSE_RATE_HZ - measured_hz).abs() < 0.05,
        "the driver simulates {} Hz, the hardware was counted at {measured_hz}",
        measured::POSE_RATE_HZ,
    );

    // And the count really was a count, not a nominal figure written down.
    let fixture = fixture();
    let samples = fixture["glasses"]["poseSampleCount"].as_f64().unwrap();
    let seconds = fixture["glasses"]["poseSampleSeconds"].as_f64().unwrap();
    assert!(
        ((samples / seconds) as f32 - measured_hz).abs() < 0.05,
        "the rate in the fixture is not what its own sample count divides to",
    );
}

#[test]
fn the_usb_identity_is_the_one_that_was_read_off_the_bus() {
    let fixture = fixture();
    assert_eq!(
        measured::USB_VENDOR_ID as u64,
        fixture["glasses"]["usbVendorId"].as_u64().unwrap(),
    );
    assert_eq!(
        measured::USB_PRODUCT_ID as u64,
        fixture["glasses"]["usbProductId"].as_u64().unwrap(),
    );
}

#[test]
fn the_panel_advertises_the_modes_it_was_seen_advertising() {
    // In 2D, and this is the list that matters: four modes, none of them wider
    // than 1920 and none of them faster than 60 Hz.
    let fixture = fixture();
    let listed = fixture["display"]["supportedModes"].as_array().unwrap();
    assert_eq!(
        listed.len(),
        measured::MODES_2D.len(),
        "the panel offered {} modes in 2D, the simulation offers {}",
        listed.len(),
        measured::MODES_2D.len(),
    );
    for (i, mode) in listed.iter().enumerate() {
        let simulated = measured::MODES_2D[i];
        assert_eq!(
            simulated.width as u64,
            mode["width"].as_u64().unwrap(),
            "mode {i} width"
        );
        assert_eq!(
            simulated.height as u64,
            mode["height"].as_u64().unwrap(),
            "mode {i} height"
        );
        let fps = mode["fps"].as_f64().unwrap() as f32;
        assert!((simulated.fps - fps).abs() < 0.001, "mode {i} refresh rate");
    }
}

#[test]
fn nothing_above_sixty_hertz_is_offered_in_two_dimensions() {
    // The glasses are rated for 120 Hz here and do not offer it, over a
    // DisplayPort link negotiated on four lanes with training successful. That
    // is a real, open discrepancy, and pinning it means a build that starts
    // offering 120 fails this test — which is exactly when somebody should be
    // told, because it would mean the way in had been found.
    let fastest = measured::MODES_2D
        .iter()
        .map(|m| m.fps)
        .fold(f32::MIN, f32::max);
    assert!(
        fastest <= 60.5,
        "2D now offers {fastest} Hz; the panel used to cap at 60 and the \
         datasheet says 120 — if this is real, the fixture and the note in \
         sim::measured need updating",
    );

    let rated = fixture()["display"]["ratedByManufacturer"]["twoDimensional"]["maxHz"]
        .as_f64()
        .unwrap();
    assert!(
        rated > fastest as f64,
        "the discrepancy is the point of this test"
    );
}

#[test]
fn side_by_side_is_the_only_place_the_wide_mode_exists() {
    // The bug this guards: a mode chosen once, at startup, is chosen from the
    // 2D list — which has nothing 3840 wide in it — and then never looked at
    // again after the panel switches and re-advertises.
    assert!(
        !measured::MODES_2D.iter().any(|m| m.width > 1920),
        "a wide mode appeared in the 2D list",
    );
    assert!(
        measured::MODES_SIDE_BY_SIDE.iter().any(|m| m.width == 3840),
        "side-by-side has no 3840-wide mode to offer",
    );
    assert_eq!(measured::modes_for(measured::MODE_2D), &measured::MODES_2D);
    assert_eq!(
        measured::modes_for(measured::MODE_SIDE_BY_SIDE),
        &measured::MODES_SIDE_BY_SIDE,
    );
    // An unknown mode byte falls back to the conservative list rather than
    // inventing one. 0x41 to 0x43 were tried on the hardware and came back
    // with the 2D modes under fresh ids.
    assert_eq!(measured::modes_for(0x41), &measured::MODES_2D);
}

#[test]
fn the_measurement_records_the_power_state_it_was_taken_in() {
    // A timing without its power state is not a measurement, on a phone that
    // throttles itself when it is low. This is the field that makes the rest of
    // the file re-readable a month from now.
    let fixture = fixture();
    let power = &fixture["power"];
    assert!(
        power["batteryPercent"].is_number(),
        "no battery level recorded"
    );
    assert!(power["charging"].is_boolean(), "no charging state recorded");
    assert!(
        power["batterySaverOn"].is_boolean(),
        "no battery saver state recorded"
    );
}
