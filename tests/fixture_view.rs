//! The renderer's half of the shared scenarios.
//!
//! `fixtures/spatial_formats.json` describes what a media centre reports about a
//! file, what the library should conclude, and what the wearer should then see.
//! The library's half is asserted in Kotlin, in `KodiToViewFixtureTest`; this is
//! the other half. Both read the same file, so a change on either side that the
//! other does not expect fails here or there.
//!
//! What is being checked is the last link in the chain: given a packing and a
//! projection — however they were arrived at — which part of the video reaches
//! which eye. That is where the swapped-eye bug lived, and it is invisible from
//! either end alone.

use std::path::PathBuf;

use viture_v2::pano::{
    sphere_mesh, uv_window, vertex_count, view_projection, Eye, Projection, StereoLayout,
    VERTEX_FLOATS,
};

const RINGS: u32 = 64;
const SECTORS: u32 = 128;

fn fixture() -> serde_json::Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("spatial_formats.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("the fixture should be valid JSON")
}

/// What the middle of the panel samples for one eye, looking straight ahead.
///
/// Built from the mesh the renderer uploads and the matrix it draws with, so
/// this is the whole chain rather than a restatement of `uv_window`.
fn centre_uv(projection: Projection, layout: StereoLayout, eye: Eye) -> [f32; 2] {
    let mut mesh = vec![0.0f32; vertex_count(RINGS, SECTORS) * VERTEX_FLOATS];
    sphere_mesh(RINGS, SECTORS, 1.0, projection, &mut mesh).expect("mesh");

    let mut mvp = [0.0f32; 16];
    view_projection([1.0, 0.0, 0.0, 0.0], 25.8, 16.0 / 9.0, &mut mvp);

    // The vertex nearest the middle of the panel.
    let mut best = ([0.0f32, 0.0], f32::MAX);
    for v in mesh.as_chunks::<VERTEX_FLOATS>().0 {
        let mut clip = [0.0f32; 4];
        for (row, out) in clip.iter_mut().enumerate() {
            *out = mvp[row] * v[0] + mvp[4 + row] * v[1] + mvp[8 + row] * v[2] + mvp[12 + row];
        }
        if clip[3] <= 1e-6 {
            continue;
        }
        let distance = (clip[0] / clip[3]).powi(2) + (clip[1] / clip[3]).powi(2);
        if distance < best.1 {
            best = ([v[3], v[4]], distance);
        }
    }

    let [us, uo, vs, vo] = uv_window(layout, eye);
    [best.0[0] * us + uo, best.0[1] * vs + vo]
}

fn layout_of(packing: u64) -> StereoLayout {
    StereoLayout::from_raw(packing as u32)
}

fn projection_of(value: u64) -> Projection {
    Projection::from_raw(value as u32)
}

#[test]
fn every_scenario_puts_the_right_half_of_the_frame_in_front_of_each_eye() {
    let fixture = fixture();
    let scenarios = fixture["scenarios"]
        .as_array()
        .expect("scenarios should be a list");
    assert!(!scenarios.is_empty(), "the fixture is empty");

    for scenario in scenarios {
        let name = scenario["name"].as_str().unwrap_or("unnamed");
        let expect = &scenario["expect"];
        let layout = layout_of(expect["packing"].as_u64().expect("packing"));
        let projection = projection_of(expect["projection"].as_u64().expect("projection"));

        for (eye, key) in [(Eye::Left, "left"), (Eye::Right, "right")] {
            let expected = scenario["centreUv"][key]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: centreUv.{key} missing"));
            let want = [
                expected[0].as_f64().unwrap() as f32,
                expected[1].as_f64().unwrap() as f32,
            ];
            let got = centre_uv(projection, layout, eye);
            // A tenth of a mesh cell. The centre of an equirectangular image is
            // straight ahead exactly, but the nearest vertex to the middle of
            // the panel is on a grid.
            let tolerance = 0.1 / SECTORS as f32;
            assert!(
                (got[0] - want[0]).abs() < tolerance && (got[1] - want[1]).abs() < tolerance,
                "{name}, {key} eye: sampled {got:?}, expected {want:?}",
            );
        }
    }
}

#[test]
fn the_two_eyes_differ_exactly_where_the_packing_says_they_should() {
    // A stereo file whose eyes sample the same place is being shown flat; a
    // monoscopic one whose eyes differ is being torn in half. Both are silent
    // failures on a panel and obvious here.
    let fixture = fixture();
    for scenario in fixture["scenarios"].as_array().unwrap() {
        let name = scenario["name"].as_str().unwrap_or("unnamed");
        let packing = scenario["expect"]["packing"].as_u64().unwrap();
        let projection = projection_of(scenario["expect"]["projection"].as_u64().unwrap());
        let layout = layout_of(packing);

        let left = centre_uv(projection, layout, Eye::Left);
        let right = centre_uv(projection, layout, Eye::Right);

        match packing {
            0 => assert_eq!(left, right, "{name}: a monoscopic frame reached the eyes split"),
            1 => {
                assert!((left[0] - right[0]).abs() < 1e-4, "{name}: split horizontally");
                assert!(
                    (left[1] - right[1] - 0.5).abs() < 1e-4,
                    "{name}: the halves are {} apart, expected 0.5 with the left eye above",
                    left[1] - right[1],
                );
            }
            2 => {
                assert!((left[1] - right[1]).abs() < 1e-4, "{name}: split vertically");
                assert!(
                    (right[0] - left[0] - 0.5).abs() < 1e-4,
                    "{name}: the halves are {} apart, expected 0.5 with the left eye first",
                    right[0] - left[0],
                );
            }
            other => panic!("{name}: unknown packing {other}"),
        }
    }
}

#[test]
fn the_fixture_covers_both_projections_and_all_three_packings() {
    // A shared fixture is only worth having if it exercises the cases that
    // differ. This fails when someone adds a case and forgets the interesting
    // one, which is the usual way a table like this rots.
    let fixture = fixture();
    let scenarios = fixture["scenarios"].as_array().unwrap();
    let packings: std::collections::BTreeSet<u64> = scenarios
        .iter()
        .map(|s| s["expect"]["packing"].as_u64().unwrap())
        .collect();
    let projections: std::collections::BTreeSet<u64> = scenarios
        .iter()
        .map(|s| s["expect"]["projection"].as_u64().unwrap())
        .collect();
    assert_eq!(packings, [0, 1, 2].into_iter().collect(), "packings covered");
    assert_eq!(projections, [0, 1].into_iter().collect(), "projections covered");
}
