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
    index_count, screen_mesh, screen_vertex_count, sphere_indices, sphere_mesh, uv_window,
    vertex_count, view_projection, Eye, Projection, StereoLayout, DEFAULT_SCREEN_DISTANCE,
    DEFAULT_SCREEN_WIDTH_DEG, VERTEX_FLOATS,
};

const RINGS: u32 = 64;
const SECTORS: u32 = 128;

/// Where the shared fixture lives.
///
/// The build directory by default, and whatever `XR_FIXTURES` says when it is
/// set. The override exists so this can be cross-compiled and run on the phone:
/// `CARGO_MANIFEST_DIR` is baked in at compile time and names a directory that
/// does not exist there, and the one test that pins the two languages together
/// is worth being able to run on the machine that has to agree.
fn fixture() -> serde_json::Value {
    let root = std::env::var_os("XR_FIXTURES")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures"));
    let path = root.join("spatial_formats.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    serde_json::from_str(&text).expect("the fixture should be valid JSON")
}

/// What the middle of the panel samples for one eye.
///
/// Built from the mesh the renderer uploads and the matrix it draws with, and
/// interpolated across the triangle the point falls in rather than snapped to
/// the nearest vertex — a screen has two rows, so its nearest vertex is always
/// an edge and never the middle of the picture.
fn centre_uv(projection: Projection, layout: StereoLayout, eye: Eye, swap: bool) -> [f32; 2] {
    // Which eye's picture this eye is shown, which is not the same question as
    // which eye is being drawn. A `right_left` file packs them the other way
    // round, and the renderer answers it in exactly this place.
    let eye = match (swap, eye) {
        (true, Eye::Left) => Eye::Right,
        (true, Eye::Right) => Eye::Left,
        (false, e) => e,
    };
    // The eye offset is left at zero even for a screen: what is being isolated
    // here is the stereo window, and a displaced camera would move the sampled
    // point for a reason that has nothing to do with it.
    let (mesh, triangles) = surface(projection);

    let mut mvp = [0.0f32; 16];
    view_projection([1.0, 0.0, 0.0, 0.0], 25.8, 16.0 / 9.0, &mut mvp);

    let project = |index: usize| -> Option<([f32; 2], [f32; 2])> {
        let v = &mesh[index * VERTEX_FLOATS..][..VERTEX_FLOATS];
        let mut clip = [0.0f32; 4];
        for (row, out) in clip.iter_mut().enumerate() {
            *out = mvp[row] * v[0] + mvp[4 + row] * v[1] + mvp[8 + row] * v[2] + mvp[12 + row];
        }
        (clip[3] > 1e-6).then(|| ([clip[0] / clip[3], clip[1] / clip[3]], [v[3], v[4]]))
    };

    let uv = triangles
        .iter()
        .find_map(|&[i, j, k]| {
            let (a, ua) = project(i)?;
            let (b, ub) = project(j)?;
            let (c, uc) = project(k)?;
            barycentric([0.0, 0.0], a, b, c).map(|[wa, wb, wc]| {
                [
                    ua[0] * wa + ub[0] * wb + uc[0] * wc,
                    ua[1] * wa + ub[1] * wb + uc[1] * wc,
                ]
            })
        })
        .expect("something should cover the middle of the panel");

    let [us, uo, vs, vo] = uv_window(layout, eye);
    [uv[0] * us + uo, uv[1] * vs + vo]
}

/// The mesh the renderer would upload for a projection, and its triangles.
fn surface(projection: Projection) -> (Vec<f32>, Vec<[usize; 3]>) {
    if projection.is_panoramic() {
        let mut mesh = vec![0.0f32; vertex_count(RINGS, SECTORS) * VERTEX_FLOATS];
        sphere_mesh(RINGS, SECTORS, 1.0, projection, &mut mesh).expect("sphere");
        let mut indices = vec![0u16; index_count(RINGS, SECTORS)];
        sphere_indices(RINGS, SECTORS, &mut indices).expect("indices");
        let triangles = indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|t| [t[0] as usize, t[1] as usize, t[2] as usize])
            .collect();
        (mesh, triangles)
    } else {
        let segments = 64;
        let mut mesh = vec![0.0f32; screen_vertex_count(segments) * VERTEX_FLOATS];
        screen_mesh(
            segments,
            DEFAULT_SCREEN_DISTANCE,
            DEFAULT_SCREEN_WIDTH_DEG,
            16.0 / 9.0,
            0.0,
            &mut mesh,
        )
        .expect("screen");
        // A triangle strip: every three consecutive vertices form a triangle.
        let count = screen_vertex_count(segments);
        let triangles = (0..count - 2).map(|i| [i, i + 1, i + 2]).collect();
        (mesh, triangles)
    }
}

/// Barycentric weights of `p` in the triangle `a b c`, or `None` if it is
/// outside. Winding-agnostic, since a strip alternates.
fn barycentric(p: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> Option<[f32; 3]> {
    let area = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1]);
    if area.abs() < 1e-9 {
        return None;
    }
    let wa = ((b[0] - p[0]) * (c[1] - p[1]) - (c[0] - p[0]) * (b[1] - p[1])) / area;
    let wb = ((c[0] - p[0]) * (a[1] - p[1]) - (a[0] - p[0]) * (c[1] - p[1])) / area;
    let wc = 1.0 - wa - wb;
    let inside = [wa, wb, wc].iter().all(|w| *w >= -1e-5);
    inside.then_some([wa, wb, wc])
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

        let swap = scenario["expect"]["swapEyes"].as_bool().unwrap_or(false);
        for (eye, key) in [(Eye::Left, "left"), (Eye::Right, "right")] {
            let expected = scenario["centreUv"][key]
                .as_array()
                .unwrap_or_else(|| panic!("{name}: centreUv.{key} missing"));
            let want = [
                expected[0].as_f64().unwrap() as f32,
                expected[1].as_f64().unwrap() as f32,
            ];
            let got = centre_uv(projection, layout, eye, swap);
            // Interpolated across the triangle, so this is exact up to the
            // mesh's own faceting: a sphere approximated by flat quads is a
            // fraction of a cell away from the ideal surface.
            let tolerance = 1e-3;
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

        let swap = scenario["expect"]["swapEyes"].as_bool().unwrap_or(false);
        let left = centre_uv(projection, layout, Eye::Left, swap);
        let right = centre_uv(projection, layout, Eye::Right, swap);
        // A swap exchanges the two halves and nothing else, so the assertions
        // below hold with the eyes put back the way the packing means them.
        let (left, right) = if swap { (right, left) } else { (left, right) };

        match packing {
            0 => assert_eq!(
                left, right,
                "{name}: a monoscopic frame reached the eyes split"
            ),
            1 => {
                assert!(
                    (left[0] - right[0]).abs() < 1e-4,
                    "{name}: split horizontally"
                );
                assert!(
                    (left[1] - right[1] - 0.5).abs() < 1e-4,
                    "{name}: the halves are {} apart, expected 0.5 with the left eye above",
                    left[1] - right[1],
                );
            }
            2 => {
                assert!(
                    (left[1] - right[1]).abs() < 1e-4,
                    "{name}: split vertically"
                );
                assert!(
                    (right[0] - left[0] - 0.5).abs() < 1e-4,
                    "{name}: the halves are {} apart, expected 0.5 with the left eye first",
                    right[0] - left[0],
                );
            }
            // Anaglyph and row-interleaved cover the whole frame with both
            // eyes and separate them per pixel, in the fragment shader. There
            // is no window to be wrong about, and asserting that the geometry
            // samples the same place for both is the true statement — not a
            // claim that the eyes see the same thing.
            3 | 4 => assert_eq!(
                left, right,
                "{name}: a packing with no window sampled two different places"
            ),
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
    assert_eq!(
        packings,
        [0, 1, 2, 3, 4].into_iter().collect(),
        "packings covered"
    );
    assert_eq!(
        projections,
        [0, 1, 2].into_iter().collect(),
        "projections covered"
    );
    // And both settings of the swap, which is a separate axis from the packing
    // and the one most easily left untested.
    let swaps: std::collections::BTreeSet<bool> = scenarios
        .iter()
        .map(|s| s["expect"]["swapEyes"].as_bool().unwrap_or(false))
        .collect();
    assert_eq!(swaps, [false, true].into_iter().collect(), "swaps covered");
}
