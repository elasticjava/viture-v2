//! What the wearer would see, computed instead of looked at.
//!
//! The other simulation tests drive the sensor. This one drives the optics: for
//! a given head orientation, projection and frame packing, it works out which
//! part of the video lands where on the panel.
//!
//! That matters because every rendering mistake made on this project so far
//! passed every automated check and was caught by putting the glasses on — a
//! sphere textured upside down, a panorama letterboxed inside a smaller band, a
//! source that turned out to be dual fisheye. Each of them is a statement about
//! which texel appears at which screen position, and each is checkable here.
//!
//! Method: take the mesh the renderer uploads, project its vertices through the
//! matrix the renderer uses, and ask which vertex lands nearest a given point on
//! screen. That covers the mesh and the camera together. A rasteriser would be
//! more faithful and is not needed — the questions are all about where things
//! are, not what colour they come out.

use viture_v2::pano::{
    self, index_count, sphere_indices, sphere_mesh, uv_window, vertex_count, view_projection, Eye,
    Projection, StereoLayout, VERTEX_FLOATS,
};

const RINGS: u32 = 64;
const SECTORS: u32 = 128;
const ASPECT: f32 = 16.0 / 9.0;

/// One vertex of the uploaded mesh: where it is, and what it samples.
#[derive(Clone, Copy, Debug)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2],
    /// Normalised device coordinates, or `None` when behind the viewer.
    ndc: Option<[f32; 2]>,
}

/// Builds the mesh and projects it, exactly as the renderer does.
fn render(head: [f32; 4], projection: Projection, fov_deg: f32) -> Vec<Vertex> {
    let mut mesh = vec![0.0f32; vertex_count(RINGS, SECTORS) * VERTEX_FLOATS];
    sphere_mesh(RINGS, SECTORS, 1.0, projection, &mut mesh).expect("mesh");

    let mut mvp = [0.0f32; 16];
    view_projection(head, fov_deg, ASPECT, &mut mvp);

    mesh.as_chunks::<VERTEX_FLOATS>()
        .0
        .iter()
        .map(|v| {
            let position = [v[0], v[1], v[2]];
            let mut clip = [0.0f32; 4];
            for (row, out) in clip.iter_mut().enumerate() {
                *out = mvp[row] * position[0]
                    + mvp[4 + row] * position[1]
                    + mvp[8 + row] * position[2]
                    + mvp[12 + row];
            }
            Vertex {
                position,
                uv: [v[3], v[4]],
                ndc: (clip[3] > 1e-6).then(|| [clip[0] / clip[3], clip[1] / clip[3]]),
            }
        })
        .collect()
}

/// The visible vertex nearest a point on screen, in normalised device
/// coordinates where `(0, 0)` is the centre, `x` grows right and `y` grows up.
fn nearest_on_screen(vertices: &[Vertex], target: [f32; 2]) -> Vertex {
    vertices
        .iter()
        .filter(|v| v.ndc.is_some())
        .min_by(|a, b| {
            let d = |v: &Vertex| {
                let n = v.ndc.unwrap();
                (n[0] - target[0]).powi(2) + (n[1] - target[1]).powi(2)
            };
            d(a).partial_cmp(&d(b)).unwrap()
        })
        .copied()
        .expect("something should be on screen")
}

/// The texture coordinate a point on screen ends up sampling, after the stereo
/// window has selected this eye's half of the frame.
fn sampled(vertices: &[Vertex], target: [f32; 2], layout: StereoLayout, eye: Eye) -> [f32; 2] {
    let v = nearest_on_screen(vertices, target);
    let [us, uo, vs, vo] = uv_window(layout, eye);
    [v.uv[0] * us + uo, v.uv[1] * vs + vo]
}

fn yaw(degrees: f32) -> [f32; 4] {
    // Right-handed about +Y. A negative angle turns the head to its right.
    let half = -degrees.to_radians() / 2.0;
    [half.cos(), 0.0, half.sin(), 0.0]
}

fn pitch(degrees: f32) -> [f32; 4] {
    let half = degrees.to_radians() / 2.0;
    [half.cos(), half.sin(), 0.0, 0.0]
}

// -- Where the image is ------------------------------------------------------

#[test]
fn looking_ahead_shows_the_middle_of_the_image() {
    let view = render([1.0, 0.0, 0.0, 0.0], Projection::Equirect360, 25.8);
    let uv = sampled(&view, [0.0, 0.0], StereoLayout::Mono, Eye::Left);
    assert!((uv[0] - 0.5).abs() < 0.01, "u {} at the centre", uv[0]);
    assert!((uv[1] - 0.5).abs() < 0.01, "v {} at the centre", uv[1]);
}

#[test]
fn the_top_of_the_panel_is_the_top_of_the_image() {
    // The bug this exists for: the sphere was built with image-orientation
    // texture coordinates, so every 360° video rendered upside down. It looked
    // entirely plausible in a still frame.
    let view = render([1.0, 0.0, 0.0, 0.0], Projection::Equirect360, 25.8);
    let top = sampled(&view, [0.0, 0.9], StereoLayout::Mono, Eye::Left);
    let bottom = sampled(&view, [0.0, -0.9], StereoLayout::Mono, Eye::Left);
    assert!(
        top[1] > bottom[1],
        "the top of the panel sampled v {} and the bottom {} — the image is upside down",
        top[1],
        bottom[1],
    );
    // And straight up in the world is the top of the image.
    let overhead = render(pitch(80.0), Projection::Equirect360, 25.8);
    let zenith = sampled(&overhead, [0.0, 0.0], StereoLayout::Mono, Eye::Left);
    assert!(zenith[1] > 0.9, "looking up sampled v {}", zenith[1]);
}

#[test]
fn the_right_of_the_panel_is_the_right_of_the_image() {
    let view = render([1.0, 0.0, 0.0, 0.0], Projection::Equirect360, 25.8);
    let right = sampled(&view, [0.9, 0.0], StereoLayout::Mono, Eye::Left);
    let left = sampled(&view, [-0.9, 0.0], StereoLayout::Mono, Eye::Left);
    assert!(
        right[0] > left[0],
        "the image is mirrored: right sampled u {} and left {}",
        right[0],
        left[0],
    );
}

#[test]
fn turning_the_head_right_brings_the_next_part_of_the_image_into_view() {
    let ahead = sampled(
        &render([1.0, 0.0, 0.0, 0.0], Projection::Equirect360, 25.8),
        [0.0, 0.0],
        StereoLayout::Mono,
        Eye::Left,
    );
    let turned = sampled(
        &render(yaw(30.0), Projection::Equirect360, 25.8),
        [0.0, 0.0],
        StereoLayout::Mono,
        Eye::Left,
    );
    // Thirty degrees of a full turn is a twelfth of the image.
    let moved = turned[0] - ahead[0];
    assert!(
        (moved - 1.0 / 12.0).abs() < 0.01,
        "a 30° turn moved the centre by {moved} of the image, expected {}",
        1.0 / 12.0,
    );
}

#[test]
fn the_field_of_view_covers_the_angle_the_panel_spans() {
    // The panorama is life-size only if the rendered angle matches the optics.
    // Rendering a wider angle into the same panel makes the world look half
    // size, which is what the first version did by taking the diagonal figure
    // for the vertical one.
    let view = render([1.0, 0.0, 0.0, 0.0], Projection::Equirect360, 25.8);
    let top = nearest_on_screen(&view, [0.0, 1.0]);
    let elevation = top.position[1].asin().to_degrees();
    // Within one ring of the mesh: 64 rings over 180° is 2.8° apart, so the
    // nearest vertex to the top of the panel can be that far from the frustum
    // edge without anything being wrong.
    let ring = 180.0 / RINGS as f32;
    assert!(
        (elevation - 12.9).abs() < ring,
        "the top of the panel is {elevation}° up; half of 25.8° is 12.9°",
    );
}

// -- Stereo ------------------------------------------------------------------

#[test]
fn over_under_gives_the_left_eye_the_top_half() {
    let view = render([1.0, 0.0, 0.0, 0.0], Projection::Equirect360, 25.8);
    let left = sampled(&view, [0.0, 0.0], StereoLayout::OverUnder, Eye::Left);
    let right = sampled(&view, [0.0, 0.0], StereoLayout::OverUnder, Eye::Right);

    // Texture space runs bottom to top, so the top half of the image is the
    // upper half of v.
    assert!(
        left[1] > 0.5,
        "the left eye sampled v {} — not the top half",
        left[1]
    );
    assert!(right[1] < 0.5, "the right eye sampled v {}", right[1]);
    // Both eyes look at the same place in the world, half a frame apart.
    assert!((left[1] - right[1] - 0.5).abs() < 0.01);
    assert!(
        (left[0] - right[0]).abs() < 0.001,
        "the eyes disagree horizontally"
    );
}

#[test]
fn side_by_side_gives_the_left_eye_the_left_half() {
    let view = render([1.0, 0.0, 0.0, 0.0], Projection::Equirect360, 25.8);
    let left = sampled(&view, [0.0, 0.0], StereoLayout::SideBySide, Eye::Left);
    let right = sampled(&view, [0.0, 0.0], StereoLayout::SideBySide, Eye::Right);
    assert!(left[0] < 0.5, "the left eye sampled u {}", left[0]);
    assert!(right[0] > 0.5, "the right eye sampled u {}", right[0]);
    assert!(
        (left[1] - right[1]).abs() < 0.001,
        "the eyes disagree vertically"
    );
}

#[test]
fn a_monoscopic_frame_reaches_both_eyes_whole() {
    let view = render([1.0, 0.0, 0.0, 0.0], Projection::Equirect360, 25.8);
    let left = sampled(&view, [0.0, 0.0], StereoLayout::Mono, Eye::Left);
    let right = sampled(&view, [0.0, 0.0], StereoLayout::Mono, Eye::Right);
    assert_eq!(left, right);
    assert!((left[0] - 0.5).abs() < 0.01 && (left[1] - 0.5).abs() < 0.01);
}

// -- Coverage ----------------------------------------------------------------

#[test]
fn vr180_puts_the_whole_image_in_front_and_nothing_behind() {
    // Played as a full sphere, a VR180 file wraps round the back at half scale —
    // plausible-looking and wrong. The hemisphere has to end.
    let view = render([1.0, 0.0, 0.0, 0.0], Projection::Equirect180, 25.8);
    let centre = sampled(&view, [0.0, 0.0], StereoLayout::Mono, Eye::Left);
    assert!(
        (centre[0] - 0.5).abs() < 0.01,
        "the centre moved to u {}",
        centre[0]
    );

    // A quarter turn reaches the edge of the footage, not the middle of it.
    let edge = sampled(
        &render(yaw(90.0), Projection::Equirect180, 25.8),
        [0.0, 0.0],
        StereoLayout::Mono,
        Eye::Left,
    );
    assert!(
        edge[0] > 0.98,
        "a 90° turn sampled u {}, expected the edge",
        edge[0]
    );

    // And behind the viewer there is no geometry at all.
    let behind = render(yaw(180.0), Projection::Equirect180, 25.8);
    let visible = behind.iter().filter(|v| {
        v.ndc
            .is_some_and(|n| n[0].abs() <= 1.0 && n[1].abs() <= 1.0)
    });
    assert_eq!(visible.count(), 0, "footage appeared behind a VR180 viewer");
}

#[test]
fn a_full_sphere_has_something_to_show_in_every_direction() {
    for heading in [0.0f32, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0] {
        let view = render(yaw(heading), Projection::Equirect360, 25.8);
        let centre = nearest_on_screen(&view, [0.0, 0.0]);
        let ndc = centre.ndc.unwrap();
        assert!(
            ndc[0].abs() < 0.05 && ndc[1].abs() < 0.05,
            "nothing at the centre when facing {heading}°",
        );
    }
}

// -- Zoom --------------------------------------------------------------------

#[test]
fn zooming_in_narrows_what_is_visible() {
    // Zoom on a panorama is a change of view angle, not of scale: there is
    // nothing to move towards.
    let wide = render(
        [1.0, 0.0, 0.0, 0.0],
        Projection::Equirect360,
        pano::fov_for_zoom(1.0),
    );
    let tight = render(
        [1.0, 0.0, 0.0, 0.0],
        Projection::Equirect360,
        pano::fov_for_zoom(2.0),
    );

    let wide_edge = sampled(&wide, [1.0, 0.0], StereoLayout::Mono, Eye::Left);
    let tight_edge = sampled(&tight, [1.0, 0.0], StereoLayout::Mono, Eye::Left);
    let wide_span = wide_edge[0] - 0.5;
    let tight_span = tight_edge[0] - 0.5;
    assert!(
        tight_span < wide_span * 0.6,
        "doubling the zoom took the visible span from {wide_span} to {tight_span}",
    );
}

// -- The mesh itself ---------------------------------------------------------

#[test]
fn the_index_buffer_addresses_the_mesh_it_was_built_for() {
    // A mismatch here draws garbage rather than nothing, which is harder to
    // recognise than a black screen.
    let mut indices = vec![0u16; index_count(RINGS, SECTORS)];
    sphere_indices(RINGS, SECTORS, &mut indices).expect("indices");
    let vertices = vertex_count(RINGS, SECTORS) as u16;
    assert!(indices.iter().all(|&i| i < vertices));
    assert_eq!(indices.len() % 3, 0, "a partial triangle would be dropped");
}
