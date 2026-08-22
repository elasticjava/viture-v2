//! Panorama geometry and camera maths for 360° playback.
//!
//! A 360° video is an equirectangular image: longitude across, latitude down.
//! Playing it back means texturing the inside of a sphere with it and putting
//! the camera at the centre, so the work splits into two halves — build the
//! sphere once, and produce one matrix per frame.
//!
//! Both halves live here rather than in the renderer because both are hot in
//! their own way. The mesh is 8k vertices that a managed runtime would build
//! element by element into a boxed buffer; the matrix is needed every frame for
//! every eye, and computing it above the JNI boundary means a pose array
//! allocation and four matrix calls per frame that no one profiles.
//!
//! # Conventions
//!
//! Right-handed world space: `+X` right, `+Y` up, `−Z` forward.
//!
//! Texture coordinates follow OpenGL rather than the image: `u` runs left to
//! right, and `v` runs *bottom to top*, so `v = 1` is the zenith. That is the
//! convention `SurfaceTexture.getTransformMatrix` expects — it maps GL-style
//! coordinates onto the decoder's buffer and carries whatever vertical flip the
//! buffer needs. Emitting image-style coordinates here instead flips the sphere,
//! and a 360° video renders upside down.
//!
//! The image centre `(0.5, 0.5)` sits straight ahead at `−Z`, which is where a
//! viewer looks when the video starts, and increasing `u` sweeps to the right.
//!
//! The sphere is wound so that its triangles are counter-clockwise *seen from
//! the centre*. Back-face culling can therefore stay on, which halves the
//! fragment work: without it the GPU shades the far wall of the sphere and then
//! throws it away.

use glam::camera::rh::proj::opengl;
use glam::{Mat4, Quat, Vec3};

/// How a stereoscopic 360° frame packs its two eyes.
///
/// There is no metadata channel here that says which one a file uses — the
/// convention lives in the filename, the container's stereo mode box, or the
/// site it came from. Over-under is what most 3D 360° material uses, because
/// halving vertical resolution costs less than halving horizontal on content
/// that is twice as wide as it is tall.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum StereoLayout {
    /// One image for both eyes. Depth comes from nothing; the scene is a flat
    /// sphere, which is still the right choice for the vast majority of 360°
    /// footage.
    Mono = 0,
    /// Left eye in the top half, right eye in the bottom.
    OverUnder = 1,
    /// Left eye in the left half, right eye in the right.
    SideBySide = 2,
}

impl StereoLayout {
    pub fn from_raw(v: u32) -> StereoLayout {
        match v {
            1 => StereoLayout::OverUnder,
            2 => StereoLayout::SideBySide,
            _ => StereoLayout::Mono,
        }
    }
}

/// How much of the sphere the footage covers.
///
/// Everything here is equirectangular; the difference is the arc. Mesh
/// projections — the `mshp` box a fisheye rig writes — are a third case that
/// needs the mesh out of the file and is not handled.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum Projection {
    /// The full sphere: 360° across, 180° down.
    Equirect360 = 0,
    /// The front half only: 180° across, 180° down.
    ///
    /// This is what VR180 rigs shoot, and it is most of the stereoscopic
    /// material that exists, because two forward-facing lenses can be given a
    /// real interocular distance where a full-sphere rig cannot. Only the
    /// hemisphere is built, so turning round shows the background rather than
    /// the footage smeared across geometry it was never meant to cover.
    Equirect180 = 1,
    /// Not a panorama at all: an ordinary rectangular picture, shown on a screen
    /// standing in the room.
    ///
    /// This is what most video is, and playing it on a sphere smears one frame
    /// across the whole world. It is here rather than as a special case
    /// elsewhere because the choice between a sphere and a screen is exactly the
    /// same decision as the choice between a full sphere and a hemisphere: how
    /// much world does this image cover.
    Flat = 2,
}

impl Projection {
    pub fn from_raw(v: u32) -> Projection {
        match v {
            1 => Projection::Equirect180,
            2 => Projection::Flat,
            _ => Projection::Equirect360,
        }
    }

    /// Whether this projection wraps around the viewer at all.
    pub fn is_panoramic(self) -> bool {
        self != Projection::Flat
    }

    /// The longitude the image spans, in radians. Meaningless for [`Flat`],
    /// which is built by [`screen_mesh`] instead.
    fn arc(self) -> f32 {
        match self {
            Projection::Equirect360 => std::f32::consts::TAU,
            _ => std::f32::consts::PI,
        }
    }
}

/// Limits on how large a screen may be asked to appear, in degrees across.
///
/// The lower bound is a screen you would have to lean towards; the upper is
/// wider than the optics can show, which is allowed because looking around a
/// screen that overflows the view is a legitimate way to watch one.
pub const MIN_SCREEN_WIDTH_DEG: f32 = 10.0;
pub const MAX_SCREEN_WIDTH_DEG: f32 = 120.0;

/// A comfortable default: about the angle a cinema screen subtends from the
/// middle of the stalls.
pub const DEFAULT_SCREEN_WIDTH_DEG: f32 = 45.0;

/// Where the screen stands, in metres. Far enough that the eyes relax, near
/// enough that it does not feel painted on the horizon.
pub const DEFAULT_SCREEN_DISTANCE: f32 = 4.0;

/// Which eye a frame is being drawn for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(i32)]
pub enum Eye {
    Left = 0,
    Right = 1,
}

/// Field of view limits, in degrees of vertical angle.
///
/// Zooming in narrows the angle; below ten degrees the head tracking's own
/// jitter becomes the dominant motion on screen and the image stops feeling
/// stable, which is where the floor comes from. The ceiling is well past
/// anything comfortable and exists only to bound the arithmetic.
pub const MIN_FOV_DEG: f32 = 10.0;
pub const MAX_FOV_DEG: f32 = 100.0;

/// Diagonal field of view of the panel, in degrees.
///
/// This is the number optics are sold on — 50° for the Pro 2, 46° for the Pro —
/// and it is always the diagonal, never the vertical. Taking it for the vertical
/// angle renders roughly twice as much world into the same optics, and the
/// panorama comes out at half life size: recognisably a picture of a place
/// rather than the place.
pub const PANEL_DIAGONAL_FOV_DEG: f32 = 50.0;

/// The vertical angle that makes a panorama life-size: the same 25.8° the panel
/// actually spans, for a 16:9 frame with [`PANEL_DIAGONAL_FOV_DEG`] across its
/// diagonal. Checked against [`vertical_fov`] in the tests rather than trusted.
pub const DEFAULT_FOV_DEG: f32 = 25.76;

/// The vertical field of view of a screen quoted by its diagonal.
///
/// The half-height of a rectangle is `1 / sqrt(1 + aspect²)` of its
/// half-diagonal, and the tangents scale with it.
pub fn vertical_fov(diagonal_deg: f32, aspect: f32) -> f32 {
    let half_diagonal = (diagonal_deg.to_radians() * 0.5).tan();
    let half_height = half_diagonal / (1.0 + aspect * aspect).sqrt();
    2.0 * half_height.atan().to_degrees()
}

/// The clipping planes.
///
/// The far plane used to be 4.0, chosen when the only thing drawn was a
/// unit-radius sphere. A screen stands at four metres by default, which put it
/// exactly on that plane and clipped it away. Nothing here is depth-tested, so
/// the planes cost nothing but have to contain what is drawn — with room for a
/// screen pushed further back than the default.
const NEAR: f32 = 0.05;
const FAR: f32 = 100.0;

/// Floats per vertex: three of position, two of texture coordinate.
pub const VERTEX_FLOATS: usize = 5;

/// Beyond this the 16-bit index buffer cannot address the mesh. 16-bit indices
/// halve index bandwidth and are what mobile GPUs are tuned for, so the limit
/// is worth keeping rather than widening to 32-bit.
const MAX_VERTICES: usize = 65_536;

/// Number of vertices [`sphere_mesh`] will write for this tessellation.
///
/// Both grid dimensions get one extra line: the last latitude closes the sphere
/// at the south pole, and the last longitude duplicates the first at `u = 1`.
/// The duplicate is the whole reason there is no visible seam behind the
/// viewer — sharing that column would make the texture coordinate jump from
/// nearly 1 back to 0 across one triangle, and the hardware would interpolate
/// the entire image across it.
pub const fn vertex_count(rings: u32, sectors: u32) -> usize {
    (rings as usize + 1) * (sectors as usize + 1)
}

/// Number of indices [`sphere_indices`] will write for this tessellation.
///
/// Independent of the projection: a hemisphere uses the same grid, stretched
/// over half the longitude.
///
/// Two triangles per grid cell, except along the two polar rows where one
/// corner of the cell collapses onto the pole and one of the two triangles has
/// no area. Those are skipped: a degenerate triangle costs a primitive-assembly
/// slot for nothing.
pub const fn index_count(rings: u32, sectors: u32) -> usize {
    if rings < 2 {
        return 0;
    }
    6 * (rings as usize - 1) * sectors as usize
}

/// Writes an inside-out UV sphere as interleaved `[x, y, z, u, v]` vertices.
///
/// Returns the number of vertices written, or `None` if the tessellation is
/// unusable — fewer than two rings or three sectors makes no closed surface,
/// more than 65 536 vertices cannot be indexed, and a short output slice would
/// leave a half-built mesh behind.
///
/// `rings` counts latitude bands and `sectors` counts longitude bands. 64 × 128
/// is a good default: the residual faceting is finer than the panel resolves,
/// and the whole mesh is under 200 kB.
pub fn sphere_mesh(
    rings: u32,
    sectors: u32,
    radius: f32,
    projection: Projection,
    out: &mut [f32],
) -> Option<usize> {
    let count = vertex_count(rings, sectors);
    if rings < 2 || sectors < 3 || count > MAX_VERTICES || out.len() < count * VERTEX_FLOATS {
        return None;
    }
    let arc = projection.arc();

    let mut w = 0;
    for ring in 0..=rings {
        // Rows run from the zenith down, so that consecutive vertices are
        // adjacent in memory the way the index buffer walks them.
        let down = ring as f32 / rings as f32;
        // Texture space is bottom-up: the zenith is v = 1.
        let v = 1.0 - down;
        let lat = (0.5 - down) * std::f32::consts::PI;
        let (sin_lat, cos_lat) = lat.sin_cos();
        let y = radius * sin_lat;
        // Radius of this latitude's circle.
        let r = radius * cos_lat;

        for sector in 0..=sectors {
            let u = sector as f32 / sectors as f32;
            // Longitude, zero straight ahead so the image centre is straight
            // ahead. The image always spans the full `u` range; what changes
            // between projections is how much world that is.
            let lon = (u - 0.5) * arc;
            let (sin_lon, cos_lon) = lon.sin_cos();

            out[w] = r * sin_lon;
            out[w + 1] = y;
            out[w + 2] = -r * cos_lon;
            out[w + 3] = u;
            out[w + 4] = v;
            w += VERTEX_FLOATS;
        }
    }
    Some(count)
}

/// Writes the triangle indices for a [`sphere_mesh`] of the same tessellation.
///
/// Returns the number of indices written, or `None` on the same conditions as
/// [`sphere_mesh`].
pub fn sphere_indices(rings: u32, sectors: u32, out: &mut [u16]) -> Option<usize> {
    let count = index_count(rings, sectors);
    if rings < 2 || sectors < 3 || vertex_count(rings, sectors) > MAX_VERTICES || out.len() < count
    {
        return None;
    }

    let stride = sectors + 1;
    let mut w = 0;
    for ring in 0..rings {
        // a --- b   ring
        // |     |
        // c --- d   ring + 1
        for sector in 0..sectors {
            let a = (ring * stride + sector) as u16;
            let b = a + 1;
            let c = a + stride as u16;
            let d = c + 1;

            // At the zenith a and b are the same point, at the nadir c and d
            // are — so one triangle of the cell is degenerate and dropped.
            if ring > 0 {
                out[w] = a;
                out[w + 1] = d;
                out[w + 2] = b;
                w += 3;
            }
            if ring + 1 < rings {
                out[w] = a;
                out[w + 1] = c;
                out[w + 2] = d;
                w += 3;
            }
        }
    }
    debug_assert_eq!(w, count);
    Some(count)
}

/// The texture window one eye samples, as `[u_scale, u_offset, v_scale, v_offset]`.
///
/// Apply it to the sphere's texture coordinate: `uv' = uv * scale + offset`. For
/// [`StereoLayout::Mono`] it is the identity, so the same shader serves both
/// mono and stereo material without a branch.
pub fn uv_window(layout: StereoLayout, eye: Eye) -> [f32; 4] {
    let second = eye == Eye::Right;
    match layout {
        StereoLayout::Mono => [1.0, 0.0, 1.0, 0.0],
        // The left eye takes the top half of the *image*, and texture space runs
        // bottom to top — so the top half is the upper half of v, and the left
        // eye gets the offset. Getting this backwards does not look broken; it
        // looks like 3D that is subtly unpleasant to watch, which is why it is
        // pinned by a test rather than left to reasoning.
        StereoLayout::OverUnder => [1.0, 0.0, 0.5, if second { 0.0 } else { 0.5 }],
        // Horizontal is unaffected by that flip: the left eye is still the left
        // half of the frame.
        StereoLayout::SideBySide => [0.5, if second { 0.5 } else { 0.0 }, 1.0, 0.0],
    }
}

/// Maps a zoom factor to a vertical field of view, in degrees.
///
/// Zooming a panorama is not a scale — there is nothing to move the camera
/// towards, since the image is at infinity — it is a narrowing of the view
/// angle, exactly like a lens. Dividing the neutral angle by the factor makes
/// the mapping feel proportional to the pinch: doubling the zoom halves the
/// visible arc, which is what a 2× lens does.
pub fn fov_for_zoom(zoom: f32) -> f32 {
    if !zoom.is_finite() || zoom <= 0.0 {
        return DEFAULT_FOV_DEG;
    }
    (DEFAULT_FOV_DEG / zoom).clamp(MIN_FOV_DEG, MAX_FOV_DEG)
}

/// The zoom factor a field of view corresponds to — the inverse of
/// [`fov_for_zoom`], for reporting the current level back to the interface.
pub fn zoom_for_fov(fov_deg: f32) -> f32 {
    if !fov_deg.is_finite() || fov_deg <= 0.0 {
        return 1.0;
    }
    DEFAULT_FOV_DEG / fov_deg
}

/// Writes the view-projection matrix for a panorama, in column-major order
/// ready for `glUniformMatrix4fv`.
///
/// `head` is the orientation of the glasses as `[w, x, y, z]`; the camera sits
/// at the centre of the sphere and only turns, so the view transform is the
/// inverse of that rotation and carries no translation.
///
/// There is deliberately no eye offset. Stereo 360° depth comes from the two
/// images in the frame, not from displacing the camera: the sphere is at a
/// fixed radius with the same geometry for both eyes, so shifting the camera
/// sideways inside it produces parallax against a wall that is not really
/// there — every object in the scene would appear at the sphere's radius, and
/// the disparity already encoded in the footage would fight it. Both eyes get
/// this matrix and differ only in [`uv_window`].
pub fn view_projection(head: [f32; 4], fov_y_deg: f32, aspect: f32, out: &mut [f32; 16]) {
    view_projection_for_eye(head, fov_y_deg, aspect, 0.0, out)
}

/// The same, with the camera displaced sideways by `eye_offset` metres.
///
/// Zero for a panorama, and that is not an oversight: a sphere is at a fixed
/// radius with the same geometry for both eyes, so moving the camera inside it
/// invents parallax against a wall that is not there, and fights the disparity
/// already in the footage.
///
/// A screen is different. It stands at a real distance in the room, so each eye
/// should see it from where that eye is — that is what makes it a screen rather
/// than a picture painted on the sky. The depth *within* the picture still comes
/// from the two half-images; this only places the screen itself.
pub fn view_projection_for_eye(
    head: [f32; 4],
    fov_y_deg: f32,
    aspect: f32,
    eye_offset: f32,
    out: &mut [f32; 16],
) {
    let [w, x, y, z] = head;
    let q = Quat::from_xyzw(x, y, z, w).normalize();
    // World-to-view is the inverse of the head's world orientation. For a unit
    // quaternion that is the conjugate, which `inverse` reduces to.
    let view = Mat4::from_translation(glam::Vec3::new(-eye_offset, 0.0, 0.0))
        * Mat4::from_quat(q.inverse());
    let fov = fov_y_deg.clamp(MIN_FOV_DEG, MAX_FOV_DEG).to_radians();
    let aspect = if aspect.is_finite() && aspect > 0.0 {
        aspect
    } else {
        1.0
    };
    let proj = opengl::perspective(fov, aspect, NEAR, FAR);
    out.copy_from_slice(&(proj * view).to_cols_array());
}

/// Vertices [`screen_mesh`] will write for a given number of segments.
///
/// Two rows of `segments + 1` columns, wound as one triangle strip — a screen
/// needs no index buffer and no tessellation vertically, because a cylinder is
/// straight up and down.
pub const fn screen_vertex_count(segments: u32) -> usize {
    2 * (segments as usize + 1)
}

/// Writes a screen standing in the room, as an interleaved `[x, y, z, u, v]`
/// triangle strip.
///
/// `width_deg` is how wide the screen should *appear* from where the viewer is,
/// which is the thing a person actually has an opinion about — "make it bigger"
/// means a wider angle, not a larger object at an unknown distance. `distance`
/// then only decides how far away it feels; the two together give the physical
/// size.
///
/// `aspect` is the picture's width over its height **after** the stereo split: a
/// side-by-side film that is 3840×1080 on disc is 16:9 per eye, and passing the
/// frame's own 32:9 would give a screen twice as wide as the picture on it.
///
/// `curvature` runs from 0 for a flat plane to 1 for an arc every part of which
/// is the same distance away. Curving trades a little geometric honesty for
/// edges that stay in focus, which is why cinema screens and monitors both do
/// it; the arc radius is `distance / curvature`, so the two ends meet
/// continuously.
///
/// Returns the vertex count, or `None` if the request or the buffer is unusable.
pub fn screen_mesh(
    segments: u32,
    distance: f32,
    width_deg: f32,
    aspect: f32,
    curvature: f32,
    out: &mut [f32],
) -> Option<usize> {
    let count = screen_vertex_count(segments);
    if segments == 0
        || !distance.is_finite()
        || distance <= 0.0
        || !aspect.is_finite()
        || aspect <= 0.0
        || out.len() < count * VERTEX_FLOATS
    {
        return None;
    }

    let width = width_deg.clamp(MIN_SCREEN_WIDTH_DEG, MAX_SCREEN_WIDTH_DEG);
    let half_width = distance * (width.to_radians() * 0.5).tan();
    let half_height = half_width / aspect;
    let curvature = curvature.clamp(0.0, 1.0);

    let mut w = 0;
    for i in 0..=segments {
        let t = i as f32 / segments as f32;
        // Arc length from the middle of the screen, so the picture is not
        // stretched towards the edges as it curves.
        let s = (t - 0.5) * 2.0 * half_width;
        let (x, z) = if curvature < 1e-4 {
            (s, -distance)
        } else {
            let radius = distance / curvature;
            let angle = s / radius;
            (
                radius * angle.sin(),
                -distance + radius * (1.0 - angle.cos()),
            )
        };

        // Top of the column first, then the bottom: the strip's first triangle
        // is then counter-clockwise from in front, and every following one
        // inherits that.
        for (y, v) in [(half_height, 1.0f32), (-half_height, 0.0f32)] {
            out[w] = x;
            out[w + 1] = y;
            out[w + 2] = z;
            out[w + 3] = t;
            out[w + 4] = v;
            w += VERTEX_FLOATS;
        }
    }
    Some(count)
}

/// Where the viewer is looking, as a point in the equirectangular image.
///
/// Useful for reporting a heading in the interface, and for deciding which part
/// of a tiled or projected source to fetch at full resolution.
pub fn gaze_uv(head: [f32; 4]) -> [f32; 2] {
    let [w, x, y, z] = head;
    let q = Quat::from_xyzw(x, y, z, w).normalize();
    let dir = q * Vec3::NEG_Z;
    let lon = dir.x.atan2(-dir.z);
    let lat = dir.y.clamp(-1.0, 1.0).asin();
    [
        (lon / std::f32::consts::TAU + 0.5).rem_euclid(1.0),
        0.5 - lat / std::f32::consts::PI,
    ]
}

// ---------------------------------------------------------------------------
// C ABI
//
// The renderer calls these across JNI once per frame at most, and the buffers
// are Java direct buffers, so nothing here allocates or copies. Every function
// validates its pointer and capacity: a wrong stride on the caller's side
// should produce a negative return, not a half-written buffer.
// ---------------------------------------------------------------------------

/// Vertices [`xr_pano_mesh`] will write, for sizing the buffer.
#[no_mangle]
pub extern "C" fn xr_pano_vertex_count(rings: u32, sectors: u32) -> u32 {
    vertex_count(rings, sectors) as u32
}

/// Indices [`xr_pano_indices`] will write, for sizing the buffer.
#[no_mangle]
pub extern "C" fn xr_pano_index_count(rings: u32, sectors: u32) -> u32 {
    index_count(rings, sectors) as u32
}

/// Fills `out` with interleaved `[x, y, z, u, v]` vertices. Returns the vertex
/// count, or -1 if the tessellation or the capacity is unusable.
///
/// `projection` is 0 for a full sphere and 1 for the VR180 front hemisphere.
///
/// # Safety
/// `out` must point to `cap_floats` writable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_pano_mesh(
    rings: u32,
    sectors: u32,
    radius: f32,
    projection: u32,
    out: *mut f32,
    cap_floats: usize,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let slice = std::slice::from_raw_parts_mut(out, cap_floats);
    sphere_mesh(
        rings,
        sectors,
        radius,
        Projection::from_raw(projection),
        slice,
    )
    .map_or(-1, |n| n as i32)
}

/// Fills `out` with triangle indices. Returns the index count, or -1.
///
/// # Safety
/// `out` must point to `cap` writable, aligned `u16`s.
#[no_mangle]
pub unsafe extern "C" fn xr_pano_indices(
    rings: u32,
    sectors: u32,
    out: *mut u16,
    cap: usize,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let slice = std::slice::from_raw_parts_mut(out, cap);
    sphere_indices(rings, sectors, slice).map_or(-1, |n| n as i32)
}

/// Writes `[u_scale, u_offset, v_scale, v_offset]` for one eye of a stereo
/// layout. `eye` is 0 for left, anything else for right.
///
/// # Safety
/// `out` must point to four writable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_pano_uv(layout: u32, eye: i32, out: *mut f32) -> i32 {
    if out.is_null() {
        return -1;
    }
    let eye = if eye == 0 { Eye::Left } else { Eye::Right };
    let w = uv_window(StereoLayout::from_raw(layout), eye);
    std::ptr::copy_nonoverlapping(w.as_ptr(), out, 4);
    0
}

/// Writes the panorama view-projection for a head orientation, column-major and
/// ready for `glUniformMatrix4fv`.
///
/// The orientation is passed in rather than read from a tracker so that the
/// caller's scene and its panorama are built from exactly the same sample. Two
/// independent reads a frame apart shear the video against anything drawn on
/// top of it, and the shear tracks head speed, so it shows up precisely when it
/// is most visible.
///
/// `eye_offset` displaces the camera sideways, in metres. Zero for a panorama,
/// where both eyes share the result — see [`view_projection_for_eye`] for why.
///
/// # Safety
/// `out` must point to sixteen writable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_pano_mvp(
    w: f32,
    x: f32,
    y: f32,
    z: f32,
    fov_y_deg: f32,
    aspect: f32,
    eye_offset: f32,
    out: *mut f32,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let mut m = [0.0f32; 16];
    view_projection_for_eye([w, x, y, z], fov_y_deg, aspect, eye_offset, &mut m);
    std::ptr::copy_nonoverlapping(m.as_ptr(), out, 16);
    0
}

/// Vertices [`xr_screen_mesh`] will write, for sizing the buffer.
#[no_mangle]
pub extern "C" fn xr_screen_vertex_count(segments: u32) -> u32 {
    screen_vertex_count(segments) as u32
}

/// Fills `out` with a screen standing in the room, as an interleaved
/// `[x, y, z, u, v]` triangle strip. Returns the vertex count, or -1.
///
/// `aspect` is the picture's shape after the stereo split, not the frame's.
///
/// # Safety
/// `out` must point to `cap_floats` writable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_screen_mesh(
    segments: u32,
    distance: f32,
    width_deg: f32,
    aspect: f32,
    curvature: f32,
    out: *mut f32,
    cap_floats: usize,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let slice = std::slice::from_raw_parts_mut(out, cap_floats);
    screen_mesh(segments, distance, width_deg, aspect, curvature, slice).map_or(-1, |n| n as i32)
}

/// Writes the `[u, v]` a head orientation is looking at.
///
/// # Safety
/// `out` must point to two writable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_pano_gaze_uv(w: f32, x: f32, y: f32, z: f32, out: *mut f32) -> i32 {
    if out.is_null() {
        return -1;
    }
    let uv = gaze_uv([w, x, y, z]);
    std::ptr::copy_nonoverlapping(uv.as_ptr(), out, 2);
    0
}

/// The vertical field of view, in degrees, for a zoom factor.
#[no_mangle]
pub extern "C" fn xr_pano_fov_for_zoom(zoom: f32) -> f32 {
    fov_for_zoom(zoom)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RINGS: u32 = 32;
    const SECTORS: u32 = 64;

    fn build() -> (Vec<f32>, Vec<u16>) {
        let mut verts = vec![0.0f32; vertex_count(RINGS, SECTORS) * VERTEX_FLOATS];
        let mut idx = vec![0u16; index_count(RINGS, SECTORS)];
        assert_eq!(
            sphere_mesh(RINGS, SECTORS, 1.0, Projection::Equirect360, &mut verts),
            Some(vertex_count(RINGS, SECTORS))
        );
        assert_eq!(
            sphere_indices(RINGS, SECTORS, &mut idx),
            Some(index_count(RINGS, SECTORS))
        );
        (verts, idx)
    }

    fn verts_of() -> Vec<f32> {
        build().0
    }

    fn pos(verts: &[f32], i: u16) -> [f32; 3] {
        let o = i as usize * VERTEX_FLOATS;
        [verts[o], verts[o + 1], verts[o + 2]]
    }

    #[test]
    fn every_vertex_sits_on_the_sphere() {
        let (verts, _) = build();
        for chunk in verts.as_chunks::<VERTEX_FLOATS>().0.iter() {
            let r = (chunk[0] * chunk[0] + chunk[1] * chunk[1] + chunk[2] * chunk[2]).sqrt();
            assert!((r - 1.0).abs() < 1e-4, "radius {r}");
        }
    }

    #[test]
    fn image_centre_is_straight_ahead() {
        // (u, v) = (0.5, 0.5) must land on -Z, or the video starts off-centre.
        let (verts, _) = build();
        let mid = verts
            .as_chunks::<VERTEX_FLOATS>()
            .0
            .iter()
            .find(|c| (c[3] - 0.5).abs() < 1e-6 && (c[4] - 0.5).abs() < 1e-6)
            .expect("the grid contains u = v = 0.5 for even ring and sector counts");
        assert!(mid[0].abs() < 1e-5, "x {}", mid[0]);
        assert!(mid[1].abs() < 1e-5, "y {}", mid[1]);
        assert!((mid[2] + 1.0).abs() < 1e-5, "z {}", mid[2]);
    }

    #[test]
    fn increasing_u_sweeps_right() {
        let (verts, _) = build();
        // Three quarters along is a quarter turn to the right: +X.
        let right = verts
            .as_chunks::<VERTEX_FLOATS>()
            .0
            .iter()
            .find(|c| (c[3] - 0.75).abs() < 1e-6 && (c[4] - 0.5).abs() < 1e-6)
            .expect("grid contains u = 0.75, v = 0.5");
        assert!(right[0] > 0.99, "x {}", right[0]);
    }

    #[test]
    fn the_top_of_the_texture_is_the_zenith() {
        // Texture space is bottom-up, so v = 1 must be overhead. Backwards is a
        // plausible-looking convention either way, and it renders the whole
        // panorama upside down.
        let rows = |target: f32| {
            *verts_of()
                .as_chunks::<VERTEX_FLOATS>()
                .0
                .iter()
                .find(|c| c[4] == target)
                .unwrap()
        };
        let up = rows(1.0);
        let down = rows(0.0);
        assert!(
            (up[1] - 1.0).abs() < 1e-5,
            "v = 1 should be up, y = {}",
            up[1]
        );
        assert!(
            (down[1] + 1.0).abs() < 1e-5,
            "v = 0 should be down, y = {}",
            down[1]
        );
    }

    #[test]
    fn seam_column_is_duplicated_not_wrapped() {
        // The first and last column must be the same point with u = 0 and u = 1,
        // otherwise the whole texture interpolates across the seam behind you.
        let (verts, _) = build();
        let stride = (SECTORS + 1) as usize * VERTEX_FLOATS;
        for ring in 0..=RINGS as usize {
            let first = ring * stride;
            let last = first + SECTORS as usize * VERTEX_FLOATS;
            for axis in 0..3 {
                assert!(
                    (verts[first + axis] - verts[last + axis]).abs() < 1e-5,
                    "ring {ring} axis {axis} not coincident"
                );
            }
            assert_eq!(verts[first + 3], 0.0);
            assert_eq!(verts[last + 3], 1.0);
        }
    }

    #[test]
    fn all_triangles_face_the_centre() {
        // The winding decides whether back-face culling shows the panorama or a
        // black screen, and it is invisible in review — so it is asserted.
        let (verts, idx) = build();
        assert_eq!(idx.len() % 3, 0);
        for (n, tri) in idx.as_chunks::<3>().0.iter().enumerate() {
            let a = pos(&verts, tri[0]);
            let b = pos(&verts, tri[1]);
            let c = pos(&verts, tri[2]);
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let normal = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let area =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            assert!(area > 1e-9, "triangle {n} is degenerate");
            // Right-hand normal pointing back towards the origin means the
            // triangle is counter-clockwise as seen from inside.
            let facing = normal[0] * a[0] + normal[1] * a[1] + normal[2] * a[2];
            assert!(facing < 0.0, "triangle {n} faces outward ({facing})");
        }
    }

    #[test]
    fn a_hemisphere_spans_half_the_longitude() {
        // VR180 puts the whole image in front of you. If the arc were still a
        // full turn the footage would wrap round the back at half scale, which
        // looks plausible enough in a screenshot to survive review.
        let mut verts = vec![0.0f32; vertex_count(RINGS, SECTORS) * VERTEX_FLOATS];
        assert!(sphere_mesh(RINGS, SECTORS, 1.0, Projection::Equirect180, &mut verts).is_some());
        let column = |u: f32| {
            *verts
                .as_chunks::<VERTEX_FLOATS>()
                .0
                .iter()
                .find(|c| (c[3] - u).abs() < 1e-6 && (c[4] - 0.5).abs() < 1e-6)
                .unwrap()
        };
        // The centre still looks straight ahead.
        let middle = column(0.5);
        assert!(
            middle[0].abs() < 1e-5 && (middle[2] + 1.0).abs() < 1e-5,
            "{middle:?}"
        );
        // The edges reach a quarter turn each way, not half.
        let right = column(1.0);
        assert!((right[0] - 1.0).abs() < 1e-4, "right edge x {}", right[0]);
        assert!(right[2].abs() < 1e-4, "right edge z {}", right[2]);
        let left = column(0.0);
        assert!((left[0] + 1.0).abs() < 1e-4, "left edge x {}", left[0]);
    }

    #[test]
    fn a_hemisphere_still_faces_inward() {
        // The winding must survive the narrower arc, or VR180 renders black.
        let mut verts = vec![0.0f32; vertex_count(RINGS, SECTORS) * VERTEX_FLOATS];
        let mut idx = vec![0u16; index_count(RINGS, SECTORS)];
        sphere_mesh(RINGS, SECTORS, 1.0, Projection::Equirect180, &mut verts).unwrap();
        sphere_indices(RINGS, SECTORS, &mut idx).unwrap();
        for tri in idx.as_chunks::<3>().0 {
            let a = pos(&verts, tri[0]);
            let b = pos(&verts, tri[1]);
            let c = pos(&verts, tri[2]);
            let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
            let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
            let n = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let facing = n[0] * a[0] + n[1] * a[1] + n[2] * a[2];
            // Polar triangles collapse to a sliver on a hemisphere; only judge
            // the ones with area worth judging.
            let area = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
            if area > 1e-6 {
                assert!(facing <= 0.0, "outward-facing triangle ({facing})");
            }
        }
    }

    #[test]
    fn a_flat_screen_stands_at_the_distance_asked_for() {
        let mut verts = vec![0.0f32; screen_vertex_count(32) * VERTEX_FLOATS];
        screen_mesh(32, 4.0, 45.0, 16.0 / 9.0, 0.0, &mut verts).unwrap();
        for c in verts.as_chunks::<VERTEX_FLOATS>().0 {
            assert!(
                (c[2] + 4.0).abs() < 1e-4,
                "a flat screen bent to z = {}",
                c[2]
            );
        }
    }

    #[test]
    fn a_fully_curved_screen_is_equidistant() {
        // The reason to curve one at all: every part of it is then the same
        // distance away, so none of it is out of focus.
        let mut verts = vec![0.0f32; screen_vertex_count(64) * VERTEX_FLOATS];
        screen_mesh(64, 4.0, 90.0, 16.0 / 9.0, 1.0, &mut verts).unwrap();
        for c in verts.as_chunks::<VERTEX_FLOATS>().0 {
            let horizontal = (c[0] * c[0] + c[2] * c[2]).sqrt();
            assert!(
                (horizontal - 4.0).abs() < 1e-3,
                "a point {horizontal} m away on a screen that should be 4 m",
            );
        }
    }

    #[test]
    fn a_screen_subtends_the_angle_it_was_asked_for() {
        // Size is expressed as an angle because that is the thing a person has
        // an opinion about; distance only decides how far away it feels.
        for width in [20.0f32, 45.0, 90.0] {
            let mut verts = vec![0.0f32; screen_vertex_count(8) * VERTEX_FLOATS];
            screen_mesh(8, 4.0, width, 16.0 / 9.0, 0.0, &mut verts).unwrap();
            let chunks = verts.as_chunks::<VERTEX_FLOATS>().0;
            let right = chunks.last().unwrap();
            let subtended = 2.0 * (right[0] / -right[2]).atan().to_degrees();
            assert!(
                (subtended - width).abs() < 0.01,
                "asked for {width}°, got {subtended}°",
            );
        }
    }

    #[test]
    fn a_screen_keeps_the_picture_s_shape() {
        // Passing the frame's own aspect for a side-by-side film would give a
        // screen twice as wide as the picture on it.
        let mut verts = vec![0.0f32; screen_vertex_count(8) * VERTEX_FLOATS];
        screen_mesh(8, 4.0, 45.0, 16.0 / 9.0, 0.0, &mut verts).unwrap();
        let chunks = verts.as_chunks::<VERTEX_FLOATS>().0;
        let half_width = chunks.last().unwrap()[0];
        let half_height = chunks[0][1];
        assert!(
            ((half_width / half_height) - 16.0 / 9.0).abs() < 1e-3,
            "the screen is {}:1 for a 16:9 picture",
            half_width / half_height,
        );
    }

    #[test]
    fn the_screen_s_texture_runs_the_same_way_as_the_sphere_s() {
        // Two conventions in one renderer is how a picture ends up upside down
        // on one surface and not the other.
        let mut verts = vec![0.0f32; screen_vertex_count(4) * VERTEX_FLOATS];
        screen_mesh(4, 4.0, 45.0, 16.0 / 9.0, 0.0, &mut verts).unwrap();
        let chunks = verts.as_chunks::<VERTEX_FLOATS>().0;
        let top = chunks
            .iter()
            .max_by(|a, b| a[1].partial_cmp(&b[1]).unwrap())
            .unwrap();
        let bottom = chunks
            .iter()
            .min_by(|a, b| a[1].partial_cmp(&b[1]).unwrap())
            .unwrap();
        assert_eq!(top[4], 1.0, "the top of the screen should be v = 1");
        assert_eq!(bottom[4], 0.0, "the bottom should be v = 0");
        // And u grows to the right, as on the sphere.
        let left = chunks
            .iter()
            .min_by(|a, b| a[0].partial_cmp(&b[0]).unwrap())
            .unwrap();
        assert_eq!(left[3], 0.0);
    }

    #[test]
    fn the_screen_faces_the_viewer() {
        // A triangle strip inherits the winding of its first triangle, so that
        // one decides whether the screen is visible at all with culling on.
        let mut verts = vec![0.0f32; screen_vertex_count(8) * VERTEX_FLOATS];
        screen_mesh(8, 4.0, 45.0, 16.0 / 9.0, 0.0, &mut verts).unwrap();
        let chunks = verts.as_chunks::<VERTEX_FLOATS>().0;
        let p = |i: usize| [chunks[i][0], chunks[i][1], chunks[i][2]];
        let (a, b, c) = (p(0), p(1), p(2));
        let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
        let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
        let n = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        // The normal must point back towards the viewer at the origin.
        let facing = n[0] * a[0] + n[1] * a[1] + n[2] * a[2];
        assert!(facing < 0.0, "the screen faces away ({facing})");
    }

    #[test]
    fn a_screen_at_its_default_distance_is_inside_the_frustum() {
        // The far plane was four metres, chosen for a unit sphere, and a screen
        // stands at four metres — so the default screen sat exactly on it. A
        // point on the plane is on the boundary of what survives clipping, which
        // is not a place to leave the thing the viewer is looking at.
        let mut verts = vec![0.0f32; screen_vertex_count(32) * VERTEX_FLOATS];
        screen_mesh(
            32,
            DEFAULT_SCREEN_DISTANCE,
            DEFAULT_SCREEN_WIDTH_DEG,
            16.0 / 9.0,
            0.0,
            &mut verts,
        )
        .unwrap();
        let mut mvp = [0.0f32; 16];
        view_projection([1.0, 0.0, 0.0, 0.0], DEFAULT_FOV_DEG, 16.0 / 9.0, &mut mvp);
        for c in verts.as_chunks::<VERTEX_FLOATS>().0 {
            let w = mvp[3] * c[0] + mvp[7] * c[1] + mvp[11] * c[2] + mvp[15];
            let z = mvp[2] * c[0] + mvp[6] * c[1] + mvp[10] * c[2] + mvp[14];
            assert!(w > 0.0, "a screen vertex fell behind the eye");
            let depth = z / w;
            assert!(
                (-1.0..1.0).contains(&depth),
                "a screen vertex clipped at depth {depth}",
            );
        }
    }

    #[test]
    fn an_unusable_screen_request_is_refused() {
        let mut verts = vec![0.0f32; screen_vertex_count(32) * VERTEX_FLOATS];
        assert_eq!(
            screen_mesh(0, 4.0, 45.0, 1.78, 0.0, &mut verts),
            None,
            "no segments"
        );
        assert_eq!(
            screen_mesh(8, 0.0, 45.0, 1.78, 0.0, &mut verts),
            None,
            "at the eye"
        );
        assert_eq!(
            screen_mesh(8, 4.0, 45.0, 0.0, 0.0, &mut verts),
            None,
            "no aspect"
        );
        let mut tiny = [0.0f32; 4];
        assert_eq!(
            screen_mesh(8, 4.0, 45.0, 1.78, 0.0, &mut tiny),
            None,
            "no room"
        );
    }

    #[test]
    fn an_eye_offset_moves_the_camera_and_nothing_else() {
        // A screen at four metres should shift a little between the eyes; a
        // panorama should not shift at all, which is why the offset is a
        // parameter rather than always applied.
        let point = [0.0f32, 0.0, -4.0];
        let mut centred = [0.0f32; 16];
        let mut right_eye = [0.0f32; 16];
        view_projection([1.0, 0.0, 0.0, 0.0], DEFAULT_FOV_DEG, 1.78, &mut centred);
        view_projection_for_eye(
            [1.0, 0.0, 0.0, 0.0],
            DEFAULT_FOV_DEG,
            1.78,
            0.0315,
            &mut right_eye,
        );

        let x_of = |m: &[f32; 16]| {
            let w = m[3] * point[0] + m[7] * point[1] + m[11] * point[2] + m[15];
            (m[0] * point[0] + m[4] * point[1] + m[8] * point[2] + m[12]) / w
        };
        let shift = x_of(&right_eye) - x_of(&centred);
        assert!(
            shift < 0.0,
            "the right eye should see the screen shifted left"
        );
        assert!(shift.abs() > 1e-4, "the eye offset did nothing");
        // And view_projection itself is still the zero-offset case.
        assert!((x_of(&centred)).abs() < 1e-6);
    }

    #[test]
    fn indices_stay_in_range() {
        let (_, idx) = build();
        let max = vertex_count(RINGS, SECTORS) as u16;
        assert!(idx.iter().all(|&i| i < max));
    }

    #[test]
    fn tessellation_is_validated() {
        let mut verts = vec![0.0f32; 1 << 20];
        let mut idx = vec![0u16; 1 << 20];
        assert_eq!(
            sphere_mesh(1, 64, 1.0, Projection::Equirect360, &mut verts),
            None,
            "one ring"
        );
        assert_eq!(
            sphere_mesh(32, 2, 1.0, Projection::Equirect360, &mut verts),
            None,
            "two sectors"
        );
        assert_eq!(
            sphere_mesh(512, 512, 1.0, Projection::Equirect360, &mut verts),
            None,
            "over 16-bit"
        );
        assert_eq!(sphere_indices(512, 512, &mut idx), None, "over 16-bit");
        // Short output slices are rejected rather than half-filled.
        let mut tiny = [0.0f32; 4];
        assert_eq!(
            sphere_mesh(RINGS, SECTORS, 1.0, Projection::Equirect360, &mut tiny),
            None
        );
        let mut tiny_idx = [0u16; 4];
        assert_eq!(sphere_indices(RINGS, SECTORS, &mut tiny_idx), None);
    }

    #[test]
    fn stereo_windows_split_the_frame_without_overlap() {
        assert_eq!(
            uv_window(StereoLayout::Mono, Eye::Left),
            [1.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            uv_window(StereoLayout::Mono, Eye::Right),
            [1.0, 0.0, 1.0, 0.0]
        );

        // The left eye reads the top half of the image, which in bottom-up
        // texture space is the upper half of v.
        let [_, _, vs, vl] = uv_window(StereoLayout::OverUnder, Eye::Left);
        let [_, _, _, vr] = uv_window(StereoLayout::OverUnder, Eye::Right);
        assert_eq!((vs, vl, vr), (0.5, 0.5, 0.0));

        let [us, ul, _, _] = uv_window(StereoLayout::SideBySide, Eye::Left);
        let [_, ur, _, _] = uv_window(StereoLayout::SideBySide, Eye::Right);
        assert_eq!((us, ul, ur), (0.5, 0.0, 0.5));
    }

    #[test]
    fn the_default_field_of_view_is_the_panel_s_own() {
        // A panorama is life-size only when the rendered angle matches the angle
        // the optics span. The constant is spelled out for readability; this
        // keeps it honest.
        let derived = vertical_fov(PANEL_DIAGONAL_FOV_DEG, 16.0 / 9.0);
        assert!(
            (derived - DEFAULT_FOV_DEG).abs() < 0.02,
            "derived {derived}, constant {DEFAULT_FOV_DEG}"
        );
        // A square screen's vertical angle is its diagonal scaled by 1/sqrt(2).
        let square = vertical_fov(90.0, 1.0);
        let expected = 2.0 * (45f32.to_radians().tan() / 2f32.sqrt()).atan().to_degrees();
        assert!((square - expected).abs() < 1e-3, "square {square}");
    }

    #[test]
    fn zoom_and_fov_are_inverses_inside_the_clamp() {
        // Inside the clamp: at a 25.8° neutral angle, 2.6x is where zooming in
        // hits the 10° floor.
        for zoom in [0.4f32, 1.0, 2.0, 2.5] {
            let fov = fov_for_zoom(zoom);
            assert!(
                (zoom_for_fov(fov) - zoom).abs() < 1e-4,
                "zoom {zoom} fov {fov}"
            );
        }
        assert_eq!(fov_for_zoom(1.0), DEFAULT_FOV_DEG);
        assert_eq!(fov_for_zoom(1000.0), MIN_FOV_DEG, "clamped in");
        assert_eq!(fov_for_zoom(0.0001), MAX_FOV_DEG, "clamped out");
        assert_eq!(fov_for_zoom(4.0), MIN_FOV_DEG, "past the floor");
        assert_eq!(fov_for_zoom(f32::NAN), DEFAULT_FOV_DEG);
        assert_eq!(fov_for_zoom(-1.0), DEFAULT_FOV_DEG);
    }

    #[test]
    fn identity_pose_looks_at_the_image_centre() {
        let mut m = [0.0f32; 16];
        view_projection([1.0, 0.0, 0.0, 0.0], DEFAULT_FOV_DEG, 16.0 / 9.0, &mut m);
        // Straight ahead must project to the middle of the screen.
        let p = project(&m, [0.0, 0.0, -1.0]);
        assert!(p[0].abs() < 1e-5 && p[1].abs() < 1e-5, "centre at {p:?}");
        // And behind the viewer must not.
        let behind = project_raw(&m, [0.0, 0.0, 1.0]);
        assert!(behind[3] < 0.0, "w {} should be negative behind", behind[3]);
    }

    #[test]
    fn turning_right_moves_the_scene_left() {
        // Yawing right by 30° must push what was ahead towards -X on screen; the
        // opposite sign means the panorama drags with the head instead of
        // staying put, which is the classic inverted-tracking bug.
        let a = (30f32.to_radians() / 2.0).sin();
        let c = (30f32.to_radians() / 2.0).cos();
        // Right-handed yaw about +Y by -30° turns the head to its right.
        let mut m = [0.0f32; 16];
        view_projection([c, 0.0, -a, 0.0], DEFAULT_FOV_DEG, 16.0 / 9.0, &mut m);
        let p = project(&m, [0.0, 0.0, -1.0]);
        assert!(p[0] < -0.1, "image centre should slide left, got {}", p[0]);
    }

    #[test]
    fn zooming_in_magnifies() {
        let point = [0.2f32, 0.0, -1.0];
        let mut wide = [0.0f32; 16];
        let mut tight = [0.0f32; 16];
        view_projection([1.0, 0.0, 0.0, 0.0], fov_for_zoom(1.0), 1.0, &mut wide);
        view_projection([1.0, 0.0, 0.0, 0.0], fov_for_zoom(2.0), 1.0, &mut tight);
        assert!(
            project(&tight, point)[0] > project(&wide, point)[0] * 1.5,
            "narrowing the field of view must spread the image out"
        );
    }

    #[test]
    fn gaze_uv_matches_the_mesh() {
        assert_eq!(gaze_uv([1.0, 0.0, 0.0, 0.0]), [0.5, 0.5]);
        // Same -30° yaw as above: looking right means a larger u.
        let a = (30f32.to_radians() / 2.0).sin();
        let c = (30f32.to_radians() / 2.0).cos();
        let uv = gaze_uv([c, 0.0, -a, 0.0]);
        assert!((uv[0] - (0.5 + 30.0 / 360.0)).abs() < 1e-4, "u {}", uv[0]);
        assert!((uv[1] - 0.5).abs() < 1e-4, "v {}", uv[1]);
    }

    /// Column-major matrix times point, returning the full clip-space vector.
    fn project_raw(m: &[f32; 16], p: [f32; 3]) -> [f32; 4] {
        let mut out = [0.0f32; 4];
        for row in 0..4 {
            out[row] = m[row] * p[0] + m[4 + row] * p[1] + m[8 + row] * p[2] + m[12 + row];
        }
        out
    }

    /// Normalised device coordinates.
    fn project(m: &[f32; 16], p: [f32; 3]) -> [f32; 2] {
        let v = project_raw(m, p);
        [v[0] / v[3], v[1] / v[3]]
    }
}
