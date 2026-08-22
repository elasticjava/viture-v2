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
    /// Both eyes in every pixel, separated by colour.
    ///
    /// Made for cardboard glasses with two coloured filters, and hostile to any
    /// other display: each eye's picture survives only in the channels its
    /// filter passes, so the colour is gone and only the brightness is left.
    /// Shown as-is on a panel it is a red-and-blue smear. Recovering a grey
    /// picture per eye is the most the frame contains, and it is worth doing —
    /// the alternative is that a whole class of old 3D files plays unwatchably.
    Anaglyph = 3,
    /// Eyes on alternating rows of the frame.
    ///
    /// What passive polarised televisions want, and unlike the other packings
    /// its two halves are interleaved rather than adjacent, so no rectangle of
    /// texture is one eye's picture. It has to be resolved per pixel.
    RowInterleaved = 4,
}

impl StereoLayout {
    pub fn from_raw(v: u32) -> StereoLayout {
        match v {
            1 => StereoLayout::OverUnder,
            2 => StereoLayout::SideBySide,
            3 => StereoLayout::Anaglyph,
            4 => StereoLayout::RowInterleaved,
            _ => StereoLayout::Mono,
        }
    }

    /// Whether each eye's picture is a rectangle of the frame, which is what
    /// [`uv_window`] can express on its own.
    ///
    /// The two that are not need the fragment shader: one separates the eyes by
    /// colour and the other by row, and neither is a window.
    pub fn is_windowed(self) -> bool {
        !matches!(self, StereoLayout::Anaglyph | StereoLayout::RowInterleaved)
    }
}

/// Which pair of complementary colours an anaglyph frame was encoded for.
///
/// Named left eye first, and that is a claim about the pixels rather than a
/// repetition of what the container calls it. Matroska names the red/cyan one
/// "anaglyph (cyan/red)", which read as left-then-right says the left eye is in
/// the green and blue channels — and it is not. Encoders put the left eye in
/// red, because red-cyan glasses are worn with the red lens over the left eye,
/// and that is what the files contain: an anaglyph made by `ffmpeg -vf
/// stereo3d=sbsl:arcg` has its left view in the red channel, checked rather
/// than assumed.
///
/// So the container's label is mapped onto these, not copied into them, and a
/// file that turns out to be the other way round is handled by swapping the
/// eyes — which is something the viewer can see and decide.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub enum AnaglyphPair {
    /// Left eye red, right eye cyan. What Matroska labels "cyan/red".
    RedCyan = 0,
    /// Left eye green, right eye magenta.
    GreenMagenta = 1,
    /// Left eye yellow, right eye blue.
    YellowBlue = 2,
}

impl AnaglyphPair {
    pub fn from_raw(v: u32) -> AnaglyphPair {
        match v {
            1 => AnaglyphPair::GreenMagenta,
            2 => AnaglyphPair::YellowBlue,
            _ => AnaglyphPair::RedCyan,
        }
    }
}

/// How much of each colour channel carries one eye's picture.
///
/// The weights sum to one, so a grey frame comes back at its own brightness
/// rather than darkened or blown out. A filter that passes two channels gets
/// half of each: both hold the same eye's image, and averaging them is a free
/// halving of the encoder's chroma noise.
pub fn anaglyph_mix(pair: AnaglyphPair, eye: Eye) -> [f32; 3] {
    let left = eye == Eye::Left;
    match (pair, left) {
        (AnaglyphPair::RedCyan, true) => [1.0, 0.0, 0.0],
        (AnaglyphPair::RedCyan, false) => [0.0, 0.5, 0.5],
        (AnaglyphPair::GreenMagenta, true) => [0.0, 1.0, 0.0],
        (AnaglyphPair::GreenMagenta, false) => [0.5, 0.0, 0.5],
        (AnaglyphPair::YellowBlue, true) => [0.5, 0.5, 0.0],
        (AnaglyphPair::YellowBlue, false) => [0.0, 0.0, 1.0],
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
    /// Six faces of a cube, as a 3x2 grid in one frame.
    ///
    /// The `cbmp` box's layout 0. Cube mapping spends its pixels far more evenly
    /// than an equirectangular image, which crowds them at the poles and starves
    /// the horizon — the part anyone actually looks at.
    Cubemap = 3,
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
            3 => Projection::Cubemap,
            _ => Projection::Equirect360,
        }
    }

    /// Whether this projection wraps around the viewer at all.
    pub fn is_panoramic(self) -> bool {
        self != Projection::Flat
    }

    /// The equirectangular bounds this projection is shorthand for, if any.
    pub fn bounds(self) -> Option<Bounds> {
        match self {
            Projection::Equirect360 => Some(Bounds::FULL),
            // The front half: a quarter of the sphere cropped from each side.
            Projection::Equirect180 => Some(Bounds {
                top: 0.0,
                bottom: 0.0,
                left: 0.25,
                right: 0.25,
            }),
            _ => None,
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

/// How much of the sphere a frame covers, as the `equi` box states it.
///
/// Each field is the proportion of the projection cropped from that edge and is
/// not covered by the video, so all zeroes is the whole sphere. The box stores
/// them as 0.32 fixed point; they arrive here as fractions.
///
/// This is what makes 360° and VR180 two points on one scale rather than two
/// cases: a hemisphere is a quarter cropped from each side, and a file that
/// covers, say, 270° by 140° is neither and is perfectly legal.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Bounds {
    pub top: f32,
    pub bottom: f32,
    pub left: f32,
    pub right: f32,
}

impl Bounds {
    /// The whole sphere.
    pub const FULL: Bounds = Bounds {
        top: 0.0,
        bottom: 0.0,
        left: 0.0,
        right: 0.0,
    };

    /// Whether the crops leave anything to draw.
    pub fn is_usable(self) -> bool {
        [self.top, self.bottom, self.left, self.right]
            .iter()
            .all(|v| v.is_finite() && (0.0..1.0).contains(v))
            && self.top + self.bottom < 1.0
            && self.left + self.right < 1.0
    }

    /// Longitude covered, in radians.
    pub fn arc(self) -> f32 {
        (1.0 - self.left - self.right) * std::f32::consts::TAU
    }

    /// Latitude covered, in radians.
    pub fn pitch_span(self) -> f32 {
        (1.0 - self.top - self.bottom) * std::f32::consts::PI
    }
}

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

/// A pair of glasses this driver knows the optics of.
///
/// The field of view cannot be worked out from inside the software. There is no
/// sensor pointed at the virtual image and nothing on the wire that reports it;
/// it is a property of lenses. What the software *can* do is recognise which
/// glasses it is talking to and look the number up, which is what this is.
///
/// The per-eye resolution is here too, and it is not decoration: the Beast is
/// 1920x1200 where everything else is 1920x1080, and a renderer that assumes
/// 16:9 draws a stretched world on it. That part is also readable from the
/// display itself, which is the better source when the two disagree — the panel
/// knows its own shape and this table is a copy.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Model {
    /// USB product id, which is how the glasses are told apart.
    pub product_id: u16,
    pub name: &'static str,
    /// Diagonal field of view in degrees, as the manufacturer publishes it.
    pub diagonal_fov_deg: f32,
    pub eye_width: u32,
    pub eye_height: u32,
    /// Whether these numbers were checked against the hardware or taken from
    /// the manufacturer. Worth carrying: a published figure is a claim, and the
    /// one model measured here already disagreed with its own datasheet about
    /// refresh rate.
    pub verified: bool,
}

/// The glasses this driver has numbers for.
///
/// Only the Pro 2 has been in front of it. The rest are from VITURE's published
/// specifications and are marked accordingly — they are far better than
/// pretending every pair is a Pro 2, and they are not measurements.
///
/// Product ids beyond the Pro 2's are not known, so those entries are matched
/// by nothing yet and exist to be filled in the moment one of these is
/// attached: the id is logged on every connection for exactly that reason.
pub const KNOWN_MODELS: [Model; 4] = [
    Model {
        product_id: 0x1301,
        name: "VITURE Pro 2",
        diagonal_fov_deg: 50.0,
        eye_width: 1920,
        eye_height: 1080,
        verified: true,
    },
    Model {
        product_id: 0,
        name: "VITURE Luma",
        diagonal_fov_deg: 52.0,
        eye_width: 1920,
        eye_height: 1080,
        verified: false,
    },
    Model {
        product_id: 0,
        name: "VITURE Luma Ultra",
        diagonal_fov_deg: 52.0,
        eye_width: 1920,
        eye_height: 1080,
        verified: false,
    },
    Model {
        product_id: 0,
        // 1920x1200 an eye, which is 16:10 and not 16:9. The one entry in this
        // table whose shape differs, and the reason the table carries shapes.
        name: "VITURE Beast",
        diagonal_fov_deg: 58.0,
        eye_width: 1920,
        eye_height: 1200,
        verified: false,
    },
];

/// The glasses with this product id, if they are known.
///
/// A product id of zero never matches, which is what keeps the unverified
/// entries above from being chosen by accident.
pub fn model_for(product_id: u16) -> Option<&'static Model> {
    if product_id == 0 {
        return None;
    }
    KNOWN_MODELS.iter().find(|m| m.product_id == product_id)
}

/// The vertical field of view to render at, for glasses whose product id is
/// `product_id` and whose panel is `eye_width` by `eye_height` per eye.
///
/// Both halves of the answer come from the best source available, which is not
/// the same source for each. The **shape** comes from the panel, because the
/// panel knows it and reports it; the **angle** comes from the table, because
/// nothing reports it. Unknown glasses get the Pro 2's angle with their own
/// shape, which is wrong by a few degrees of scale rather than wrong by a
/// stretch — the mistake nobody notices instead of the one everybody does.
pub fn fov_for_panel(product_id: u16, eye_width: u32, eye_height: u32) -> f32 {
    let diagonal = model_for(product_id)
        .map(|m| m.diagonal_fov_deg)
        .unwrap_or(PANEL_DIAGONAL_FOV_DEG);
    let aspect = if eye_height > 0 {
        eye_width as f32 / eye_height as f32
    } else {
        16.0 / 9.0
    };
    vertical_fov(diagonal, aspect)
}

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

/// How many indices to make room for at this tessellation.
///
/// An upper bound, not an exact count: two triangles per grid cell everywhere.
/// Where the mesh reaches a pole one corner of the cell collapses and one of
/// the two triangles has no area, and those are skipped — a degenerate triangle
/// costs a primitive-assembly slot for nothing. How many are skipped depends on
/// whether the covered patch reaches the poles at all, which the buffer has to
/// be allocated without knowing. Draw with the count [`sphere_indices`]
/// returns, not with this.
pub const fn index_count(rings: u32, sectors: u32) -> usize {
    if rings < 2 {
        return 0;
    }
    6 * rings as usize * sectors as usize
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
    sphere_mesh_bounded(rings, sectors, radius, projection.bounds()?, out)
}

/// The same, over an arbitrary patch of the sphere.
///
/// This is the general case and [`sphere_mesh`] is two named points on it. A
/// file is free to cover 270° by 140°, and the `equi` box says so in exactly
/// these terms; treating coverage as a pair of presets meant such a file was
/// stretched over whichever preset was nearest.
pub fn sphere_mesh_bounded(
    rings: u32,
    sectors: u32,
    radius: f32,
    bounds: Bounds,
    out: &mut [f32],
) -> Option<usize> {
    let count = vertex_count(rings, sectors);
    if rings < 2 || sectors < 3 || count > MAX_VERTICES || out.len() < count * VERTEX_FLOATS {
        return None;
    }
    if !bounds.is_usable() {
        return None;
    }
    let arc = bounds.arc();
    // Where the covered patch sits. A crop that is not symmetric moves the
    // middle of the picture off the straight-ahead direction, which is correct:
    // the file says which part of the sphere it holds, not merely how much.
    let lon_centre = (bounds.left - bounds.right) * std::f32::consts::PI;
    let lat_top = (0.5 - bounds.top) * std::f32::consts::PI;
    let lat_span = bounds.pitch_span();

    let mut w = 0;
    for ring in 0..=rings {
        // Rows run from the top of the covered patch down, so that consecutive
        // vertices are adjacent in memory the way the index buffer walks them.
        let down = ring as f32 / rings as f32;
        // Texture space is bottom-up: the first row is v = 1.
        let v = 1.0 - down;
        let lat = lat_top - down * lat_span;
        let (sin_lat, cos_lat) = lat.sin_cos();
        let y = radius * sin_lat;
        // Radius of this latitude's circle.
        let r = radius * cos_lat;

        for sector in 0..=sectors {
            let u = sector as f32 / sectors as f32;
            // Longitude. The image always spans the full `u` range; the bounds
            // say how much world that is and where it sits.
            let lon = lon_centre + (u - 0.5) * arc;
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
    sphere_indices_bounded(rings, sectors, Bounds::FULL, out)
}

/// The same, for a mesh built by [`sphere_mesh_bounded`].
///
/// The bounds matter here for one reason: the top and bottom rows of a full
/// sphere collapse onto the poles, so half of each of those cells is a triangle
/// with no area and is dropped. Crop the sphere and those rows are ordinary
/// quads that need both triangles — dropping them would leave two visible bands
/// of nothing along the top and bottom of the picture.
pub fn sphere_indices_bounded(
    rings: u32,
    sectors: u32,
    bounds: Bounds,
    out: &mut [u16],
) -> Option<usize> {
    if rings < 2 || sectors < 3 || vertex_count(rings, sectors) > MAX_VERTICES {
        return None;
    }
    if !bounds.is_usable() {
        return None;
    }
    // Only an uncropped edge actually reaches a pole.
    let at_zenith = bounds.top == 0.0;
    let at_nadir = bounds.bottom == 0.0;
    let count = index_count(rings, sectors)
        - 3 * sectors as usize * (usize::from(at_zenith) + usize::from(at_nadir));
    if out.len() < count {
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

            if ring > 0 || !at_zenith {
                out[w] = a;
                out[w + 1] = d;
                out[w + 2] = b;
                w += 3;
            }
            if ring + 1 < rings || !at_nadir {
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

/// One face of a cube map: where it points, and where its picture sits.
///
/// `centre` is the direction the middle of the face lies in; `right` and `up`
/// are the world directions that image-right and image-up point along once you
/// are looking at it. Straight ahead is `-Z`, up is `+Y`, and turning right
/// goes towards `+X`, which is the convention [`sphere_mesh_bounded`] builds
/// to.
struct CubeFace {
    centre: [f32; 3],
    right: [f32; 3],
    up: [f32; 3],
    column: u32,
    row: u32,
}

/// The six faces, in the order and orientation the `cbmp` box's layout 0 packs
/// them into a 3x2 grid.
///
/// The two odd ones are the poles. Tilt your head back to look at the ceiling
/// and your head's up direction swings from `+Y` towards `-Z`, so the top of
/// the ceiling image points forward; look down and it swings towards `+Z`, so
/// the top of the floor image points backwards. The spec words this as "top of
/// face forward" and "top of face backward", and this is what that means.
const CUBE_FACES: [CubeFace; 6] = [
    CubeFace {
        centre: [1.0, 0.0, 0.0],
        right: [0.0, 0.0, 1.0],
        up: [0.0, 1.0, 0.0],
        column: 0,
        row: 0,
    },
    CubeFace {
        centre: [-1.0, 0.0, 0.0],
        right: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        column: 1,
        row: 0,
    },
    CubeFace {
        centre: [0.0, 1.0, 0.0],
        right: [-1.0, 0.0, 0.0],
        up: [0.0, 0.0, -1.0],
        column: 2,
        row: 0,
    },
    CubeFace {
        centre: [0.0, -1.0, 0.0],
        right: [-1.0, 0.0, 0.0],
        up: [0.0, 0.0, 1.0],
        column: 0,
        row: 1,
    },
    CubeFace {
        centre: [0.0, 0.0, -1.0],
        right: [1.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        column: 1,
        row: 1,
    },
    CubeFace {
        centre: [0.0, 0.0, 1.0],
        right: [-1.0, 0.0, 0.0],
        up: [0.0, 1.0, 0.0],
        column: 2,
        row: 1,
    },
];

/// Columns and rows the `cbmp` layout 0 grid has.
const CUBE_COLUMNS: f32 = 3.0;
const CUBE_ROWS: f32 = 2.0;

/// Vertices [`cubemap_mesh`] writes: six independent faces, each a grid.
///
/// The faces cannot share vertices even where they meet in space, because they
/// do not meet in the texture — a shared corner would need six texture
/// coordinates at once.
pub const fn cubemap_vertex_count(cells: u32) -> usize {
    6 * (cells as usize + 1) * (cells as usize + 1)
}

/// Indices [`cubemap_indices`] writes: two triangles per cell, six faces.
pub const fn cubemap_index_count(cells: u32) -> usize {
    36 * cells as usize * cells as usize
}

/// Writes the six faces of a cube as interleaved `[x, y, z, u, v]` vertices.
///
/// A cube map spends its pixels far more evenly than an equirectangular image,
/// which crowds them into the poles nobody looks at and starves the horizon
/// everybody does. Recent 360 material increasingly ships this way.
///
/// `padding` is the proportion of each grid cell to ignore along each of its
/// edges, from the box's pixel count divided by the face's pixel size. Encoders
/// pad the faces because bilinear filtering at a face edge would otherwise
/// fetch texels belonging to a face pointing somewhere else entirely, which
/// shows up as a bright seam along every cube edge.
///
/// The faces are flat and the texture maps onto them linearly, so one cell per
/// face is already exact; `cells` exists for the same reason a screen is
/// subdivided, to keep the interpolation of anything applied per-vertex honest.
pub fn cubemap_mesh(cells: u32, radius: f32, padding: f32, out: &mut [f32]) -> Option<usize> {
    let count = cubemap_vertex_count(cells);
    if cells == 0 || count > MAX_VERTICES || out.len() < count * VERTEX_FLOATS {
        return None;
    }
    if !padding.is_finite() || !(0.0..0.5).contains(&padding) || !radius.is_finite() {
        return None;
    }

    let mut w = 0;
    for face in &CUBE_FACES {
        // The cell this face occupies, shrunk by the padding. Texture space is
        // bottom-up, so the grid's first row is the upper half of v.
        let (column, row) = (face.column as f32, face.row as f32);
        let u0 = (column + padding) / CUBE_COLUMNS;
        let u1 = (column + 1.0 - padding) / CUBE_COLUMNS;
        let v1 = (CUBE_ROWS - row - padding) / CUBE_ROWS;
        let v0 = (CUBE_ROWS - row - 1.0 + padding) / CUBE_ROWS;

        for y in 0..=cells {
            let t = y as f32 / cells as f32;
            for x in 0..=cells {
                let s = x as f32 / cells as f32;
                // From the middle of the face out to its edges, in the two
                // directions the image runs.
                let a = 2.0 * s - 1.0;
                let b = 2.0 * t - 1.0;
                for axis in 0..3 {
                    out[w + axis] =
                        radius * (face.centre[axis] + face.right[axis] * a + face.up[axis] * b);
                }
                out[w + 3] = u0 + (u1 - u0) * s;
                out[w + 4] = v0 + (v1 - v0) * t;
                w += VERTEX_FLOATS;
            }
        }
    }
    Some(count)
}

/// Writes the triangle indices for a [`cubemap_mesh`] of the same tessellation.
pub fn cubemap_indices(cells: u32, out: &mut [u16]) -> Option<usize> {
    let count = cubemap_index_count(cells);
    if cells == 0 || cubemap_vertex_count(cells) > MAX_VERTICES || out.len() < count {
        return None;
    }

    let stride = cells + 1;
    let per_face = (stride * stride) as u16;
    let mut w = 0;
    for face in 0..6u16 {
        let base = face * per_face;
        for y in 0..cells {
            for x in 0..cells {
                let a = base + (y * stride + x) as u16;
                let b = a + 1;
                let c = a + stride as u16;
                let d = c + 1;
                // Wound counter-clockwise *seen from the centre*, which is
                // where the viewer is. The obvious order — a, c, d — is
                // counter-clockwise seen from outside, and outside is where
                // nobody is standing: with back-face culling on, every face of
                // the cube faced away and the whole thing rendered as black.
                out[w] = a;
                out[w + 1] = d;
                out[w + 2] = c;
                out[w + 3] = a;
                out[w + 4] = b;
                out[w + 5] = d;
                w += 6;
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
        // Both eyes cover the whole frame; what separates them is colour or
        // row, and the fragment shader does that. Returning the identity means
        // one geometry path serves every packing.
        StereoLayout::Anaglyph | StereoLayout::RowInterleaved => [1.0, 0.0, 1.0, 0.0],
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

/// The vertical field of view to render at, in degrees, for these glasses and
/// this panel.
///
/// The shape comes from the panel because the panel knows it; the angle comes
/// from a table because nothing reports it. Glasses this build has never heard
/// of get the Pro 2's angle with their own shape — wrong by a little scale
/// rather than by a stretch.
#[no_mangle]
pub extern "C" fn xr_fov_for_panel(product_id: u32, eye_width: u32, eye_height: u32) -> f32 {
    fov_for_panel(product_id as u16, eye_width, eye_height)
}

/// Writes what is known about the glasses with this product id, for logging:
/// `[diagonal_fov_deg, eye_width, eye_height, verified]`. Returns 0 when they
/// are recognised and -1 when they are not, in which case nothing is written.
///
/// # Safety
/// `out` must point to four writable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_model_info(product_id: u32, out: *mut f32) -> i32 {
    if out.is_null() {
        return -1;
    }
    let Some(model) = model_for(product_id as u16) else {
        return -1;
    };
    let slice = std::slice::from_raw_parts_mut(out, 4);
    slice.copy_from_slice(&[
        model.diagonal_fov_deg,
        model.eye_width as f32,
        model.eye_height as f32,
        f32::from(u8::from(model.verified)),
    ]);
    0
}

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

/// Reads a `[top, bottom, left, right]` bounds pointer, treating null as the
/// whole sphere.
///
/// # Safety
/// `p` must be null or point to four readable, aligned `f32`s.
unsafe fn bounds_from(p: *const f32) -> Bounds {
    if p.is_null() {
        return Bounds::FULL;
    }
    let b = std::slice::from_raw_parts(p, 4);
    Bounds {
        top: b[0],
        bottom: b[1],
        left: b[2],
        right: b[3],
    }
}

/// Writes the `[top, bottom, left, right]` bounds a projection is shorthand
/// for. Returns 0, or -1 for a projection that is not a patch of a sphere.
///
/// The renderer converts once, here, and passes bounds everywhere after that —
/// so a file that states its own coverage in the `equi` box travels the same
/// path as one that only says "360" or "180".
///
/// # Safety
/// `out` must point to four writable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_pano_bounds(projection: u32, out: *mut f32) -> i32 {
    if out.is_null() {
        return -1;
    }
    match Projection::from_raw(projection).bounds() {
        Some(b) => {
            let slice = std::slice::from_raw_parts_mut(out, 4);
            slice.copy_from_slice(&[b.top, b.bottom, b.left, b.right]);
            0
        }
        None => -1,
    }
}

/// Fills `out` with interleaved `[x, y, z, u, v]` vertices. Returns the vertex
/// count, or -1 if the tessellation, the bounds or the capacity is unusable.
///
/// `bounds` is `[top, bottom, left, right]`, the proportion of the sphere the
/// video does *not* cover on each side, or null for all of it.
///
/// # Safety
/// `out` must point to `cap_floats` writable, aligned `f32`s, and `bounds` must
/// be null or point to four readable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_pano_mesh(
    rings: u32,
    sectors: u32,
    radius: f32,
    bounds: *const f32,
    out: *mut f32,
    cap_floats: usize,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let slice = std::slice::from_raw_parts_mut(out, cap_floats);
    sphere_mesh_bounded(rings, sectors, radius, bounds_from(bounds), slice).map_or(-1, |n| n as i32)
}

/// Fills `out` with triangle indices. Returns the index count, or -1.
///
/// The count depends on the bounds — a cropped sphere needs the polar rows a
/// full one drops — so draw with what this returns, not with
/// [`xr_pano_index_count`], which is only large enough for either.
///
/// # Safety
/// `out` must point to `cap` writable, aligned `u16`s, and `bounds` must be
/// null or point to four readable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_pano_indices(
    rings: u32,
    sectors: u32,
    bounds: *const f32,
    out: *mut u16,
    cap: usize,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let slice = std::slice::from_raw_parts_mut(out, cap);
    sphere_indices_bounded(rings, sectors, bounds_from(bounds), slice).map_or(-1, |n| n as i32)
}

/// Writes the three channel weights one eye's picture survives in, for an
/// anaglyph frame. `pair` is 0 red/cyan, 1 green/magenta, 2 yellow/blue, each
/// named left eye first; `eye` is 0 for left. Returns 0, or -1.
///
/// # Safety
/// `out` must point to three writable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_anaglyph_mix(pair: u32, eye: i32, out: *mut f32) -> i32 {
    if out.is_null() {
        return -1;
    }
    let eye = if eye == 0 { Eye::Left } else { Eye::Right };
    let mix = anaglyph_mix(AnaglyphPair::from_raw(pair), eye);
    std::slice::from_raw_parts_mut(out, 3).copy_from_slice(&mix);
    0
}

/// Vertices [`xr_cube_mesh`] will write, for sizing the buffer.
#[no_mangle]
pub extern "C" fn xr_cube_vertex_count(cells: u32) -> u32 {
    cubemap_vertex_count(cells) as u32
}

/// Indices [`xr_cube_indices`] will write, for sizing the buffer.
#[no_mangle]
pub extern "C" fn xr_cube_index_count(cells: u32) -> u32 {
    cubemap_index_count(cells) as u32
}

/// Fills `out` with the six faces of a cube map, as interleaved
/// `[x, y, z, u, v]` vertices. Returns the vertex count, or -1.
///
/// `padding` is the `cbmp` box's pixel padding divided by the pixel size of one
/// face — the caller has the frame's dimensions and this does not.
///
/// # Safety
/// `out` must point to `cap_floats` writable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_cube_mesh(
    cells: u32,
    radius: f32,
    padding: f32,
    out: *mut f32,
    cap_floats: usize,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let slice = std::slice::from_raw_parts_mut(out, cap_floats);
    cubemap_mesh(cells, radius, padding, slice).map_or(-1, |n| n as i32)
}

/// Fills `out` with the cube map's triangle indices. Returns the count, or -1.
///
/// # Safety
/// `out` must point to `cap` writable, aligned `u16`s.
#[no_mangle]
pub unsafe extern "C" fn xr_cube_indices(cells: u32, out: *mut u16, cap: usize) -> i32 {
    if out.is_null() {
        return -1;
    }
    let slice = std::slice::from_raw_parts_mut(out, cap);
    cubemap_indices(cells, slice).map_or(-1, |n| n as i32)
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
        // A full sphere drops the half of each polar cell that has no area, so
        // it writes fewer indices than the buffer holds.
        let written = sphere_indices(RINGS, SECTORS, &mut idx).expect("indices");
        assert_eq!(written, index_count(RINGS, SECTORS) - 6 * SECTORS as usize);
        idx.truncate(written);
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

    // -- which glasses ------------------------------------------------------

    #[test]
    fn the_pro_2_is_the_one_that_was_actually_measured() {
        let pro2 = model_for(0x1301).expect("the Pro 2 should be known");
        assert_eq!(pro2.diagonal_fov_deg, PANEL_DIAGONAL_FOV_DEG);
        assert!(
            pro2.verified,
            "the Pro 2 is the model this was built against"
        );
        // And it is the only one, which is the honest state of affairs. If a
        // second becomes verified, that is a measurement somebody took and this
        // line should be updated deliberately rather than drift.
        assert_eq!(KNOWN_MODELS.iter().filter(|m| m.verified).count(), 1);
    }

    #[test]
    fn glasses_nobody_has_plugged_in_cannot_be_matched_by_accident() {
        // The unverified entries carry a product id of zero because nobody has
        // read theirs off a bus. Zero must never match, or an unrecognised pair
        // would silently be treated as whichever placeholder came first.
        assert_eq!(model_for(0), None);
        assert_eq!(model_for(0xFFFF), None);
        for model in KNOWN_MODELS.iter().filter(|m| !m.verified) {
            assert_eq!(
                model.product_id, 0,
                "{} claims an id nobody read",
                model.name
            );
        }
    }

    #[test]
    fn the_panel_decides_the_shape_and_the_table_decides_the_angle() {
        // The two halves of the answer come from different places on purpose.
        // A wrong angle scales the world; a wrong shape stretches it, and only
        // one of those is noticed. So the shape is taken from the panel even
        // for glasses the table has never heard of.
        let known_16_9 = fov_for_panel(0x1301, 1920, 1080);
        let unknown_16_9 = fov_for_panel(0xBEEF, 1920, 1080);
        assert_eq!(known_16_9, unknown_16_9, "same optics, same answer");

        // A 16:10 panel is a different vertical angle for the same diagonal,
        // whether or not the glasses are recognised.
        let unknown_16_10 = fov_for_panel(0xBEEF, 1920, 1200);
        assert!(
            unknown_16_10 > unknown_16_9,
            "a taller panel spans more vertical angle for the same diagonal",
        );

        // The Beast is both wider in angle and taller in shape, so it must
        // differ from the Pro 2 on both counts once its id is known.
        let beast = KNOWN_MODELS
            .iter()
            .find(|m| m.name.contains("Beast"))
            .unwrap();
        assert!(beast.diagonal_fov_deg > 50.0);
        assert_eq!(beast.eye_height, 1200);
    }

    #[test]
    fn a_panel_that_reports_nothing_falls_back_rather_than_dividing_by_zero() {
        let fallback = fov_for_panel(0x1301, 0, 0);
        assert!(fallback.is_finite() && fallback > 0.0);
        assert!((fallback - DEFAULT_FOV_DEG).abs() < 0.1, "{fallback}");
    }

    // -- bounded spheres ----------------------------------------------------

    /// The direction of a vertex, and its texture coordinate.
    fn vertex(verts: &[f32], i: usize) -> ([f32; 3], [f32; 2]) {
        let o = i * VERTEX_FLOATS;
        (
            [verts[o], verts[o + 1], verts[o + 2]],
            [verts[o + 3], verts[o + 4]],
        )
    }

    fn bounded(bounds: Bounds) -> Vec<f32> {
        let mut verts = vec![0.0f32; vertex_count(RINGS, SECTORS) * VERTEX_FLOATS];
        sphere_mesh_bounded(RINGS, SECTORS, 1.0, bounds, &mut verts).expect("bounded sphere");
        verts
    }

    #[test]
    fn the_named_projections_are_the_bounds_they_stand_for() {
        // If these two ever disagree, a file that states its coverage and an
        // identical one that only says "180" would render differently.
        for projection in [Projection::Equirect360, Projection::Equirect180] {
            let mut named = vec![0.0f32; vertex_count(RINGS, SECTORS) * VERTEX_FLOATS];
            sphere_mesh(RINGS, SECTORS, 1.0, projection, &mut named).unwrap();
            let stated = bounded(projection.bounds().unwrap());
            assert_eq!(named, stated, "{projection:?} via bounds");
        }
        assert_eq!(Projection::Flat.bounds(), None);
        assert_eq!(Projection::Cubemap.bounds(), None);
    }

    #[test]
    fn cropped_bounds_narrow_the_sphere_without_moving_its_middle() {
        // A symmetric crop covers less world but is still centred straight
        // ahead — the middle of the picture is the middle of the picture.
        let bounds = Bounds {
            top: 0.1,
            bottom: 0.1,
            left: 0.125,
            right: 0.125,
        };
        let verts = bounded(bounds);
        let stride = (SECTORS + 1) as usize;
        let middle = (RINGS as usize / 2) * stride + SECTORS as usize / 2;
        let (dir, uv) = vertex(&verts, middle);
        assert!((uv[0] - 0.5).abs() < 1e-6 && (uv[1] - 0.5).abs() < 1e-6);
        assert!(
            dir[0].abs() < 1e-5 && dir[1].abs() < 1e-5,
            "not centred: {dir:?}"
        );
        assert!(dir[2] < -0.99, "not straight ahead: {dir:?}");

        // Three quarters of the longitude, four fifths of the latitude.
        assert!((bounds.arc() - 0.75 * std::f32::consts::TAU).abs() < 1e-5);
        assert!((bounds.pitch_span() - 0.8 * std::f32::consts::PI).abs() < 1e-5);
    }

    #[test]
    fn an_off_centre_crop_moves_the_picture_where_the_file_says() {
        // Cropping only one side means the covered patch is not centred, and
        // pretending otherwise would swing the whole world sideways.
        let verts = bounded(Bounds {
            top: 0.0,
            bottom: 0.0,
            left: 0.25,
            right: 0.0,
        });
        let stride = (SECTORS + 1) as usize;
        let middle = (RINGS as usize / 2) * stride + SECTORS as usize / 2;
        let (dir, _) = vertex(&verts, middle);
        // Left-cropped by a quarter turn: the middle of what remains sits 45°
        // to the right of straight ahead.
        let heading = dir[0].atan2(-dir[2]).to_degrees();
        assert!((heading - 45.0).abs() < 0.5, "heading {heading}");
    }

    #[test]
    fn a_cropped_sphere_keeps_the_triangles_a_full_one_drops() {
        // The polar rows of a full sphere collapse to a point and half of each
        // cell is dropped. Crop the poles away and those rows are ordinary
        // quads; dropping them would leave a band of nothing top and bottom.
        let mut idx = vec![0u16; index_count(RINGS, SECTORS)];
        let full = sphere_indices_bounded(RINGS, SECTORS, Bounds::FULL, &mut idx).unwrap();
        let cropped = sphere_indices_bounded(
            RINGS,
            SECTORS,
            Bounds {
                top: 0.05,
                bottom: 0.05,
                left: 0.0,
                right: 0.0,
            },
            &mut idx,
        )
        .unwrap();
        assert_eq!(cropped, index_count(RINGS, SECTORS));
        assert_eq!(full, cropped - 6 * SECTORS as usize);

        // One pole cropped, one not.
        let half = sphere_indices_bounded(
            RINGS,
            SECTORS,
            Bounds {
                top: 0.05,
                bottom: 0.0,
                left: 0.0,
                right: 0.0,
            },
            &mut idx,
        )
        .unwrap();
        assert_eq!(half, cropped - 3 * SECTORS as usize);
    }

    #[test]
    fn bounds_that_leave_nothing_are_refused() {
        let mut verts = vec![0.0f32; vertex_count(RINGS, SECTORS) * VERTEX_FLOATS];
        for bad in [
            Bounds {
                top: 0.5,
                bottom: 0.5,
                left: 0.0,
                right: 0.0,
            },
            Bounds {
                top: 0.0,
                bottom: 0.0,
                left: 0.9,
                right: 0.2,
            },
            Bounds {
                top: -0.1,
                bottom: 0.0,
                left: 0.0,
                right: 0.0,
            },
            Bounds {
                top: f32::NAN,
                bottom: 0.0,
                left: 0.0,
                right: 0.0,
            },
        ] {
            assert!(!bad.is_usable(), "{bad:?} should be refused");
            assert_eq!(
                sphere_mesh_bounded(RINGS, SECTORS, 1.0, bad, &mut verts),
                None
            );
        }
        assert!(Bounds::FULL.is_usable());
    }

    // -- cube maps -----------------------------------------------------------

    /// One vertex of one face, by its place in the grid `cubemap_mesh` writes:
    /// `x` runs along image-right and `y` along image-up, both from 0 to
    /// `cells`.
    fn cube_vertex(verts: &[f32], cells: u32, face: usize, x: u32, y: u32) -> ([f32; 3], [f32; 2]) {
        let stride = (cells + 1) as usize;
        vertex(
            verts,
            face * stride * stride + y as usize * stride + x as usize,
        )
    }

    fn cube(cells: u32, padding: f32) -> Vec<f32> {
        let mut verts = vec![0.0f32; cubemap_vertex_count(cells) * VERTEX_FLOATS];
        assert_eq!(
            cubemap_mesh(cells, 1.0, padding, &mut verts),
            Some(cubemap_vertex_count(cells))
        );
        verts
    }

    #[test]
    fn the_cube_faces_land_where_the_spec_packs_them() {
        // The 3x2 grid of layout 0, read as the spec words it: Right, Left, Up
        // across the top row of the image and Down, Front, Back across the
        // bottom. Texture space is bottom-up, so the image's top row is the
        // upper half of v — the flip that has caught every other surface here.
        let verts = cube(2, 0.0);
        for (face, name, dir, column, row) in [
            (0, "right", [1.0, 0.0, 0.0], 0.0, 0.0),
            (1, "left", [-1.0, 0.0, 0.0], 1.0, 0.0),
            (2, "up", [0.0, 1.0, 0.0], 2.0, 0.0),
            (3, "down", [0.0, -1.0, 0.0], 0.0, 1.0),
            (4, "front", [0.0, 0.0, -1.0], 1.0, 1.0),
            (5, "back", [0.0, 0.0, 1.0], 2.0, 1.0),
        ] {
            // The middle of a face points along its own axis and nothing else.
            let (centre, uv) = cube_vertex(&verts, 2, face, 1, 1);
            assert_eq!(centre, dir, "{name} centre");
            let want = [(column + 0.5) / 3.0, (2.0 - row - 0.5) / 2.0];
            assert!(
                (uv[0] - want[0]).abs() < 1e-6 && (uv[1] - want[1]).abs() < 1e-6,
                "{name}: sampled {uv:?}, expected {want:?}",
            );
        }
    }

    #[test]
    fn the_poles_are_oriented_the_way_a_tilted_head_expects() {
        // Tilt your head back to look at the ceiling and your up direction
        // swings from up towards forward, so the top of the ceiling image
        // points forward; look down and it swings backwards. The spec words it
        // as "top of face forward" and "top of face backward". Get it wrong and
        // the sky is rotated half a turn against the walls, which reads as the
        // world tearing overhead — visible only while wearing the thing.
        let verts = cube(2, 0.0);
        let up_face = 2;
        let down_face = 3;

        // Top edge of the ceiling image, halfway along: forward.
        let (top_of_ceiling, uv) = cube_vertex(&verts, 2, up_face, 1, 2);
        assert_eq!(
            top_of_ceiling,
            [0.0, 1.0, -1.0],
            "ceiling top is not forward"
        );
        assert!(
            (uv[1] - 1.0).abs() < 1e-6,
            "not the top of the image: {uv:?}"
        );

        // Top edge of the floor image: backward.
        let (top_of_floor, uv) = cube_vertex(&verts, 2, down_face, 1, 2);
        assert_eq!(top_of_floor, [0.0, -1.0, 1.0], "floor top is not backward");
        assert!(
            (uv[1] - 0.5).abs() < 1e-6,
            "not the top of that cell: {uv:?}"
        );

        // And the ceiling's forward edge is the same place in the world as the
        // front wall's top edge, so the two pictures meet rather than overlap.
        let (front_top, _) = cube_vertex(&verts, 2, 4, 1, 2);
        assert_eq!(
            front_top, top_of_ceiling,
            "ceiling and front wall do not meet"
        );
    }

    #[test]
    fn every_cube_edge_is_shared_by_exactly_two_faces() {
        // A cube map is only a closed world if the faces agree on where they
        // meet. Each of the twelve edges is built twice, once by each face
        // that owns it, and a sign error in the table shows up as an edge no
        // second face reaches.
        let cells = 2;
        let verts = cube(cells, 0.0);
        let key = |p: [f32; 3]| p.map(|v| (v * 1e4).round() as i32);
        let mut corners = std::collections::BTreeMap::new();
        for face in 0..6 {
            for (x, y) in [(0, 0), (cells, 0), (0, cells), (cells, cells)] {
                let (p, _) = cube_vertex(&verts, cells, face, x, y);
                *corners.entry(key(p)).or_insert(0) += 1;
            }
        }
        assert_eq!(corners.len(), 8, "a cube has eight corners");
        for (corner, faces) in corners {
            assert_eq!(faces, 3, "corner {corner:?} is built by {faces} faces");
        }
    }

    #[test]
    fn every_cube_triangle_faces_the_viewer_at_the_centre() {
        // The sphere has this test and the cube did not, which is why the cube
        // shipped inside out: every face wound away from the middle, back-face
        // culling discarded all six, and the panel went black with the geometry
        // uploaded and nothing to say about it.
        //
        // The check is the sign of the triangle's normal against the direction
        // to it. Positive means it faces the centre, which is where the viewer
        // is standing.
        let cells = 2;
        let verts = cube(cells, 0.0);
        let mut idx = vec![0u16; cubemap_index_count(cells)];
        cubemap_indices(cells, &mut idx).unwrap();

        for (n, triangle) in idx.as_chunks::<3>().0.iter().enumerate() {
            let p = triangle.map(|i| {
                let o = i as usize * VERTEX_FLOATS;
                [verts[o], verts[o + 1], verts[o + 2]]
            });
            let edge1 = [p[1][0] - p[0][0], p[1][1] - p[0][1], p[1][2] - p[0][2]];
            let edge2 = [p[2][0] - p[0][0], p[2][1] - p[0][1], p[2][2] - p[0][2]];
            let normal = [
                edge1[1] * edge2[2] - edge1[2] * edge2[1],
                edge1[2] * edge2[0] - edge1[0] * edge2[2],
                edge1[0] * edge2[1] - edge1[1] * edge2[0],
            ];
            // Any point of the triangle serves as the direction from the centre.
            let facing = normal[0] * p[0][0] + normal[1] * p[0][1] + normal[2] * p[0][2];
            assert!(
                facing < 0.0,
                "triangle {n} faces away from the centre ({facing}); the cube is inside out",
            );
        }
    }

    #[test]
    fn cube_padding_shrinks_every_face_towards_its_middle() {
        // Padding exists so that filtering at a face edge cannot reach across
        // into a face pointing somewhere else. Trimming it moves the texture
        // coordinates in and leaves the geometry alone.
        let plain = cube(1, 0.0);
        let padded = cube(1, 0.1);

        for i in 0..cubemap_vertex_count(1) {
            let (plain_dir, plain_uv) = vertex(&plain, i);
            let (padded_dir, padded_uv) = vertex(&padded, i);
            assert_eq!(plain_dir, padded_dir, "geometry moved");
            // Every vertex of a single-cell face is a corner, so each moves
            // inwards by a tenth of a cell in both directions.
            assert!((plain_uv[0] - padded_uv[0]).abs() > 1e-4);
            assert!((plain_uv[1] - padded_uv[1]).abs() > 1e-4);
        }

        let mut scratch = plain.clone();
        assert_eq!(
            cubemap_mesh(1, 1.0, 0.5, &mut scratch),
            None,
            "half is not a face"
        );
        assert_eq!(cubemap_mesh(0, 1.0, 0.0, &mut scratch), None, "no cells");
    }

    #[test]
    fn cube_indices_cover_every_face_and_stay_in_range() {
        for cells in [1u32, 4, 16] {
            let mut idx = vec![0u16; cubemap_index_count(cells)];
            assert_eq!(cubemap_indices(cells, &mut idx), Some(idx.len()));
            let limit = cubemap_vertex_count(cells) as u16;
            assert!(idx.iter().all(|&i| i < limit), "index out of range");
            let per_face = (cells as u16 + 1) * (cells as u16 + 1);
            for face in 0..6u16 {
                assert!(
                    idx.iter().any(|&i| i / per_face == face),
                    "face {face} has no triangles",
                );
            }
        }
    }

    // -- packings that are not windows ---------------------------------------

    #[test]
    fn the_packings_that_need_a_shader_say_so() {
        for layout in [
            StereoLayout::Mono,
            StereoLayout::OverUnder,
            StereoLayout::SideBySide,
        ] {
            assert!(layout.is_windowed(), "{layout:?}");
        }
        for layout in [StereoLayout::Anaglyph, StereoLayout::RowInterleaved] {
            assert!(!layout.is_windowed(), "{layout:?}");
            // Both eyes see the whole frame; the difference is per pixel.
            for eye in [Eye::Left, Eye::Right] {
                assert_eq!(uv_window(layout, eye), [1.0, 0.0, 1.0, 0.0]);
            }
        }
        assert_eq!(StereoLayout::from_raw(3), StereoLayout::Anaglyph);
        assert_eq!(StereoLayout::from_raw(4), StereoLayout::RowInterleaved);
    }

    #[test]
    fn each_anaglyph_filter_takes_the_channels_the_other_leaves() {
        // The two eyes must not share a channel — that is the whole mechanism —
        // and each eye's weights must sum to one, or one eye comes out darker
        // than the other and the pair is unfusable.
        for pair in [
            AnaglyphPair::RedCyan,
            AnaglyphPair::GreenMagenta,
            AnaglyphPair::YellowBlue,
        ] {
            let left = anaglyph_mix(pair, Eye::Left);
            let right = anaglyph_mix(pair, Eye::Right);
            assert!(
                (left.iter().sum::<f32>() - 1.0).abs() < 1e-6,
                "{pair:?} left"
            );
            assert!(
                (right.iter().sum::<f32>() - 1.0).abs() < 1e-6,
                "{pair:?} right"
            );
            for channel in 0..3 {
                assert!(
                    left[channel] == 0.0 || right[channel] == 0.0,
                    "{pair:?} shares channel {channel}",
                );
            }
            // And between them they use all of it, or part of the picture is
            // being thrown away.
            for channel in 0..3 {
                assert!(
                    left[channel] + right[channel] > 0.0,
                    "{pair:?} drops {channel}"
                );
            }
            // The name is a claim about which eye is where, and it is the claim
            // that was wrong first time round: the left eye lives in the
            // channel the pair is named after.
            let named_channel = match pair {
                AnaglyphPair::RedCyan => 0,
                AnaglyphPair::GreenMagenta => 1,
                AnaglyphPair::YellowBlue => 0,
            };
            assert!(
                left[named_channel] > 0.0,
                "{pair:?}: the left eye is not in the channel the name gives it",
            );
        }
    }
}
