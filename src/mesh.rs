//! Mesh projections: the geometry a spherical video carries inside itself.
//!
//! Most 360° footage is equirectangular, and a sphere renders it. Fisheye rigs
//! — the VR180 cameras, and anything that stitches in the camera rather than in
//! post — instead ship the mapping as an actual triangle mesh in the file, in an
//! `sv3d`/`proj`/`mshp` box. Its vertices carry both a direction and the texture
//! coordinate that belongs there, so replaying it is exact: no assumption about
//! how the lens distorts, because the file already says.
//!
//! Rendered on a sphere instead, such a file is visibly wrong — straight lines
//! bow, and the edges stretch — which is the whole reason for this module.
//!
//! # Format
//!
//! Defined by Google's Spherical Video V2 metadata. A `mesh` box holds a pool of
//! `f32` coordinates, then a vertex list that indexes into it five values at a
//! time (x, y, z, u, v), then one or more sub-meshes that index into *those*.
//! Both index lists are bit-packed to the width the counts need, zigzagged, and
//! stored as deltas — so a mesh whose vertices are enumerated in a sensible
//! order costs a few bits each.
//!
//! The whole payload may be raw or deflated.
//!
//! The reference implementation is ExoPlayer's `ProjectionDecoder`, which is
//! package-private and therefore unusable from outside its own view. This is a
//! reimplementation against the same format, kept deliberately close to it so
//! the two can be compared.

use crate::pano::{Bounds, VERTEX_FLOATS};

/// Triangle layout of a sub-mesh, matching the values in the file.
pub mod draw_mode {
    pub const TRIANGLES: u8 = 0;
    pub const TRIANGLE_STRIP: u8 = 1;
    pub const TRIANGLE_FAN: u8 = 2;
}

/// One drawable run: interleaved `[x, y, z, u, v]` vertices, ready to upload.
///
/// The file stores indices, but they are expanded here for the same reason the
/// reference does: sub-meshes index a shared vertex pool with per-sub-mesh
/// deltas, so the indices are not reusable as a GL element buffer without
/// rebuilding them anyway, and a fisheye mesh is a few thousand vertices at
/// most.
pub struct SubMesh {
    pub texture_id: u8,
    pub mode: u8,
    /// `vertices.len() / VERTEX_FLOATS` triangles' worth of vertices.
    pub vertices: Vec<f32>,
}

impl SubMesh {
    pub fn vertex_count(&self) -> usize {
        self.vertices.len() / VERTEX_FLOATS
    }
}

/// Everything one eye draws.
pub struct Mesh {
    pub submeshes: Vec<SubMesh>,
}

/// The meshes a `proj` box contains: one for monoscopic or a shared view, two
/// when the file gives each eye its own geometry.
pub struct Projection {
    pub meshes: Vec<Mesh>,
}

impl Projection {
    /// Whether both eyes share one mesh, in which case the caller still has to
    /// split the frame by the stereo mode. With two meshes the texture
    /// coordinates already point each eye at its own half.
    pub fn single_mesh(&self) -> bool {
        self.meshes.len() == 1
    }
}

/// What a `proj` box turned out to say.
///
/// The three are alternatives in the format and alternatives here. A file
/// carries exactly one of them, and which one decides what geometry is built —
/// so this is the one place the question is answered, rather than three
/// half-answers spread across the renderer.
pub enum Spherical {
    /// An equirectangular image over the stated patch of the sphere.
    ///
    /// The bounds come from the `equi` box, which every conforming spherical
    /// file has even when it covers everything — in which case they are all
    /// zero and this is a plain 360 video.
    Equirect(Bounds),
    /// Six cube faces packed into one frame.
    Cubemap {
        /// 0 is the only layout defined: a 3x2 grid.
        layout: u32,
        /// Pixels of padding around each face, which the renderer must trim or
        /// the filtering will drag one face's texels onto the next.
        padding: u32,
    },
    /// Geometry the file carries itself, for lenses no formula describes.
    Mesh(Projection),
}

// Box types, as big-endian FourCCs.
const TYPE_PROJ: u32 = u32::from_be_bytes(*b"proj");
const TYPE_EQUI: u32 = u32::from_be_bytes(*b"equi");
const TYPE_CBMP: u32 = u32::from_be_bytes(*b"cbmp");
const TYPE_MSHP: u32 = u32::from_be_bytes(*b"mshp");
const TYPE_YTMP: u32 = u32::from_be_bytes(*b"ytmp");
const TYPE_MESH: u32 = u32::from_be_bytes(*b"mesh");
const TYPE_RAW: u32 = u32::from_be_bytes(*b"raw ");
const TYPE_DFL8: u32 = u32::from_be_bytes(*b"dfl8");

/// Bounds on a file's own numbers, so a corrupt or hostile one cannot ask for
/// an arbitrary allocation. The same limits the reference decoder uses.
const MAX_COORDINATES: u32 = 10_000;
const MAX_VERTICES: u32 = 32_000;
const MAX_TRIANGLE_INDICES: u32 = 128_000;

/// Padding, in pixels, beyond which a `cbmp` box is not describing a frame any
/// display will ever carry — a face of an 8K cube map is under 3000 pixels
/// across, so this is already generous.
const MAX_CUBE_PADDING: u32 = 4096;

/// Ceiling on the inflated size of a deflated mesh, so a small box cannot
/// expand into an arbitrarily large allocation.
const MAX_INFLATED_BYTES: usize = 16 << 20;

/// Parses the contents of a `proj` box — what ExoPlayer hands over as
/// `Format.projectionData`, header included.
///
/// Returns `None` for anything malformed rather than a partial answer: half a
/// projection renders as a torn hole in the world, which is worse than falling
/// back to a sphere.
pub fn parse_spherical(data: &[u8]) -> Option<Spherical> {
    let mut r = Reader::new(data);
    // The box's own size field is not needed; the slice bounds it.
    r.skip(4)?;
    if r.u32()? != TYPE_PROJ {
        return None;
    }

    while r.remaining() >= 8 {
        let start = r.position();
        let size = r.u32()? as usize;
        let child_type = r.u32()?;
        let end = start.checked_add(size)?;
        if size < 8 || end > data.len() {
            return None;
        }
        let body = &data[r.position()..end];
        match child_type {
            TYPE_MSHP | TYPE_YTMP => {
                let meshes = parse_mshp(body)?;
                if meshes.is_empty() {
                    return None;
                }
                return Some(Spherical::Mesh(Projection { meshes }));
            }
            TYPE_EQUI => return parse_equi(body).map(Spherical::Equirect),
            TYPE_CBMP => return parse_cbmp(body),
            // `prhd`, and anything a later revision adds.
            _ => {}
        }
        r.seek(end)?;
    }
    None
}

/// The mesh, if that is what the box holds. Kept for the callers that can only
/// use geometry and fall back to a sphere for everything else.
pub fn parse_projection(data: &[u8]) -> Option<Projection> {
    match parse_spherical(data)? {
        Spherical::Mesh(p) => Some(p),
        _ => None,
    }
}

/// The `equi` payload: a full-box header, then how much of the sphere is
/// missing from each edge.
///
/// The four values are 0.32 fixed point — the whole 32 bits are fraction, so
/// the range is `[0, 1)` and the unit is the whole sphere. All zero is an
/// ordinary 360 video, and that is what a conforming file writes when it covers
/// everything; the values only become interesting for footage that does not,
/// which until now was stretched over whichever preset was nearest.
fn parse_equi(data: &[u8]) -> Option<Bounds> {
    let mut r = Reader::new(data);
    if r.u8()? != 0 {
        return None;
    }
    r.skip(3)?; // flags
    let scale = |v: u32| v as f32 / 4_294_967_296.0;
    let bounds = Bounds {
        top: scale(r.u32()?),
        bottom: scale(r.u32()?),
        left: scale(r.u32()?),
        right: scale(r.u32()?),
    };
    // A file claiming to have cropped away a whole dimension has nothing to
    // show, and would divide the mesh by zero.
    bounds.is_usable().then_some(bounds)
}

/// The `cbmp` payload: a full-box header, the grid layout, and the padding.
fn parse_cbmp(data: &[u8]) -> Option<Spherical> {
    let mut r = Reader::new(data);
    if r.u8()? != 0 {
        return None;
    }
    r.skip(3)?; // flags
    let layout = r.u32()?;
    let padding = r.u32()?;
    // Layout 0 is the only one defined, and a cube face cannot be more than
    // half padding.
    if layout != 0 || padding > MAX_CUBE_PADDING {
        return None;
    }
    Some(Spherical::Cubemap { layout, padding })
}

/// The `mshp` payload: a header, then mesh boxes, optionally deflated.
fn parse_mshp(data: &[u8]) -> Option<Vec<Mesh>> {
    let mut r = Reader::new(data);
    if r.u8()? != 0 {
        // Version 0 is the only one defined; a later one may mean anything.
        return None;
    }
    r.skip(7)?; // flags and crc
    let encoding = r.u32()?;
    let rest = &data[r.position()..];

    match encoding {
        TYPE_RAW => parse_meshes(rest),
        TYPE_DFL8 => {
            // Raw deflate, no zlib wrapper — `new Inflater(true)` in the
            // reference.
            let inflated =
                miniz_oxide::inflate::decompress_to_vec_with_limit(rest, MAX_INFLATED_BYTES)
                    .ok()?;
            parse_meshes(&inflated)
        }
        _ => None,
    }
}

/// A run of boxes, of which the `mesh` ones are collected in order.
fn parse_meshes(data: &[u8]) -> Option<Vec<Mesh>> {
    let mut meshes = Vec::new();
    let mut position = 0usize;
    while position + 8 <= data.len() {
        let size = u32::from_be_bytes(data.get(position..position + 4)?.try_into().ok()?) as usize;
        let child_type = u32::from_be_bytes(data.get(position + 4..position + 8)?.try_into().ok()?);
        let end = position.checked_add(size)?;
        if size < 8 || end > data.len() {
            return None;
        }
        if child_type == TYPE_MESH {
            meshes.push(parse_mesh(&data[position + 8..end])?);
        }
        position = end;
    }
    Some(meshes)
}

fn parse_mesh(data: &[u8]) -> Option<Mesh> {
    let mut r = Reader::new(data);

    let coordinate_count = r.u32()?;
    if coordinate_count == 0 || coordinate_count > MAX_COORDINATES {
        return None;
    }
    let mut coordinates = Vec::with_capacity(coordinate_count as usize);
    for _ in 0..coordinate_count {
        coordinates.push(f32::from_bits(r.u32()?));
    }

    let vertex_count = r.u32()?;
    if vertex_count == 0 || vertex_count > MAX_VERTICES {
        return None;
    }

    // Index widths follow the pool sizes: enough bits for the count, doubled to
    // leave room for the sign that zigzag folds in.
    let coordinate_bits = bit_width(coordinate_count);
    let mut bits = BitReader::new(data, r.position());

    // Five interleaved fields per vertex, each delta-coded against the previous
    // vertex's value for the same field.
    let mut vertices = vec![0.0f32; vertex_count as usize * VERTEX_FLOATS];
    let mut running = [0i64; VERTEX_FLOATS];
    let mut w = 0usize;
    for _ in 0..vertex_count {
        for previous in running.iter_mut() {
            let index = *previous + zigzag(bits.read(coordinate_bits)?) as i64;
            if index < 0 || index >= coordinate_count as i64 {
                return None;
            }
            vertices[w] = coordinates[index as usize];
            w += 1;
            *previous = index;
        }
    }

    bits.align_to_byte();

    let submesh_count = bits.read(32)?;
    if submesh_count == 0 {
        return None;
    }
    let vertex_bits = bit_width(vertex_count);
    let mut submeshes = Vec::with_capacity(submesh_count as usize);
    for _ in 0..submesh_count {
        let texture_id = bits.read(8)? as u8;
        let mode = bits.read(8)? as u8;
        let index_count = bits.read(32)?;
        if index_count == 0 || index_count > MAX_TRIANGLE_INDICES {
            return None;
        }

        let mut out = vec![0.0f32; index_count as usize * VERTEX_FLOATS];
        let mut index = 0i64;
        for triangle_vertex in 0..index_count as usize {
            index += zigzag(bits.read(vertex_bits)?) as i64;
            if index < 0 || index >= vertex_count as i64 {
                return None;
            }
            let src = index as usize * VERTEX_FLOATS;
            let dst = triangle_vertex * VERTEX_FLOATS;
            out[dst..dst + VERTEX_FLOATS].copy_from_slice(&vertices[src..src + VERTEX_FLOATS]);
        }
        submeshes.push(SubMesh {
            texture_id,
            mode,
            vertices: out,
        });
    }
    Some(Mesh { submeshes })
}

/// Bits needed to index a pool of `count`, with room for a signed delta.
fn bit_width(count: u32) -> u32 {
    (2.0 * count as f64).log2().ceil() as u32
}

/// Undoes zigzag encoding, which maps signed values onto unsigned ones so that
/// small negatives stay small.
#[inline]
fn zigzag(n: u32) -> i32 {
    ((n >> 1) as i32) ^ -((n & 1) as i32)
}

/// A byte reader that returns `None` past the end rather than panicking, since
/// every length in this format comes from the file.
struct Reader<'a> {
    data: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn new(data: &'a [u8]) -> Reader<'a> {
        Reader { data, position: 0 }
    }

    fn position(&self) -> usize {
        self.position
    }

    fn remaining(&self) -> usize {
        self.data.len() - self.position
    }

    fn seek(&mut self, to: usize) -> Option<()> {
        if to > self.data.len() {
            return None;
        }
        self.position = to;
        Some(())
    }

    fn skip(&mut self, n: usize) -> Option<()> {
        self.seek(self.position.checked_add(n)?)
    }

    fn u8(&mut self) -> Option<u8> {
        let v = *self.data.get(self.position)?;
        self.position += 1;
        Some(v)
    }

    fn u32(&mut self) -> Option<u32> {
        let end = self.position.checked_add(4)?;
        let v = u32::from_be_bytes(self.data.get(self.position..end)?.try_into().ok()?);
        self.position = end;
        Some(v)
    }
}

/// Most-significant-bit-first bit reader, matching the format's packing.
struct BitReader<'a> {
    data: &'a [u8],
    bit: usize,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8], byte_offset: usize) -> BitReader<'a> {
        BitReader {
            data,
            bit: byte_offset * 8,
        }
    }

    /// Reads up to 32 bits. `None` once the data runs out.
    fn read(&mut self, count: u32) -> Option<u32> {
        debug_assert!(count <= 32);
        let mut value: u32 = 0;
        for _ in 0..count {
            let byte = *self.data.get(self.bit >> 3)?;
            let shift = 7 - (self.bit & 7);
            value = (value << 1) | ((byte >> shift) & 1) as u32;
            self.bit += 1;
        }
        Some(value)
    }

    fn align_to_byte(&mut self) {
        self.bit = (self.bit + 7) & !7;
    }
}

// ---------------------------------------------------------------------------
// C ABI
//
// Meshes are sized by the file, so they cross the boundary as an opaque handle:
// parse, ask how much there is, copy into buffers the caller owns, free. Four
// calls per video, none of them per frame.
// ---------------------------------------------------------------------------

/// What a `proj` box says, without building anything: 0 equirectangular,
/// 1 cube map, 2 a carried mesh, -1 for absent or malformed.
///
/// `out` receives four floats whose meaning follows the return value:
/// `[top, bottom, left, right]` bounds for equirectangular, `[layout, padding,
/// 0, 0]` for a cube map, zeroes for a mesh. The mesh itself still comes from
/// [`xr_mesh_parse`] — this is the cheap question the renderer asks first, once
/// per video, to decide which of the three it is about to draw.
///
/// # Safety
/// `data` must point to `len` readable bytes and `out` to four writable,
/// aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_proj_kind(data: *const u8, len: usize, out: *mut f32) -> i32 {
    if data.is_null() || len == 0 || out.is_null() {
        return -1;
    }
    let bytes = std::slice::from_raw_parts(data, len);
    let slice = std::slice::from_raw_parts_mut(out, 4);
    match parse_spherical(bytes) {
        Some(Spherical::Equirect(b)) => {
            slice.copy_from_slice(&[b.top, b.bottom, b.left, b.right]);
            0
        }
        Some(Spherical::Cubemap { layout, padding }) => {
            slice.copy_from_slice(&[layout as f32, padding as f32, 0.0, 0.0]);
            1
        }
        Some(Spherical::Mesh(_)) => {
            slice.fill(0.0);
            2
        }
        None => -1,
    }
}

/// Parses a `proj` box. Returns a handle to free with [`xr_mesh_free`], or null
/// if the box is absent, malformed or in a form this does not read.
///
/// # Safety
/// `data` must point to `len` readable bytes.
#[no_mangle]
pub unsafe extern "C" fn xr_mesh_parse(data: *const u8, len: usize) -> *mut Projection {
    if data.is_null() || len == 0 {
        return std::ptr::null_mut();
    }
    let bytes = std::slice::from_raw_parts(data, len);
    match parse_projection(bytes) {
        Some(p) => Box::into_raw(Box::new(p)),
        None => std::ptr::null_mut(),
    }
}

/// How many meshes the file carries: 1 when both eyes share one, 2 when each has
/// its own.
///
/// # Safety
/// `p` must be from [`xr_mesh_parse`], or null.
#[no_mangle]
pub unsafe extern "C" fn xr_mesh_count(p: *const Projection) -> i32 {
    p.as_ref().map(|p| p.meshes.len() as i32).unwrap_or(0)
}

/// How many drawable runs one mesh has.
///
/// # Safety
/// `p` must be from [`xr_mesh_parse`], or null.
#[no_mangle]
pub unsafe extern "C" fn xr_mesh_submesh_count(p: *const Projection, mesh: i32) -> i32 {
    let Some(p) = p.as_ref() else { return 0 };
    p.meshes
        .get(mesh.max(0) as usize)
        .map(|m| m.submeshes.len() as i32)
        .unwrap_or(0)
}

/// Writes `[draw_mode, vertex_count, texture_id]` for one sub-mesh. `draw_mode`
/// is 0 triangles, 1 strip, 2 fan.
///
/// # Safety
/// `p` must be from [`xr_mesh_parse`]; `out` must point to three writable `i32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_mesh_submesh_info(
    p: *const Projection,
    mesh: i32,
    submesh: i32,
    out: *mut i32,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let Some(sub) = p
        .as_ref()
        .and_then(|p| p.meshes.get(mesh.max(0) as usize))
        .and_then(|m| m.submeshes.get(submesh.max(0) as usize))
    else {
        return -1;
    };
    let info = [
        sub.mode as i32,
        sub.vertex_count() as i32,
        sub.texture_id as i32,
    ];
    std::ptr::copy_nonoverlapping(info.as_ptr(), out, info.len());
    0
}

/// Copies a sub-mesh as interleaved `[x, y, z, u, v]`, the same layout the
/// sphere uses, so one vertex-attribute setup serves both. Returns the vertex
/// count, or -1 if the buffer is too small.
///
/// # Safety
/// `p` must be from [`xr_mesh_parse`]; `out` must point to `cap_floats`
/// writable, aligned `f32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_mesh_copy(
    p: *const Projection,
    mesh: i32,
    submesh: i32,
    out: *mut f32,
    cap_floats: usize,
) -> i32 {
    if out.is_null() {
        return -1;
    }
    let Some(sub) = p
        .as_ref()
        .and_then(|p| p.meshes.get(mesh.max(0) as usize))
        .and_then(|m| m.submeshes.get(submesh.max(0) as usize))
    else {
        return -1;
    };
    if cap_floats < sub.vertices.len() {
        return -1;
    }
    std::ptr::copy_nonoverlapping(sub.vertices.as_ptr(), out, sub.vertices.len());
    sub.vertex_count() as i32
}

/// Releases a handle from [`xr_mesh_parse`].
///
/// # Safety
/// `p` must be from [`xr_mesh_parse`] and not already freed, or null.
#[no_mangle]
pub unsafe extern "C" fn xr_mesh_free(p: *mut Projection) {
    if !p.is_null() {
        drop(Box::from_raw(p));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a box: big-endian size, FourCC, payload.
    fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend_from_slice(kind);
        out.extend_from_slice(payload);
        out
    }

    /// Most-significant-bit-first writer, the inverse of [`BitReader`].
    #[derive(Default)]
    struct BitWriter {
        bytes: Vec<u8>,
        bit: usize,
    }

    impl BitWriter {
        fn write(&mut self, value: u32, count: u32) {
            for i in (0..count).rev() {
                if self.bit.is_multiple_of(8) {
                    self.bytes.push(0);
                }
                let set = (value >> i) & 1 == 1;
                if set {
                    let last = self.bytes.len() - 1;
                    self.bytes[last] |= 1 << (7 - (self.bit % 8));
                }
                self.bit += 1;
            }
        }

        fn align(&mut self) {
            while !self.bit.is_multiple_of(8) {
                self.write(0, 1);
            }
        }
    }

    fn zigzag_encode(v: i32) -> u32 {
        ((v << 1) ^ (v >> 31)) as u32
    }

    /// A mesh with the given coordinate pool, vertices as five indices each, and
    /// one sub-mesh listing `indices` into those vertices.
    fn build_mesh(
        coordinates: &[f32],
        vertices: &[[u32; 5]],
        mode: u8,
        indices: &[u32],
    ) -> Vec<u8> {
        let mut payload = (coordinates.len() as u32).to_be_bytes().to_vec();
        for c in coordinates {
            payload.extend_from_slice(&c.to_bits().to_be_bytes());
        }
        payload.extend_from_slice(&(vertices.len() as u32).to_be_bytes());

        let coordinate_bits = bit_width(coordinates.len() as u32);
        let mut bits = BitWriter::default();
        let mut previous = [0i32; 5];
        for vertex in vertices {
            for (field, &index) in vertex.iter().enumerate() {
                bits.write(
                    zigzag_encode(index as i32 - previous[field]),
                    coordinate_bits,
                );
                previous[field] = index as i32;
            }
        }
        bits.align();
        bits.write(1, 32); // one sub-mesh
        bits.write(7, 8); // texture id, deliberately not zero
        bits.write(mode as u32, 8);
        bits.write(indices.len() as u32, 32);
        let vertex_bits = bit_width(vertices.len() as u32);
        let mut running = 0i32;
        for &index in indices {
            bits.write(zigzag_encode(index as i32 - running), vertex_bits);
            running = index as i32;
        }
        payload.extend_from_slice(&bits.bytes);
        boxed(b"mesh", &payload)
    }

    fn build_proj(mesh_boxes: &[Vec<u8>], deflate: bool) -> Vec<u8> {
        let mut meshes = Vec::new();
        for m in mesh_boxes {
            meshes.extend_from_slice(m);
        }
        let mut mshp = vec![0u8]; // version
        mshp.extend_from_slice(&[0; 7]); // flags + crc
        if deflate {
            mshp.extend_from_slice(b"dfl8");
            mshp.extend_from_slice(&miniz_oxide::deflate::compress_to_vec(&meshes, 6));
        } else {
            mshp.extend_from_slice(b"raw ");
            mshp.extend_from_slice(&meshes);
        }
        boxed(b"proj", &boxed(b"mshp", &mshp))
    }

    /// Two triangles' worth of a quad, as the file would store it.
    fn sample() -> (Vec<f32>, Vec<[u32; 5]>, Vec<u32>) {
        // Coordinate pool: positions and texture coordinates share it.
        let coordinates = vec![-1.0f32, 1.0, 0.0, 0.5, -0.5];
        let vertices = vec![
            [0, 1, 2, 3, 4], // (-1,  1, 0) uv (0.5, -0.5)
            [1, 1, 2, 4, 3],
            [1, 0, 2, 3, 3],
        ];
        let indices = vec![0u32, 1, 2];
        (coordinates, vertices, indices)
    }

    #[test]
    fn parses_a_raw_mesh() {
        let (coordinates, vertices, indices) = sample();
        let data = build_proj(
            &[build_mesh(
                &coordinates,
                &vertices,
                draw_mode::TRIANGLES,
                &indices,
            )],
            false,
        );
        let projection = parse_projection(&data).expect("raw mesh should parse");
        assert_eq!(projection.meshes.len(), 1);
        assert!(projection.single_mesh());

        let sub = &projection.meshes[0].submeshes[0];
        assert_eq!(sub.texture_id, 7);
        assert_eq!(sub.mode, draw_mode::TRIANGLES);
        assert_eq!(sub.vertex_count(), 3);
        // First vertex resolves its five indices through the coordinate pool.
        assert_eq!(&sub.vertices[..5], &[-1.0, 1.0, 0.0, 0.5, -0.5]);
        // Third: indices [1, 0, 2, 3, 3].
        assert_eq!(&sub.vertices[10..15], &[1.0, -1.0, 0.0, 0.5, 0.5]);
    }

    #[test]
    fn parses_a_deflated_mesh() {
        // Real files are deflated; the raw path mostly exists for tests like the
        // one above.
        let (coordinates, vertices, indices) = sample();
        let data = build_proj(
            &[build_mesh(
                &coordinates,
                &vertices,
                draw_mode::TRIANGLE_STRIP,
                &indices,
            )],
            true,
        );
        let projection = parse_projection(&data).expect("deflated mesh should parse");
        let sub = &projection.meshes[0].submeshes[0];
        assert_eq!(sub.mode, draw_mode::TRIANGLE_STRIP);
        assert_eq!(sub.vertex_count(), 3);
    }

    #[test]
    fn two_meshes_mean_one_per_eye() {
        let (coordinates, vertices, indices) = sample();
        let one = build_mesh(&coordinates, &vertices, draw_mode::TRIANGLES, &indices);
        let data = build_proj(&[one.clone(), one], false);
        let projection = parse_projection(&data).unwrap();
        assert_eq!(projection.meshes.len(), 2);
        assert!(!projection.single_mesh(), "two meshes are not shared");
    }

    #[test]
    fn rejects_a_box_that_is_not_proj() {
        let data = boxed(b"equi", &[0; 16]);
        assert!(parse_projection(&data).is_none());
    }

    #[test]
    fn rejects_a_truncated_mesh() {
        let (coordinates, vertices, indices) = sample();
        let data = build_proj(
            &[build_mesh(
                &coordinates,
                &vertices,
                draw_mode::TRIANGLES,
                &indices,
            )],
            false,
        );
        // Every prefix must fail cleanly. Half a projection renders as a torn
        // hole in the world, so there is no partial success to fall back on.
        for cut in 1..data.len() {
            let _ = parse_projection(&data[..cut]);
        }
        assert!(parse_projection(&data[..data.len() - 4]).is_none());
    }

    #[test]
    fn rejects_absurd_counts() {
        // A coordinate count larger than any real mesh must be refused before it
        // becomes an allocation.
        let mut payload = (MAX_COORDINATES + 1).to_be_bytes().to_vec();
        payload.extend_from_slice(&[0; 16]);
        let data = build_proj(&[boxed(b"mesh", &payload)], false);
        assert!(parse_projection(&data).is_none());
    }

    #[test]
    fn zigzag_round_trips() {
        for v in [-5000i32, -2, -1, 0, 1, 2, 5000] {
            assert_eq!(zigzag(zigzag_encode(v)), v, "value {v}");
        }
    }

    #[test]
    fn bit_width_leaves_room_for_the_sign() {
        // The width has to hold a zigzagged delta, which is why the count is
        // doubled before the logarithm.
        assert_eq!(bit_width(1), 1);
        assert_eq!(bit_width(2), 2);
        assert_eq!(bit_width(3), 3);
        assert_eq!(bit_width(4), 3);
        assert_eq!(bit_width(5), 4);
    }

    #[test]
    fn the_bit_reader_is_most_significant_first() {
        let mut writer = BitWriter::default();
        writer.write(0b101, 3);
        writer.write(0b1100, 4);
        let mut reader = BitReader::new(&writer.bytes, 0);
        assert_eq!(reader.read(3), Some(0b101));
        assert_eq!(reader.read(4), Some(0b1100));
    }

    // -- the projections that are not meshes ---------------------------------

    /// A `proj` box wrapping the given children.
    fn proj(children: &[Vec<u8>]) -> Vec<u8> {
        boxed(b"proj", &children.concat())
    }

    /// A `prhd` box, which every conforming file carries and nothing here
    /// reads: pose yaw, pitch and roll as 16.16 fixed point.
    fn prhd() -> Vec<u8> {
        boxed(b"prhd", &[0u8; 4 + 12])
    }

    /// An `equi` box, from four fractions of the sphere.
    fn equi(top: f64, bottom: f64, left: f64, right: f64) -> Vec<u8> {
        let mut payload = vec![0u8; 4]; // version and flags
        for value in [top, bottom, left, right] {
            let fixed = (value * 4_294_967_296.0) as u32;
            payload.extend_from_slice(&fixed.to_be_bytes());
        }
        boxed(b"equi", &payload)
    }

    fn cbmp(layout: u32, padding: u32) -> Vec<u8> {
        let mut payload = vec![0u8; 4];
        payload.extend_from_slice(&layout.to_be_bytes());
        payload.extend_from_slice(&padding.to_be_bytes());
        boxed(b"cbmp", &payload)
    }

    #[test]
    fn an_uncropped_equi_box_is_a_plain_full_sphere() {
        // What a conforming 360 file actually writes: the box is present and
        // every bound is zero. Reading that as anything but a full sphere would
        // break the common case in the name of the rare one.
        let data = proj(&[prhd(), equi(0.0, 0.0, 0.0, 0.0)]);
        match parse_spherical(&data) {
            Some(Spherical::Equirect(b)) => assert_eq!(b, Bounds::FULL),
            _ => panic!("not read as a full sphere"),
        }
    }

    #[test]
    fn equi_bounds_come_back_as_the_fractions_they_encode() {
        // 0.32 fixed point: the whole word is fraction, so a quarter is a
        // quarter of the sphere. A file cropped a quarter from each side is a
        // hemisphere, which is what VR180 is — and it says so this way rather
        // than by being named.
        let data = proj(&[prhd(), equi(0.0, 0.0, 0.25, 0.25)]);
        let Some(Spherical::Equirect(bounds)) = parse_spherical(&data) else {
            panic!("not equirectangular");
        };
        assert!((bounds.left - 0.25).abs() < 1e-6);
        assert!((bounds.right - 0.25).abs() < 1e-6);
        assert_eq!(
            bounds,
            crate::pano::Projection::Equirect180.bounds().unwrap()
        );

        // And an asymmetric crop survives as itself.
        let data = proj(&[equi(0.125, 0.0625, 0.0, 0.5)]);
        let Some(Spherical::Equirect(bounds)) = parse_spherical(&data) else {
            panic!("not equirectangular");
        };
        assert!((bounds.top - 0.125).abs() < 1e-6);
        assert!((bounds.bottom - 0.0625).abs() < 1e-6);
        assert!((bounds.right - 0.5).abs() < 1e-6);
    }

    #[test]
    fn an_equi_box_that_crops_everything_away_is_refused() {
        // Half from the top and half from the bottom leaves a sphere of zero
        // height, which would divide the mesh by nothing.
        assert!(parse_spherical(&proj(&[equi(0.5, 0.5, 0.0, 0.0)])).is_none());
        assert!(parse_spherical(&proj(&[equi(0.0, 0.0, 0.75, 0.5)])).is_none());
    }

    #[test]
    fn a_cbmp_box_reads_its_layout_and_padding() {
        let Some(Spherical::Cubemap { layout, padding }) =
            parse_spherical(&proj(&[prhd(), cbmp(0, 8)]))
        else {
            panic!("not a cube map");
        };
        assert_eq!((layout, padding), (0, 8));

        // Layout 0 is the only one defined; guessing at another would draw the
        // faces in the wrong places, which is worse than falling back.
        assert!(parse_spherical(&proj(&[cbmp(1, 0)])).is_none());
        assert!(parse_spherical(&proj(&[cbmp(0, 100_000)])).is_none());
    }

    #[test]
    fn a_truncated_projection_box_yields_nothing() {
        // Files get cut off mid-download, and half a projection renders as a
        // hole in the world.
        let full = proj(&[prhd(), equi(0.0, 0.0, 0.25, 0.25)]);
        for cut in 1..full.len() {
            let _ = parse_spherical(&full[..cut]);
        }
        // A child claiming to be longer than the box that holds it.
        let mut lying = proj(&[equi(0.0, 0.0, 0.0, 0.0)]);
        let child = 8;
        lying[child..child + 4].copy_from_slice(&9999u32.to_be_bytes());
        assert!(parse_spherical(&lying).is_none());
    }

    #[test]
    fn the_mesh_path_still_only_answers_for_meshes() {
        // `parse_projection` is what the geometry callers use, and an
        // equirectangular file must not come back from it looking like a mesh
        // of no triangles.
        let data = proj(&[equi(0.0, 0.0, 0.0, 0.0)]);
        assert!(parse_projection(&data).is_none());
        assert!(parse_spherical(&data).is_some());
    }
}
