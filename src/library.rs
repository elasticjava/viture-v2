//! Working out how a library's files are packed, a listing at a time.
//!
//! A media centre answers a browse with hundreds or thousands of entries, and
//! each one has to be turned into a geometry and a packing before anything can
//! be drawn. The answer comes from two places: what the catalogue says, which
//! is authoritative when it says anything at all, and what the name and the
//! frame's shape suggest, which is a guess and is usually all there is.
//!
//! # Why this is here and not in the caller
//!
//! The work per item is small and there are a great many items, which is the
//! shape of problem that gets lost in per-item overhead. Three things were
//! costing more than the work:
//!
//! * **A case-folded copy of every name.** Thirty substring searches over a
//!   name want it lowercased, and doing that by allocating a second string
//!   means one allocation per item for a value discarded immediately. Here the
//!   fold happens a byte at a time, during the comparison, and nothing is
//!   allocated.
//!
//! * **Thirty substring searches.** Each marker is a needle in a haystack of
//!   twenty-odd bytes, and almost none of them are there. [`Signature`] rejects
//!   most of them with one `and`.
//!
//! * **One boundary crossing per item.** A whole listing goes across in one
//!   call, in buffers neither side copies — the same submission-and-completion
//!   arrangement the pose ring uses, for the same reason.
//!
//! What is left is a linear scan over one contiguous blob, which is what the
//! hardware is good at, split across cores when there is enough of it to be
//! worth the threads.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How a file maps onto geometry, and how confidently that is known.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Format {
    /// [`crate::pano::StereoLayout`] as a raw value.
    pub packing: u32,
    /// [`crate::pano::Projection`] as a raw value.
    pub projection: u32,
    /// Whether the eyes are the other way round from what the packing implies.
    pub swap_eyes: bool,
    /// [`crate::pano::AnaglyphPair`] as a raw value; only meaningful when the
    /// packing is anaglyph.
    pub anaglyph_pair: u32,
    /// Where the answer came from. Weakest to strongest, which is what decides
    /// who wins when two sources disagree.
    pub verdict: Verdict,
}

/// Where a reading of a file came from.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Default)]
#[repr(u32)]
pub enum Verdict {
    /// Inferred from the file name and the shape of the frame.
    #[default]
    Guessed = 0,
    /// Read from a catalogue that scanned the file.
    Catalogued = 1,
    /// Read from the file's own metadata.
    Declared = 2,
    /// Chosen by the person looking at the result, who can see it.
    Chosen = 3,
}

impl Format {
    /// Packs into one word, so a batch of answers is one buffer of them.
    ///
    /// Eleven bits used of thirty-two. The room is deliberate: a packing or a
    /// projection added later widens a field rather than the record, and a
    /// caller reading an older field out of a newer word still gets the right
    /// answer.
    pub fn pack(self) -> u32 {
        (self.packing & 0x7)
            | ((self.projection & 0x7) << 3)
            | ((self.swap_eyes as u32) << 6)
            | ((self.anaglyph_pair & 0x3) << 7)
            | ((self.verdict as u32 & 0x3) << 9)
    }

    /// The inverse of [`Format::pack`].
    pub fn unpack(word: u32) -> Format {
        Format {
            packing: word & 0x7,
            projection: (word >> 3) & 0x7,
            swap_eyes: (word >> 6) & 1 == 1,
            anaglyph_pair: (word >> 7) & 0x3,
            verdict: match (word >> 9) & 0x3 {
                1 => Verdict::Catalogued,
                2 => Verdict::Declared,
                3 => Verdict::Chosen,
                _ => Verdict::Guessed,
            },
        }
    }
}

// Packings, mirroring `pano::StereoLayout`. Repeated as plain numbers because
// this module is compiled without the `render` feature too, and the numbers are
// the shared contract either way.
const MONO: u32 = 0;
const OVER_UNDER: u32 = 1;
const SIDE_BY_SIDE: u32 = 2;
const ANAGLYPH: u32 = 3;
const ROW_INTERLEAVED: u32 = 4;

// Projections, mirroring `pano::Projection`.
const EQUIRECT_360: u32 = 0;
const EQUIRECT_180: u32 = 1;
const FLAT: u32 = 2;

/// A one-word summary of which characters a string contains.
///
/// A Bloom filter of a single hash over sixty-four buckets, and the properties
/// that come with one are exactly the properties wanted here: a marker whose
/// signature has a bit the name's does not **cannot** be in the name, so the
/// rejection is certain and the scan is skipped. The reverse is not certain,
/// and a false positive costs only the scan that would have happened anyway.
///
/// Cheap enough to be worth it because the markers' signatures are computed
/// once, at compile time, and a name's is one pass over bytes that are about to
/// be read anyway.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Signature(u64);

impl Signature {
    /// Which bit a byte claims. Letters, digits and the three punctuation marks
    /// that appear in the markers get their own; everything else shares the
    /// last, which costs nothing because no marker contains anything else.
    const fn bit(byte: u8) -> u64 {
        let lowered = fold(byte);
        let index = match lowered {
            b'a'..=b'z' => (lowered - b'a') as u64,
            b'0'..=b'9' => 26 + (lowered - b'0') as u64,
            b'_' => 36,
            b'-' => 37,
            b'.' => 38,
            _ => 39,
        };
        1u64 << index
    }

    pub const fn of(bytes: &[u8]) -> Signature {
        let mut mask = 0u64;
        let mut i = 0;
        while i < bytes.len() {
            mask |= Signature::bit(bytes[i]);
            i += 1;
        }
        Signature(mask)
    }

    /// Whether `needle` could possibly occur in a string with this signature.
    ///
    /// No false negatives: a byte present in the needle and absent from the
    /// haystack settles it.
    #[inline]
    pub fn may_contain(self, needle: Signature) -> bool {
        needle.0 & !self.0 == 0
    }
}

/// A substring that means something, and the signature that lets it be skipped.
struct Marker {
    needle: &'static [u8],
    signature: Signature,
}

impl Marker {
    const fn new(needle: &'static str) -> Marker {
        Marker {
            needle: needle.as_bytes(),
            signature: Signature::of(needle.as_bytes()),
        }
    }
}

/// Lower case, for ASCII, and identity for everything else.
///
/// Not `byte | 0x20`, which is the usual trick and is wrong the moment a name
/// contains an underscore: `_` is 0x5F and the spare bit turns it into 0x7F.
/// Every marker here is written with underscores, so that mistake silently
/// disables most of them.
#[inline]
const fn fold(byte: u8) -> u8 {
    if byte.is_ascii_uppercase() {
        byte | 0x20
    } else {
        byte
    }
}

/// Whether `haystack` contains `needle`, comparing case-insensitively without
/// folding either into a new buffer.
///
/// `needle` must already be lower case, which every marker here is.
///
/// The first byte is checked before any comparison, which is what makes the
/// common case — a marker that is not there at all — one pass over the name
/// rather than one per position.
fn contains_fold(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() || needle.len() > haystack.len() {
        return false;
    }
    let first = needle[0];
    let limit = haystack.len() - needle.len();
    for start in 0..=limit {
        if fold(haystack[start]) != first {
            continue;
        }
        if haystack[start..]
            .iter()
            .zip(needle)
            .all(|(a, b)| fold(*a) == *b)
        {
            return true;
        }
    }
    false
}

/// Whether two strings are the same, ignoring case. `expected` must already be
/// lower case.
fn eq_fold(actual: &[u8], expected: &[u8]) -> bool {
    actual.len() == expected.len() && actual.iter().zip(expected).all(|(a, b)| fold(*a) == *b)
}

/// Whether any marker in `markers` occurs in a name with this signature.
fn any_marker(name: &[u8], signature: Signature, markers: &[Marker]) -> bool {
    markers
        .iter()
        .any(|m| signature.may_contain(m.signature) && contains_fold(name, m.needle))
}

/// What cameras and editors write into a name when the eyes are stacked.
///
/// All lower case: [`contains_fold`] folds the haystack towards lower case, so
/// a needle with a capital in it could never match.
const OVER_UNDER_MARKERS: [Marker; 9] = [
    Marker::new("_tb"),
    Marker::new("-tb"),
    Marker::new("_ou"),
    Marker::new("-ou"),
    Marker::new("overunder"),
    Marker::new("over-under"),
    Marker::new("topbottom"),
    Marker::new("top-bottom"),
    Marker::new("3dv"),
];

const SIDE_BY_SIDE_MARKERS: [Marker; 7] = [
    Marker::new("_sbs"),
    Marker::new("-sbs"),
    Marker::new("_lr"),
    Marker::new("-lr"),
    Marker::new("sidebyside"),
    Marker::new("side-by-side"),
    Marker::new("3dh"),
];

const VR180_MARKERS: [Marker; 5] = [
    Marker::new("vr180"),
    Marker::new("_180"),
    Marker::new("-180"),
    Marker::new("180x180"),
    Marker::new("half-sphere"),
];

const PANORAMA_MARKERS: [Marker; 7] = [
    Marker::new("360"),
    Marker::new("equirect"),
    Marker::new("spherical"),
    Marker::new("_vr."),
    Marker::new("-vr."),
    Marker::new("insta360"),
    Marker::new("theta"),
];

/// How close an eye's aspect must be to 2:1 or 1:1 before it counts as evidence
/// of a panorama. Tight, because ordinary films sit at 1.78 and 2.39 and a
/// loose bound would sweep them onto a sphere.
const EQUIRECT_TOLERANCE: f32 = 0.06;

/// No camera shoots taller than square; stacking two pictures does.
const TALLER_THAN_ANY_PICTURE: f32 = 1.0;

/// Wider than Ultra Panavision, the widest format anyone shoots.
const WIDER_THAN_ANY_PICTURE: f32 = 2.9;

/// Where each eye's image sits, from the name first and the frame's shape
/// second.
///
/// The name comes first because it is the only place anyone ever *states* the
/// answer: `_TB`, `_SBS` and their variants are what cameras and editors write.
///
/// The shape is a much weaker signal than it looks. It reads a frame as though
/// it were equirectangular — 2:1 for one eye — so anything narrower looks
/// stacked and anything wider looks side-by-side, and a four-by-three film
/// comes out as over-under 3D. So it is only consulted for frames an ordinary
/// picture never takes: much taller than wide, or wider than any cinema format.
pub fn packing_for(name: &[u8], signature: Signature, width: u32, height: u32) -> u32 {
    if any_marker(name, signature, &OVER_UNDER_MARKERS) {
        return OVER_UNDER;
    }
    if any_marker(name, signature, &SIDE_BY_SIDE_MARKERS) {
        return SIDE_BY_SIDE;
    }
    if width == 0 || height == 0 {
        return MONO;
    }
    let aspect = width as f32 / height as f32;
    if aspect < TALLER_THAN_ANY_PICTURE {
        OVER_UNDER
    } else if aspect > WIDER_THAN_ANY_PICTURE {
        SIDE_BY_SIDE
    } else {
        MONO
    }
}

/// How much world the picture covers: a screen, a hemisphere or a sphere.
///
/// The default is a screen, and that is the important part. Most video is an
/// ordinary picture, and playing one on a sphere smears a single frame across
/// the whole world. So a panorama has to be *shown*, not merely not ruled out.
///
/// Two things show it. A name that says so, which is what exporters write. Or a
/// frame whose shape is characteristic: an equirectangular image covers 360°
/// across and 180° down, so one eye's worth of a full sphere is 2:1 almost
/// exactly and one eye's worth of a hemisphere is square. Ordinary films
/// cluster at 1.78 and 2.39 and land on neither.
///
/// The square case additionally requires a stereo packing, because a monoscopic
/// VR180 is not a thing anyone shoots — two forward-facing lenses are the whole
/// point — whereas a square flat video is merely unusual.
pub fn projection_for(
    name: &[u8],
    signature: Signature,
    packing: u32,
    width: u32,
    height: u32,
) -> u32 {
    if any_marker(name, signature, &VR180_MARKERS) {
        return EQUIRECT_180;
    }
    if any_marker(name, signature, &PANORAMA_MARKERS) {
        return EQUIRECT_360;
    }
    if width == 0 || height == 0 {
        return FLAT;
    }
    let eye_aspect = match packing {
        // Two pictures stacked, and row-interleaving is the same halving of the
        // height by another arrangement.
        OVER_UNDER | ROW_INTERLEAVED => width as f32 / (height as f32 / 2.0),
        SIDE_BY_SIDE => (width as f32 / 2.0) / height as f32,
        // An anaglyph frame is full size for both eyes; only the colour is
        // shared.
        _ => width as f32 / height as f32,
    };
    if (eye_aspect - 2.0).abs() < EQUIRECT_TOLERANCE {
        EQUIRECT_360
    } else if packing != MONO && (eye_aspect - 1.0).abs() < EQUIRECT_TOLERANCE {
        EQUIRECT_180
    } else {
        FLAT
    }
}

/// The packing a catalogue's stereo mode states, or `None` when it states
/// nothing usable.
///
/// The vocabulary is Kodi's, which is also Matroska's. Two cases return `None`
/// for quite different reasons, and both matter.
///
/// An **absent or empty** mode is not a finding. A scanner writes the same empty
/// string whether it looked and saw a flat file or never looked at all, so
/// reading it as "monoscopic" would let an absence override a name somebody
/// deliberately marked `_TB` — breaking exactly the files whose owners took
/// care. An explicit `mono` is different: that is a reading, and it counts.
///
/// **Checkerboard and column-interleaved** are real and are not handled: both
/// would need their own fragment path, and neither has turned up in a library
/// here. They fall through to inference, which plays the file flat rather than
/// wrongly.
///
/// The `_rl` and `bottom_top` variants come back as their `lr` counterpart with
/// the eyes swapped, because that is what they are: the frame is split
/// identically and only the labels differ.
pub fn packing_for_stereo_mode(mode: &[u8]) -> Option<(u32, bool, u32)> {
    let trimmed = trim(mode);
    if trimmed.is_empty() {
        return None;
    }
    let named = |name: &str| eq_fold(trimmed, name.as_bytes());

    if named("mono") {
        return Some((MONO, false, 0));
    }
    for name in ["left_right", "lr", "sbs", "side_by_side"] {
        if named(name) {
            return Some((SIDE_BY_SIDE, false, 0));
        }
    }
    for name in ["right_left", "rl"] {
        if named(name) {
            return Some((SIDE_BY_SIDE, true, 0));
        }
    }
    for name in ["top_bottom", "tb", "over_under"] {
        if named(name) {
            return Some((OVER_UNDER, false, 0));
        }
    }
    if named("bottom_top") {
        return Some((OVER_UNDER, true, 0));
    }
    if named("row_interleaved_lr") {
        return Some((ROW_INTERLEAVED, false, 0));
    }
    if named("row_interleaved_rl") {
        return Some((ROW_INTERLEAVED, true, 0));
    }
    // Matroska's "cyan/red" is this player's red/cyan: the label reads
    // left-then-right but the pixels do not agree with it, and the pixels win.
    // See `pano::AnaglyphPair`.
    for (name, pair) in [
        ("anaglyph_cyan_red", 0),
        ("anaglyph_green_magenta", 1),
        ("anaglyph_yellow_blue", 2),
    ] {
        if named(name) {
            return Some((ANAGLYPH, false, pair));
        }
    }
    None
}

fn trim(bytes: &[u8]) -> &[u8] {
    let start = bytes.iter().position(|b| !b.is_ascii_whitespace());
    let end = bytes.iter().rposition(|b| !b.is_ascii_whitespace());
    match (start, end) {
        (Some(s), Some(e)) => &bytes[s..=e],
        _ => &[],
    }
}

/// Everything known about one entry, read once.
pub fn infer(name: &[u8], mode: &[u8], width: u32, height: u32) -> Format {
    let signature = Signature::of(name);
    match packing_for_stereo_mode(mode) {
        // The catalogue scanned the file, so its packing beats the name. The
        // projection is still inferred: no catalogue records it.
        Some((packing, swap_eyes, anaglyph_pair)) => Format {
            packing,
            projection: projection_for(name, signature, packing, width, height),
            swap_eyes,
            anaglyph_pair,
            verdict: Verdict::Catalogued,
        },
        None => {
            let packing = packing_for(name, signature, width, height);
            Format {
                packing,
                projection: projection_for(name, signature, packing, width, height),
                swap_eyes: false,
                anaglyph_pair: 0,
                verdict: Verdict::Guessed,
            }
        }
    }
}

/// Words of description per entry in a batch: where its name and stereo mode
/// are in the blob, and how big its frame is.
pub const META_WORDS: usize = 6;

/// One entry's slice of the submission buffer.
struct Entry<'a> {
    name: &'a [u8],
    mode: &'a [u8],
    width: u32,
    height: u32,
}

fn entry<'a>(blob: &'a [u8], meta: &[u32], index: usize) -> Option<Entry<'a>> {
    let m = meta.get(index * META_WORDS..index * META_WORDS + META_WORDS)?;
    let slice = |offset: u32, len: u32| -> Option<&'a [u8]> {
        let start = offset as usize;
        let end = start.checked_add(len as usize)?;
        blob.get(start..end)
    };
    Some(Entry {
        name: slice(m[0], m[1])?,
        mode: slice(m[2], m[3])?,
        width: m[4],
        height: m[5],
    })
}

/// How a batch should be split on *this* machine.
///
/// Splitting a listing across cores is obviously right and, on the hardware
/// this was written for, obviously slower. Rather than settle that with a
/// constant — which would be a measurement of one phone in one year, wrong on
/// the next one in either direction — the machine is asked.
///
/// See [`calibrate`] for how, and [`start_calibration`] for when.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Plan {
    /// Entries above which threading has been measured to win, or
    /// [`usize::MAX`] when it never does.
    pub threshold: usize,
    /// How many threads to use once it does.
    pub threads: usize,
}

impl Plan {
    /// What to do before anything has been measured, and if measuring fails:
    /// one thread, always. The conservative choice, because a listing read on
    /// one core is merely not as fast as it could be, whereas one split across
    /// cores that cost more than they save is slower than doing nothing clever
    /// at all.
    pub const SERIAL: Plan = Plan {
        threshold: usize::MAX,
        threads: 1,
    };
}

static PLAN: OnceLock<Plan> = OnceLock::new();
static CALIBRATING: AtomicBool = AtomicBool::new(false);

/// The plan for this machine, or [`Plan::SERIAL`] until one has been measured.
pub fn plan() -> Plan {
    PLAN.get().copied().unwrap_or(Plan::SERIAL)
}

/// Measures this machine and settles the plan, on a thread of its own.
///
/// Returns at once. The measurement costs a few milliseconds — it has to start
/// threads to find out what starting threads costs — which is nothing spread
/// over a session and far too much on the first browse, so it does not happen
/// there. Until it finishes, [`plan`] says serial and listings are read on one
/// core, which is what they would have been anyway.
///
/// Calling it more than once is harmless; only the first does anything.
pub fn start_calibration() {
    if PLAN.get().is_some() || CALIBRATING.swap(true, Ordering::AcqRel) {
        return;
    }
    // A detached thread rather than a scope: nothing waits for this, and the
    // result is published through `PLAN` when it is ready.
    let _ = std::thread::Builder::new()
        .name("library-calibrate".into())
        .spawn(|| {
            let measured = calibrate();
            let _ = PLAN.set(measured);
        });
}

/// Works out, by measurement, the batch size at which threads start to pay.
///
/// Not by modelling it. The first version of this timed the threaded path at
/// two sizes, fitted a line and solved for where it crossed the serial one —
/// which is a reasonable thing to do and was wrong about one run in three,
/// because both timings are dominated by the fixed cost of starting the threads
/// and their difference is mostly noise. It announced crossovers at forty and
/// fifty thousand entries on hardware where repeated measurement says there is
/// none at any size.
///
/// So nothing is predicted. The two paths are raced at a ladder of sizes, and
/// the threshold is the smallest size at which the threaded one was **seen** to
/// win. A path nobody has watched win does not get switched on.
///
/// The ladder stops at [`LADDER`]'s largest rung, which is already far past any
/// listing a media library returns. Hardware on which threads only pay beyond
/// that is hardware on which, for this purpose, they do not pay.
///
/// The sample is synthetic and unlike a real listing in one way only: every
/// name is distinct, so nothing can be cached between entries and the per-entry
/// cost is not flattered.
pub fn calibrate() -> Plan {
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    if threads < 2 {
        return Plan::SERIAL;
    }

    let largest = *LADDER.last().expect("the ladder has rungs");
    let (blob, meta) = calibration_sample(largest);
    let mut out = vec![0u32; largest];

    // Once through to warm the caches and let the governor notice there is
    // work, so the first timed run is not measuring a sleeping core.
    infer_serial(&blob, &meta, &mut out);

    for &size in LADDER {
        let serial = best_of(|| infer_serial(&blob, &meta, &mut out[..size]));
        let threaded = best_of(|| infer_threaded(&blob, &meta, &mut out[..size], threads));
        if threaded.as_secs_f64() * WORTH_IT < serial.as_secs_f64() {
            return Plan {
                threshold: size,
                threads,
            };
        }
    }
    Plan {
        threshold: usize::MAX,
        threads,
    }
}

/// Batch sizes the two paths are raced at, smallest first.
///
/// The rungs are far apart because the answer is not needed precisely. What is
/// being decided is whether to use threads on this machine at all, and being a
/// factor of four late to start using them costs a fraction of a millisecond on
/// a listing nobody has.
///
/// The top rung is deliberately past plausibility. A folder of thirty thousand
/// files is not something a person browses, so hardware that needs more than
/// that before threads pay is hardware where the answer is no.
const LADDER: &[usize] = &[8192, 32_768];

/// How much the threaded path has to win by before it counts as having won.
///
/// A win by a hair is a coincidence; five runs of the same measurement on the
/// same phone will produce one if allowed to. Ten per cent is comfortably
/// outside the run-to-run spread and comfortably inside any real speed-up:
/// hardware where threads help at all helps by much more than this.
const WORTH_IT: f64 = 1.1;

/// How many times each measurement is repeated. A phone is a busy machine: a
/// run can be interrupted by anything, and a single timing that happened to
/// land next to something else is how a calibration decides to use threads on
/// hardware where they lose.
const CALIBRATION_RUNS: usize = 3;

/// The fastest of several runs.
///
/// The minimum, not the mean: what is wanted is how fast the machine *can* do
/// this, and every source of error here — a scheduler preemption, another app
/// waking up, a core parked at a low clock — makes a run slower and none makes
/// one faster. Averaging folds that noise into the answer; taking the best
/// discards it.
fn best_of(mut work: impl FnMut()) -> Duration {
    (0..CALIBRATION_RUNS)
        .map(|_| {
            let start = Instant::now();
            work();
            start.elapsed()
        })
        .min()
        .unwrap_or_default()
}

/// A synthetic listing of the shape a real one has: mostly ordinary names, a
/// few marked up, every one distinct.
fn calibration_sample(count: usize) -> (Vec<u8>, Vec<u32>) {
    const NAMES: [&str; 4] = [
        "An Ordinary Film (2016) 2160p",
        "Holiday_TB.mkv",
        "climb_vr180_sbs.mp4",
        "The Longest Title Anyone Would Reasonably Give A File",
    ];
    const MODES: [&str; 3] = ["", "mono", "left_right"];
    let mut blob = Vec::with_capacity(count * 48);
    let mut meta = Vec::with_capacity(count * META_WORDS);
    for i in 0..count {
        let name_offset = blob.len() as u32;
        blob.extend_from_slice(NAMES[i % NAMES.len()].as_bytes());
        // A serial number, so no two entries are the same bytes.
        blob.extend_from_slice(i.to_string().as_bytes());
        let name_len = blob.len() as u32 - name_offset;
        let mode = MODES[i % MODES.len()].as_bytes();
        let mode_offset = blob.len() as u32;
        blob.extend_from_slice(mode);
        meta.extend_from_slice(&[
            name_offset,
            name_len,
            mode_offset,
            mode.len() as u32,
            3840,
            1920,
        ]);
    }
    (blob, meta)
}

/// Entries one thread claims at a time.
///
/// Small relative to the batch on purpose. The cores are not alike — on this
/// generation, one fast, three middling and four slow — so an even split would
/// finish when the slowest core finished its eighth, which is the wrong answer
/// by a factor of two. Claiming in small chunks lets a slow core take fewer of
/// them, and the batch finishes when the *work* runs out rather than when the
/// worst core does.
///
/// A chunk is 256 bytes of output, which is four cache lines, so two threads
/// working on neighbouring chunks are not writing to the same one.
const CHUNK: usize = 64;

fn infer_serial(blob: &[u8], meta: &[u32], out: &mut [u32]) {
    for (i, slot) in out.iter_mut().enumerate() {
        let e = entry(blob, meta, i).expect("validated by the caller");
        *slot = infer(e.name, e.mode, e.width, e.height).pack();
    }
}

fn infer_threaded(blob: &[u8], meta: &[u32], out: &mut [u32], threads: usize) {
    // `chunks_mut` hands out disjoint slices, so the borrow checker already
    // knows no two threads can write the same element and there is nothing here
    // to get wrong by hand.
    let queue = Mutex::new(out.chunks_mut(CHUNK).enumerate().collect::<Vec<_>>());
    let queue = &queue;
    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(move || loop {
                // Taking one chunk under a lock costs tens of nanoseconds
                // against a chunk's tens of microseconds of work, so the
                // contention is not worth engineering away.
                let Some((index, chunk)) = queue.lock().ok().and_then(|mut q| q.pop()) else {
                    return;
                };
                let base = index * CHUNK;
                for (offset, slot) in chunk.iter_mut().enumerate() {
                    let e = entry(blob, meta, base + offset).expect("validated by the caller");
                    *slot = infer(e.name, e.mode, e.width, e.height).pack();
                }
            });
        }
    });
}

/// Reads a whole listing.
///
/// `blob` holds every name and stereo mode end to end; `meta` describes each
/// entry with [`META_WORDS`] words; `out` receives one packed [`Format`] each.
/// Returns how many were written, which is `out.len()` unless the description
/// is short or inconsistent, in which case nothing is written at all — a
/// partly-filled listing would be read as a complete one.
///
/// Whether the work is split across cores is [`plan`]'s decision, and that is a
/// measurement of the machine rather than an assumption about it. Both paths
/// produce identical output; only the timing differs.
///
/// # What was worth doing here
///
/// Measured on the Pixel 9 this targets, with `examples/library_bench`:
///
/// * The [`Signature`] filter is worth **5.2×** — 56 ns an entry against 294 ns
///   with it forced to always pass. It is the one that mattered.
/// * A five-hundred-entry folder costs **0.23 ms** in total, in one crossing
///   rather than five hundred.
/// * Threading **loses at every size**, and keeps losing: 0.06× at 4096
///   entries, 0.65× at 262 144, where the fixed cost is long since amortised.
///   Starting eight threads and joining them costs 3.7 ms on that generation,
///   and four of its eight cores are the slow ones.
///
/// That last figure is why the split is measured rather than assumed. It is a
/// fact about one phone in one year, and hard-coding it would make the next
/// generation inherit a decision taken about hardware it is not.
pub fn infer_batch(blob: &[u8], meta: &[u32], out: &mut [u32]) -> usize {
    infer_batch_using(plan(), blob, meta, out)
}

/// The same, under a plan of the caller's choosing rather than the measured
/// one.
///
/// Exists so that both paths can be run deliberately — by the tests, which must
/// prove they agree, and by `examples/library_bench`, which has to be able to
/// check that the crossover [`calibrate`] predicts is really where the two
/// curves cross. A model nobody measures against is a guess with arithmetic in
/// front of it.
pub fn infer_batch_using(plan: Plan, blob: &[u8], meta: &[u32], out: &mut [u32]) -> usize {
    let count = out.len().min(meta.len() / META_WORDS);
    if count == 0 {
        return 0;
    }
    // Validated up front so the loop cannot fail halfway and leave a listing
    // half-read.
    if (0..count).any(|i| entry(blob, meta, i).is_none()) {
        return 0;
    }

    if count >= plan.threshold && plan.threads > 1 {
        infer_threaded(blob, meta, &mut out[..count], plan.threads);
    } else {
        infer_serial(blob, meta, &mut out[..count]);
    }
    count
}

// ---------------------------------------------------------------------------
// C ABI
//
// One call per listing, over buffers neither side copies. The alternative was a
// call and a string allocation per entry, which for a library of any size costs
// more than the reading does.
// ---------------------------------------------------------------------------

/// Reads a listing. Returns the number of entries written, or -1.
///
/// # Safety
/// `blob` must point to `blob_len` readable bytes, `meta` to `meta_len`
/// readable `u32`s, and `out` to `out_len` writable `u32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_library_infer(
    blob: *const u8,
    blob_len: usize,
    meta: *const u32,
    meta_len: usize,
    out: *mut u32,
    out_len: usize,
) -> i32 {
    if meta.is_null() || out.is_null() || (blob.is_null() && blob_len != 0) {
        return -1;
    }
    let blob = if blob_len == 0 {
        &[][..]
    } else {
        std::slice::from_raw_parts(blob, blob_len)
    };
    let meta = std::slice::from_raw_parts(meta, meta_len);
    let out = std::slice::from_raw_parts_mut(out, out_len);
    infer_batch(blob, meta, out) as i32
}

/// Starts measuring how this machine should split a listing, on a thread of its
/// own. Returns at once; call it once, early.
///
/// Skipping it is not an error and costs nothing but the chance of a faster
/// batch: until the measurement lands, listings are read on one core.
#[no_mangle]
pub extern "C" fn xr_library_calibrate() {
    start_calibration();
}

/// Writes `[threshold, threads]` — what the measurement decided, or
/// `[0xFFFFFFFF, 1]` until it has. For logging; nothing depends on it.
///
/// # Safety
/// `out` must point to two writable, aligned `u32`s.
#[no_mangle]
pub unsafe extern "C" fn xr_library_plan(out: *mut u32) -> i32 {
    if out.is_null() {
        return -1;
    }
    let plan = plan();
    let slice = std::slice::from_raw_parts_mut(out, 2);
    slice[0] = plan.threshold.min(u32::MAX as usize) as u32;
    slice[1] = plan.threads.min(u32::MAX as usize) as u32;
    0
}

/// Reads one entry, for the caller that has exactly one. Returns the packed
/// [`Format`].
///
/// # Safety
/// `name` must point to `name_len` readable bytes and `mode` to `mode_len`.
#[no_mangle]
pub unsafe extern "C" fn xr_library_infer_one(
    name: *const u8,
    name_len: usize,
    mode: *const u8,
    mode_len: usize,
    width: u32,
    height: u32,
) -> u32 {
    let bytes = |p: *const u8, len: usize| -> &[u8] {
        if p.is_null() || len == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(p, len)
        }
    };
    infer(bytes(name, name_len), bytes(mode, mode_len), width, height).pack()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn guess(name: &str, width: u32, height: u32) -> Format {
        infer(name.as_bytes(), b"", width, height)
    }

    #[test]
    fn a_signature_never_rejects_a_substring_that_is_there() {
        // The one property the filter must have. Everything else it does is an
        // optimisation; this is correctness.
        let names = [
            "Holiday_TB.mkv",
            "climb_vr180_sbs.mp4",
            "Arrival (2016)",
            "insta360 x4 clip",
            "SIDE-BY-SIDE.MP4",
            "",
        ];
        let all = OVER_UNDER_MARKERS
            .iter()
            .chain(&SIDE_BY_SIDE_MARKERS)
            .chain(&VR180_MARKERS)
            .chain(&PANORAMA_MARKERS);
        for marker in all {
            for name in names {
                let signature = Signature::of(name.as_bytes());
                if contains_fold(name.as_bytes(), marker.needle) {
                    assert!(
                        signature.may_contain(marker.signature),
                        "{name:?} contains {:?} but the signature denied it",
                        std::str::from_utf8(marker.needle).unwrap(),
                    );
                }
            }
        }
    }

    #[test]
    fn the_signature_actually_rejects_most_markers() {
        // If it rejected nothing it would be pure overhead. An ordinary film
        // name shares few letters with `sidebyside` or `equirect`.
        let name = "Arrival (2016)";
        let signature = Signature::of(name.as_bytes());
        let all: Vec<&Marker> = OVER_UNDER_MARKERS
            .iter()
            .chain(&SIDE_BY_SIDE_MARKERS)
            .chain(&VR180_MARKERS)
            .chain(&PANORAMA_MARKERS)
            .collect();
        let survivors = all
            .iter()
            .filter(|m| signature.may_contain(m.signature))
            .count();
        assert!(
            survivors * 4 < all.len(),
            "{survivors} of {} markers survived the filter",
            all.len(),
        );
    }

    #[test]
    fn folding_happens_without_a_second_copy_of_the_name() {
        assert!(contains_fold(b"Holiday_TB.mkv", b"_tb"));
        assert!(contains_fold(b"HOLIDAY_tb.MKV", b"_tb"));
        assert!(!contains_fold(b"Holiday.mkv", b"_tb"));
        // A needle longer than the haystack, and an empty one.
        assert!(!contains_fold(b"ab", b"abc"));
        assert!(!contains_fold(b"abc", b""));
        // A near miss that shares its first byte with the needle at every
        // position, which is where a first-byte scan alone would go wrong.
        assert!(!contains_fold(b"____", b"_tb"));
    }

    #[test]
    fn a_name_that_states_the_packing_outranks_the_frame_shape() {
        let format = guess("Holiday_TB.mkv", 3840, 1920);
        assert_eq!(format.packing, OVER_UNDER);
        // 3840x1920 is 2:1, which would otherwise read as a monoscopic sphere.
        assert_eq!(format.verdict, Verdict::Guessed);
    }

    #[test]
    fn an_ordinary_film_is_a_screen_and_not_a_sphere() {
        for (name, w, h) in [
            ("Arrival (2016)", 1920, 1080),
            ("Dune", 3840, 1608),
            ("Casablanca", 1440, 1080),
        ] {
            let format = guess(name, w, h);
            assert_eq!(format.projection, FLAT, "{name}");
            assert_eq!(format.packing, MONO, "{name}");
        }
    }

    #[test]
    fn a_stereo_mode_the_catalogue_states_outranks_the_name() {
        let format = infer(b"Holiday_TB.mkv", b"left_right", 3840, 1920);
        assert_eq!(format.packing, SIDE_BY_SIDE);
        assert_eq!(format.verdict, Verdict::Catalogued);
    }

    #[test]
    fn an_empty_stereo_mode_is_an_absence_and_not_a_reading() {
        // A scanner writes the same empty string whether it looked and saw a
        // flat file or never looked at all. Reading it as monoscopic would
        // override exactly the names somebody took care to mark.
        for mode in ["", "   "] {
            let format = infer(b"Holiday_TB.mkv", mode.as_bytes(), 3840, 1920);
            assert_eq!(format.packing, OVER_UNDER, "mode {mode:?}");
            assert_eq!(format.verdict, Verdict::Guessed, "mode {mode:?}");
        }
        // An explicit `mono`, by contrast, is a reading and it counts.
        let format = infer(b"Holiday_TB.mkv", b"mono", 3840, 1920);
        assert_eq!(format.packing, MONO);
        assert_eq!(format.verdict, Verdict::Catalogued);
    }

    #[test]
    fn the_reversed_variants_are_a_swap_and_not_a_packing() {
        for (mode, packing) in [
            ("right_left", SIDE_BY_SIDE),
            ("bottom_top", OVER_UNDER),
            ("row_interleaved_rl", ROW_INTERLEAVED),
        ] {
            let format = infer(b"Film", mode.as_bytes(), 1920, 1080);
            assert_eq!(format.packing, packing, "{mode}");
            assert!(format.swap_eyes, "{mode} should swap the eyes");
        }
        for (mode, packing) in [
            ("left_right", SIDE_BY_SIDE),
            ("top_bottom", OVER_UNDER),
            ("row_interleaved_lr", ROW_INTERLEAVED),
        ] {
            let format = infer(b"Film", mode.as_bytes(), 1920, 1080);
            assert_eq!(format.packing, packing, "{mode}");
            assert!(!format.swap_eyes, "{mode} should not swap the eyes");
        }
    }

    #[test]
    fn each_anaglyph_mode_carries_its_own_pair_of_colours() {
        for (mode, pair) in [
            ("anaglyph_cyan_red", 0),
            ("anaglyph_green_magenta", 1),
            ("anaglyph_yellow_blue", 2),
        ] {
            let format = infer(b"Documentary", mode.as_bytes(), 1920, 1080);
            assert_eq!(format.packing, ANAGLYPH, "{mode}");
            assert_eq!(format.anaglyph_pair, pair, "{mode}");
            assert_eq!(format.verdict, Verdict::Catalogued, "{mode}");
        }
    }

    #[test]
    fn the_modes_nothing_here_can_split_fall_through_to_inference() {
        // Checkerboard and column-interleaved are real, and pretending to
        // handle them would show a chequerboard of two eyes to both.
        for mode in ["checkerboard_lr", "col_interleaved_lr", "block_lr"] {
            let format = infer(b"Documentary", mode.as_bytes(), 3840, 1920);
            assert_eq!(format.verdict, Verdict::Guessed, "{mode}");
            assert_eq!(format.packing, MONO, "{mode}");
        }
    }

    #[test]
    fn a_row_interleaved_frame_is_half_height_per_eye() {
        // Interleaving halves the height per eye exactly as stacking does, so a
        // 3840x3840 file is a full sphere for each eye. Reading the frame at
        // full height instead would call it a flat 1:1 picture and put a
        // panorama on a screen.
        let format = infer(b"Reef", b"row_interleaved_lr", 3840, 3840);
        assert_eq!(format.packing, ROW_INTERLEAVED);
        assert_eq!(format.projection, EQUIRECT_360);

        // And an ordinary 3D television file is still a screen: 1920x1080
        // interleaved is 1920x540 per eye, which is nothing like a sphere.
        let film = infer(b"Gravity", b"row_interleaved_lr", 1920, 1080);
        assert_eq!(film.projection, FLAT);
    }

    #[test]
    fn packing_and_unpacking_a_format_is_the_identity() {
        let format = Format {
            packing: ROW_INTERLEAVED,
            projection: EQUIRECT_180,
            swap_eyes: true,
            anaglyph_pair: 2,
            verdict: Verdict::Chosen,
        };
        assert_eq!(Format::unpack(format.pack()), format);
        assert_eq!(Format::unpack(Format::default().pack()), Format::default());
    }

    /// Builds a submission buffer the way the caller would.
    fn batch(entries: &[(&str, &str, u32, u32)]) -> (Vec<u8>, Vec<u32>) {
        let mut blob = Vec::new();
        let mut meta = Vec::with_capacity(entries.len() * META_WORDS);
        for (name, mode, width, height) in entries {
            let name_offset = blob.len() as u32;
            blob.extend_from_slice(name.as_bytes());
            let mode_offset = blob.len() as u32;
            blob.extend_from_slice(mode.as_bytes());
            meta.extend_from_slice(&[
                name_offset,
                name.len() as u32,
                mode_offset,
                mode.len() as u32,
                *width,
                *height,
            ]);
        }
        (blob, meta)
    }

    #[test]
    fn a_batch_reads_the_same_as_the_entries_read_one_at_a_time() {
        // The batch path exists for speed, and the only thing that would make
        // it worth having is that it cannot disagree with the simple one.
        let entries: Vec<(String, String, u32, u32)> = (0..1000)
            .map(|i| {
                let names = [
                    "Arrival (2016)",
                    "Holiday_TB.mkv",
                    "climb_vr180_sbs.mp4",
                    "insta360 dive",
                    "Dune",
                ];
                let modes = ["", "mono", "left_right", "row_interleaved_rl", "  "];
                (
                    format!("{}{i}", names[i % names.len()]),
                    modes[i % modes.len()].to_string(),
                    if i % 3 == 0 { 3840 } else { 1920 },
                    if i % 3 == 0 { 1920 } else { 1080 },
                )
            })
            .collect();
        let borrowed: Vec<(&str, &str, u32, u32)> = entries
            .iter()
            .map(|(n, m, w, h)| (n.as_str(), m.as_str(), *w, *h))
            .collect();

        let (blob, meta) = batch(&borrowed);
        let mut out = vec![0u32; borrowed.len()];
        assert_eq!(infer_batch(&blob, &meta, &mut out), borrowed.len());

        for (i, (name, mode, width, height)) in borrowed.iter().enumerate() {
            let alone = infer(name.as_bytes(), mode.as_bytes(), *width, *height);
            assert_eq!(Format::unpack(out[i]), alone, "entry {i}: {name:?}");
        }
    }

    #[test]
    fn a_batch_far_larger_than_any_real_listing_is_still_read_whole() {
        // Two orders of magnitude past a plausible folder. Nothing here scales
        // with the batch, and the point is that nothing starts to.
        let entries: Vec<(&str, &str, u32, u32)> =
            (0..65_536).map(|_| ("Reef_TB", "", 3840, 3840)).collect();
        let (blob, meta) = batch(&entries);
        let mut out = vec![u32::MAX; entries.len()];
        assert_eq!(infer_batch(&blob, &meta, &mut out), entries.len());
        let expected = infer(b"Reef_TB", b"", 3840, 3840).pack();
        assert!(
            out.iter().all(|&w| w == expected),
            "an entry went unwritten"
        );
    }

    #[test]
    fn a_description_that_points_outside_the_blob_yields_nothing() {
        // A listing half-read is worse than one not read: the caller cannot
        // tell which half.
        let (blob, mut meta) = batch(&[("Reef", "", 3840, 1920), ("Dive", "", 3840, 3840)]);
        meta[6] = blob.len() as u32 + 10;
        let mut out = vec![0u32; 2];
        assert_eq!(infer_batch(&blob, &meta, &mut out), 0);
        assert_eq!(out, vec![0, 0], "nothing should have been written");
    }

    #[test]
    fn an_empty_batch_is_not_an_error() {
        let mut out: Vec<u32> = Vec::new();
        assert_eq!(infer_batch(&[], &[], &mut out), 0);
    }

    #[test]
    fn the_threaded_path_reads_exactly_what_the_serial_one_does() {
        // Whether the batch is split is a timing decision, so the two paths
        // must be indistinguishable in their output. A chunk written twice, or
        // an off-by-one in the base index, shows up here and nowhere else — the
        // threaded path may never run in production on this generation.
        let entries: Vec<(&str, &str, u32, u32)> = (0..1000)
            .map(|i| {
                let names = ["Arrival", "Holiday_TB.mkv", "climb_vr180_sbs.mp4", "Dune"];
                let modes = ["", "mono", "row_interleaved_rl", "anaglyph_green_magenta"];
                (
                    names[i % names.len()],
                    modes[i % modes.len()],
                    if i % 2 == 0 { 3840 } else { 1920 },
                    if i % 2 == 0 { 1920 } else { 1080 },
                )
            })
            .collect();
        let (blob, meta) = batch(&entries);

        let mut serial = vec![0u32; entries.len()];
        infer_serial(&blob, &meta, &mut serial);
        for threads in [2usize, 3, 8] {
            let mut threaded = vec![u32::MAX; entries.len()];
            infer_threaded(&blob, &meta, &mut threaded, threads);
            assert_eq!(threaded, serial, "with {threads} threads");
        }
    }

    #[test]
    fn a_partial_chunk_at_the_end_is_not_lost() {
        // The batch is claimed in chunks of CHUNK, and a listing is not a
        // multiple of it. The last, short chunk is the one an index arithmetic
        // slip drops.
        let count = CHUNK * 3 + 7;
        let entries: Vec<(&str, &str, u32, u32)> =
            (0..count).map(|_| ("Reef_TB", "", 3840, 3840)).collect();
        let (blob, meta) = batch(&entries);
        let mut out = vec![u32::MAX; count];
        infer_threaded(&blob, &meta, &mut out, 4);
        let expected = infer(b"Reef_TB", b"", 3840, 3840).pack();
        assert!(
            out.iter().all(|&w| w == expected),
            "an entry went unwritten"
        );
    }

    #[test]
    fn nothing_is_split_until_the_machine_has_been_measured() {
        // The conservative default matters more than the fast one: a listing
        // read on one core is merely not as fast as it could be, whereas one
        // split across cores that cost more than they save is slower than doing
        // nothing clever at all.
        assert_eq!(Plan::SERIAL.threads, 1);
        assert_eq!(Plan::SERIAL.threshold, usize::MAX);
    }

    #[test]
    fn measuring_the_machine_yields_a_plan_that_can_be_acted_on() {
        // What the numbers come out as depends on the machine and is not
        // asserted. That they are usable is: a threshold of zero would split
        // every listing however small, and zero threads would read none of it.
        let plan = calibrate();
        assert!(plan.threads >= 1, "{plan:?}");
        assert!(plan.threshold > 0, "{plan:?}");
        if plan.threads == 1 {
            assert_eq!(plan.threshold, usize::MAX, "one thread is not a split");
        }
    }

    #[test]
    fn calibration_can_be_asked_for_more_than_once() {
        // It is started from whichever caller gets there first, and a second
        // caller must not start a second measurement or block on the first.
        start_calibration();
        start_calibration();
        let before = plan();
        // Whatever it settles on, asking twice gives the same answer: the plan
        // is published once and never revised.
        assert_eq!(plan(), before);
    }
}
