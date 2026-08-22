//! Measures reading a library listing, on whatever machine it is run on.
//!
//! The numbers that matter are the phone's, not the build machine's: the target
//! has one fast core, three middling and four slow, and a thread started there
//! costs far more than one started on a desktop. So this is built for the
//! device and run on it — `cargo run --example library_bench` locally tells you
//! about the wrong computer.
//!
//! What it answers: how long one entry takes, and above what batch size the
//! threads have paid for themselves. The second is what [`PARALLEL_THRESHOLD`]
//! should be, and it is not something to guess at.
//!
//! ```text
//! cargo build --release --target aarch64-linux-android --example library_bench
//! adb push target/aarch64-linux-android/release/examples/library_bench /data/local/tmp/
//! adb shell /data/local/tmp/library_bench
//! ```

use std::time::Instant;

use viture_v2::library::{self, Plan, META_WORDS};

/// Names of the shape a real library holds: mostly ordinary films, a few marked
/// up by whoever exported them.
const NAMES: [&str; 8] = [
    "Arrival (2016) 2160p HDR",
    "Dune Part Two",
    "Holiday_TB.mkv",
    "climb_vr180_sbs.mp4",
    "insta360 x4 dive clip",
    "Casablanca (1942)",
    "The Lord of the Rings - The Fellowship of the Ring",
    "reef_360_3d_ou.mp4",
];

const MODES: [&str; 5] = ["", "mono", "left_right", "top_bottom", "row_interleaved_lr"];

fn build(count: usize) -> (Vec<u8>, Vec<u32>) {
    let mut blob = Vec::new();
    let mut meta = Vec::with_capacity(count * META_WORDS);
    for i in 0..count {
        // A serial number on each name, so no two entries are the same string
        // and nothing can be cached between them.
        let name = format!("{} [{i}]", NAMES[i % NAMES.len()]);
        let mode = MODES[i % MODES.len()];
        let name_offset = blob.len() as u32;
        blob.extend_from_slice(name.as_bytes());
        let mode_offset = blob.len() as u32;
        blob.extend_from_slice(mode.as_bytes());
        meta.extend_from_slice(&[
            name_offset,
            name.len() as u32,
            mode_offset,
            mode.len() as u32,
            if i % 3 == 0 { 3840 } else { 1920 },
            if i % 3 == 0 { 1920 } else { 1080 },
        ]);
    }
    (blob, meta)
}

/// The batch, timed over enough repetitions to be worth timing.
fn time(count: usize, repeats: usize) -> f64 {
    let (blob, meta) = build(count);
    let mut out = vec![0u32; count];
    // Once to warm the caches and let the branch predictors settle.
    library::infer_batch(&blob, &meta, &mut out);

    let start = Instant::now();
    for _ in 0..repeats {
        library::infer_batch(&blob, &meta, &mut out);
        std::hint::black_box(&out);
    }
    start.elapsed().as_secs_f64() / repeats as f64
}

/// The same work with no batching machinery at all: one entry at a time, on
/// this thread. The floor everything else has to beat.
fn time_serial(count: usize, repeats: usize) -> f64 {
    let (blob, meta) = build(count);
    let mut out = vec![0u32; count];
    let one = |i: usize, out: &mut [u32]| {
        let m = &meta[i * META_WORDS..][..META_WORDS];
        let name = &blob[m[0] as usize..][..m[1] as usize];
        let mode = &blob[m[2] as usize..][..m[3] as usize];
        out[i] = library::infer(name, mode, m[4], m[5]).pack();
    };
    for i in 0..count {
        one(i, &mut out);
    }

    let start = Instant::now();
    for _ in 0..repeats {
        for i in 0..count {
            one(i, &mut out);
        }
        std::hint::black_box(&out);
    }
    start.elapsed().as_secs_f64() / repeats as f64
}

/// What it costs merely to start and join threads, doing nothing.
fn time_spawn(threads: usize, repeats: usize) -> f64 {
    let start = Instant::now();
    for _ in 0..repeats {
        std::thread::scope(|scope| {
            for _ in 0..threads {
                scope.spawn(|| std::hint::black_box(0u32));
            }
        });
    }
    start.elapsed().as_secs_f64() / repeats as f64
}

/// Times both paths on the same batches, so the crossover [`library::calibrate`]
/// predicts can be checked against the crossover that actually happens.
///
/// The prediction is two timings and a straight line through them. That is a
/// reasonable model and it is not obviously true, so it is worth an honest look
/// at whether the threaded path really does win where the model says it starts
/// to — and, more importantly, that it does not lose anywhere the model says it
/// wins.
fn check_the_crossover(measured: Plan) {
    let forced = Plan {
        threshold: 0,
        threads: measured.threads.max(2),
    };
    println!(
        "\n{:>8}  {:>12}  {:>12}  {:>8}",
        "entries", "one core", "threaded", "ratio"
    );
    for count in [4096usize, 16_384, 65_536, 262_144] {
        let (blob, meta) = build(count);
        let mut out = vec![0u32; count];
        library::infer_batch_using(Plan::SERIAL, &blob, &meta, &mut out);

        let repeats = (2_000_000 / count).clamp(3, 100);
        let serial = {
            let start = Instant::now();
            for _ in 0..repeats {
                library::infer_batch_using(Plan::SERIAL, &blob, &meta, &mut out);
                std::hint::black_box(&out);
            }
            start.elapsed().as_secs_f64() / repeats as f64
        };
        let threaded = {
            let start = Instant::now();
            for _ in 0..repeats {
                library::infer_batch_using(forced, &blob, &meta, &mut out);
                std::hint::black_box(&out);
            }
            start.elapsed().as_secs_f64() / repeats as f64
        };
        let marker = if count >= measured.threshold {
            " <- predicted to win"
        } else {
            ""
        };
        println!(
            "{count:>8}  {:>9.3} ms  {:>9.3} ms  {:>7.2}x{marker}",
            serial * 1e3,
            threaded * 1e3,
            serial / threaded,
        );
    }
}

fn main() {
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(0);
    println!("cores reported: {cores}");
    println!(
        "{:>8}  {:>12}  {:>12}  {:>12}",
        "entries", "batch", "serial", "per entry"
    );

    for count in [16usize, 64, 256, 1024, 4096, 16_384, 65_536] {
        let repeats = (1_000_000 / count).clamp(5, 2000);
        let batched = time(count, repeats);
        let serial = time_serial(count, repeats);
        println!(
            "{count:>8}  {:>10.3} ms  {:>10.3} ms  {:>9.1} ns",
            batched * 1e3,
            serial * 1e3,
            serial / count as f64 * 1e9,
        );
    }

    // What the machine decides for itself, which is the number that ships.
    let measured = library::calibrate();
    check_the_crossover(measured);
    println!(
        "\nmeasured plan: threads {}, threshold {}",
        measured.threads,
        if measured.threshold == usize::MAX {
            "never — one core wins at every size".to_string()
        } else {
            format!("{} entries", measured.threshold)
        },
    );

    for threads in [2usize, 4, 8] {
        println!(
            "starting {threads} threads and joining them: {:.3} ms",
            time_spawn(threads, 200) * 1e3,
        );
    }

    // What a browse actually looks like: one folder, once.
    println!(
        "\na 500-entry folder: {:.3} ms",
        time_serial(500, 200) * 1e3
    );
}
