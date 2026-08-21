//! `viture-v2 probe | info | pose | raw | pointer`
//!
//! Shape: a reader thread pushes events into a lock-free ring buffer and the
//! consumer drains them in batches. There is no sleep and no lock anywhere
//! between the kernel and the consumer.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use viture_v2::hidraw::{self, Hidraw};
use viture_v2::pointer::{PhoneSensor, Pointer};
use viture_v2::ring::Ring;
use viture_v2::uring::Uring;
use viture_v2::usbfs::{self, Usbfs};
use viture_v2::{Device, Event, Pose, Rate, Raw, Streams, Transport};

const VID: u16 = 0x35CA;
const PID_PRO2: u16 = 0x1301;
const RING_CAP: usize = 1024;

/// What the reader thread puts into the ring.
#[derive(Clone, Copy)]
enum Sample {
    Pose(Pose),
    Raw(Raw),
}

fn main() {
    // `termux-usb -r -e ./viture-v2 /dev/bus/usb/B/D` calls us with the
    // descriptor as the first argument. It is only valid for the lifetime of
    // this process, so everything has to happen here.
    if let Some(fd) = std::env::args().nth(1).and_then(|a| a.parse::<i32>().ok()) {
        return termux(fd);
    }

    let mut args = std::env::args().skip(1);
    let cmd = args.next().unwrap_or_else(|| "info".into());
    let secs: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(5);
    let backend = args.next().unwrap_or_else(|| "uring".into());
    let rate = match args.next().as_deref() {
        Some("60") => Rate::Hz60,
        Some("90") => Rate::Hz90,
        Some("240") => Rate::Hz240,
        Some("500") => Rate::Hz500,
        Some("1000") => Rate::Hz1000,
        _ => Rate::Hz120,
    };

    match cmd.as_str() {
        "probe" => probe(),
        "info" => info(&mut Device::new(Hidraw::new(hid_fd()))),
        "pose" => run(&backend, Streams::POSE, secs, rate),
        "raw" => run(&backend, Streams::RAW, secs, rate),
        "pointer" => pointer(Device::new(Hidraw::new(hid_fd())), secs),
        other => {
            eprintln!("unknown: {other} (probe | info | pose | raw | pointer)");
            std::process::exit(2);
        }
    }
}

/// Entry point for `termux-usb`, where the descriptor is already open.
fn termux(fd: i32) {
    println!("termux-usb: descriptor {fd}");
    let usb = match Usbfs::new(fd, 0) {
        Ok(u) => u,
        Err(e) => {
            eprintln!("usbfs transport failed: {e}");
            std::process::exit(4);
        }
    };
    let mut dev = Device::new(usb);

    let mut buf = [0u8; 64];
    match dev.firmware_version(&mut buf) {
        Ok(v) => println!("firmware       {v}"),
        Err(e) => println!("firmware       — ({e})"),
    }
    report("brightness", dev.brightness());
    match dev.display_mode() {
        Ok(m) => println!("display mode   {m:?}"),
        Err(e) => println!("display mode   — ({e})"),
    }
    match dev.worn() {
        Ok(w) => println!("worn           {}", if w { "yes" } else { "no" }),
        Err(e) => println!("worn           — ({e})"),
    }

    let secs: u64 = std::env::var("VITURE_SECS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(5);
    if std::env::var("VITURE_MODE").as_deref() == Ok("pointer") {
        return pointer(dev, secs);
    }

    println!("\n{secs} s of pose stream …");
    pump(dev, Streams::POSE, secs, Rate::Hz120, |d| {
        let s = d.transport_mut().stats;
        format!(
            "{} URB submits, {} reaps, {} waits",
            s.submits, s.reaps, s.waits
        )
    });
}

/// Checks the syscall layer and transport availability without a device.
/// Useful for measuring a new platform before hardware is involved.
fn probe() {
    println!("architecture     {}", std::env::consts::ARCH);

    // ppoll: query stdin with no wait. The call has to return; the answer is
    // beside the point.
    match viture_v2::sys::wait_readable(0, 0) {
        Ok(r) => println!("ppoll            ok (stdin readable: {r})"),
        Err(e) => println!("ppoll            FAILED: {e}"),
    }

    // io_uring: blocked by seccomp inside the Android app sandbox.
    let mut params = [0u8; 120];
    match unsafe { viture_v2::sys::io_uring_setup(8, params.as_mut_ptr()) } {
        Ok(fd) => {
            println!("io_uring_setup   ok (fd {fd})");
            let _ = viture_v2::sys::close(fd);
        }
        Err(e) => println!("io_uring_setup   unavailable: {e}"),
    }

    match hidraw::find_fd(VID, PID_PRO2) {
        Ok(fd) => {
            println!("hidraw node      found");
            let _ = viture_v2::sys::close(fd);
        }
        Err(e) => println!("hidraw node      {e}"),
    }
    match usbfs::find_fd(VID, PID_PRO2) {
        Ok(fd) => {
            println!("usbfs node       found");
            let _ = viture_v2::sys::close(fd);
        }
        Err(e) => println!("usbfs node       {e}"),
    }
    match PhoneSensor::start(20) {
        Ok(s) => {
            std::thread::sleep(Duration::from_millis(600));
            println!("phone sensor     {} samples via termux-sensor", s.samples());
        }
        Err(e) => println!("phone sensor     unavailable: {e}"),
    }
}

fn info(dev: &mut Device<Hidraw>) {
    let mut buf = [0u8; 64];
    match dev.firmware_version(&mut buf) {
        Ok(v) => println!("firmware       {v}"),
        Err(e) => println!("firmware       — ({e})"),
    }
    let mut buf = [0u8; 64];
    match dev.serial(&mut buf) {
        // The serial travels in plaintext; only show a fragment.
        Ok(s) => println!(
            "serial         {}… ({} chars)",
            &s[..s.len().min(4)],
            s.len()
        ),
        Err(e) => println!("serial         — ({e})"),
    }
    report("brightness", dev.brightness());
    report("volume", dev.volume());
    report("duty cycle", dev.duty_cycle());
    match dev.display_mode() {
        Ok(m) => println!("display mode   {m:?}"),
        Err(e) => println!("display mode   — ({e})"),
    }
    match dev.worn() {
        Ok(w) => println!("worn           {}", if w { "yes" } else { "no" }),
        Err(e) => println!("worn           — ({e})"),
    }
}

fn report(label: &str, v: viture_v2::Result<u8>) {
    match v {
        Ok(v) => println!("{label:<14} {v}"),
        Err(e) => println!("{label:<14} — ({e})"),
    }
}

/// Opens the hidraw node or gives up.
fn hid_fd() -> i32 {
    hidraw::find_fd(VID, PID_PRO2).unwrap_or_else(|e| {
        eprintln!("device unreachable: {e}");
        std::process::exit(1);
    })
}

fn run(backend: &str, streams: Streams, secs: u64, rate: Rate) {
    match backend {
        "hidraw" => {
            let dev = Device::new(Hidraw::new(hid_fd()));
            pump(dev, streams, secs, rate, |d| {
                format!("{} syscalls", d.transport_mut().syscalls)
            });
        }
        "usbfs" => match usbfs::find_fd(VID, PID_PRO2).and_then(|fd| Usbfs::new(fd, 0)) {
            Ok(u) => {
                let dev = Device::new(u);
                pump(dev, streams, secs, rate, |d| {
                    let s = d.transport_mut().stats;
                    format!(
                        "{} URB submits, {} reaps, {} waits, largest batch {}",
                        s.submits, s.reaps, s.waits, s.max_batch
                    )
                });
            }
            Err(e) => {
                eprintln!("usbfs unavailable: {e}");
                std::process::exit(4);
            }
        },
        _ => match Uring::new(hid_fd()) {
            Ok(u) => {
                let dev = Device::new(u);
                pump(dev, streams, secs, rate, |d| {
                    let s = d.transport_mut().stats;
                    format!(
                        "{} io_uring_enter, {} completions, {} batches, largest batch {}",
                        s.enters, s.completions, s.batches, s.max_batch
                    )
                });
            }
            Err(e) => {
                eprintln!("io_uring unavailable ({e}) — falling back to hidraw");
                let dev = Device::new(Hidraw::new(hid_fd()));
                pump(dev, streams, secs, rate, |d| {
                    format!("{} syscalls", d.transport_mut().syscalls)
                });
            }
        },
    }
}

/// Spawns the reader thread that fills `ring` until `stop` is set.
fn spawn_reader<T: Transport + Send + 'static>(
    mut dev: Device<T>,
    ring: Arc<Ring<Sample, RING_CAP>>,
    stop: Arc<AtomicBool>,
    consumer: std::thread::Thread,
) -> std::thread::JoinHandle<Device<T>> {
    std::thread::spawn(move || {
        // Coalesce notifications: only wake once enough has piled up. That is
        // the io_uring batching idea one level up. 1 means wake per event,
        // which is the lowest latency and the highest CPU cost.
        let coalesce: usize = std::env::var("VITURE_COALESCE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);
        // The 50 ms window only exists so the stop flag is noticed promptly;
        // it is not polling, the thread sleeps in the kernel.
        while !stop.load(Ordering::Relaxed) {
            match dev.next_event(50_000_000) {
                Ok(Some(Event::Pose(p))) => {
                    ring.push(Sample::Pose(p));
                    if ring.len() >= coalesce {
                        consumer.unpark();
                    }
                }
                Ok(Some(Event::Raw(r))) => {
                    ring.push(Sample::Raw(r));
                    if ring.len() >= coalesce {
                        consumer.unpark();
                    }
                }
                Ok(Some(Event::Other(_))) => {}
                Ok(None) => consumer.unpark(),
                Err(e) => {
                    eprintln!("read error: {e}");
                    break;
                }
            }
        }
        let _ = dev.set_imu(Streams::OFF, Rate::Hz120);
        dev
    })
}

/// Reader thread fills the ring, main thread drains it in batches.
fn pump<T, F>(mut dev: Device<T>, streams: Streams, secs: u64, rate: Rate, summary: F)
where
    T: Transport + Send + 'static,
    F: FnOnce(&mut Device<T>) -> String,
{
    if let Err(e) = dev.set_imu(streams, rate) {
        eprintln!("could not start the IMU: {e}");
        std::process::exit(3);
    }

    let ring: Arc<Ring<Sample, RING_CAP>> = Arc::new(Ring::new());
    let stop = Arc::new(AtomicBool::new(false));
    let cpu_start = cpu_ns();
    let reader = spawn_reader(
        dev,
        Arc::clone(&ring),
        Arc::clone(&stop),
        std::thread::current(),
    );

    let start = Instant::now();
    let limit = Duration::from_secs(secs);
    let (mut total, mut batches, mut biggest) = (0u64, 0u64, 0usize);
    let mut last_print = Instant::now();

    while start.elapsed() < limit {
        let mut newest: Option<Sample> = None;
        let n = ring.drain(|s| newest = Some(s));
        if n > 0 {
            total += n as u64;
            batches += 1;
            biggest = biggest.max(n);
        } else {
            // No sleep: the thread parks and the producer wakes it. The window
            // is only there to notice the end of the run.
            std::thread::park_timeout(Duration::from_millis(20));
            continue;
        }
        if last_print.elapsed() >= Duration::from_secs(1) {
            last_print = Instant::now();
            match newest {
                Some(Sample::Pose(p)) => {
                    let [r, pi, y] = p.euler_deg();
                    println!("pose  roll={r:7.2} pitch={pi:7.2} yaw={y:7.2}   (batch {n})");
                }
                Some(Sample::Raw(r)) => {
                    let a = r.accel;
                    let mag = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                    println!(
                        "raw   gyro=[{:8.4}{:8.4}{:8.4}] |a|={mag:.3}g   (batch {n})",
                        r.gyro[0], r.gyro[1], r.gyro[2]
                    );
                }
                None => {}
            }
        }
    }

    stop.store(true, Ordering::Relaxed);
    let mut dev = reader.join().expect("reader thread");
    let cpu = cpu_ns().saturating_sub(cpu_start);
    let dt = start.elapsed().as_secs_f64();

    println!("\n── {dt:.1}s ──");
    println!("events           {total}  ({:.0} Hz)", total as f64 / dt);
    println!(
        "ring             {batches} drains, largest {biggest}, {} dropped",
        ring.dropped()
    );
    println!("kernel path      {}", summary(&mut dev));
    println!(
        "cpu time         {:.1} ms  ({:.2} % of one core)",
        cpu as f64 / 1e6,
        cpu as f64 / (dt * 1e9) * 100.0
    );
    if total > 0 {
        println!("cpu per event    {:.1} µs", cpu as f64 / 1e3 / total as f64);
    }
}

/// Head-relative pointing: glasses IMU for the head, phone IMU for the hand.
///
/// Prints both sources side by side, so it is visible that they are live and
/// independent, plus the cursor that results from combining them.
fn pointer<T: Transport + Send + 'static>(mut dev: Device<T>, secs: u64) {
    let phone = match PhoneSensor::start(16) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("phone sensor unavailable ({e}) — is termux-api installed?");
            std::process::exit(5);
        }
    };
    if let Err(e) = dev.set_imu(Streams::POSE, Rate::Hz120) {
        eprintln!("could not start the IMU: {e}");
        std::process::exit(3);
    }

    let ring: Arc<Ring<Sample, RING_CAP>> = Arc::new(Ring::new());
    let stop = Arc::new(AtomicBool::new(false));
    let reader = spawn_reader(
        dev,
        Arc::clone(&ring),
        Arc::clone(&stop),
        std::thread::current(),
    );

    let mut cursor = Pointer::default();
    if let Ok(Ok(d)) = std::env::var("VITURE_DISTANCE").map(|v| v.parse::<f32>()) {
        cursor.distance = d;
    }

    let start = Instant::now();
    let limit = Duration::from_secs(secs);
    let mut head = Pose {
        tick: 0,
        q: [1.0, 0.0, 0.0, 0.0],
    };
    let (mut head_samples, mut centred) = (0u64, false);
    let mut last_print = Instant::now();

    println!("\nhold the phone towards the screen centre — recentring in 2 s\n");

    while start.elapsed() < limit {
        let mut newest = None;
        let n = ring.drain(|s| {
            if let Sample::Pose(p) = s {
                newest = Some(p)
            }
        });
        if let Some(p) = newest {
            head = p;
            head_samples += n as u64;
        } else {
            std::thread::park_timeout(Duration::from_millis(8));
        }

        let ph = phone.read();
        if !centred && start.elapsed() > Duration::from_secs(2) && ph.seq > 0 {
            cursor.recentre(head.q, ph.q);
            centred = true;
            println!("recentred.\n");
        }

        if centred && last_print.elapsed() >= Duration::from_millis(250) {
            last_print = Instant::now();
            let [_, hp, hy] = head.euler_deg();
            let phone_pose = Pose { tick: 0, q: ph.q };
            let [_, pp, py] = phone_pose.euler_deg();
            match cursor.cursor(head.q, ph.q) {
                Some((x, y)) => println!(
                    "head yaw={hy:7.2} pitch={hp:6.2} [{head_samples:5}]   \
                     phone yaw={py:7.2} pitch={pp:6.2} [{:5}]   cursor x={x:+6.2} y={y:+6.2}",
                    ph.seq
                ),
                None => println!(
                    "head yaw={hy:7.2} pitch={hp:6.2} [{head_samples:5}]   \
                     phone yaw={py:7.2} pitch={pp:6.2} [{:5}]   cursor off screen",
                    ph.seq
                ),
            }
        }
    }

    stop.store(true, Ordering::Relaxed);
    let _ = reader.join();
    println!(
        "\n{head_samples} glasses samples, {} phone samples in {:.0}s",
        phone.samples(),
        start.elapsed().as_secs_f32()
    );
}

/// Process CPU time in nanoseconds — field 1 of `/proc/self/schedstat`.
fn cpu_ns() -> u64 {
    std::fs::read_to_string("/proc/self/schedstat")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(0)
}
