//! `viture-v2 info | pose | raw [Sekunden] [uring|hidraw]`
//!
//! Aufbau: ein Lese-Thread schiebt Ereignisse in einen sperrfreien Ringpuffer,
//! der Verbraucher holt sie im Batch ab. Zwischen Kernel und Verbraucher liegt
//! damit kein einziges Sleep und keine Sperre.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use viture_v2::hidraw::{self, Hidraw};
use viture_v2::ring::Ring;
use viture_v2::uring::Uring;
use viture_v2::usbfs::{self, Usbfs};
use viture_v2::{Device, Event, Pose, Raw, Rate, Streams, Transport};

const VID: u16 = 0x35CA;
const PID_PRO2: u16 = 0x1301;
const RING_CAP: usize = 1024;

/// Was der Lese-Thread in den Ring legt.
#[derive(Clone, Copy)]
enum Sample {
    Pose(Pose),
    Raw(Raw),
}

fn main() {
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
        other => {
            eprintln!("unbekannt: {other} (info | pose [s] [uring|hidraw] | raw [s] [...])");
            std::process::exit(2);
        }
    }
}

fn info(dev: &mut Device<Hidraw>) {
    let mut buf = [0u8; 64];
    match dev.firmware_version(&mut buf) {
        Ok(v) => println!("Firmware       {v}"),
        Err(e) => println!("Firmware       — ({e})"),
    }
    let mut buf = [0u8; 64];
    match dev.serial(&mut buf) {
        Ok(s) => println!("Seriennummer   {}… ({} Zeichen)", &s[..s.len().min(4)], s.len()),
        Err(e) => println!("Seriennummer   — ({e})"),
    }
    report("Helligkeit", dev.brightness());
    report("Lautstärke", dev.volume());
    report("Duty-Cycle", dev.duty_cycle());
    match dev.display_mode() {
        Ok(m) => println!("Anzeigemodus   {m:?}"),
        Err(e) => println!("Anzeigemodus   — ({e})"),
    }
    match dev.worn() {
        Ok(w) => println!("Getragen       {}", if w { "ja" } else { "nein" }),
        Err(e) => println!("Getragen       — ({e})"),
    }
}

fn report(label: &str, v: viture_v2::Result<u8>) {
    match v {
        Ok(v) => println!("{label:<14} {v}"),
        Err(e) => println!("{label:<14} — ({e})"),
    }
}

/// Prüft die Syscall-Schicht und die Transportverfügbarkeit — ohne Gerät.
/// Nützlich, um eine neue Plattform zu vermessen, bevor Hardware im Spiel ist.
fn probe() {
    println!("Architektur      {}", std::env::consts::ARCH);

    // ppoll: stdin ohne Wartezeit abfragen. Der Aufruf muss zurückkehren,
    // nicht das Ergebnis ist interessant.
    match viture_v2::sys::wait_readable(0, 0) {
        Ok(r) => println!("ppoll            ok (stdin lesbar: {r})"),
        Err(e) => println!("ppoll            FEHLER: {e}"),
    }

    // io_uring: auf Android sperrt die App-Sandbox das per seccomp.
    let mut params = [0u8; 120];
    match unsafe { viture_v2::sys::io_uring_setup(8, params.as_mut_ptr()) } {
        Ok(fd) => {
            println!("io_uring_setup   ok (fd {fd})");
            let _ = viture_v2::sys::close(fd);
        }
        Err(e) => println!("io_uring_setup   nicht verfügbar: {e}"),
    }

    match hidraw::find_fd(VID, PID_PRO2) {
        Ok(fd) => {
            println!("hidraw-Knoten    gefunden");
            let _ = viture_v2::sys::close(fd);
        }
        Err(e) => println!("hidraw-Knoten    {e}"),
    }
    match usbfs::find_fd(VID, PID_PRO2) {
        Ok(fd) => {
            println!("usbfs-Knoten     gefunden");
            let _ = viture_v2::sys::close(fd);
        }
        Err(e) => println!("usbfs-Knoten     {e}"),
    }
}

/// Öffnet den hidraw-Knoten oder bricht ab.
fn hid_fd() -> i32 {
    hidraw::find_fd(VID, PID_PRO2).unwrap_or_else(|e| {
        eprintln!("Gerät nicht erreichbar: {e}");
        std::process::exit(1);
    })
}

fn run(backend: &str, streams: Streams, secs: u64, rate: Rate) {
    match backend {
        "hidraw" => {
            let dev = Device::new(Hidraw::new(hid_fd()));
            pump(dev, streams, secs, rate, |d| format!("{} Syscalls", d.transport_mut().syscalls));
        }
        "usbfs" => {
            // Android-Pfad: dort kommt der Deskriptor aus `termux-usb`.
            match usbfs::find_fd(VID, PID_PRO2).and_then(|fd| Usbfs::new(fd, 0)) {
                Ok(u) => {
                    let dev = Device::new(u);
                    pump(dev, streams, secs, rate, |d| {
                        let s = d.transport_mut().stats;
                        format!(
                            "{} URB-Submits, {} Reaps, {} Wartezyklen, größter Batch {}",
                            s.submits, s.reaps, s.waits, s.max_batch
                        )
                    });
                }
                Err(e) => {
                    eprintln!("usbfs nicht verfügbar: {e}");
                    std::process::exit(4);
                }
            }
        }
        _ => match Uring::new(hid_fd()) {
            Ok(u) => {
                let dev = Device::new(u);
                pump(dev, streams, secs, rate, |d| {
                    let s = d.transport_mut().stats;
                    format!(
                        "{} io_uring_enter, {} Completions, {} Batches, größter Batch {}",
                        s.enters, s.completions, s.batches, s.max_batch
                    )
                });
            }
            Err(e) => {
                eprintln!("io_uring nicht verfügbar ({e}) — weiche auf hidraw aus");
                let dev = Device::new(Hidraw::new(hid_fd()));
                pump(dev, streams, secs, rate, |d| format!("{} Syscalls", d.transport_mut().syscalls));
            }
        },
    }
}

/// Lese-Thread füllt den Ring, Hauptthread holt im Batch ab.
fn pump<T, F>(mut dev: Device<T>, streams: Streams, secs: u64, rate: Rate, summary: F)
where
    T: Transport + Send + 'static,
    F: FnOnce(&mut Device<T>) -> String,
{
    if let Err(e) = dev.set_imu(streams, rate) {
        eprintln!("IMU-Start fehlgeschlagen: {e}");
        std::process::exit(3);
    }

    let ring: Arc<Ring<Sample, RING_CAP>> = Arc::new(Ring::new());
    let stop = Arc::new(AtomicBool::new(false));
    let cpu_start = cpu_ns();

    let consumer = std::thread::current();
    let reader = {
        let ring = Arc::clone(&ring);
        let stop = Arc::clone(&stop);
        std::thread::spawn(move || {
            // 50 ms Wartefenster, damit das Stoppsignal zeitnah greift. Das ist
            // kein Polling: der Thread schläft im Kernel, bis Daten kommen.
            // Benachrichtigung koaleszieren: erst wecken, wenn genug im Ring
            // liegt. Das ist der Batch-Gedanke aus io_uring, eine Ebene höher.
            // 1 = wecken pro Ereignis (niedrigste Latenz).
            let coalesce: usize = std::env::var("VITURE_COALESCE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1);
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
                        eprintln!("Lesefehler: {e}");
                        break;
                    }
                }
            }
            let _ = dev.set_imu(Streams::OFF, Rate::Hz120);
            dev
        })
    };

    // Verbraucher: 60-mal je Sekunde alles abholen, was da ist.
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
            // Kein Sleep: der Thread parkt und wird vom Erzeuger geweckt.
            // Das Zeitfenster dient nur dazu, das Laufzeitende zu bemerken.
            std::thread::park_timeout(Duration::from_millis(20));
            continue;
        }
        if last_print.elapsed() >= Duration::from_secs(1) {
            last_print = Instant::now();
            match newest {
                Some(Sample::Pose(p)) => {
                    let [r, pi, y] = p.euler_deg();
                    println!("pose  roll={r:7.2} pitch={pi:7.2} yaw={y:7.2}   (Batch {n})");
                }
                Some(Sample::Raw(r)) => {
                    let a = r.accel;
                    let mag = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
                    println!(
                        "raw   gyro=[{:8.4}{:8.4}{:8.4}] |a|={mag:.3}g   (Batch {n})",
                        r.gyro[0], r.gyro[1], r.gyro[2]
                    );
                }
                None => {}
            }
        }
    }

    stop.store(true, Ordering::Relaxed);
    let mut dev = reader.join().expect("Lese-Thread");
    let cpu = cpu_ns().saturating_sub(cpu_start);
    let dt = start.elapsed().as_secs_f64();

    println!("\n── {dt:.1}s ──");
    println!("Ereignisse       {total}  ({:.0} Hz)", total as f64 / dt);
    println!("Ring             {batches} Abholungen, größte {biggest}, {} verworfen", ring.dropped());
    println!("Kernelpfad       {}", summary(&mut dev));
    println!("CPU-Zeit         {:.1} ms  ({:.2} % einer Kernlast)", cpu as f64 / 1e6, cpu as f64 / (dt * 1e9) * 100.0);
    if total > 0 {
        println!("CPU je Ereignis  {:.1} µs", cpu as f64 / 1e3 / total as f64);
    }
}

/// CPU-Zeit dieses Prozesses in Nanosekunden — Feld 1 aus `/proc/self/schedstat`.
fn cpu_ns() -> u64 {
    std::fs::read_to_string("/proc/self/schedstat")
        .ok()
        .and_then(|s| s.split_whitespace().next().and_then(|v| v.parse().ok()))
        .unwrap_or(0)
}
