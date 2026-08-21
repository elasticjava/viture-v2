//! Blockierender `hidraw`-Transport — der Fallback dort, wo io_uring nicht
//! erlaubt ist (Android-App-Sandbox sperrt `io_uring_setup` per seccomp).
//!
//! Gewartet wird mit `ppoll`, nicht mit einer Schleife: zwei Syscalls pro
//! Ereignis, dazwischen schläft der Thread im Kernel. `hidraw` hält selbst
//! einen Ringpuffer über mehrere Reports, es geht also nichts verloren,
//! solange der Verbraucher im Mittel Schritt hält.

use std::io::ErrorKind;

use crate::sys;
use crate::{Error, Result, Transport, FRAME_MAX};

/// Öffnet einen hidraw-Knoten und liefert den rohen Deskriptor.
pub fn open_fd(path: &str) -> Result<i32> {
    use std::fs::OpenOptions;
    use std::os::fd::IntoRawFd;
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    Ok(file.into_raw_fd())
}

/// Sucht den hidraw-Knoten zur gesuchten VID:PID.
pub fn find_fd(vid: u16, pid: u16) -> Result<i32> {
    let needle = format!("{vid:08X}:{pid:08X}");
    for entry in std::fs::read_dir("/sys/class/hidraw")? {
        let name = entry?.file_name();
        let name = name.to_string_lossy().into_owned();
        let uevent = format!("/sys/class/hidraw/{name}/device/uevent");
        let Ok(text) = std::fs::read_to_string(&uevent) else { continue };
        if text.lines().any(|l| l.starts_with("HID_ID=") && l.ends_with(&needle)) {
            return open_fd(&format!("/dev/{name}"));
        }
    }
    Err(Error::Io(std::io::Error::new(
        ErrorKind::NotFound,
        "kein hidraw-Knoten mit passender VID:PID",
    )))
}

pub struct Hidraw {
    fd: i32,
    /// Syscall-Zähler, um gegen den io_uring-Pfad vergleichen zu können.
    pub syscalls: u64,
}

impl Hidraw {
    /// Übernimmt den Deskriptor; schließt ihn beim Verwerfen.
    pub fn new(fd: i32) -> Self {
        Hidraw { fd, syscalls: 0 }
    }

    pub fn raw_fd(&self) -> i32 {
        self.fd
    }
}

impl Drop for Hidraw {
    fn drop(&mut self) {
        let _ = sys::close(self.fd);
    }
}

impl Transport for Hidraw {
    #[inline]
    fn send(&mut self, frame: &[u8]) -> Result<()> {
        // hidraw erwartet ein führendes Report-ID-Byte; das Gerät benutzt keine
        // nummerierten Reports, also 0.
        let mut buf = [0u8; FRAME_MAX + 1];
        buf[1..1 + frame.len()].copy_from_slice(frame);
        self.syscalls += 1;
        sys::write(self.fd, &buf[..1 + frame.len()]).map_err(Error::Io)?;
        Ok(())
    }

    #[inline]
    fn recv(&mut self, out: &mut [u8; FRAME_MAX], timeout_ns: u64) -> Result<usize> {
        self.syscalls += 1;
        if !sys::wait_readable(self.fd, timeout_ns).map_err(Error::Io)? {
            return Ok(0);
        }
        self.syscalls += 1;
        match sys::read(self.fd, out) {
            Ok(n) => Ok(n),
            Err(e) if e.kind() == ErrorKind::WouldBlock => Ok(0),
            Err(e) => Err(Error::Io(e)),
        }
    }
}
