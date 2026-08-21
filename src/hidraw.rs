//! Blocking `hidraw` transport — the fallback wherever io_uring is not allowed.
//!
//! Waiting is done with `ppoll`, not with a loop: two syscalls per event, and
//! the thread sleeps in the kernel in between. `hidraw` keeps a ring of several
//! reports itself, so nothing is lost as long as the consumer keeps up on
//! average.

use std::io::ErrorKind;

use crate::sys;
use crate::{Error, Result, Transport, FRAME_MAX};

/// Opens a hidraw node and returns the raw descriptor.
pub fn open_fd(path: &str) -> Result<i32> {
    use std::fs::OpenOptions;
    use std::os::fd::IntoRawFd;
    let file = OpenOptions::new().read(true).write(true).open(path)?;
    Ok(file.into_raw_fd())
}

/// Finds the hidraw node matching the given VID:PID.
pub fn find_fd(vid: u16, pid: u16) -> Result<i32> {
    let needle = format!("{vid:08X}:{pid:08X}");
    for entry in std::fs::read_dir("/sys/class/hidraw")? {
        let name = entry?.file_name();
        let name = name.to_string_lossy().into_owned();
        let uevent = format!("/sys/class/hidraw/{name}/device/uevent");
        let Ok(text) = std::fs::read_to_string(&uevent) else {
            continue;
        };
        if text
            .lines()
            .any(|l| l.starts_with("HID_ID=") && l.ends_with(&needle))
        {
            return open_fd(&format!("/dev/{name}"));
        }
    }
    Err(Error::Io(std::io::Error::new(
        ErrorKind::NotFound,
        "no hidraw node with matching VID:PID",
    )))
}

pub struct Hidraw {
    fd: i32,
    /// Syscall counter, so this path can be compared against io_uring.
    pub syscalls: u64,
}

impl Hidraw {
    /// Takes ownership of the descriptor and closes it on drop.
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
        // hidraw expects a leading report-ID byte; this device does not use
        // numbered reports, so it is zero.
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
