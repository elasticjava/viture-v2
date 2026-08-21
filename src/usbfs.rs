//! usbfs transport — the same ring idea as io_uring, but permitted everywhere.
//!
//! `USBDEVFS_SUBMITURB` queues transfers, `USBDEVFS_REAPURBNDELAY` collects
//! finished ones. That is a submission queue and a completion queue, expressed
//! as ioctls instead of shared memory. `DEPTH` interrupt-IN URBs sit in the
//! kernel at all times; every completion carries its own buffer and is
//! re-queued immediately.
//!
//! This is the path for **Android/Termux**: `termux-usb -r -e prog
//! /dev/bus/usb/…` hands over exactly this descriptor, and io_uring is blocked
//! by seccomp there — verified on a Pixel 9 running Android 17, where
//! `io_uring_setup` returns EPERM even in the `shell` domain.
//!
//! Note that the kernel driver (`usbhid`) has to be detached from the
//! interface. It is handed back on drop.

use crate::sys;
use crate::{Error, Result, Transport, FRAME_MAX};

/// Read URBs queued at once.
pub const DEPTH: usize = 16;

const EP_IN: u8 = 0x81;
const EP_OUT: u8 = 0x01;
const URB_TYPE_INTERRUPT: u8 = 1;

const USBDEVFS_TYPE: u8 = b'U';

/// `struct usbdevfs_urb` in its 64-bit layout: 56 bytes.
#[repr(C)]
#[derive(Clone, Copy)]
struct Urb {
    typ: u8,
    endpoint: u8,
    _pad0: [u8; 2],
    status: i32,
    flags: u32,
    _pad1: u32,
    buffer: *mut u8,
    buffer_length: i32,
    actual_length: i32,
    start_frame: i32,
    number_of_packets: i32,
    error_count: i32,
    signr: u32,
    usercontext: *mut u8,
}

/// `struct usbdevfs_ioctl` — the carrier for DISCONNECT/CONNECT.
#[repr(C)]
struct UsbdevfsIoctl {
    ifno: i32,
    ioctl_code: i32,
    data: *mut u8,
}

fn submiturb() -> usize {
    sys::ioc(sys::IOC_READ, USBDEVFS_TYPE, 10, core::mem::size_of::<Urb>())
}
fn reapurbndelay() -> usize {
    sys::ioc(sys::IOC_WRITE, USBDEVFS_TYPE, 13, core::mem::size_of::<usize>())
}
fn discardurb() -> usize {
    sys::ioc(sys::IOC_NONE, USBDEVFS_TYPE, 11, 0)
}
fn claiminterface() -> usize {
    sys::ioc(sys::IOC_READ, USBDEVFS_TYPE, 15, 4)
}
fn releaseinterface() -> usize {
    sys::ioc(sys::IOC_READ, USBDEVFS_TYPE, 16, 4)
}
fn usbdevfs_ioctl() -> usize {
    sys::ioc(
        sys::IOC_READ | sys::IOC_WRITE,
        USBDEVFS_TYPE,
        18,
        core::mem::size_of::<UsbdevfsIoctl>(),
    )
}
const IOCTL_DISCONNECT: i32 = 0x5516; // _IO('U', 22)
const IOCTL_CONNECT: i32 = 0x5517; // _IO('U', 23)

#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    pub submits: u64,
    pub reaps: u64,
    pub waits: u64,
    pub max_batch: u32,
}

/// Finds the usbfs node for a VID:PID and opens it.
///
/// Not needed under Termux: there the descriptor arrives ready-made from
/// `termux-usb -r -e`, because `/dev/bus/usb` is not traversable for apps.
pub fn find_fd(vid: u16, pid: u16) -> Result<i32> {
    use std::fs::{read_dir, read_to_string, OpenOptions};
    use std::os::fd::IntoRawFd;

    for entry in read_dir("/sys/bus/usb/devices")? {
        let dir = entry?.path();
        let hex = |name: &str| {
            read_to_string(dir.join(name)).ok().and_then(|s| u16::from_str_radix(s.trim(), 16).ok())
        };
        if hex("idVendor") != Some(vid) || hex("idProduct") != Some(pid) {
            continue;
        }
        let dec = |name: &str| {
            read_to_string(dir.join(name)).ok().and_then(|s| s.trim().parse::<u32>().ok())
        };
        let (Some(bus), Some(dev)) = (dec("busnum"), dec("devnum")) else { continue };
        let path = format!("/dev/bus/usb/{bus:03}/{dev:03}");
        let file = OpenOptions::new().read(true).write(true).open(&path)?;
        return Ok(file.into_raw_fd());
    }
    Err(Error::Io(std::io::Error::new(
        std::io::ErrorKind::NotFound,
        "no usbfs node with matching VID:PID",
    )))
}

pub struct Usbfs {
    fd: i32,
    ifno: i32,
    /// Pinned buffers; the URBs point at them, so they must not move.
    bufs: Box<[[u8; FRAME_MAX]; DEPTH]>,
    urbs: Box<[Urb; DEPTH]>,
    out_buf: Box<[u8; FRAME_MAX]>,
    out_urb: Box<Urb>,
    out_pending: bool,
    reattach: bool,
    pub stats: Stats,
}

// SAFETY: as with the io_uring transport — moved as a whole into exactly one
// thread, never shared.
unsafe impl Send for Usbfs {}

impl Usbfs {
    /// Takes an open usbfs descriptor, e.g. the one from `termux-usb`.
    pub fn new(fd: i32, ifno: i32) -> Result<Self> {
        // Detach the kernel driver, otherwise CLAIMINTERFACE refuses.
        let mut req =
            UsbdevfsIoctl { ifno, ioctl_code: IOCTL_DISCONNECT, data: core::ptr::null_mut() };
        let reattach =
            sys::ioctl(fd, usbdevfs_ioctl(), &mut req as *mut UsbdevfsIoctl as usize).is_ok();

        sys::ioctl(fd, claiminterface(), &ifno as *const i32 as usize).map_err(Error::Io)?;

        let mut u = Usbfs {
            fd,
            ifno,
            bufs: Box::new([[0u8; FRAME_MAX]; DEPTH]),
            urbs: Box::new(
                [Urb {
                    typ: URB_TYPE_INTERRUPT,
                    endpoint: EP_IN,
                    _pad0: [0; 2],
                    status: 0,
                    flags: 0,
                    _pad1: 0,
                    buffer: core::ptr::null_mut(),
                    buffer_length: FRAME_MAX as i32,
                    actual_length: 0,
                    start_frame: 0,
                    number_of_packets: 0,
                    error_count: 0,
                    signr: 0,
                    usercontext: core::ptr::null_mut(),
                }; DEPTH],
            ),
            out_buf: Box::new([0u8; FRAME_MAX]),
            out_urb: Box::new(Urb {
                typ: URB_TYPE_INTERRUPT,
                endpoint: EP_OUT,
                _pad0: [0; 2],
                status: 0,
                flags: 0,
                _pad1: 0,
                buffer: core::ptr::null_mut(),
                buffer_length: 0,
                actual_length: 0,
                start_frame: 0,
                number_of_packets: 0,
                error_count: 0,
                signr: 0,
                usercontext: usize::MAX as *mut u8,
            }),
            out_pending: false,
            reattach,
            stats: Stats::default(),
        };

        for i in 0..DEPTH {
            u.submit(i)?;
        }
        Ok(u)
    }

    /// Queues the read URB for buffer `idx`.
    #[inline]
    fn submit(&mut self, idx: usize) -> Result<()> {
        let buf = self.bufs[idx].as_mut_ptr();
        let urb = &mut self.urbs[idx];
        urb.buffer = buf;
        urb.buffer_length = FRAME_MAX as i32;
        urb.actual_length = 0;
        urb.status = 0;
        urb.usercontext = idx as *mut u8;
        self.stats.submits += 1;
        sys::ioctl(self.fd, submiturb(), urb as *mut Urb as usize).map_err(Error::Io)?;
        Ok(())
    }

    /// Collects one finished URB without blocking.
    #[inline]
    fn reap(&mut self) -> Result<Option<*mut Urb>> {
        let mut ptr: usize = 0;
        match sys::ioctl(self.fd, reapurbndelay(), &mut ptr as *mut usize as usize) {
            Ok(_) => {
                self.stats.reaps += 1;
                Ok(Some(ptr as *mut Urb))
            }
            // EAGAIN: nothing finished right now.
            Err(e) if e.raw_os_error() == Some(11) => Ok(None),
            Err(e) => Err(Error::Io(e)),
        }
    }
}

impl Drop for Usbfs {
    fn drop(&mut self) {
        for i in 0..DEPTH {
            let urb = &mut self.urbs[i] as *mut Urb;
            let _ = sys::ioctl(self.fd, discardurb(), urb as usize);
        }
        let _ = sys::ioctl(self.fd, releaseinterface(), &self.ifno as *const i32 as usize);
        if self.reattach {
            let mut req = UsbdevfsIoctl {
                ifno: self.ifno,
                ioctl_code: IOCTL_CONNECT,
                data: core::ptr::null_mut(),
            };
            let _ = sys::ioctl(self.fd, usbdevfs_ioctl(), &mut req as *mut UsbdevfsIoctl as usize);
        }
        let _ = sys::close(self.fd);
    }
}

impl Transport for Usbfs {
    fn send(&mut self, frame: &[u8]) -> Result<()> {
        // On the wire the frame goes out without a hidraw report-ID byte.
        self.out_buf[..frame.len()].copy_from_slice(frame);
        let ptr = self.out_buf.as_mut_ptr();
        self.out_urb.buffer = ptr;
        self.out_urb.buffer_length = frame.len() as i32;
        self.out_urb.status = 0;
        self.stats.submits += 1;
        sys::ioctl(self.fd, submiturb(), &mut *self.out_urb as *mut Urb as usize)
            .map_err(Error::Io)?;
        self.out_pending = true;
        Ok(())
    }

    fn recv(&mut self, out: &mut [u8; FRAME_MAX], timeout_ns: u64) -> Result<usize> {
        let mut batch = 0u32;
        loop {
            while let Some(p) = self.reap()? {
                batch += 1;
                self.stats.max_batch = self.stats.max_batch.max(batch);
                // SAFETY: the pointer comes out of our own URB array.
                let (ctx, len, status) =
                    unsafe { ((*p).usercontext as usize, (*p).actual_length, (*p).status) };
                if ctx == usize::MAX {
                    self.out_pending = false; // write URB, nothing to deliver
                    continue;
                }
                let idx = ctx;
                if status == 0 && len > 0 {
                    let n = (len as usize).min(FRAME_MAX);
                    out[..n].copy_from_slice(&self.bufs[idx][..n]);
                    self.submit(idx)?;
                    return Ok(n);
                }
                self.submit(idx)?;
            }
            if timeout_ns == 0 {
                return Ok(0);
            }
            self.stats.waits += 1;
            // usbfs signals finished URBs as POLLOUT.
            if !sys::wait_events(self.fd, sys::POLLOUT, timeout_ns).map_err(Error::Io)? {
                return Ok(0);
            }
        }
    }
}
