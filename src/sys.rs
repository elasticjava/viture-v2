//! Rohe Linux-Syscalls ohne `libc`.
//!
//! Nur die Handvoll Aufrufe, die der Treiber braucht. Die Syscall-Nummern von
//! `io_uring_*` sind auf x86_64 und aarch64 identisch (generischer Bereich),
//! die klassischen unterscheiden sich.

#![allow(dead_code)]

use core::arch::asm;

// ---- Syscall-Nummern -------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod nr {
    pub const READ: usize = 0;
    pub const WRITE: usize = 1;
    pub const CLOSE: usize = 3;
    pub const MMAP: usize = 9;
    pub const MUNMAP: usize = 11;
    pub const PPOLL: usize = 271;
}

#[cfg(target_arch = "aarch64")]
mod nr {
    pub const READ: usize = 63;
    pub const WRITE: usize = 64;
    pub const CLOSE: usize = 57;
    pub const MMAP: usize = 222;
    pub const MUNMAP: usize = 215;
    pub const PPOLL: usize = 73;
}

/// Auf beiden Architekturen gleich.
pub const NR_IO_URING_SETUP: usize = 425;
pub const NR_IO_URING_ENTER: usize = 426;
pub const NR_IO_URING_REGISTER: usize = 427;

// ---- Syscall-Einsprung -----------------------------------------------------

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn syscall6(
    n: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
    a6: usize,
) -> isize {
    let ret: isize;
    asm!(
        "syscall",
        inlateout("rax") n as isize => ret,
        in("rdi") a1,
        in("rsi") a2,
        in("rdx") a3,
        in("r10") a4,
        in("r8") a5,
        in("r9") a6,
        lateout("rcx") _,
        lateout("r11") _,
        options(nostack)
    );
    ret
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn syscall6(
    n: usize,
    a1: usize,
    a2: usize,
    a3: usize,
    a4: usize,
    a5: usize,
    a6: usize,
) -> isize {
    let ret: isize;
    asm!(
        "svc 0",
        in("x8") n,
        inlateout("x0") a1 => ret,
        in("x1") a2,
        in("x2") a3,
        in("x3") a4,
        in("x4") a5,
        in("x5") a6,
        options(nostack)
    );
    ret
}

/// Wandelt einen negativen Syscall-Rückgabewert in einen `io::Error`.
#[inline]
fn wrap(ret: isize) -> std::io::Result<usize> {
    if ret < 0 {
        Err(std::io::Error::from_raw_os_error(-ret as i32))
    } else {
        Ok(ret as usize)
    }
}

// ---- Dünne Hüllen ----------------------------------------------------------

pub const PROT_READ: usize = 1;
pub const PROT_WRITE: usize = 2;
pub const MAP_SHARED: usize = 1;
pub const MAP_POPULATE: usize = 0x8000;

#[inline]
pub unsafe fn mmap(len: usize, prot: usize, flags: usize, fd: i32, off: u64) -> std::io::Result<*mut u8> {
    let r = syscall6(nr::MMAP, 0, len, prot, flags, fd as usize, off as usize);
    if r < 0 {
        Err(std::io::Error::from_raw_os_error(-r as i32))
    } else {
        Ok(r as *mut u8)
    }
}

#[inline]
pub unsafe fn munmap(addr: *mut u8, len: usize) -> std::io::Result<()> {
    wrap(syscall6(nr::MUNMAP, addr as usize, len, 0, 0, 0, 0)).map(|_| ())
}

#[inline]
pub fn read(fd: i32, buf: &mut [u8]) -> std::io::Result<usize> {
    unsafe { wrap(syscall6(nr::READ, fd as usize, buf.as_mut_ptr() as usize, buf.len(), 0, 0, 0)) }
}

#[inline]
pub fn write(fd: i32, buf: &[u8]) -> std::io::Result<usize> {
    unsafe { wrap(syscall6(nr::WRITE, fd as usize, buf.as_ptr() as usize, buf.len(), 0, 0, 0)) }
}

#[inline]
pub fn close(fd: i32) -> std::io::Result<()> {
    unsafe { wrap(syscall6(nr::CLOSE, fd as usize, 0, 0, 0, 0, 0)).map(|_| ()) }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct PollFd {
    pub fd: i32,
    pub events: i16,
    pub revents: i16,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Timespec {
    pub sec: i64,
    pub nsec: i64,
}

pub const POLLIN: i16 = 0x001;
/// usbfs meldet fertige URBs als „beschreibbar".
pub const POLLOUT: i16 = 0x004;

/// Wartet auf Lesbarkeit. `timeout_ns == u64::MAX` wartet unbegrenzt.
/// Liefert `true`, wenn Daten anliegen.
pub fn wait_readable(fd: i32, timeout_ns: u64) -> std::io::Result<bool> {
    wait_events(fd, POLLIN, timeout_ns)
}

/// Wartet auf beliebige Poll-Ereignisse.
pub fn wait_events(fd: i32, events: i16, timeout_ns: u64) -> std::io::Result<bool> {
    let mut pfd = PollFd { fd, events, revents: 0 };
    let ts = Timespec { sec: (timeout_ns / 1_000_000_000) as i64, nsec: (timeout_ns % 1_000_000_000) as i64 };
    let tsp = if timeout_ns == u64::MAX { 0 } else { &ts as *const Timespec as usize };
    loop {
        let r = unsafe {
            syscall6(nr::PPOLL, &mut pfd as *mut PollFd as usize, 1, tsp, 0, 8, 0)
        };
        if r == -4 {
            continue; // EINTR
        }
        return wrap(r).map(|n| n > 0 && (pfd.revents & events) != 0);
    }
}

#[cfg(target_arch = "x86_64")]
const NR_IOCTL: usize = 16;
#[cfg(target_arch = "aarch64")]
const NR_IOCTL: usize = 29;

#[inline]
pub fn ioctl(fd: i32, request: usize, arg: usize) -> std::io::Result<usize> {
    loop {
        let r = unsafe { syscall6(NR_IOCTL, fd as usize, request, arg, 0, 0, 0) };
        if r == -4 {
            continue; // EINTR
        }
        return wrap(r);
    }
}

/// Baut eine `_IOC`-Nummer wie `linux/ioctl.h`.
pub const fn ioc(dir: usize, typ: u8, nr: u8, size: usize) -> usize {
    (dir << 30) | (size << 16) | ((typ as usize) << 8) | (nr as usize)
}

pub const IOC_NONE: usize = 0;
pub const IOC_WRITE: usize = 1;
pub const IOC_READ: usize = 2;

// ---- io_uring --------------------------------------------------------------

#[inline]
pub unsafe fn io_uring_setup(entries: u32, params: *mut u8) -> std::io::Result<i32> {
    wrap(syscall6(NR_IO_URING_SETUP, entries as usize, params as usize, 0, 0, 0, 0)).map(|v| v as i32)
}

#[inline]
pub unsafe fn io_uring_enter(
    fd: i32,
    to_submit: u32,
    min_complete: u32,
    flags: u32,
) -> std::io::Result<usize> {
    io_uring_enter_arg(fd, to_submit, min_complete, flags, 0, 0)
}

/// Variante mit `IORING_ENTER_EXT_ARG`: `arg` zeigt auf `io_uring_getevents_arg`
/// und trägt den Timeout, ohne dafür einen eigenen SQE zu verbrauchen.
#[inline]
pub unsafe fn io_uring_enter_arg(
    fd: i32,
    to_submit: u32,
    min_complete: u32,
    flags: u32,
    arg: usize,
    argsz: usize,
) -> std::io::Result<usize> {
    loop {
        let r = syscall6(
            NR_IO_URING_ENTER,
            fd as usize,
            to_submit as usize,
            min_complete as usize,
            flags as usize,
            arg,
            argsz,
        );
        match r {
            -4 => continue,       // EINTR
            -62 => return Ok(0),  // ETIME: Timeout, keine Completion
            _ => return wrap(r),
        }
    }
}
