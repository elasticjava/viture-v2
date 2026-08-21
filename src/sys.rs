//! Raw Linux syscalls, no `libc`.
//!
//! Only the handful of calls this driver needs. The `io_uring_*` numbers are
//! identical on x86-64 and aarch64 (they live in the generic range); the
//! classic ones differ.

#![allow(dead_code)]

use core::arch::asm;

// ---- Syscall numbers -------------------------------------------------------

#[cfg(target_arch = "x86_64")]
mod nr {
    pub const READ: usize = 0;
    pub const WRITE: usize = 1;
    pub const CLOSE: usize = 3;
    pub const IOCTL: usize = 16;
    pub const MMAP: usize = 9;
    pub const MUNMAP: usize = 11;
    pub const PPOLL: usize = 271;
}

#[cfg(target_arch = "aarch64")]
mod nr {
    pub const READ: usize = 63;
    pub const WRITE: usize = 64;
    pub const CLOSE: usize = 57;
    pub const IOCTL: usize = 29;
    pub const MMAP: usize = 222;
    pub const MUNMAP: usize = 215;
    pub const PPOLL: usize = 73;
}

/// Same on both architectures.
pub const NR_IO_URING_SETUP: usize = 425;
pub const NR_IO_URING_ENTER: usize = 426;
pub const NR_IO_URING_REGISTER: usize = 427;

// ---- Syscall entry ---------------------------------------------------------

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

/// Turns a negative syscall return value into an `io::Error`.
#[inline]
fn wrap(ret: isize) -> std::io::Result<usize> {
    if ret < 0 {
        Err(std::io::Error::from_raw_os_error(-ret as i32))
    } else {
        Ok(ret as usize)
    }
}

// ---- Thin wrappers ---------------------------------------------------------

pub const PROT_READ: usize = 1;
pub const PROT_WRITE: usize = 2;
pub const MAP_SHARED: usize = 1;
pub const MAP_POPULATE: usize = 0x8000;

/// # Safety
/// Maps kernel memory; the caller owns the returned range until `munmap`.
#[inline]
pub unsafe fn mmap(
    len: usize,
    prot: usize,
    flags: usize,
    fd: i32,
    off: u64,
) -> std::io::Result<*mut u8> {
    let r = syscall6(nr::MMAP, 0, len, prot, flags, fd as usize, off as usize);
    if r < 0 {
        Err(std::io::Error::from_raw_os_error(-r as i32))
    } else {
        Ok(r as *mut u8)
    }
}

/// # Safety
/// `addr`/`len` must describe a mapping obtained from [`mmap`].
#[inline]
pub unsafe fn munmap(addr: *mut u8, len: usize) -> std::io::Result<()> {
    wrap(syscall6(nr::MUNMAP, addr as usize, len, 0, 0, 0, 0)).map(|_| ())
}

#[inline]
pub fn read(fd: i32, buf: &mut [u8]) -> std::io::Result<usize> {
    unsafe {
        wrap(syscall6(
            nr::READ,
            fd as usize,
            buf.as_mut_ptr() as usize,
            buf.len(),
            0,
            0,
            0,
        ))
    }
}

#[inline]
pub fn write(fd: i32, buf: &[u8]) -> std::io::Result<usize> {
    unsafe {
        wrap(syscall6(
            nr::WRITE,
            fd as usize,
            buf.as_ptr() as usize,
            buf.len(),
            0,
            0,
            0,
        ))
    }
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
/// usbfs reports completed URBs as "writable".
pub const POLLOUT: i16 = 0x004;

/// Waits for readability. `timeout_ns == u64::MAX` waits forever.
pub fn wait_readable(fd: i32, timeout_ns: u64) -> std::io::Result<bool> {
    wait_events(fd, POLLIN, timeout_ns)
}

/// Waits for arbitrary poll events.
pub fn wait_events(fd: i32, events: i16, timeout_ns: u64) -> std::io::Result<bool> {
    let mut pfd = PollFd {
        fd,
        events,
        revents: 0,
    };
    let ts = Timespec {
        sec: (timeout_ns / 1_000_000_000) as i64,
        nsec: (timeout_ns % 1_000_000_000) as i64,
    };
    let tsp = if timeout_ns == u64::MAX {
        0
    } else {
        &ts as *const Timespec as usize
    };
    loop {
        let r = unsafe { syscall6(nr::PPOLL, &mut pfd as *mut PollFd as usize, 1, tsp, 0, 8, 0) };
        if r == -4 {
            continue; // EINTR
        }
        return wrap(r).map(|n| n > 0 && (pfd.revents & events) != 0);
    }
}

#[inline]
pub fn ioctl(fd: i32, request: usize, arg: usize) -> std::io::Result<usize> {
    loop {
        let r = unsafe { syscall6(nr::IOCTL, fd as usize, request, arg, 0, 0, 0) };
        if r == -4 {
            continue; // EINTR
        }
        return wrap(r);
    }
}

/// Builds an `_IOC` number the way `linux/ioctl.h` does.
pub const fn ioc(dir: usize, typ: u8, nr: u8, size: usize) -> usize {
    (dir << 30) | (size << 16) | ((typ as usize) << 8) | (nr as usize)
}

pub const IOC_NONE: usize = 0;
pub const IOC_WRITE: usize = 1;
pub const IOC_READ: usize = 2;

// ---- Terminal --------------------------------------------------------------

const TCGETS: usize = 0x5401;
const TCSETS: usize = 0x5402;
const TIOCGWINSZ: usize = 0x5413;

const ICANON: u32 = 0o000002;
const ECHO: u32 = 0o000010;
const VTIME: usize = 5;
const VMIN: usize = 6;

/// The kernel's `struct termios` (asm-generic, `NCCS = 19`).
#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Termios {
    pub c_iflag: u32,
    pub c_oflag: u32,
    pub c_cflag: u32,
    pub c_lflag: u32,
    pub c_line: u8,
    pub c_cc: [u8; 19],
}

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct WinSize {
    pub rows: u16,
    pub cols: u16,
    pub xpixel: u16,
    pub ypixel: u16,
}

pub fn tcgetattr(fd: i32) -> std::io::Result<Termios> {
    let mut t = Termios::default();
    ioctl(fd, TCGETS, &mut t as *mut Termios as usize)?;
    Ok(t)
}

pub fn tcsetattr(fd: i32, t: &Termios) -> std::io::Result<()> {
    ioctl(fd, TCSETS, t as *const Termios as usize).map(|_| ())
}

/// Switches the terminal to raw-ish mode: no line buffering, no echo, reads
/// return immediately. Returns the previous settings for restoring later.
pub fn enter_raw_mode(fd: i32) -> std::io::Result<Termios> {
    let previous = tcgetattr(fd)?;
    let mut raw = previous;
    raw.c_lflag &= !(ICANON | ECHO);
    raw.c_cc[VMIN] = 0;
    raw.c_cc[VTIME] = 0;
    tcsetattr(fd, &raw)?;
    Ok(previous)
}

pub fn window_size(fd: i32) -> std::io::Result<WinSize> {
    let mut w = WinSize::default();
    ioctl(fd, TIOCGWINSZ, &mut w as *mut WinSize as usize)?;
    Ok(w)
}

// ---- io_uring --------------------------------------------------------------

/// # Safety
/// `params` must point at a writable `io_uring_params` (120 bytes).
#[inline]
pub unsafe fn io_uring_setup(entries: u32, params: *mut u8) -> std::io::Result<i32> {
    wrap(syscall6(
        NR_IO_URING_SETUP,
        entries as usize,
        params as usize,
        0,
        0,
        0,
        0,
    ))
    .map(|v| v as i32)
}

/// # Safety
/// `fd` must be a ring from [`io_uring_setup`].
#[inline]
pub unsafe fn io_uring_enter(
    fd: i32,
    to_submit: u32,
    min_complete: u32,
    flags: u32,
) -> std::io::Result<usize> {
    io_uring_enter_arg(fd, to_submit, min_complete, flags, 0, 0)
}

/// Variant using `IORING_ENTER_EXT_ARG`: `arg` points at an
/// `io_uring_getevents_arg` carrying the timeout, so no SQE is spent on it.
/// # Safety
/// `fd` must be a ring from [`io_uring_setup`]; `arg`/`argsz` must describe a
/// valid `io_uring_getevents_arg` or both be zero.
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
            -4 => continue,      // EINTR
            -62 => return Ok(0), // ETIME: timed out, nothing completed
            _ => return wrap(r),
        }
    }
}
