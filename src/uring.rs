//! io_uring-Transport: ein Ring vorab bewaffneter Lesepuffer.
//!
//! Prinzip: `DEPTH` Reads stehen dauerhaft beim Kernel. Kommt ein Report,
//! landet er in dem Puffer, der zu der Completion gehört; wir reichen ihn nach
//! oben und bewaffnen denselben Puffer sofort neu. Zwischen zwei Reports wird
//! nie gepollt und nie geschlafen — `io_uring_enter` legt den Thread schlafen,
//! bis der Kernel etwas hat.
//!
//! Auf Android ist `io_uring_setup` per seccomp gesperrt (ENOSYS/EPERM). Dort
//! greift der blockierende Transport in [`crate::hidraw`], der dieselbe
//! Semantik ohne Ring liefert.

use core::sync::atomic::{fence, Ordering};

use crate::sys;
use crate::{Error, Result, Transport, FRAME_MAX};

/// Anzahl gleichzeitig beim Kernel liegender Reads.
pub const DEPTH: usize = 16;

const IORING_OP_READ: u8 = 22;
const IORING_ENTER_GETEVENTS: u32 = 1 << 0;
const IORING_ENTER_EXT_ARG: u32 = 1 << 3;
const IORING_FEAT_SINGLE_MMAP: u32 = 1 << 0;
const IORING_FEAT_EXT_ARG: u32 = 1 << 8;
const IORING_OFF_SQ_RING: u64 = 0;
const IORING_OFF_CQ_RING: u64 = 0x800_0000;
const IORING_OFF_SQES: u64 = 0x1000_0000;
const SQE_SIZE: usize = 64;
const CQE_SIZE: usize = 16;

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SqOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    flags: u32,
    dropped: u32,
    array: u32,
    resv1: u32,
    user_addr: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct CqOffsets {
    head: u32,
    tail: u32,
    ring_mask: u32,
    ring_entries: u32,
    overflow: u32,
    cqes: u32,
    flags: u32,
    resv1: u32,
    user_addr: u64,
}

#[repr(C)]
#[derive(Default, Clone, Copy)]
struct Params {
    sq_entries: u32,
    cq_entries: u32,
    flags: u32,
    sq_thread_cpu: u32,
    sq_thread_idle: u32,
    features: u32,
    wq_fd: u32,
    resv: [u32; 3],
    sq_off: SqOffsets,
    cq_off: CqOffsets,
}

#[repr(C)]
struct GetEventsArg {
    sigmask: u64,
    sigmask_sz: u32,
    pad: u32,
    ts: u64,
}

/// Zählt, was tatsächlich in den Kernel geht — die interessante Kennzahl.
#[derive(Default, Clone, Copy, Debug)]
pub struct Stats {
    pub enters: u64,
    pub completions: u64,
    pub batches: u64,
    pub max_batch: u32,
}

pub struct Uring {
    ring_fd: i32,
    dev_fd: i32,
    sq_map: (*mut u8, usize),
    cq_map: Option<(*mut u8, usize)>,
    sqe_map: (*mut u8, usize),

    sq_tail: *mut u32,
    sq_mask: u32,
    sq_array: *mut u32,
    sq_local_tail: u32,

    cq_head: *mut u32,
    cq_tail: *const u32,
    cq_mask: u32,
    cqes: *const u8,

    bufs: Box<[[u8; FRAME_MAX]; DEPTH]>,
    to_submit: u32,
    ext_arg: bool,
    pub stats: Stats,
}

impl Uring {
    /// Legt den Ring an und bewaffnet alle Lesepuffer.
    ///
    /// `dev_fd` muss offen bleiben, solange der Ring lebt.
    pub fn new(dev_fd: i32) -> Result<Self> {
        // Bewusst ohne SINGLE_ISSUER/DEFER_TASKRUN: die binden den Ring an
        // genau eine Task, unser Kommandopfad läuft aber vor dem Umzug in den
        // Lese-Thread. Der Gewinn wäre hier ohnehin klein.
        let mut p = Params::default();
        let entries = (DEPTH * 2) as u32;
        let ring_fd = unsafe {
            sys::io_uring_setup(entries, &mut p as *mut Params as *mut u8).map_err(Error::Io)?
        };

        let mut sq_len = p.sq_off.array as usize + p.sq_entries as usize * 4;
        let cq_len = p.cq_off.cqes as usize + p.cq_entries as usize * CQE_SIZE;
        let single = p.features & IORING_FEAT_SINGLE_MMAP != 0;
        if single && cq_len > sq_len {
            sq_len = cq_len;
        }

        let prot = sys::PROT_READ | sys::PROT_WRITE;
        let flags = sys::MAP_SHARED | sys::MAP_POPULATE;
        let sq_ptr = unsafe { sys::mmap(sq_len, prot, flags, ring_fd, IORING_OFF_SQ_RING) }
            .map_err(Error::Io)?;
        let cq_map = if single {
            None
        } else {
            let ptr = unsafe { sys::mmap(cq_len, prot, flags, ring_fd, IORING_OFF_CQ_RING) }
                .map_err(Error::Io)?;
            Some((ptr, cq_len))
        };
        let cq_ptr = cq_map.map(|(p, _)| p).unwrap_or(sq_ptr);

        let sqe_len = p.sq_entries as usize * SQE_SIZE;
        let sqe_ptr = unsafe { sys::mmap(sqe_len, prot, flags, ring_fd, IORING_OFF_SQES) }
            .map_err(Error::Io)?;

        // SAFETY: Offsets stammen vom Kernel und liegen innerhalb der Mappings.
        let mut u = unsafe {
            Uring {
                ring_fd,
                dev_fd,
                sq_map: (sq_ptr, sq_len),
                cq_map,
                sqe_map: (sqe_ptr, sqe_len),
                sq_tail: sq_ptr.add(p.sq_off.tail as usize) as *mut u32,
                sq_mask: *(sq_ptr.add(p.sq_off.ring_mask as usize) as *const u32),
                sq_array: sq_ptr.add(p.sq_off.array as usize) as *mut u32,
                sq_local_tail: *(sq_ptr.add(p.sq_off.tail as usize) as *const u32),
                cq_head: cq_ptr.add(p.cq_off.head as usize) as *mut u32,
                cq_tail: cq_ptr.add(p.cq_off.tail as usize) as *const u32,
                cq_mask: *(cq_ptr.add(p.cq_off.ring_mask as usize) as *const u32),
                cqes: cq_ptr.add(p.cq_off.cqes as usize),
                bufs: Box::new([[0u8; FRAME_MAX]; DEPTH]),
                to_submit: 0,
                ext_arg: p.features & IORING_FEAT_EXT_ARG != 0,
                stats: Stats::default(),
            }
        };

        for i in 0..DEPTH {
            u.arm(i);
        }
        Ok(u)
    }

    /// Legt einen Read für Puffer `idx` in die Submission Queue.
    #[inline]
    fn arm(&mut self, idx: usize) {
        let slot = (self.sq_local_tail & self.sq_mask) as usize;
        // SAFETY: slot liegt per Maske im Ring; SQE-Layout laut linux/io_uring.h.
        unsafe {
            let sqe = self.sqe_map.0.add(slot * SQE_SIZE);
            core::ptr::write_bytes(sqe, 0, SQE_SIZE);
            sqe.write(IORING_OP_READ);
            (sqe.add(4) as *mut i32).write(self.dev_fd);
            (sqe.add(8) as *mut u64).write(u64::MAX); // off = -1: aktuelle Position
            (sqe.add(16) as *mut u64).write(self.bufs[idx].as_mut_ptr() as u64);
            (sqe.add(24) as *mut u32).write(FRAME_MAX as u32);
            (sqe.add(32) as *mut u64).write(idx as u64); // user_data
            self.sq_array.add(slot).write(self.sq_local_tail & self.sq_mask);
        }
        self.sq_local_tail = self.sq_local_tail.wrapping_add(1);
        self.to_submit += 1;
    }

    /// Veröffentlicht den lokalen Tail für den Kernel.
    #[inline]
    fn publish(&mut self) {
        fence(Ordering::Release);
        unsafe { core::ptr::write_volatile(self.sq_tail, self.sq_local_tail) };
    }

    /// Holt eine fertige Completion, ohne zu blockieren.
    #[inline]
    fn reap(&mut self) -> Option<(usize, i32)> {
        let head = unsafe { core::ptr::read_volatile(self.cq_head) };
        let tail = unsafe { core::ptr::read_volatile(self.cq_tail) };
        if head == tail {
            return None;
        }
        fence(Ordering::Acquire);
        let slot = (head & self.cq_mask) as usize;
        // SAFETY: CQE-Layout: u64 user_data, i32 res, u32 flags.
        let (user_data, res) = unsafe {
            let e = self.cqes.add(slot * CQE_SIZE);
            ((e as *const u64).read(), (e.add(8) as *const i32).read())
        };
        fence(Ordering::Release);
        unsafe { core::ptr::write_volatile(self.cq_head, head.wrapping_add(1)) };
        self.stats.completions += 1;
        Some((user_data as usize, res))
    }

    /// Reicht Bewaffnungen ein und wartet auf mindestens eine Completion.
    fn submit_and_wait(&mut self, timeout_ns: u64) -> Result<()> {
        self.publish();
        let submit = core::mem::take(&mut self.to_submit);
        self.stats.enters += 1;

        if timeout_ns == 0 {
            unsafe { sys::io_uring_enter(self.ring_fd, submit, 0, 0) }.map_err(Error::Io)?;
            return Ok(());
        }

        if self.ext_arg && timeout_ns != u64::MAX {
            let ts = sys::Timespec {
                sec: (timeout_ns / 1_000_000_000) as i64,
                nsec: (timeout_ns % 1_000_000_000) as i64,
            };
            let arg = GetEventsArg {
                sigmask: 0,
                sigmask_sz: 8,
                pad: 0,
                ts: &ts as *const sys::Timespec as u64,
            };
            unsafe {
                sys::io_uring_enter_arg(
                    self.ring_fd,
                    submit,
                    1,
                    IORING_ENTER_GETEVENTS | IORING_ENTER_EXT_ARG,
                    &arg as *const GetEventsArg as usize,
                    core::mem::size_of::<GetEventsArg>(),
                )
            }
            .map_err(Error::Io)?;
        } else {
            unsafe { sys::io_uring_enter(self.ring_fd, submit, 1, IORING_ENTER_GETEVENTS) }
                .map_err(Error::Io)?;
        }
        Ok(())
    }
}

// SAFETY: Die Zeiger adressieren Mappings, die exklusiv zu diesem Ring
// gehören. `Uring` wird als Ganzes in genau einen Thread verschoben und dort
// nur von diesem benutzt — geteilt wird er nie.
unsafe impl Send for Uring {}

impl Drop for Uring {
    fn drop(&mut self) {
        unsafe {
            let _ = sys::munmap(self.sqe_map.0, self.sqe_map.1);
            if let Some((p, l)) = self.cq_map {
                let _ = sys::munmap(p, l);
            }
            let _ = sys::munmap(self.sq_map.0, self.sq_map.1);
        }
        let _ = sys::close(self.ring_fd);
    }
}

impl Transport for Uring {
    #[inline]
    fn send(&mut self, frame: &[u8]) -> Result<()> {
        // Schreiben geht direkt: ein einzelner kurzer Write pro Kommando,
        // ein SQE dafür wäre teurer als der Syscall.
        let mut buf = [0u8; FRAME_MAX + 1];
        buf[1..1 + frame.len()].copy_from_slice(frame);
        sys::write(self.dev_fd, &buf[..1 + frame.len()]).map_err(Error::Io)?;
        Ok(())
    }

    fn recv(&mut self, out: &mut [u8; FRAME_MAX], timeout_ns: u64) -> Result<usize> {
        let mut batch = 0u32;
        loop {
            if let Some((idx, res)) = self.reap() {
                batch += 1;
                if batch == 1 {
                    self.stats.batches += 1;
                }
                self.stats.max_batch = self.stats.max_batch.max(batch);
                if res > 0 {
                    let n = (res as usize).min(FRAME_MAX);
                    out[..n].copy_from_slice(&self.bufs[idx][..n]);
                    self.arm(idx);
                    return Ok(n);
                }
                // Fehler oder EOF: Puffer neu bewaffnen und weitersuchen.
                self.arm(idx);
                continue;
            }
            if timeout_ns == 0 {
                if self.to_submit > 0 {
                    self.submit_and_wait(0)?;
                }
                return Ok(0);
            }
            self.submit_and_wait(timeout_ns)?;
            if self.reap_is_empty() {
                return Ok(0); // Zeitüberschreitung
            }
        }
    }
}

impl Uring {
    #[inline]
    fn reap_is_empty(&self) -> bool {
        let head = unsafe { core::ptr::read_volatile(self.cq_head) };
        let tail = unsafe { core::ptr::read_volatile(self.cq_tail) };
        head == tail
    }
}
