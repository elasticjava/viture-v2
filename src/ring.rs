//! Lock-free SPSC ring buffer — the same idea as the io_uring queues, but
//! between two threads inside one address space.
//!
//! One producer (the reader thread), one consumer (renderer, fusion, input).
//! No mutex, no syscall, no allocation while running. Head and tail sit on
//! separate cache lines so the two threads do not steal the line from each
//! other.
//!
//! Capacity is a power of two so masking replaces division.
//!
//! On overflow the **newest** value is dropped and counted. That is
//! deliberate: only the consumer may write `tail`, otherwise there would be two
//! writers on the same index and therefore a data race. Anyone who does not
//! want a backlog should use [`Ring::take_latest`] — then the ring never fills
//! up in the first place. Nothing ever blocks: the reader thread must never
//! wait on the consumer.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU64, Ordering};

#[repr(align(64))]
struct CacheLine<T>(T);

/// Fixed-capacity ring buffer for `Copy` payloads.
pub struct Ring<T: Copy, const N: usize> {
    slots: UnsafeCell<[MaybeUninit<T>; N]>,
    /// Written by the producer, read by the consumer.
    head: CacheLine<AtomicU64>,
    /// Written by the consumer, read by the producer.
    tail: CacheLine<AtomicU64>,
    /// How often the producer had to drop a value.
    dropped: CacheLine<AtomicU64>,
}

// SAFETY: access is disciplined by the indices; exactly one producer and
// exactly one consumer (SPSC).
unsafe impl<T: Copy + Send, const N: usize> Send for Ring<T, N> {}
unsafe impl<T: Copy + Send, const N: usize> Sync for Ring<T, N> {}

impl<T: Copy, const N: usize> Ring<T, N> {
    const MASK: u64 = (N as u64) - 1;

    pub const fn new() -> Self {
        assert!(N.is_power_of_two(), "capacity must be a power of two");
        Ring {
            slots: UnsafeCell::new([MaybeUninit::uninit(); N]),
            head: CacheLine(AtomicU64::new(0)),
            tail: CacheLine(AtomicU64::new(0)),
            dropped: CacheLine(AtomicU64::new(0)),
        }
    }

    /// Producer side. Returns `false` if the ring was full and the value was
    /// dropped. Never waits.
    #[inline]
    pub fn push(&self, value: T) -> bool {
        let head = self.head.0.load(Ordering::Relaxed);
        let tail = self.tail.0.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= N as u64 {
            self.dropped.0.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        // SAFETY: only the producer writes, and only into the slot at head,
        // which the fill check just proved the consumer is not reading.
        unsafe {
            let slots = &mut *self.slots.get();
            slots[(head & Self::MASK) as usize] = MaybeUninit::new(value);
        }
        self.head.0.store(head.wrapping_add(1), Ordering::Release);
        true
    }

    /// Consumer side, one element.
    #[inline]
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.0.load(Ordering::Relaxed);
        let head = self.head.0.load(Ordering::Acquire);
        if tail == head {
            return None;
        }
        // SAFETY: the producer wrote the slot before its release store.
        let value = unsafe {
            let slots = &*self.slots.get();
            slots[(tail & Self::MASK) as usize].assume_init()
        };
        self.tail.0.store(tail.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    /// Consumer side, everything at once — the batching move from io_uring:
    /// one acquire, then n elements, then one release.
    #[inline]
    pub fn drain(&self, mut f: impl FnMut(T)) -> usize {
        let mut tail = self.tail.0.load(Ordering::Relaxed);
        let head = self.head.0.load(Ordering::Acquire);
        let n = head.wrapping_sub(tail);
        if n == 0 {
            return 0;
        }
        // SAFETY: see pop().
        let slots = unsafe { &*self.slots.get() };
        for _ in 0..n {
            f(unsafe { slots[(tail & Self::MASK) as usize].assume_init() });
            tail = tail.wrapping_add(1);
        }
        self.tail.0.store(tail, Ordering::Release);
        n as usize
    }

    /// Only the newest entry; the rest is discarded. For renderers that care
    /// about the current value and nothing else.
    #[inline]
    pub fn take_latest(&self) -> Option<T> {
        let head = self.head.0.load(Ordering::Acquire);
        let tail = self.tail.0.load(Ordering::Relaxed);
        if head == tail {
            return None;
        }
        // SAFETY: see pop().
        let value = unsafe {
            let slots = &*self.slots.get();
            slots[((head - 1) & Self::MASK) as usize].assume_init()
        };
        self.tail.0.store(head, Ordering::Release);
        Some(value)
    }

    #[inline]
    pub fn len(&self) -> usize {
        let head = self.head.0.load(Ordering::Acquire);
        let tail = self.tail.0.load(Ordering::Acquire);
        head.wrapping_sub(tail) as usize
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// How many entries the producer had to drop because nobody collected.
    #[inline]
    pub fn dropped(&self) -> u64 {
        self.dropped.0.load(Ordering::Relaxed)
    }
}

impl<T: Copy, const N: usize> Default for Ring<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_fifo_order() {
        let r: Ring<u32, 8> = Ring::new();
        for i in 0..5 {
            r.push(i);
        }
        let mut out = Vec::new();
        r.drain(|v| out.push(v));
        assert_eq!(out, vec![0, 1, 2, 3, 4]);
        assert!(r.is_empty());
    }

    #[test]
    fn overflow_drops_newest_and_counts() {
        let r: Ring<u32, 4> = Ring::new();
        let accepted = (0..6).filter(|&i| r.push(i)).count();
        assert_eq!(accepted, 4);
        assert_eq!(r.dropped(), 2);
        let mut out = Vec::new();
        r.drain(|v| out.push(v));
        assert_eq!(out, vec![0, 1, 2, 3]);
    }

    #[test]
    fn take_latest_skips_backlog() {
        let r: Ring<u32, 8> = Ring::new();
        for i in 0..5 {
            r.push(i);
        }
        assert_eq!(r.take_latest(), Some(4));
        assert!(r.is_empty());
    }

    /// Producer and consumer running concurrently: order must be strictly
    /// increasing, and everything must either arrive or be counted as dropped.
    #[test]
    fn spsc_order_and_accounting() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        const N: u64 = 200_000;
        let r: Arc<Ring<u64, 1024>> = Arc::new(Ring::new());
        let done = Arc::new(AtomicBool::new(false));

        let producer = {
            let r = Arc::clone(&r);
            let done = Arc::clone(&done);
            std::thread::spawn(move || {
                let mut accepted = 0u64;
                for i in 0..N {
                    if r.push(i) {
                        accepted += 1;
                    }
                }
                done.store(true, Ordering::Release);
                accepted
            })
        };

        let mut collected = 0u64;
        let mut last: Option<u64> = None;
        loop {
            let n = r.drain(|v| {
                if let Some(l) = last {
                    assert!(v > l, "order violated: {l} -> {v}");
                }
                last = Some(v);
            });
            collected += n as u64;
            if n == 0 {
                if done.load(Ordering::Acquire) && r.is_empty() {
                    break;
                }
                std::thread::yield_now();
            }
        }

        let accepted = producer.join().unwrap();
        assert_eq!(collected, accepted, "everything accepted must arrive");
        assert_eq!(accepted + r.dropped(), N, "accounting must balance");
    }
}
