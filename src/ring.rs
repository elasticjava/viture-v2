//! Sperrfreier SPSC-Ringpuffer — dasselbe Prinzip wie die io_uring-Queues,
//! nur zwischen zwei Threads im eigenen Adressraum.
//!
//! Ein Erzeuger (der Lese-Thread), ein Verbraucher (Renderer, Fusion, Eingabe).
//! Kein Mutex, kein Syscall, keine Allokation im Betrieb. Head und Tail liegen
//! auf getrennten Cache-Zeilen, damit sich die beiden Threads nicht gegenseitig
//! die Zeile aus dem Cache reißen.
//!
//! Kapazität ist eine Zweierpotenz, damit die Maskierung ohne Division läuft.
//!
//! Überlauf verwirft den **neuesten** Wert und zählt ihn. Das ist bewusst so:
//! Nur der Verbraucher darf `tail` schreiben, sonst gäbe es zwei Schreiber auf
//! demselben Index und damit ein Datenrennen. Wer keinen Rückstand will, holt
//! mit [`Ring::take_latest`] ab — dann läuft der Ring gar nicht erst voll.
//! Blockiert wird nie: der Lese-Thread darf niemals auf den Verbraucher warten.

use core::cell::UnsafeCell;
use core::mem::MaybeUninit;
use core::sync::atomic::{AtomicU64, Ordering};

#[repr(align(64))]
struct CacheLine<T>(T);

/// Ringpuffer fester Kapazität für `Copy`-Nutzlasten.
pub struct Ring<T: Copy, const N: usize> {
    slots: UnsafeCell<[MaybeUninit<T>; N]>,
    /// Vom Erzeuger geschrieben, vom Verbraucher gelesen.
    head: CacheLine<AtomicU64>,
    /// Vom Verbraucher geschrieben, vom Erzeuger gelesen.
    tail: CacheLine<AtomicU64>,
    /// Wie oft der Erzeuger den Verbraucher überholt hat.
    dropped: CacheLine<AtomicU64>,
}

// SAFETY: Zugriff ist durch die Indizes diszipliniert; genau ein Erzeuger und
// genau ein Verbraucher (SPSC).
unsafe impl<T: Copy + Send, const N: usize> Send for Ring<T, N> {}
unsafe impl<T: Copy + Send, const N: usize> Sync for Ring<T, N> {}

impl<T: Copy, const N: usize> Ring<T, N> {
    const MASK: u64 = (N as u64) - 1;

    pub const fn new() -> Self {
        assert!(N.is_power_of_two(), "Kapazität muss eine Zweierpotenz sein");
        Ring {
            slots: UnsafeCell::new([MaybeUninit::uninit(); N]),
            head: CacheLine(AtomicU64::new(0)),
            tail: CacheLine(AtomicU64::new(0)),
            dropped: CacheLine(AtomicU64::new(0)),
        }
    }

    /// Erzeugerseite. `false`, wenn der Ring voll war und der Wert verworfen
    /// wurde. Wartet nie.
    #[inline]
    pub fn push(&self, value: T) -> bool {
        let head = self.head.0.load(Ordering::Relaxed);
        let tail = self.tail.0.load(Ordering::Acquire);
        if head.wrapping_sub(tail) >= N as u64 {
            self.dropped.0.fetch_add(1, Ordering::Relaxed);
            return false;
        }
        // SAFETY: Nur der Erzeuger schreibt, und nur in den Slot bei head, der
        // wegen der Füllstandsprüfung gerade nicht vom Verbraucher gelesen wird.
        unsafe {
            let slots = &mut *self.slots.get();
            slots[(head & Self::MASK) as usize] = MaybeUninit::new(value);
        }
        self.head.0.store(head.wrapping_add(1), Ordering::Release);
        true
    }

    /// Verbraucherseite, ein Element.
    #[inline]
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.0.load(Ordering::Relaxed);
        let head = self.head.0.load(Ordering::Acquire);
        if tail == head {
            return None;
        }
        // SAFETY: Slot wurde vom Erzeuger vor dem Release-Store beschrieben.
        let value = unsafe {
            let slots = &*self.slots.get();
            slots[(tail & Self::MASK) as usize].assume_init()
        };
        self.tail.0.store(tail.wrapping_add(1), Ordering::Release);
        Some(value)
    }

    /// Verbraucherseite, alles auf einmal — der Batch-Zug aus io_uring:
    /// ein Acquire, dann n Elemente, dann ein Release.
    #[inline]
    pub fn drain(&self, mut f: impl FnMut(T)) -> usize {
        let mut tail = self.tail.0.load(Ordering::Relaxed);
        let head = self.head.0.load(Ordering::Acquire);
        let n = head.wrapping_sub(tail);
        if n == 0 {
            return 0;
        }
        // SAFETY: siehe pop().
        let slots = unsafe { &*self.slots.get() };
        for _ in 0..n {
            f(unsafe { slots[(tail & Self::MASK) as usize].assume_init() });
            tail = tail.wrapping_add(1);
        }
        self.tail.0.store(tail, Ordering::Release);
        n as usize
    }

    /// Nur der jüngste Eintrag; der Rest wird verworfen. Für Lagedaten, bei
    /// denen ein Renderer ohnehin nur den aktuellen Wert braucht.
    #[inline]
    pub fn take_latest(&self) -> Option<T> {
        let head = self.head.0.load(Ordering::Acquire);
        let tail = self.tail.0.load(Ordering::Relaxed);
        if head == tail {
            return None;
        }
        // SAFETY: siehe pop().
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

    /// Wie viele Einträge der Erzeuger verworfen hat, weil niemand abgeholt hat.
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
    fn fifo_reihenfolge() {
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
    fn ueberlauf_verwirft_neueste_und_zaehlt() {
        let r: Ring<u32, 4> = Ring::new();
        let angenommen = (0..6).filter(|&i| r.push(i)).count();
        assert_eq!(angenommen, 4);
        assert_eq!(r.dropped(), 2);
        let mut out = Vec::new();
        r.drain(|v| out.push(v));
        assert_eq!(out, vec![0, 1, 2, 3]);
    }

    #[test]
    fn jüngster_eintrag() {
        let r: Ring<u32, 8> = Ring::new();
        for i in 0..5 {
            r.push(i);
        }
        assert_eq!(r.take_latest(), Some(4));
        assert!(r.is_empty());
    }

    /// Erzeuger und Verbraucher gleichzeitig: Reihenfolge muss streng steigen,
    /// und alles muss entweder angekommen oder gezählt verworfen sein.
    #[test]
    fn spsc_reihenfolge_und_bilanz() {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Arc;

        const N: u64 = 200_000;
        let r: Arc<Ring<u64, 1024>> = Arc::new(Ring::new());
        let fertig = Arc::new(AtomicBool::new(false));

        let prod = {
            let r = Arc::clone(&r);
            let fertig = Arc::clone(&fertig);
            std::thread::spawn(move || {
                let mut angenommen = 0u64;
                for i in 0..N {
                    if r.push(i) {
                        angenommen += 1;
                    }
                }
                fertig.store(true, Ordering::Release);
                angenommen
            })
        };

        let mut geholt = 0u64;
        let mut last: Option<u64> = None;
        loop {
            let n = r.drain(|v| {
                if let Some(l) = last {
                    assert!(v > l, "Reihenfolge verletzt: {l} -> {v}");
                }
                last = Some(v);
            });
            geholt += n as u64;
            if n == 0 {
                if fertig.load(Ordering::Acquire) && r.is_empty() {
                    break;
                }
                std::thread::yield_now();
            }
        }

        let angenommen = prod.join().unwrap();
        assert_eq!(geholt, angenommen, "alles Angenommene muss ankommen");
        assert_eq!(angenommen + r.dropped(), N, "Bilanz muss aufgehen");
    }
}
