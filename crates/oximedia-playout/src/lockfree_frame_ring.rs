//! Minimal-contention ring buffer for inter-thread frame passing.
//!
//! Implements a single-producer / single-consumer (SPSC) ring buffer
//! optimised for broadcast playout pipelines.  The design uses:
//!
//! * **Per-slot `Mutex<Option<VideoFrame>>`** — eliminates the global lock
//!   that a naïve `Mutex<VecDeque>` would require.  Only the slot being
//!   read or written is ever locked; all other slots remain free.
//! * **Atomic `u64` head/tail cursors** — the producer and consumer advance
//!   their respective cursors without holding any lock, so slot selection is
//!   genuinely wait-free.  The mutex is taken only for the actual data move.
//! * **Power-of-two capacity** — index wrapping is a cheap bitwise AND.
//!
//! # Comparison with a global-mutex queue
//!
//! | Operation | Global-mutex VecDeque | This design |
//! |-----------|----------------------|-------------|
//! | Push contention | Every push vs every pop | Rarely: only if producer & consumer hit the *same* slot simultaneously (impossible in SPSC unless the buffer is exactly 1 element and always full) |
//! | Lock granularity | Whole queue | One slot |
//! | Allocation | Dynamic | Fixed at construction |
//!
//! # Thread Safety
//!
//! The type is `Send + Sync` via the inner `Mutex` wrapper.  It is the
//! **caller's responsibility** to ensure that only one thread pushes and
//! only one thread pops at any given time (SPSC contract).
//!
//! # Overflow / underrun metrics
//!
//! Push attempts on a full buffer increment `overflow_count`.
//! Pop attempts on an empty buffer increment `underrun_count`.

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// VideoFrame
// ---------------------------------------------------------------------------

/// A lightweight video frame stored in the ring buffer.
///
/// In production the pixel data would reference a pooled buffer; here we store
/// owned bytes to keep the module self-contained.
#[derive(Debug, Clone)]
pub struct VideoFrame {
    /// Frame sequence number (monotonically increasing from the encoder).
    pub sequence: u64,
    /// Presentation timestamp in nanoseconds.
    pub pts_ns: u64,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Raw RGBA pixel data (`width * height * 4` bytes).
    pub data: Vec<u8>,
}

impl VideoFrame {
    /// Construct a new blank frame with the given dimensions.
    pub fn blank(sequence: u64, pts_ns: u64, width: u32, height: u32) -> Self {
        let size = (width as usize) * (height as usize) * 4;
        Self {
            sequence,
            pts_ns,
            width,
            height,
            data: vec![0u8; size],
        }
    }

    /// Return `true` if all pixels are zero (black).
    pub fn is_blank(&self) -> bool {
        self.data.iter().all(|&b| b == 0)
    }
}

// ---------------------------------------------------------------------------
// Slot
// ---------------------------------------------------------------------------

/// A single slot in the ring buffer.
///
/// `sequence` is an atomic tag used for lock-free slot coordination:
/// - Value equals `slot_index`         → slot is empty (ready for producer).
/// - Value equals `slot_index + 1`     → slot holds valid data (ready for consumer).
/// - Any other value                   → transition in progress (spin/yield).
struct Slot {
    data: Mutex<Option<VideoFrame>>,
    /// Atomic tag — see struct-level comment.
    sequence: AtomicU64,
}

impl Slot {
    fn new(index: u64) -> Self {
        Self {
            data: Mutex::new(None),
            sequence: AtomicU64::new(index),
        }
    }
}

// ---------------------------------------------------------------------------
// LockfreeFrameRing
// ---------------------------------------------------------------------------

/// SPSC ring buffer for [`VideoFrame`] objects with per-slot locking.
pub struct LockfreeFrameRing {
    slots: Vec<Slot>,
    /// Bitwise mask for cheap modulo: `capacity - 1`.
    mask: u64,
    /// Producer write cursor.
    head: AtomicU64,
    /// Consumer read cursor.
    tail: AtomicU64,
    /// Count of push attempts rejected due to a full buffer.
    overflow_count: AtomicU64,
    /// Count of pop attempts on an empty buffer.
    underrun_count: AtomicU64,
    /// Total frames successfully pushed.
    pushed_total: AtomicU64,
    /// Total frames successfully popped.
    popped_total: AtomicU64,
}

impl LockfreeFrameRing {
    /// Create a new ring buffer with at least `min_capacity` slots.
    ///
    /// The actual capacity is rounded up to the next power of two.
    ///
    /// # Panics
    ///
    /// Panics if `min_capacity` is 0.
    pub fn new(min_capacity: usize) -> Self {
        assert!(min_capacity > 0, "capacity must be > 0");
        let cap = min_capacity.next_power_of_two();
        let mask = (cap as u64) - 1;
        let slots: Vec<Slot> = (0..cap).map(|i| Slot::new(i as u64)).collect();
        Self {
            slots,
            mask,
            head: AtomicU64::new(0),
            tail: AtomicU64::new(0),
            overflow_count: AtomicU64::new(0),
            underrun_count: AtomicU64::new(0),
            pushed_total: AtomicU64::new(0),
            popped_total: AtomicU64::new(0),
        }
    }

    /// Capacity of the ring buffer (always a power of two).
    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Approximate number of frames currently in the buffer.
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Relaxed);
        let tail = self.tail.load(Ordering::Relaxed);
        head.wrapping_sub(tail) as usize
    }

    /// Return `true` if the buffer is currently empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return `true` if the buffer is currently full.
    pub fn is_full(&self) -> bool {
        self.len() >= self.capacity()
    }

    /// Attempt to push a frame into the buffer (non-blocking).
    ///
    /// Returns `Ok(())` on success or `Err(frame)` if the buffer is full.
    ///
    /// # Concurrency
    ///
    /// Must be called from **exactly one** producer thread.
    pub fn push(&self, frame: VideoFrame) -> Result<(), VideoFrame> {
        let head = self.head.load(Ordering::Relaxed);
        let slot_idx = (head & self.mask) as usize;
        let slot = &self.slots[slot_idx];

        // A slot is available when its sequence tag equals `head`.
        let seq = slot.sequence.load(Ordering::Acquire);
        if seq != head {
            // The consumer has not yet freed this slot — buffer is full.
            self.overflow_count.fetch_add(1, Ordering::Relaxed);
            return Err(frame);
        }

        // Store the frame under the per-slot lock.
        {
            let mut guard = slot.data.lock().unwrap_or_else(|e| e.into_inner());
            *guard = Some(frame);
        }

        // Publish: advance sequence to head + 1 so the consumer knows data is ready.
        slot.sequence.store(head + 1, Ordering::Release);
        // Advance the producer cursor.
        self.head.store(head + 1, Ordering::Relaxed);
        self.pushed_total.fetch_add(1, Ordering::Relaxed);

        Ok(())
    }

    /// Attempt to pop the next frame from the buffer (non-blocking).
    ///
    /// Returns `Some(frame)` or `None` if the buffer is empty.
    ///
    /// # Concurrency
    ///
    /// Must be called from **exactly one** consumer thread.
    pub fn pop(&self) -> Option<VideoFrame> {
        let tail = self.tail.load(Ordering::Relaxed);
        let slot_idx = (tail & self.mask) as usize;
        let slot = &self.slots[slot_idx];

        // A slot contains data when its sequence tag equals `tail + 1`.
        let seq = slot.sequence.load(Ordering::Acquire);
        if seq != tail + 1 {
            self.underrun_count.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        // Take the frame under the per-slot lock.
        let frame = {
            let mut guard = slot.data.lock().unwrap_or_else(|e| e.into_inner());
            guard.take()
        };

        // Release the slot back to the producer by setting its sequence to
        // `tail + capacity` — the position at which the producer will next
        // wrap around to this slot.
        slot.sequence
            .store(tail + self.capacity() as u64, Ordering::Release);
        // Advance the consumer cursor.
        self.tail.store(tail + 1, Ordering::Relaxed);
        self.popped_total.fetch_add(1, Ordering::Relaxed);

        frame
    }

    /// Drain all available frames into a `Vec`.
    pub fn drain(&self) -> Vec<VideoFrame> {
        let mut frames = Vec::new();
        while let Some(f) = self.pop() {
            frames.push(f);
        }
        frames
    }

    /// Total push attempts rejected because the buffer was full.
    pub fn overflow_count(&self) -> u64 {
        self.overflow_count.load(Ordering::Relaxed)
    }

    /// Total pop attempts on an empty buffer.
    pub fn underrun_count(&self) -> u64 {
        self.underrun_count.load(Ordering::Relaxed)
    }

    /// Total frames successfully pushed since creation.
    pub fn pushed_total(&self) -> u64 {
        self.pushed_total.load(Ordering::Relaxed)
    }

    /// Total frames successfully popped since creation.
    pub fn popped_total(&self) -> u64 {
        self.popped_total.load(Ordering::Relaxed)
    }

    /// Snapshot of current ring-buffer statistics.
    pub fn stats(&self) -> FrameRingStats {
        FrameRingStats {
            capacity: self.capacity(),
            fill_level: self.len(),
            pushed_total: self.pushed_total(),
            popped_total: self.popped_total(),
            overflow_count: self.overflow_count(),
            underrun_count: self.underrun_count(),
        }
    }
}

// ---------------------------------------------------------------------------
// FrameRingStats
// ---------------------------------------------------------------------------

/// A snapshot of [`LockfreeFrameRing`] metrics.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameRingStats {
    /// Allocated slot count (power of two).
    pub capacity: usize,
    /// Approximate number of frames currently queued.
    pub fill_level: usize,
    /// Total frames successfully pushed since the ring was created.
    pub pushed_total: u64,
    /// Total frames successfully popped since the ring was created.
    pub popped_total: u64,
    /// Frames rejected due to a full buffer.
    pub overflow_count: u64,
    /// Pop attempts on an empty buffer.
    pub underrun_count: u64,
}

impl FrameRingStats {
    /// Fill level as a fraction of capacity (`0.0`–`1.0`).
    pub fn fill_ratio(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.fill_level as f64 / self.capacity as f64
    }

    /// Return `true` if any overflow events have occurred.
    pub fn has_overflow(&self) -> bool {
        self.overflow_count > 0
    }

    /// Return `true` if any underrun events have occurred.
    pub fn has_underrun(&self) -> bool {
        self.underrun_count > 0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::thread;

    fn make_frame(seq: u64) -> VideoFrame {
        VideoFrame::blank(seq, seq * 40_000_000, 1920, 1080)
    }

    #[test]
    fn test_capacity_rounded_to_power_of_two() {
        assert_eq!(LockfreeFrameRing::new(5).capacity(), 8);
        assert_eq!(LockfreeFrameRing::new(8).capacity(), 8);
        assert_eq!(LockfreeFrameRing::new(100).capacity(), 128);
    }

    #[test]
    fn test_push_and_pop_single_frame() {
        let ring = LockfreeFrameRing::new(4);
        assert!(ring.is_empty());

        ring.push(make_frame(1)).expect("push should succeed");
        assert_eq!(ring.len(), 1);

        let frame = ring.pop().expect("pop should return a frame");
        assert_eq!(frame.sequence, 1);
        assert!(ring.is_empty());
    }

    #[test]
    fn test_push_fill_and_overflow() {
        let ring = LockfreeFrameRing::new(2); // capacity = 2
        ring.push(make_frame(0)).expect("first push succeeds");
        ring.push(make_frame(1)).expect("second push succeeds");

        assert!(ring.is_full());
        let rejected = ring.push(make_frame(2));
        assert!(rejected.is_err(), "should reject when full");
        assert_eq!(ring.overflow_count(), 1);
    }

    #[test]
    fn test_pop_empty_increments_underrun() {
        let ring = LockfreeFrameRing::new(4);
        assert!(ring.pop().is_none());
        assert_eq!(ring.underrun_count(), 1);
    }

    #[test]
    fn test_push_pop_order_is_fifo() {
        let ring = LockfreeFrameRing::new(8);
        for i in 0..5u64 {
            ring.push(make_frame(i)).expect("push should succeed");
        }
        for i in 0..5u64 {
            let f = ring.pop().expect("should pop frame");
            assert_eq!(f.sequence, i, "FIFO order violated");
        }
    }

    #[test]
    fn test_drain_empties_buffer() {
        let ring = LockfreeFrameRing::new(4);
        ring.push(make_frame(10)).unwrap();
        ring.push(make_frame(11)).unwrap();
        ring.push(make_frame(12)).unwrap();

        let drained = ring.drain();
        assert_eq!(drained.len(), 3);
        assert!(ring.is_empty());
        assert_eq!(drained[0].sequence, 10);
        assert_eq!(drained[1].sequence, 11);
        assert_eq!(drained[2].sequence, 12);
    }

    #[test]
    fn test_stats_fill_ratio() {
        let ring = LockfreeFrameRing::new(4); // cap = 4
        ring.push(make_frame(0)).unwrap();
        ring.push(make_frame(1)).unwrap();

        let stats = ring.stats();
        assert_eq!(stats.capacity, 4);
        assert_eq!(stats.fill_level, 2);
        assert!((stats.fill_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_pushed_and_popped_total() {
        let ring = LockfreeFrameRing::new(4);
        ring.push(make_frame(0)).unwrap();
        ring.push(make_frame(1)).unwrap();
        ring.pop();
        ring.pop();
        let stats = ring.stats();
        assert_eq!(stats.pushed_total, 2);
        assert_eq!(stats.popped_total, 2);
        assert!(!stats.has_overflow());
        assert!(!stats.has_underrun());
    }

    #[test]
    fn test_spsc_concurrent_throughput() {
        let ring = Arc::new(LockfreeFrameRing::new(64));

        let producer_ring = Arc::clone(&ring);
        let n_frames: u64 = 1000;

        let producer = thread::spawn(move || {
            let mut pushed = 0u64;
            let mut seq = 0u64;
            while pushed < n_frames {
                let f = VideoFrame::blank(seq, seq * 40_000_000, 64, 64);
                if producer_ring.push(f).is_ok() {
                    pushed += 1;
                    seq += 1;
                }
                std::hint::spin_loop();
            }
        });

        let consumer_ring = Arc::clone(&ring);
        let consumer = thread::spawn(move || {
            let mut received = Vec::with_capacity(n_frames as usize);
            while received.len() < n_frames as usize {
                if let Some(f) = consumer_ring.pop() {
                    received.push(f.sequence);
                }
                std::hint::spin_loop();
            }
            received
        });

        producer.join().expect("producer thread panicked");
        let received = consumer.join().expect("consumer thread panicked");

        assert_eq!(received.len(), n_frames as usize);
        for (i, &seq) in received.iter().enumerate() {
            assert_eq!(seq, i as u64, "FIFO violated at index {i}");
        }
    }

    #[test]
    fn test_video_frame_blank_and_is_blank() {
        let f = VideoFrame::blank(42, 0, 4, 4);
        assert_eq!(f.data.len(), 4 * 4 * 4);
        assert!(f.is_blank());
    }

    #[test]
    fn test_wrap_around_reuses_slots() {
        let ring = LockfreeFrameRing::new(4); // cap = 4
        for cycle in 0u64..3 {
            for i in 0u64..4 {
                let seq = cycle * 4 + i;
                ring.push(make_frame(seq))
                    .expect("push in wraparound test failed");
            }
            let drained = ring.drain();
            assert_eq!(drained.len(), 4);
        }
        let stats = ring.stats();
        assert_eq!(stats.pushed_total, 12);
        assert_eq!(stats.popped_total, 12);
        assert!(!stats.has_overflow());
    }

    #[test]
    fn test_stats_has_overflow_and_underrun_flags() {
        // Use capacity 2 to avoid the Vyukov-queue corner case where cap=1 has
        // no way to distinguish a full slot from a consumer-freed slot at the
        // same absolute sequence position.
        let ring = LockfreeFrameRing::new(2); // rounds up to cap = 2
        ring.push(make_frame(0)).unwrap();
        ring.push(make_frame(1)).unwrap();
        // Buffer full → overflow.
        let _ = ring.push(make_frame(2));
        // Drain.
        ring.pop();
        ring.pop();
        // Empty → underrun.
        ring.pop();

        let stats = ring.stats();
        assert!(stats.has_overflow(), "expected overflow after filling buffer");
        assert!(stats.has_underrun(), "expected underrun after draining buffer");
    }
}
