//! NDI frame buffering with configurable strategies.
//!
//! Provides a ring-buffer-style frame store with strategy selection for
//! different latency/quality trade-offs, plus statistics tracking.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Strategy controlling how the frame buffer handles overflow and underrun.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferStrategy {
    /// Drop the oldest frame when the buffer is full (live/low-latency).
    DropOldest,
    /// Drop the incoming frame when the buffer is full (preserve existing).
    DropNewest,
    /// Block until space becomes available (only meaningful in async contexts).
    Blocking,
    /// Dynamically grow the buffer up to a maximum depth.
    DynamicGrow { max_depth: usize },
}

impl BufferStrategy {
    /// Returns whether this strategy allows dropping frames silently.
    pub fn may_drop_frames(self) -> bool {
        matches!(self, Self::DropOldest | Self::DropNewest)
    }
}

/// A lightweight NDI video/audio frame stored in the buffer.
#[derive(Debug, Clone)]
pub struct NdiBufferedFrame {
    /// Frame sequence number.
    pub sequence: u64,
    /// Presentation timestamp in nanoseconds.
    pub pts_ns: i64,
    /// Raw data payload (video or audio samples).
    pub data: Vec<u8>,
    /// When this frame arrived at the buffer.
    pub arrival: Instant,
}

impl NdiBufferedFrame {
    /// Create a new buffered frame.
    pub fn new(sequence: u64, pts_ns: i64, data: Vec<u8>) -> Self {
        Self {
            sequence,
            pts_ns,
            data,
            arrival: Instant::now(),
        }
    }

    /// Return the latency of this frame (time since it arrived in the buffer).
    pub fn latency(&self) -> Duration {
        self.arrival.elapsed()
    }

    /// Return the size of the payload in bytes.
    pub fn data_len(&self) -> usize {
        self.data.len()
    }
}

/// Statistics counters for a `NdiFrameBuffer`.
#[derive(Debug, Clone, Default)]
pub struct FrameBufferStats {
    /// Total frames pushed into the buffer.
    pub frames_in: u64,
    /// Total frames successfully consumed.
    pub frames_out: u64,
    /// Frames dropped due to buffer overflow.
    pub frames_dropped: u64,
    /// Total bytes that have passed through the buffer.
    pub bytes_total: u64,
    /// Peak depth (maximum simultaneous frames observed).
    pub peak_depth: usize,
}

impl FrameBufferStats {
    /// Compute the drop rate as a fraction in [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    pub fn drop_rate(&self) -> f64 {
        if self.frames_in == 0 {
            0.0
        } else {
            self.frames_dropped as f64 / self.frames_in as f64
        }
    }

    /// Compute the fill rate (frames_out / frames_in).
    #[allow(clippy::cast_precision_loss)]
    pub fn fill_rate(&self) -> f64 {
        if self.frames_in == 0 {
            0.0
        } else {
            self.frames_out as f64 / self.frames_in as f64
        }
    }
}

/// A fixed-depth (or dynamically growing) NDI frame buffer.
#[derive(Debug)]
pub struct NdiFrameBuffer {
    /// Internal storage.
    queue: VecDeque<NdiBufferedFrame>,
    /// Nominal maximum depth (before strategy kicks in).
    nominal_capacity: usize,
    /// Current effective capacity (may grow under `DynamicGrow`).
    effective_capacity: usize,
    /// The buffer overflow / underrun strategy.
    strategy: BufferStrategy,
    /// Accumulated statistics.
    stats: FrameBufferStats,
}

impl NdiFrameBuffer {
    /// Create a new frame buffer with a given nominal capacity and strategy.
    pub fn new(capacity: usize, strategy: BufferStrategy) -> Self {
        let effective = match strategy {
            BufferStrategy::DynamicGrow { .. } => capacity,
            _ => capacity,
        };
        Self {
            queue: VecDeque::with_capacity(capacity),
            nominal_capacity: capacity,
            effective_capacity: effective,
            strategy,
            stats: FrameBufferStats::default(),
        }
    }

    /// Push a frame into the buffer.
    ///
    /// Returns `true` if the frame was accepted, `false` if it was dropped.
    pub fn push(&mut self, frame: NdiBufferedFrame) -> bool {
        let frame_len = frame.data_len() as u64;
        self.stats.frames_in += 1;
        self.stats.bytes_total += frame_len;

        if self.queue.len() < self.effective_capacity {
            self.queue.push_back(frame);
            let depth = self.queue.len();
            if depth > self.stats.peak_depth {
                self.stats.peak_depth = depth;
            }
            return true;
        }

        // Buffer full — apply strategy.
        match self.strategy {
            BufferStrategy::DropOldest => {
                self.queue.pop_front();
                self.stats.frames_dropped += 1;
                self.queue.push_back(frame);
                let depth = self.queue.len();
                if depth > self.stats.peak_depth {
                    self.stats.peak_depth = depth;
                }
                true
            }
            BufferStrategy::DropNewest => {
                self.stats.frames_dropped += 1;
                false
            }
            BufferStrategy::Blocking => {
                // In a sync context, we simply drop the frame (caller handles blocking).
                self.stats.frames_dropped += 1;
                false
            }
            BufferStrategy::DynamicGrow { max_depth } => {
                if self.effective_capacity < max_depth {
                    self.effective_capacity += 1;
                    self.queue.push_back(frame);
                    let depth = self.queue.len();
                    if depth > self.stats.peak_depth {
                        self.stats.peak_depth = depth;
                    }
                    true
                } else {
                    self.stats.frames_dropped += 1;
                    false
                }
            }
        }
    }

    /// Pop the oldest frame from the buffer.
    pub fn pop(&mut self) -> Option<NdiBufferedFrame> {
        let frame = self.queue.pop_front()?;
        self.stats.frames_out += 1;
        Some(frame)
    }

    /// Peek at the oldest frame without removing it.
    pub fn peek(&self) -> Option<&NdiBufferedFrame> {
        self.queue.front()
    }

    /// Returns the current number of frames in the buffer.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Returns true if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Returns the current effective capacity.
    pub fn capacity(&self) -> usize {
        self.effective_capacity
    }

    /// Returns a reference to the accumulated statistics.
    pub fn stats(&self) -> &FrameBufferStats {
        &self.stats
    }

    /// Reset statistics (does not clear frame data).
    pub fn reset_stats(&mut self) {
        self.stats = FrameBufferStats::default();
    }

    /// Clear all frames from the buffer.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Returns fill level as a percentage in [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    pub fn fill_fraction(&self) -> f64 {
        if self.effective_capacity == 0 {
            return 0.0;
        }
        self.queue.len() as f64 / self.effective_capacity as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(seq: u64, size: usize) -> NdiBufferedFrame {
        NdiBufferedFrame::new(seq, seq as i64 * 33_333_333, vec![0u8; size])
    }

    #[test]
    fn test_buffer_strategy_may_drop() {
        assert!(BufferStrategy::DropOldest.may_drop_frames());
        assert!(BufferStrategy::DropNewest.may_drop_frames());
        assert!(!BufferStrategy::Blocking.may_drop_frames());
        assert!(!BufferStrategy::DynamicGrow { max_depth: 32 }.may_drop_frames());
    }

    #[test]
    fn test_buffer_push_pop_basic() {
        let mut buf = NdiFrameBuffer::new(4, BufferStrategy::DropOldest);
        assert!(buf.push(make_frame(1, 100)));
        assert!(buf.push(make_frame(2, 100)));
        assert_eq!(buf.len(), 2);
        let f = buf.pop().expect("expected non-empty buffer");
        assert_eq!(f.sequence, 1);
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_buffer_drop_oldest() {
        let mut buf = NdiFrameBuffer::new(2, BufferStrategy::DropOldest);
        buf.push(make_frame(1, 10));
        buf.push(make_frame(2, 10));
        buf.push(make_frame(3, 10)); // should evict frame 1
        assert_eq!(buf.len(), 2);
        let f = buf.pop().expect("expected non-empty buffer");
        assert_eq!(f.sequence, 2); // frame 1 was dropped
        assert_eq!(buf.stats().frames_dropped, 1);
    }

    #[test]
    fn test_buffer_drop_newest() {
        let mut buf = NdiFrameBuffer::new(2, BufferStrategy::DropNewest);
        buf.push(make_frame(1, 10));
        buf.push(make_frame(2, 10));
        let accepted = buf.push(make_frame(3, 10));
        assert!(!accepted);
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.stats().frames_dropped, 1);
    }

    #[test]
    fn test_buffer_dynamic_grow() {
        let mut buf = NdiFrameBuffer::new(2, BufferStrategy::DynamicGrow { max_depth: 4 });
        buf.push(make_frame(1, 10));
        buf.push(make_frame(2, 10));
        let grew = buf.push(make_frame(3, 10));
        assert!(grew);
        assert_eq!(buf.len(), 3);
        assert_eq!(buf.capacity(), 3);
    }

    #[test]
    fn test_buffer_dynamic_grow_stops_at_max() {
        let mut buf = NdiFrameBuffer::new(2, BufferStrategy::DynamicGrow { max_depth: 2 });
        buf.push(make_frame(1, 10));
        buf.push(make_frame(2, 10));
        let accepted = buf.push(make_frame(3, 10));
        assert!(!accepted); // max_depth == nominal_capacity so can't grow
        assert_eq!(buf.stats().frames_dropped, 1);
    }

    #[test]
    fn test_buffer_peek() {
        let mut buf = NdiFrameBuffer::new(4, BufferStrategy::DropOldest);
        buf.push(make_frame(10, 50));
        let peeked = buf.peek().expect("expected non-empty buffer");
        assert_eq!(peeked.sequence, 10);
        assert_eq!(buf.len(), 1); // peek doesn't consume
    }

    #[test]
    fn test_buffer_clear() {
        let mut buf = NdiFrameBuffer::new(4, BufferStrategy::DropOldest);
        buf.push(make_frame(1, 10));
        buf.push(make_frame(2, 10));
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_stats_drop_rate() {
        let mut buf = NdiFrameBuffer::new(1, BufferStrategy::DropNewest);
        buf.push(make_frame(1, 10));
        buf.push(make_frame(2, 10)); // dropped
        let stats = buf.stats();
        assert_eq!(stats.frames_in, 2);
        assert_eq!(stats.frames_dropped, 1);
        assert!((stats.drop_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_fill_rate() {
        let mut buf = NdiFrameBuffer::new(4, BufferStrategy::DropOldest);
        buf.push(make_frame(1, 10));
        buf.push(make_frame(2, 10));
        buf.pop();
        let stats = buf.stats();
        assert_eq!(stats.frames_out, 1);
        assert!((stats.fill_rate() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_stats_peak_depth() {
        let mut buf = NdiFrameBuffer::new(10, BufferStrategy::DropOldest);
        buf.push(make_frame(1, 10));
        buf.push(make_frame(2, 10));
        buf.push(make_frame(3, 10));
        buf.pop();
        assert_eq!(buf.stats().peak_depth, 3);
    }

    #[test]
    fn test_stats_bytes_total() {
        let mut buf = NdiFrameBuffer::new(4, BufferStrategy::DropOldest);
        buf.push(make_frame(1, 200));
        buf.push(make_frame(2, 300));
        assert_eq!(buf.stats().bytes_total, 500);
    }

    #[test]
    fn test_fill_fraction() {
        let mut buf = NdiFrameBuffer::new(4, BufferStrategy::DropOldest);
        buf.push(make_frame(1, 10));
        buf.push(make_frame(2, 10));
        assert!((buf.fill_fraction() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_reset_stats() {
        let mut buf = NdiFrameBuffer::new(1, BufferStrategy::DropNewest);
        buf.push(make_frame(1, 10));
        buf.push(make_frame(2, 10));
        buf.reset_stats();
        assert_eq!(buf.stats().frames_in, 0);
        assert_eq!(buf.stats().frames_dropped, 0);
    }

    #[test]
    fn test_frame_data_len() {
        let f = make_frame(0, 512);
        assert_eq!(f.data_len(), 512);
    }
}
