//! Port buffering strategies for graph connections.
//!
//! This module provides configurable buffering between graph ports,
//! controlling how many frames can be queued between an output port
//! and its connected input port. Different strategies allow trade-offs
//! between latency and throughput.

use std::collections::VecDeque;
use std::fmt;

/// Buffering strategy for a port connection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferStrategy {
    /// No buffering; frames are passed directly (synchronous).
    Direct,
    /// Fixed-size ring buffer with the given capacity.
    Ring {
        /// Maximum number of frames in the buffer.
        capacity: usize,
    },
    /// Unbounded buffer (grows as needed).
    Unbounded,
    /// Double-buffer: producer writes to back, consumer reads from front.
    DoubleBuffer,
}

impl BufferStrategy {
    /// Get the capacity hint for this strategy (0 for unbounded/direct).
    pub fn capacity_hint(&self) -> usize {
        match self {
            Self::Direct => 0,
            Self::Ring { capacity } => *capacity,
            Self::Unbounded => 0,
            Self::DoubleBuffer => 2,
        }
    }
}

impl Default for BufferStrategy {
    fn default() -> Self {
        Self::Ring { capacity: 4 }
    }
}

impl fmt::Display for BufferStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Direct => write!(f, "Direct"),
            Self::Ring { capacity } => write!(f, "Ring({capacity})"),
            Self::Unbounded => write!(f, "Unbounded"),
            Self::DoubleBuffer => write!(f, "DoubleBuffer"),
        }
    }
}

/// Status of a port buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferStatus {
    /// Buffer is empty; consumer will block or return None.
    Empty,
    /// Buffer has items but is not full.
    Partial,
    /// Buffer is at capacity.
    Full,
}

impl fmt::Display for BufferStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => write!(f, "Empty"),
            Self::Partial => write!(f, "Partial"),
            Self::Full => write!(f, "Full"),
        }
    }
}

/// A frame token used within the port buffer (lightweight handle).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrameToken {
    /// Sequence number of the frame.
    pub sequence: u64,
    /// Presentation timestamp in microseconds.
    pub pts_us: i64,
    /// Size of the frame data in bytes.
    pub size: usize,
}

impl FrameToken {
    /// Create a new frame token.
    pub fn new(sequence: u64, pts_us: i64, size: usize) -> Self {
        Self { sequence, pts_us, size }
    }
}

impl fmt::Display for FrameToken {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Frame(seq={}, pts={}us, {}B)", self.sequence, self.pts_us, self.size)
    }
}

/// A buffer that sits between an output port and an input port.
pub struct PortBuffer {
    /// The buffering strategy.
    strategy: BufferStrategy,
    /// The queue of frame tokens.
    queue: VecDeque<FrameToken>,
    /// Maximum capacity (0 = unbounded).
    max_capacity: usize,
    /// Total frames pushed since creation.
    total_pushed: u64,
    /// Total frames popped since creation.
    total_popped: u64,
    /// Number of frames dropped due to overflow.
    dropped: u64,
    /// Label for debugging.
    label: String,
}

impl PortBuffer {
    /// Create a new port buffer with the given strategy.
    pub fn new(strategy: BufferStrategy, label: &str) -> Self {
        let max_capacity = match strategy {
            BufferStrategy::Direct => 1,
            BufferStrategy::Ring { capacity } => capacity,
            BufferStrategy::Unbounded => 0,
            BufferStrategy::DoubleBuffer => 2,
        };
        Self {
            strategy,
            queue: VecDeque::with_capacity(max_capacity.min(1024)),
            max_capacity,
            total_pushed: 0,
            total_popped: 0,
            dropped: 0,
            label: label.to_string(),
        }
    }

    /// Push a frame token into the buffer.
    ///
    /// Returns `true` if the frame was accepted, `false` if it was dropped
    /// due to overflow (for bounded strategies).
    pub fn push(&mut self, token: FrameToken) -> bool {
        if self.max_capacity > 0 && self.queue.len() >= self.max_capacity {
            // For ring buffers, drop the oldest frame
            if self.strategy == BufferStrategy::Ring { capacity: self.max_capacity } {
                self.queue.pop_front();
                self.dropped += 1;
            } else {
                self.dropped += 1;
                return false;
            }
        }
        self.queue.push_back(token);
        self.total_pushed += 1;
        true
    }

    /// Pop the next frame token from the buffer.
    pub fn pop(&mut self) -> Option<FrameToken> {
        let token = self.queue.pop_front();
        if token.is_some() {
            self.total_popped += 1;
        }
        token
    }

    /// Peek at the next frame token without removing it.
    pub fn peek(&self) -> Option<&FrameToken> {
        self.queue.front()
    }

    /// Get the current number of frames in the buffer.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// Check whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }

    /// Get the current buffer status.
    pub fn status(&self) -> BufferStatus {
        if self.queue.is_empty() {
            BufferStatus::Empty
        } else if self.max_capacity > 0 && self.queue.len() >= self.max_capacity {
            BufferStatus::Full
        } else {
            BufferStatus::Partial
        }
    }

    /// Get the buffering strategy.
    pub fn strategy(&self) -> BufferStrategy {
        self.strategy
    }

    /// Get the maximum capacity (0 for unbounded).
    pub fn max_capacity(&self) -> usize {
        self.max_capacity
    }

    /// Get the total number of frames pushed.
    pub fn total_pushed(&self) -> u64 {
        self.total_pushed
    }

    /// Get the total number of frames popped.
    pub fn total_popped(&self) -> u64 {
        self.total_popped
    }

    /// Get the number of dropped frames.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Get the buffer label.
    pub fn label(&self) -> &str {
        &self.label
    }

    /// Clear all buffered frames.
    pub fn clear(&mut self) {
        self.queue.clear();
    }

    /// Drain all frames from the buffer as a vector.
    pub fn drain_all(&mut self) -> Vec<FrameToken> {
        let items: Vec<_> = self.queue.drain(..).collect();
        self.total_popped += items.len() as u64;
        items
    }

    /// Get fill ratio (0.0 to 1.0). Returns 0.0 for unbounded buffers.
    #[allow(clippy::cast_precision_loss)]
    pub fn fill_ratio(&self) -> f64 {
        if self.max_capacity == 0 {
            return 0.0;
        }
        self.queue.len() as f64 / self.max_capacity as f64
    }
}

impl fmt::Display for PortBuffer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "PortBuffer[{}]: {} ({}/{} frames, {} dropped)",
            self.label, self.strategy, self.queue.len(),
            if self.max_capacity > 0 { self.max_capacity.to_string() } else { "inf".to_string() },
            self.dropped)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_strategy_default() {
        let s = BufferStrategy::default();
        assert_eq!(s, BufferStrategy::Ring { capacity: 4 });
    }

    #[test]
    fn test_buffer_strategy_display() {
        assert_eq!(format!("{}", BufferStrategy::Direct), "Direct");
        assert_eq!(format!("{}", BufferStrategy::Ring { capacity: 8 }), "Ring(8)");
        assert_eq!(format!("{}", BufferStrategy::Unbounded), "Unbounded");
        assert_eq!(format!("{}", BufferStrategy::DoubleBuffer), "DoubleBuffer");
    }

    #[test]
    fn test_capacity_hint() {
        assert_eq!(BufferStrategy::Direct.capacity_hint(), 0);
        assert_eq!(BufferStrategy::Ring { capacity: 16 }.capacity_hint(), 16);
        assert_eq!(BufferStrategy::Unbounded.capacity_hint(), 0);
        assert_eq!(BufferStrategy::DoubleBuffer.capacity_hint(), 2);
    }

    #[test]
    fn test_frame_token_new() {
        let tok = FrameToken::new(42, 1_000_000, 4096);
        assert_eq!(tok.sequence, 42);
        assert_eq!(tok.pts_us, 1_000_000);
        assert_eq!(tok.size, 4096);
    }

    #[test]
    fn test_frame_token_display() {
        let tok = FrameToken::new(1, 500, 256);
        assert_eq!(format!("{tok}"), "Frame(seq=1, pts=500us, 256B)");
    }

    #[test]
    fn test_port_buffer_push_pop() {
        let mut buf = PortBuffer::new(BufferStrategy::Ring { capacity: 4 }, "test");
        assert!(buf.push(FrameToken::new(0, 0, 100)));
        assert!(buf.push(FrameToken::new(1, 1000, 100)));
        assert_eq!(buf.len(), 2);
        let tok = buf.pop().expect("pop should succeed");
        assert_eq!(tok.sequence, 0);
        assert_eq!(buf.len(), 1);
    }

    #[test]
    fn test_port_buffer_peek() {
        let mut buf = PortBuffer::new(BufferStrategy::Ring { capacity: 4 }, "test");
        assert!(buf.peek().is_none());
        buf.push(FrameToken::new(5, 0, 10));
        assert_eq!(buf.peek().expect("peek should succeed").sequence, 5);
        assert_eq!(buf.len(), 1); // Peek doesn't consume
    }

    #[test]
    fn test_port_buffer_ring_overflow_drops_oldest() {
        let mut buf = PortBuffer::new(BufferStrategy::Ring { capacity: 2 }, "ring");
        buf.push(FrameToken::new(0, 0, 10));
        buf.push(FrameToken::new(1, 0, 10));
        buf.push(FrameToken::new(2, 0, 10)); // Should drop seq=0
        assert_eq!(buf.len(), 2);
        assert_eq!(buf.dropped(), 1);
        assert_eq!(buf.pop().expect("pop should succeed").sequence, 1);
    }

    #[test]
    fn test_port_buffer_unbounded() {
        let mut buf = PortBuffer::new(BufferStrategy::Unbounded, "unb");
        for i in 0..100 {
            assert!(buf.push(FrameToken::new(i, 0, 10)));
        }
        assert_eq!(buf.len(), 100);
        assert_eq!(buf.dropped(), 0);
    }

    #[test]
    fn test_port_buffer_status() {
        let mut buf = PortBuffer::new(BufferStrategy::Ring { capacity: 2 }, "s");
        assert_eq!(buf.status(), BufferStatus::Empty);
        buf.push(FrameToken::new(0, 0, 1));
        assert_eq!(buf.status(), BufferStatus::Partial);
        buf.push(FrameToken::new(1, 0, 1));
        assert_eq!(buf.status(), BufferStatus::Full);
    }

    #[test]
    fn test_port_buffer_clear() {
        let mut buf = PortBuffer::new(BufferStrategy::Ring { capacity: 8 }, "c");
        buf.push(FrameToken::new(0, 0, 1));
        buf.push(FrameToken::new(1, 0, 1));
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_port_buffer_drain_all() {
        let mut buf = PortBuffer::new(BufferStrategy::Ring { capacity: 8 }, "d");
        buf.push(FrameToken::new(0, 0, 1));
        buf.push(FrameToken::new(1, 0, 1));
        buf.push(FrameToken::new(2, 0, 1));
        let drained = buf.drain_all();
        assert_eq!(drained.len(), 3);
        assert!(buf.is_empty());
        assert_eq!(buf.total_popped(), 3);
    }

    #[test]
    fn test_port_buffer_fill_ratio() {
        let mut buf = PortBuffer::new(BufferStrategy::Ring { capacity: 4 }, "f");
        assert!((buf.fill_ratio() - 0.0).abs() < f64::EPSILON);
        buf.push(FrameToken::new(0, 0, 1));
        buf.push(FrameToken::new(1, 0, 1));
        assert!((buf.fill_ratio() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_port_buffer_fill_ratio_unbounded() {
        let buf = PortBuffer::new(BufferStrategy::Unbounded, "u");
        assert!((buf.fill_ratio() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_port_buffer_total_counters() {
        let mut buf = PortBuffer::new(BufferStrategy::Ring { capacity: 8 }, "tc");
        buf.push(FrameToken::new(0, 0, 1));
        buf.push(FrameToken::new(1, 0, 1));
        buf.pop();
        assert_eq!(buf.total_pushed(), 2);
        assert_eq!(buf.total_popped(), 1);
    }

    #[test]
    fn test_buffer_status_display() {
        assert_eq!(format!("{}", BufferStatus::Empty), "Empty");
        assert_eq!(format!("{}", BufferStatus::Partial), "Partial");
        assert_eq!(format!("{}", BufferStatus::Full), "Full");
    }
}
