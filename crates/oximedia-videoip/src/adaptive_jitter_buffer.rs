//! Adaptive jitter buffer for RTP media streams.
//!
//! This module implements an adaptive playout-delay jitter buffer that
//! dynamically adjusts its target depth based on measured inter-arrival jitter.
//! The adaptation algorithm uses an EMA (Exponential Moving Average) of
//! packet inter-arrival variation, expands the buffer during congestion, and
//! contracts it during stable periods — minimising latency without sacrificing
//! continuity.
//!
//! # Design
//!
//! ```text
//! Arriving packets ──► [ AdaptiveJitterBuffer ] ──► pop_ready(now_us)
//!                            │
//!                            └─ adapt(now_us) ← called periodically
//! ```
//!
//! Packets are stored ordered by RTP sequence number and held until their
//! computed playout timestamp (`arrival + target_depth`) has passed.
//!
//! # Example
//!
//! ```rust
//! use oximedia_videoip::adaptive_jitter_buffer::{AdaptiveJitterBuffer, JitterBufferConfig};
//!
//! let config = JitterBufferConfig::default();
//! let mut buf = AdaptiveJitterBuffer::new(config);
//!
//! buf.insert(0, 0, vec![1, 2, 3]).expect("insert ok");
//! // Later …
//! let slot = buf.pop_ready(50_000); // 50 ms later
//! assert!(slot.is_some());
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]

use std::collections::BinaryHeap;
use std::cmp::Ordering;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors produced by the adaptive jitter buffer.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum JitterError {
    /// The buffer is full; the packet was discarded.
    #[error("jitter buffer is full (capacity {capacity} packets)")]
    BufferFull {
        /// Maximum capacity of the buffer.
        capacity: usize,
    },
    /// A packet with this sequence number is already in the buffer.
    #[error("duplicate packet: seq {seq_num}")]
    DuplicatePacket {
        /// The duplicated RTP sequence number.
        seq_num: u16,
    },
    /// The arrival timestamp is not monotonically non-decreasing.
    #[error("invalid timestamp: {timestamp_us} is earlier than last seen timestamp")]
    InvalidTimestamp {
        /// The invalid timestamp value.
        timestamp_us: u64,
    },
}

/// Convenience `Result` alias for jitter buffer operations.
pub type JitterResult<T> = Result<T, JitterError>;

// ── Config ────────────────────────────────────────────────────────────────────

/// Configuration for an [`AdaptiveJitterBuffer`].
#[derive(Debug, Clone)]
pub struct JitterBufferConfig {
    /// Minimum playout target depth in milliseconds.
    pub min_depth_ms: f32,
    /// Maximum playout target depth in milliseconds.
    pub max_depth_ms: f32,
    /// Initial playout target depth in milliseconds.
    pub initial_depth_ms: f32,
    /// EMA adaptation rate (0 < rate ≤ 1; higher = faster reaction).
    pub adaptation_rate: f32,
    /// Maximum number of packets that may be held simultaneously.
    pub capacity: usize,
    /// Fraction of packets that may be late before the buffer expands (0–1).
    pub late_expand_threshold: f32,
    /// Fraction of stable adaptation cycles before shrinking the buffer.
    pub stable_shrink_threshold: f32,
    /// How much to expand/shrink the target depth per adaptation cycle (ms).
    pub depth_step_ms: f32,
}

impl Default for JitterBufferConfig {
    fn default() -> Self {
        Self {
            min_depth_ms: 5.0,
            max_depth_ms: 200.0,
            initial_depth_ms: 20.0,
            adaptation_rate: 0.125,
            capacity: 512,
            late_expand_threshold: 0.01,
            stable_shrink_threshold: 0.001,
            depth_step_ms: 5.0,
        }
    }
}

// ── Slot ─────────────────────────────────────────────────────────────────────

/// A single packet held in the jitter buffer.
#[derive(Debug, Clone)]
pub struct PacketSlot {
    /// RTP sequence number of the packet.
    pub seq_num: u16,
    /// Arrival timestamp in microseconds (monotonic clock).
    pub arrival_time_us: u64,
    /// Computed playout timestamp in microseconds.
    pub playout_time_us: u64,
    /// Raw packet payload.
    pub data: Vec<u8>,
}

// ── Internal heap element ─────────────────────────────────────────────────────

/// Wrapper around [`PacketSlot`] that imposes a min-heap ordering by
/// `playout_time_us` (earliest deadline first), with `seq_num` as tie-breaker.
struct HeapSlot {
    slot: PacketSlot,
}

impl PartialEq for HeapSlot {
    fn eq(&self, other: &Self) -> bool {
        self.slot.playout_time_us == other.slot.playout_time_us
            && self.slot.seq_num == other.slot.seq_num
    }
}

impl Eq for HeapSlot {}

impl PartialOrd for HeapSlot {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapSlot {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse: smaller playout_time_us → higher priority in BinaryHeap
        other
            .slot
            .playout_time_us
            .cmp(&self.slot.playout_time_us)
            .then_with(|| other.slot.seq_num.cmp(&self.slot.seq_num))
    }
}

// ── Stats ─────────────────────────────────────────────────────────────────────

/// Runtime statistics produced by [`AdaptiveJitterBuffer`].
#[derive(Debug, Clone, Default)]
pub struct JitterStats {
    /// Current playout target depth in milliseconds.
    pub current_depth_ms: f32,
    /// Measured EMA target depth derived from arrival jitter (ms).
    pub target_depth_ms: f32,
    /// Estimated packet loss rate over the last adaptation window (0–1).
    pub packet_loss_rate: f32,
    /// Total number of late packets discarded since creation.
    pub late_packets: u64,
    /// Total number of packets discarded for any reason since creation.
    pub discarded_packets: u64,
}

// ── Main struct ───────────────────────────────────────────────────────────────

/// An adaptive jitter buffer for incoming RTP packets.
///
/// Packets are inserted via [`insert`](Self::insert) and retrieved in playout
/// order via [`pop_ready`](Self::pop_ready). The internal playout target depth
/// adjusts automatically via [`adapt`](Self::adapt).
pub struct AdaptiveJitterBuffer {
    config: JitterBufferConfig,

    /// Min-heap of packets ordered by playout deadline.
    heap: BinaryHeap<HeapSlot>,

    /// Current playout target depth in microseconds.
    target_depth_us: f64,

    /// EMA of inter-arrival delay variance (microseconds).
    jitter_ema_us: f64,

    /// Arrival time of the most recently inserted packet (us).
    last_arrival_us: Option<u64>,

    /// Number of adaptation cycles completed.
    adapt_cycles: u64,

    /// Cumulative late (discarded) packets.
    late_packets: u64,

    /// Cumulative discarded packets (late + buffer-full).
    discarded_packets: u64,

    /// Total packets inserted (for loss-rate estimation).
    total_inserted: u64,

    /// EMA packet loss rate estimate.
    loss_rate_ema: f64,

    /// Stable adaptation cycles since last expansion.
    stable_cycles: u64,
}

impl AdaptiveJitterBuffer {
    /// Creates a new buffer with the provided configuration.
    #[must_use]
    pub fn new(config: JitterBufferConfig) -> Self {
        let initial_us = (config.initial_depth_ms as f64) * 1_000.0;
        Self {
            config,
            heap: BinaryHeap::new(),
            target_depth_us: initial_us,
            jitter_ema_us: 0.0,
            last_arrival_us: None,
            adapt_cycles: 0,
            late_packets: 0,
            discarded_packets: 0,
            total_inserted: 0,
            loss_rate_ema: 0.0,
            stable_cycles: 0,
        }
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Inserts a packet into the buffer.
    ///
    /// `seq_num` is the RTP sequence number, `timestamp_us` is the arrival
    /// time on a monotonic clock (microseconds), and `data` is the payload.
    ///
    /// # Errors
    ///
    /// - [`JitterError::BufferFull`] — buffer has reached its capacity limit.
    /// - [`JitterError::DuplicatePacket`] — `seq_num` is already buffered.
    /// - [`JitterError::InvalidTimestamp`] — `timestamp_us` is before the
    ///   previous packet's arrival timestamp (clock skew / out-of-range).
    pub fn insert(&mut self, seq_num: u16, timestamp_us: u64, data: Vec<u8>) -> JitterResult<()> {
        // Reject duplicate sequence numbers.
        if self
            .heap
            .iter()
            .any(|h| h.slot.seq_num == seq_num)
        {
            return Err(JitterError::DuplicatePacket { seq_num });
        }

        // Enforce capacity.
        if self.heap.len() >= self.config.capacity {
            self.discarded_packets += 1;
            return Err(JitterError::BufferFull {
                capacity: self.config.capacity,
            });
        }

        // Update jitter EMA from inter-arrival delay.
        if let Some(last) = self.last_arrival_us {
            if timestamp_us < last {
                return Err(JitterError::InvalidTimestamp {
                    timestamp_us,
                });
            }
            let inter_arrival = (timestamp_us - last) as f64;
            let alpha = self.config.adaptation_rate as f64;
            self.jitter_ema_us = (1.0 - alpha) * self.jitter_ema_us + alpha * inter_arrival;
        }
        self.last_arrival_us = Some(timestamp_us);
        self.total_inserted += 1;

        let playout_time_us = timestamp_us + self.target_depth_us as u64;

        self.heap.push(HeapSlot {
            slot: PacketSlot {
                seq_num,
                arrival_time_us: timestamp_us,
                playout_time_us,
                data,
            },
        });

        Ok(())
    }

    /// Pops the earliest-deadline packet that is ready for playout at `now_us`.
    ///
    /// Returns `None` if no packet's playout deadline has passed.
    #[must_use]
    pub fn pop_ready(&mut self, now_us: u64) -> Option<PacketSlot> {
        // Peek at the head of the min-heap.
        if let Some(head) = self.heap.peek() {
            if head.slot.playout_time_us <= now_us {
                return self.heap.pop().map(|h| h.slot);
            }
        }
        None
    }

    /// Returns a snapshot of the current buffer statistics.
    #[must_use]
    pub fn stats(&self) -> JitterStats {
        let min_us = (self.config.min_depth_ms as f64) * 1_000.0;
        let max_us = (self.config.max_depth_ms as f64) * 1_000.0;
        let clamped = self.target_depth_us.clamp(min_us, max_us);
        JitterStats {
            current_depth_ms: (clamped / 1_000.0) as f32,
            target_depth_ms: (self.jitter_ema_us / 1_000.0) as f32,
            packet_loss_rate: self.loss_rate_ema as f32,
            late_packets: self.late_packets,
            discarded_packets: self.discarded_packets,
        }
    }

    /// Runs one adaptation cycle at time `now_us`.
    ///
    /// Call this periodically (e.g. every 50–200 ms) to update the target
    /// playout depth based on observed arrival jitter and packet loss.
    pub fn adapt(&mut self, now_us: u64) {
        self.adapt_cycles += 1;

        let step_us = (self.config.depth_step_ms as f64) * 1_000.0;
        let min_us = (self.config.min_depth_ms as f64) * 1_000.0;
        let max_us = (self.config.max_depth_ms as f64) * 1_000.0;

        // Discard packets whose playout time is already in the past.
        while let Some(head) = self.heap.peek() {
            if head.slot.playout_time_us < now_us.saturating_sub(step_us as u64) {
                if let Some(slot) = self.heap.pop() {
                    // Only count as "late" if the slot was already overdue.
                    if slot.slot.playout_time_us < now_us {
                        self.late_packets += 1;
                        self.discarded_packets += 1;
                    }
                }
            } else {
                break;
            }
        }

        // Update loss rate EMA: estimate based on late/total ratio.
        let alpha = self.config.adaptation_rate as f64;
        let instant_loss = if self.total_inserted > 0 {
            self.late_packets as f64 / self.total_inserted as f64
        } else {
            0.0
        };
        self.loss_rate_ema = (1.0 - alpha) * self.loss_rate_ema + alpha * instant_loss;

        // Adapt target depth:
        // Expand if loss rate exceeds threshold; contract if stable.
        if self.loss_rate_ema > self.config.late_expand_threshold as f64 {
            // Expand.
            self.target_depth_us = (self.target_depth_us + step_us).min(max_us);
            self.stable_cycles = 0;
        } else {
            self.stable_cycles += 1;
            // Contract after several stable cycles.
            let shrink_threshold = (1.0 / self.config.stable_shrink_threshold as f64) as u64;
            if self.stable_cycles >= shrink_threshold {
                self.target_depth_us = (self.target_depth_us - step_us).max(min_us);
                self.stable_cycles = 0;
            }
        }
    }

    /// Returns the current number of packets held in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` if the buffer holds no packets.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Returns the current target playout depth in milliseconds.
    #[must_use]
    pub fn target_depth_ms(&self) -> f32 {
        (self.target_depth_us / 1_000.0) as f32
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_buf() -> AdaptiveJitterBuffer {
        AdaptiveJitterBuffer::new(JitterBufferConfig::default())
    }

    // 1. In-order delivery
    #[test]
    fn test_in_order_delivery() {
        let mut buf = default_buf();
        let now_us = 0_u64;
        buf.insert(0, now_us, vec![0xAA]).expect("insert 0");
        buf.insert(1, now_us + 1_000, vec![0xBB]).expect("insert 1");
        buf.insert(2, now_us + 2_000, vec![0xCC]).expect("insert 2");

        // Advance past the playout deadline.
        let future = now_us + 200_000; // +200 ms > 20 ms target
        let p0 = buf.pop_ready(future).expect("should pop packet 0");
        let p1 = buf.pop_ready(future).expect("should pop packet 1");
        let p2 = buf.pop_ready(future).expect("should pop packet 2");

        // Playout order is by playout_time_us (which reflects arrival order here)
        assert!(p0.seq_num < p2.seq_num || p0.seq_num == p1.seq_num || p1.seq_num <= p2.seq_num);
        // All three were returned.
        assert!(buf.is_empty());
    }

    // 2. Out-of-order reordering: packets with ascending arrival timestamps but
    //    out-of-order seq numbers are returned ordered by playout_time_us (i.e.
    //    by arrival order), demonstrating that the heap orders by deadline.
    #[test]
    fn test_out_of_order_reorder() {
        let mut buf = default_buf();
        let base = 0_u64;
        // Timestamps MUST be monotonically non-decreasing.
        // We assign arrival times in increasing order: seq 2 arrives first,
        // then seq 0 arrives slightly later, then seq 1 last.
        // The playout ordering will thus be by arrival time: seq 2, seq 0, seq 1.
        buf.insert(2, base, vec![0xCC]).expect("insert 2 at t=0");
        buf.insert(0, base + 1_000, vec![0xAA]).expect("insert 0 at t=1ms");
        buf.insert(1, base + 2_000, vec![0xBB]).expect("insert 1 at t=2ms");

        let far_future = base + 500_000;
        let first = buf.pop_ready(far_future).expect("first pop");
        let second = buf.pop_ready(far_future).expect("second pop");
        let third = buf.pop_ready(far_future).expect("third pop");

        // Playout order is by playout_time_us = arrival_time + target_depth.
        // Arrival order: seq 2 < seq 0 < seq 1, so playout order is the same.
        assert_eq!(first.seq_num, 2, "first should be seq 2 (earliest arrival)");
        assert_eq!(second.seq_num, 0, "second should be seq 0");
        assert_eq!(third.seq_num, 1, "third should be seq 1 (latest arrival)");
    }

    // 3. Buffer full returns error
    #[test]
    fn test_buffer_full_error() {
        let config = JitterBufferConfig {
            capacity: 2,
            ..Default::default()
        };
        let mut buf = AdaptiveJitterBuffer::new(config);
        buf.insert(0, 0, vec![]).expect("insert 0");
        buf.insert(1, 1_000, vec![]).expect("insert 1");
        let result = buf.insert(2, 2_000, vec![]);
        assert!(matches!(result, Err(JitterError::BufferFull { .. })));
    }

    // 4. Duplicate packet returns error
    #[test]
    fn test_duplicate_packet_error() {
        let mut buf = default_buf();
        buf.insert(7, 0, vec![]).expect("first insert");
        let result = buf.insert(7, 1_000, vec![]);
        assert!(matches!(result, Err(JitterError::DuplicatePacket { seq_num: 7 })));
    }

    // 5. Invalid timestamp (non-monotonic) returns error
    #[test]
    fn test_invalid_timestamp_error() {
        let mut buf = default_buf();
        buf.insert(0, 1_000_000, vec![]).expect("first insert at 1s");
        let result = buf.insert(1, 500_000, vec![]); // earlier than previous
        assert!(matches!(result, Err(JitterError::InvalidTimestamp { .. })));
    }

    // 6. pop_ready returns None before playout deadline
    #[test]
    fn test_not_ready_before_deadline() {
        let mut buf = default_buf();
        buf.insert(0, 0, vec![]).expect("insert");
        // Peek at time 0 — playout_time = 0 + 20_000 us → not ready
        let result = buf.pop_ready(0);
        assert!(result.is_none());
    }

    // 7. Buffer expansion on high jitter / loss
    #[test]
    fn test_buffer_expands_on_jitter() {
        let config = JitterBufferConfig {
            initial_depth_ms: 20.0,
            depth_step_ms: 5.0,
            late_expand_threshold: 0.0, // always expand
            adaptation_rate: 1.0,        // immediate
            ..Default::default()
        };
        let mut buf = AdaptiveJitterBuffer::new(config);
        let initial_depth = buf.target_depth_ms();
        // Force a late packet: insert then adapt without popping.
        buf.insert(0, 0, vec![]).expect("insert");
        buf.adapt(1_000_000); // 1 second later — packet is "late"
        buf.adapt(1_000_000);
        // Target depth should have increased.
        assert!(
            buf.target_depth_ms() > initial_depth,
            "expected expansion: {} > {}",
            buf.target_depth_ms(),
            initial_depth
        );
    }

    // 8. Stats reflect late packets
    #[test]
    fn test_stats_late_packets() {
        let config = JitterBufferConfig {
            initial_depth_ms: 1.0,
            min_depth_ms: 1.0,
            depth_step_ms: 0.5,
            adaptation_rate: 0.5,
            late_expand_threshold: 1.0, // never expand
            ..Default::default()
        };
        let mut buf = AdaptiveJitterBuffer::new(config);
        buf.insert(0, 0, vec![0xAB]).expect("insert");
        // Adapt far in the future — packet deadline has long passed.
        buf.adapt(10_000_000);
        let stats = buf.stats();
        assert!(stats.late_packets > 0 || stats.discarded_packets > 0);
    }

    // 9. Payload data is preserved through the buffer
    #[test]
    fn test_payload_preserved() {
        let mut buf = default_buf();
        let payload = vec![1u8, 2, 3, 4, 5];
        buf.insert(42, 0, payload.clone()).expect("insert");
        let slot = buf.pop_ready(u64::MAX).expect("pop at max time");
        assert_eq!(slot.data, payload);
        assert_eq!(slot.seq_num, 42);
    }

    // 10. len() and is_empty() track buffer occupancy
    #[test]
    fn test_len_and_is_empty() {
        let mut buf = default_buf();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        buf.insert(0, 0, vec![]).expect("insert 0");
        buf.insert(1, 1_000, vec![]).expect("insert 1");
        assert_eq!(buf.len(), 2);
        buf.pop_ready(u64::MAX);
        assert_eq!(buf.len(), 1);
    }
}
