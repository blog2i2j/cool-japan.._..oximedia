//! RTP jitter buffer with packet reordering, configurable depth, late-packet
//! handling, and playout-timestamp tracking.
//!
//! The buffer accepts incoming RTP packets (identified by their 16-bit sequence
//! number and 32-bit timestamp), reorders them, and releases them in sequence
//! number order once enough packets have accumulated to cover the configured
//! playout delay.
//!
//! # Design
//!
//! Packets arrive with an RTP sequence number and timestamp.  The buffer holds
//! up to `capacity` packets; once the head-of-line packet's playout time is
//! reached (or the depth is exceeded), it is removed from the buffer and
//! handed to the caller.  Packets that arrive too late (their sequence number
//! has already been released) are discarded and counted as late.
//!
//! The playout delay can be adjusted at runtime via [`JitterBuffer::set_depth`].

use crate::error::{VideoIpError, VideoIpResult};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};

// ─────────────────────────────────────────────────────────────────────────────
// Public types
// ─────────────────────────────────────────────────────────────────────────────

/// A single RTP packet stored in the jitter buffer.
#[derive(Debug, Clone)]
pub struct RtpPacket {
    /// RTP sequence number (16-bit, wraps).
    pub seq: u16,
    /// RTP timestamp (clock-rate-dependent).
    pub timestamp: u32,
    /// SSRC of the sender.
    pub ssrc: u32,
    /// Payload type.
    pub payload_type: u8,
    /// Raw payload bytes (without RTP header).
    pub payload: Vec<u8>,
    /// Wall-clock time at which this packet was received.
    pub received_at: Instant,
}

impl RtpPacket {
    /// Creates a new packet with `received_at` set to `Instant::now()`.
    #[must_use]
    pub fn new(seq: u16, timestamp: u32, ssrc: u32, payload_type: u8, payload: Vec<u8>) -> Self {
        Self {
            seq,
            timestamp,
            ssrc,
            payload_type,
            payload,
            received_at: Instant::now(),
        }
    }

    /// Creates a packet with an explicit receive time (useful in tests).
    #[must_use]
    pub fn with_receive_time(
        seq: u16,
        timestamp: u32,
        ssrc: u32,
        payload_type: u8,
        payload: Vec<u8>,
        received_at: Instant,
    ) -> Self {
        Self {
            seq,
            timestamp,
            ssrc,
            payload_type,
            payload,
            received_at,
        }
    }
}

/// Statistics accumulated by the jitter buffer.
#[derive(Debug, Clone, Default)]
pub struct JitterStats {
    /// Total packets inserted.
    pub inserted: u64,
    /// Total packets removed by the caller via [`JitterBuffer::pop`].
    pub popped: u64,
    /// Packets dropped because they arrived after their playout deadline.
    pub late_dropped: u64,
    /// Packets dropped because the buffer exceeded its `capacity`.
    pub overflow_dropped: u64,
    /// Duplicate sequence numbers received (second packet discarded).
    pub duplicates: u64,
    /// Current number of packets held in the buffer.
    pub current_depth: usize,
    /// Maximum depth ever reached.
    pub peak_depth: usize,
    /// Running estimate of network jitter (in RTP timestamp ticks) using an
    /// RFC 3550 §A.8 inter-arrival exponential moving average.
    pub jitter_estimate: f64,
}

/// Outcome of a [`JitterBuffer::insert`] call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertOutcome {
    /// Packet accepted into the buffer.
    Accepted,
    /// Packet was too late and has been discarded.
    Late,
    /// Buffer was full; oldest packet evicted to make room.
    OverflowEvicted,
    /// Packet was a duplicate and has been discarded.
    Duplicate,
}

/// Outcome of a [`JitterBuffer::pop`] call.
#[derive(Debug)]
pub enum PopOutcome {
    /// A packet is ready for playout.
    Packet(RtpPacket),
    /// The next packet is buffered but not yet due for playout.
    NotYetDue,
    /// The buffer is empty.
    Empty,
}

// ─────────────────────────────────────────────────────────────────────────────
// JitterBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the jitter buffer.
#[derive(Debug, Clone)]
pub struct JitterBufferConfig {
    /// Maximum number of packets to hold in the buffer at any one time.
    /// When the buffer is full and a new packet arrives, the oldest buffered
    /// packet is evicted.
    pub capacity: usize,
    /// Target playout delay: packets are held for at least this duration
    /// before being released.
    pub playout_delay: Duration,
    /// RTP clock rate (Hz) used for jitter estimation.
    pub clock_rate: u32,
}

impl JitterBufferConfig {
    /// Creates a configuration suitable for 1080p60 video (90 kHz clock, 80 ms
    /// playout delay, capacity 256 packets).
    #[must_use]
    pub const fn video_hd() -> Self {
        Self {
            capacity: 256,
            playout_delay: Duration::from_millis(80),
            clock_rate: 90_000,
        }
    }

    /// Creates a configuration suitable for Opus audio (48 kHz clock, 40 ms
    /// playout delay, capacity 128 packets).
    #[must_use]
    pub const fn audio_opus() -> Self {
        Self {
            capacity: 128,
            playout_delay: Duration::from_millis(40),
            clock_rate: 48_000,
        }
    }
}

impl Default for JitterBufferConfig {
    fn default() -> Self {
        Self {
            capacity: 256,
            playout_delay: Duration::from_millis(60),
            clock_rate: 90_000,
        }
    }
}

/// RTP jitter buffer.
///
/// Packets are indexed by their sequence number, stored in sorted order.
/// [`pop`](JitterBuffer::pop) returns the packet with the smallest sequence
/// number if it has been held for at least the playout delay.
pub struct JitterBuffer {
    /// Sorted map from sequence number → packet.
    ///
    /// We use a `BTreeMap<u32, RtpPacket>` where the key is a normalised
    /// "extended sequence number" to handle the 16-bit wrap-around naturally.
    map: BTreeMap<u32, RtpPacket>,
    /// Configuration.
    config: JitterBufferConfig,
    /// The highest sequence number we have released so far (extended, 32-bit).
    /// `None` if nothing has been popped yet.
    last_released_ext: Option<u32>,
    /// The extended sequence number of the first packet ever inserted.
    base_ext: Option<u32>,
    /// Accumulated statistics.
    stats: JitterStats,
    /// RFC 3550 §A.8 transit time of the last processed packet (in RTP ticks).
    last_transit: Option<i64>,
}

impl JitterBuffer {
    /// The amount added to the base extended seq when a new "epoch" begins
    /// (full 16-bit wrap).
    const EPOCH: u32 = 0x0001_0000;

    /// Creates a new jitter buffer with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::InvalidState`] when `capacity` is zero.
    pub fn new(config: JitterBufferConfig) -> VideoIpResult<Self> {
        if config.capacity == 0 {
            return Err(VideoIpError::InvalidState(
                "jitter buffer capacity must be > 0".to_string(),
            ));
        }
        Ok(Self {
            map: BTreeMap::new(),
            config,
            last_released_ext: None,
            base_ext: None,
            stats: JitterStats::default(),
            last_transit: None,
        })
    }

    /// Returns a reference to the current statistics.
    #[must_use]
    pub fn stats(&self) -> &JitterStats {
        &self.stats
    }

    /// Updates the playout delay.
    pub fn set_depth(&mut self, delay: Duration) {
        self.config.playout_delay = delay;
    }

    /// Returns the current playout delay.
    #[must_use]
    pub fn playout_delay(&self) -> Duration {
        self.config.playout_delay
    }

    /// Converts a raw 16-bit sequence number to an extended (monotonically
    /// increasing) 32-bit sequence number, using `base_ext` as the
    /// reference point.
    fn extend_seq(&self, seq: u16) -> u32 {
        match self.base_ext {
            None => u32::from(seq),
            Some(base) => {
                // Current epoch (upper bits of base).
                let epoch = base & !0xFFFF;
                let candidate = epoch | u32::from(seq);
                // Check whether wrapping to the next epoch makes more sense.
                // If the distance forward is smaller with the next epoch, use
                // it.  This handles packets arriving just after a wrap.
                let max_ext = self.map.keys().copied().next_back().unwrap_or(base);
                // If candidate is more than 2^15 behind the current max,
                // bump it to the next epoch.
                if max_ext > candidate && max_ext - candidate > (Self::EPOCH / 2) {
                    candidate + Self::EPOCH
                } else {
                    candidate
                }
            }
        }
    }

    /// Inserts a packet into the jitter buffer.
    ///
    /// Returns an [`InsertOutcome`] indicating what happened.
    pub fn insert(&mut self, packet: RtpPacket) -> InsertOutcome {
        let ext_seq = self.extend_seq(packet.seq);

        // Initialise base_ext on first insert.
        if self.base_ext.is_none() {
            self.base_ext = Some(ext_seq);
        }

        // ── Late-packet detection ─────────────────────────────────────────
        if let Some(last_rel) = self.last_released_ext {
            if ext_seq <= last_rel {
                self.stats.late_dropped += 1;
                return InsertOutcome::Late;
            }
        }

        // ── Duplicate detection ───────────────────────────────────────────
        if self.map.contains_key(&ext_seq) {
            self.stats.duplicates += 1;
            return InsertOutcome::Duplicate;
        }

        // ── Overflow handling ─────────────────────────────────────────────
        let outcome = if self.map.len() >= self.config.capacity {
            // Evict the oldest packet (smallest key).
            if let Some((&oldest_key, _)) = self.map.iter().next() {
                self.map.remove(&oldest_key);
                self.stats.overflow_dropped += 1;
            }
            InsertOutcome::OverflowEvicted
        } else {
            InsertOutcome::Accepted
        };

        // ── Update jitter estimate (RFC 3550 §A.8) ────────────────────────
        // We approximate arrival time in RTP ticks.
        let now_ticks = {
            let elapsed = packet.received_at.elapsed();
            (elapsed.as_secs_f64() * self.config.clock_rate as f64) as i64
        };
        // Use RTP timestamp as send time.
        let transit = now_ticks - packet.timestamp as i64;
        if let Some(last) = self.last_transit {
            let d = (transit - last).unsigned_abs() as f64;
            self.stats.jitter_estimate += (d - self.stats.jitter_estimate) / 16.0;
        }
        self.last_transit = Some(transit);

        self.map.insert(ext_seq, packet);
        self.stats.inserted += 1;
        let depth = self.map.len();
        self.stats.current_depth = depth;
        if depth > self.stats.peak_depth {
            self.stats.peak_depth = depth;
        }
        outcome
    }

    /// Attempts to pop the next in-sequence packet from the buffer.
    ///
    /// A packet is released when it has been held for at least the configured
    /// playout delay.  If the next packet is not yet due, returns
    /// [`PopOutcome::NotYetDue`].  If the buffer is empty returns
    /// [`PopOutcome::Empty`].
    pub fn pop(&mut self) -> PopOutcome {
        // Peek at the head-of-line packet.
        let (&head_key, head) = match self.map.iter().next() {
            Some(kv) => kv,
            None => return PopOutcome::Empty,
        };

        // Check playout deadline.
        if head.received_at.elapsed() < self.config.playout_delay {
            return PopOutcome::NotYetDue;
        }

        // Remove and return it.
        let pkt = self.map.remove(&head_key).expect("key was just peeked");
        self.last_released_ext = Some(head_key);
        self.stats.popped += 1;
        self.stats.current_depth = self.map.len();
        PopOutcome::Packet(pkt)
    }

    /// Forcibly removes and returns the head-of-line packet regardless of the
    /// playout deadline.  Returns `None` when the buffer is empty.
    pub fn pop_force(&mut self) -> Option<RtpPacket> {
        let (&head_key, _) = self.map.iter().next()?;
        let pkt = self.map.remove(&head_key)?;
        self.last_released_ext = Some(head_key);
        self.stats.popped += 1;
        self.stats.current_depth = self.map.len();
        Some(pkt)
    }

    /// Returns the number of packets currently held.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` when no packets are held.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Flushes all buffered packets in sequence order, returning them.
    pub fn flush(&mut self) -> Vec<RtpPacket> {
        let pkts: Vec<RtpPacket> = self.map.values().cloned().collect();
        let count = pkts.len() as u64;
        if let Some((&last_key, _)) = self.map.iter().next_back() {
            self.last_released_ext = Some(last_key);
        }
        self.map.clear();
        self.stats.popped += count;
        self.stats.current_depth = 0;
        pkts
    }

    /// Peeks at the sequence number of the head-of-line packet without
    /// removing it.  Returns `None` when the buffer is empty.
    #[must_use]
    pub fn peek_seq(&self) -> Option<u16> {
        self.map.values().next().map(|p| p.seq)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    fn make_packet(seq: u16, ts: u32) -> RtpPacket {
        // Use a fixed receive time so playout-delay tests are deterministic.
        RtpPacket::with_receive_time(
            seq,
            ts,
            0xDEAD_BEEF,
            96,
            vec![0u8; 16],
            Instant::now() - Duration::from_secs(10), // always "old"
        )
    }

    fn buf_zero_delay() -> JitterBuffer {
        JitterBuffer::new(JitterBufferConfig {
            capacity: 64,
            playout_delay: Duration::ZERO,
            clock_rate: 90_000,
        })
        .expect("valid jitter buffer configuration")
    }

    // ── Basic insertion and pop ───────────────────────────────────────────────

    #[test]
    fn test_insert_and_pop_in_order() {
        let mut buf = buf_zero_delay();
        for i in 0u16..5 {
            let outcome = buf.insert(make_packet(i, u32::from(i) * 3000));
            assert_eq!(outcome, InsertOutcome::Accepted);
        }
        assert_eq!(buf.len(), 5);
        for expected_seq in 0u16..5 {
            match buf.pop() {
                PopOutcome::Packet(p) => assert_eq!(p.seq, expected_seq),
                other => panic!("unexpected: {other:?}"),
            }
        }
        assert_eq!(buf.len(), 0);
    }

    // ── Reordering ────────────────────────────────────────────────────────────

    #[test]
    fn test_out_of_order_reordering() {
        let mut buf = buf_zero_delay();
        // Insert out of order: 3 2 0 1 4
        for &seq in &[3u16, 2, 0, 1, 4] {
            buf.insert(make_packet(seq, u32::from(seq) * 3000));
        }
        let mut seqs = Vec::new();
        while let PopOutcome::Packet(p) = buf.pop() {
            seqs.push(p.seq);
        }
        assert_eq!(seqs, vec![0u16, 1, 2, 3, 4]);
    }

    // ── Duplicate detection ───────────────────────────────────────────────────

    #[test]
    fn test_duplicate_rejected() {
        let mut buf = buf_zero_delay();
        let first = buf.insert(make_packet(10, 0));
        assert_eq!(first, InsertOutcome::Accepted);
        let second = buf.insert(make_packet(10, 0));
        assert_eq!(second, InsertOutcome::Duplicate);
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.stats().duplicates, 1);
    }

    // ── Late-packet handling ──────────────────────────────────────────────────

    #[test]
    fn test_late_packet_rejected() {
        let mut buf = buf_zero_delay();
        buf.insert(make_packet(0, 0));
        buf.pop_force(); // releases seq 0
        // Now insert seq 0 again — should be "late"
        let outcome = buf.insert(make_packet(0, 0));
        assert_eq!(outcome, InsertOutcome::Late);
        assert_eq!(buf.stats().late_dropped, 1);
    }

    // ── Overflow handling ─────────────────────────────────────────────────────

    #[test]
    fn test_overflow_evicts_oldest() {
        let mut buf = JitterBuffer::new(JitterBufferConfig {
            capacity: 4,
            playout_delay: Duration::ZERO,
            clock_rate: 90_000,
        })
        .expect("valid jitter buffer configuration");
        for i in 0u16..4 {
            buf.insert(make_packet(i, 0));
        }
        assert_eq!(buf.len(), 4);
        let outcome = buf.insert(make_packet(4, 0));
        assert_eq!(outcome, InsertOutcome::OverflowEvicted);
        // Buffer still at capacity
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.stats().overflow_dropped, 1);
    }

    // ── Playout-delay gate ────────────────────────────────────────────────────

    #[test]
    fn test_not_yet_due_when_packet_too_fresh() {
        let mut buf = JitterBuffer::new(JitterBufferConfig {
            capacity: 64,
            playout_delay: Duration::from_secs(3600), // absurdly large
            clock_rate: 90_000,
        })
        .expect("valid jitter buffer configuration");
        // Packet received right now (elapsed ≈ 0)
        let fresh = RtpPacket::new(0, 0, 0, 96, vec![]);
        buf.insert(fresh);
        assert!(matches!(buf.pop(), PopOutcome::NotYetDue));
    }

    // ── Flush ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_flush_returns_ordered_packets() {
        let mut buf = buf_zero_delay();
        for &seq in &[5u16, 3, 1, 4, 2] {
            buf.insert(make_packet(seq, 0));
        }
        let flushed = buf.flush();
        let seqs: Vec<u16> = flushed.iter().map(|p| p.seq).collect();
        assert_eq!(seqs, vec![1u16, 2, 3, 4, 5]);
        assert!(buf.is_empty());
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    #[test]
    fn test_statistics_tracking() {
        let mut buf = buf_zero_delay();
        buf.insert(make_packet(0, 0));
        buf.insert(make_packet(1, 3000));
        buf.pop_force();
        buf.pop_force();
        let s = buf.stats();
        assert_eq!(s.inserted, 2);
        assert_eq!(s.popped, 2);
        assert_eq!(s.current_depth, 0);
    }

    // ── Zero-capacity rejected ────────────────────────────────────────────────

    #[test]
    fn test_zero_capacity_returns_error() {
        let cfg = JitterBufferConfig {
            capacity: 0,
            playout_delay: Duration::ZERO,
            clock_rate: 90_000,
        };
        assert!(JitterBuffer::new(cfg).is_err());
    }

    // ── Sequence wrap-around ──────────────────────────────────────────────────

    #[test]
    fn test_sequence_wrap() {
        let mut buf = buf_zero_delay();
        // Insert packets around the 16-bit wrap.
        buf.insert(make_packet(0xFFFE, 0));
        buf.insert(make_packet(0xFFFF, 3000));
        buf.insert(make_packet(0x0000, 6000));
        buf.insert(make_packet(0x0001, 9000));
        let seqs: Vec<u16> = buf.flush().iter().map(|p| p.seq).collect();
        assert_eq!(seqs, vec![0xFFFEu16, 0xFFFF, 0x0000, 0x0001]);
    }

    // ── depth adjustment ─────────────────────────────────────────────────────

    #[test]
    fn test_set_depth() {
        let mut buf = buf_zero_delay();
        buf.set_depth(Duration::from_millis(100));
        assert_eq!(buf.playout_delay(), Duration::from_millis(100));
    }
}
