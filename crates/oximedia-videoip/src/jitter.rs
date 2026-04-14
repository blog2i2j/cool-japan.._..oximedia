//! Jitter buffer for packet reordering and delay compensation.

use crate::error::{VideoIpError, VideoIpResult};
use crate::packet::Packet;
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::time::{Duration, Instant};

/// Wrapper for packets in the jitter buffer with ordering.
struct JitterPacket {
    packet: Packet,
    arrival_time: Instant,
}

impl PartialEq for JitterPacket {
    fn eq(&self, other: &Self) -> bool {
        self.packet.header.sequence == other.packet.header.sequence
    }
}

impl Eq for JitterPacket {}

impl PartialOrd for JitterPacket {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for JitterPacket {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap (earliest sequence first)
        other
            .packet
            .header
            .sequence
            .cmp(&self.packet.header.sequence)
    }
}

/// Jitter buffer for reordering packets and compensating for network jitter.
pub struct JitterBuffer {
    /// Buffer of packets waiting to be played out.
    buffer: BinaryHeap<JitterPacket>,
    /// Maximum buffer size in packets.
    max_size: usize,
    /// Target buffer delay in milliseconds.
    target_delay_ms: u64,
    /// Expected next sequence number.
    next_sequence: Option<u16>,
    /// Statistics.
    stats: JitterStats,
}

/// Statistics for jitter buffer.
#[derive(Debug, Clone, Default)]
pub struct JitterStats {
    /// Number of packets added.
    pub packets_added: u64,
    /// Number of packets played out.
    pub packets_played: u64,
    /// Number of packets dropped due to buffer overflow.
    pub packets_dropped: u64,
    /// Number of packets played out of order.
    pub packets_out_of_order: u64,
    /// Number of duplicate packets.
    pub packets_duplicate: u64,
    /// Current buffer occupancy.
    pub buffer_occupancy: usize,
}

impl JitterBuffer {
    /// Creates a new jitter buffer.
    ///
    /// # Arguments
    ///
    /// * `max_size` - Maximum number of packets to buffer
    /// * `target_delay_ms` - Target buffering delay in milliseconds
    #[must_use]
    pub fn new(max_size: usize, target_delay_ms: u64) -> Self {
        Self {
            buffer: BinaryHeap::new(),
            max_size,
            target_delay_ms,
            next_sequence: None,
            stats: JitterStats::default(),
        }
    }

    /// Adds a packet to the buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is full.
    pub fn add_packet(&mut self, packet: Packet) -> VideoIpResult<()> {
        // Check for duplicates
        if self
            .buffer
            .iter()
            .any(|jp| jp.packet.header.sequence == packet.header.sequence)
        {
            self.stats.packets_duplicate += 1;
            return Ok(());
        }

        // Check buffer size
        if self.buffer.len() >= self.max_size {
            self.stats.packets_dropped += 1;
            return Err(VideoIpError::BufferOverflow);
        }

        // Initialize next_sequence on first packet
        if self.next_sequence.is_none() {
            self.next_sequence = Some(packet.header.sequence);
        }

        let jitter_packet = JitterPacket {
            packet,
            arrival_time: Instant::now(),
        };

        self.buffer.push(jitter_packet);
        self.stats.packets_added += 1;
        self.stats.buffer_occupancy = self.buffer.len();

        Ok(())
    }

    /// Retrieves the next packet if it's ready to be played out.
    ///
    /// Returns `None` if no packet is ready or if the target delay hasn't been reached.
    #[must_use]
    pub fn get_packet(&mut self) -> Option<Packet> {
        if self.buffer.is_empty() {
            return None;
        }

        // Check if the oldest packet has been buffered long enough
        let oldest = self.buffer.peek()?;
        let buffered_duration = oldest.arrival_time.elapsed();

        if buffered_duration < Duration::from_millis(self.target_delay_ms) {
            return None;
        }

        // Get the packet with the earliest sequence number
        if let Some(jitter_packet) = self.buffer.pop() {
            let packet = jitter_packet.packet;
            let sequence = packet.header.sequence;

            // Check if this is the expected sequence number
            if let Some(expected) = self.next_sequence {
                if sequence != expected {
                    self.stats.packets_out_of_order += 1;
                }
                self.next_sequence = Some(expected.wrapping_add(1));
            } else {
                self.next_sequence = Some(sequence.wrapping_add(1));
            }

            self.stats.packets_played += 1;
            self.stats.buffer_occupancy = self.buffer.len();

            Some(packet)
        } else {
            None
        }
    }

    /// Tries to get a packet immediately, bypassing the delay check.
    ///
    /// This is useful when the buffer is getting too full.
    #[must_use]
    pub fn get_packet_immediate(&mut self) -> Option<Packet> {
        if let Some(jitter_packet) = self.buffer.pop() {
            let packet = jitter_packet.packet;
            self.stats.packets_played += 1;
            self.stats.buffer_occupancy = self.buffer.len();
            Some(packet)
        } else {
            None
        }
    }

    /// Returns the number of packets currently in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns true if the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns the buffer statistics.
    #[must_use]
    pub const fn stats(&self) -> &JitterStats {
        &self.stats
    }

    /// Clears the buffer.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.next_sequence = None;
        self.stats.buffer_occupancy = 0;
    }

    /// Sets the target delay.
    pub fn set_target_delay(&mut self, delay_ms: u64) {
        self.target_delay_ms = delay_ms;
    }

    /// Returns the current target delay in milliseconds.
    #[must_use]
    pub const fn target_delay(&self) -> u64 {
        self.target_delay_ms
    }

    /// Adjusts the buffer delay dynamically based on network conditions.
    ///
    /// This implements a simple adaptive algorithm that increases delay when
    /// packets are arriving out of order and decreases it when the buffer is stable.
    pub fn adjust_delay(&mut self) {
        const MIN_DELAY_MS: u64 = 5;
        const MAX_DELAY_MS: u64 = 100;
        const ADJUSTMENT_STEP: u64 = 5;

        // Increase delay if we're seeing lots of out-of-order packets
        let out_of_order_ratio = if self.stats.packets_played > 0 {
            self.stats.packets_out_of_order as f64 / self.stats.packets_played as f64
        } else {
            0.0
        };

        if out_of_order_ratio > 0.1 && self.target_delay_ms < MAX_DELAY_MS {
            self.target_delay_ms = (self.target_delay_ms + ADJUSTMENT_STEP).min(MAX_DELAY_MS);
        } else if out_of_order_ratio < 0.01 && self.target_delay_ms > MIN_DELAY_MS {
            self.target_delay_ms = (self.target_delay_ms - ADJUSTMENT_STEP).max(MIN_DELAY_MS);
        }
    }

    /// Removes packets older than the specified age.
    pub fn cleanup_old_packets(&mut self, max_age: Duration) {
        let now = Instant::now();
        let mut new_buffer = BinaryHeap::new();
        let mut dropped = 0;

        while let Some(jitter_packet) = self.buffer.pop() {
            if now.duration_since(jitter_packet.arrival_time) <= max_age {
                new_buffer.push(jitter_packet);
            } else {
                dropped += 1;
            }
        }

        self.buffer = new_buffer;
        self.stats.packets_dropped += dropped;
        self.stats.buffer_occupancy = self.buffer.len();
    }
}

// ── NetworkAwareJitterBuffer ──────────────────────────────────────────────────

/// Network condition snapshot supplied to [`NetworkAwareJitterBuffer::adapt`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetworkCondition {
    /// Measured RTT in milliseconds.
    pub rtt_ms: f64,
    /// RTT variance (standard deviation) in milliseconds.
    pub rtt_variance_ms: f64,
    /// Packet loss rate (0.0 – 1.0).
    pub loss_rate: f64,
    /// `true` when a congestion event has been signalled externally
    /// (e.g. BBR drain phase or AIMD multiplicative decrease).
    pub congested: bool,
}

impl Default for NetworkCondition {
    fn default() -> Self {
        Self {
            rtt_ms: 10.0,
            rtt_variance_ms: 2.0,
            loss_rate: 0.0,
            congested: false,
        }
    }
}

/// Configuration for [`NetworkAwareJitterBuffer`].
#[derive(Debug, Clone)]
pub struct NetworkAwareJitterConfig {
    /// Minimum allowed target depth in milliseconds.
    pub min_depth_ms: u64,
    /// Maximum allowed target depth in milliseconds.
    pub max_depth_ms: u64,
    /// Initial target depth in milliseconds.
    pub initial_depth_ms: u64,
    /// Maximum number of packets to hold simultaneously.
    pub capacity: usize,
    /// RTT variance multiplier used in the ideal-depth formula.
    pub variance_multiplier: f64,
    /// Extra milliseconds added per percentage point of packet loss.
    pub loss_penalty_ms_per_pct: f64,
    /// Extra milliseconds added when the network is congested.
    pub congestion_penalty_ms: f64,
    /// EMA smoothing factor for the depth target (0 < α ≤ 1; higher = faster).
    pub depth_ema_alpha: f64,
    /// Milliseconds to expand per adaptation cycle when above the ideal.
    pub expand_step_ms: u64,
    /// Milliseconds to shrink per adaptation cycle when below the ideal.
    pub shrink_step_ms: u64,
    /// Consecutive stable cycles required before shrinking.
    pub stable_cycles_before_shrink: u32,
}

impl Default for NetworkAwareJitterConfig {
    fn default() -> Self {
        Self {
            min_depth_ms: 5,
            max_depth_ms: 150,
            initial_depth_ms: 20,
            capacity: 512,
            variance_multiplier: 3.0,
            loss_penalty_ms_per_pct: 2.0,
            congestion_penalty_ms: 20.0,
            depth_ema_alpha: 0.25,
            expand_step_ms: 5,
            shrink_step_ms: 2,
            stable_cycles_before_shrink: 10,
        }
    }
}

/// Statistics reported by [`NetworkAwareJitterBuffer`].
#[derive(Debug, Clone, Default)]
pub struct NetworkAwareJitterStats {
    /// Current target delay in milliseconds.
    pub current_depth_ms: u64,
    /// Last EMA-smoothed ideal depth (ms, floating-point).
    pub ideal_depth_ms: f64,
    /// Total adaptation expand steps.
    pub expand_steps: u64,
    /// Total adaptation shrink steps.
    pub shrink_steps: u64,
    /// Consecutive stable cycles (resets when an expand step is taken).
    pub stable_cycles: u32,
    /// Total packets added.
    pub packets_added: u64,
    /// Total packets played out.
    pub packets_played: u64,
    /// Total packets dropped due to buffer overflow.
    pub packets_dropped: u64,
    /// Total duplicate packets silently discarded.
    pub packets_duplicate: u64,
}

/// A jitter buffer whose playout depth adapts based on externally supplied
/// [`NetworkCondition`] measurements.
///
/// The target depth expands when the network is congested, has high RTT
/// variance, or is experiencing packet loss, and contracts slowly during
/// stable periods to minimise end-to-end latency.
///
/// # Depth computation
///
/// Each [`adapt`](Self::adapt) cycle the *ideal depth* is:
///
/// ```text
/// ideal = rtt_variance_ms × variance_multiplier
///       + loss_rate_pct × loss_penalty_ms_per_pct
///       + congestion_penalty_ms   (only when congested == true)
/// ```
///
/// This is then smoothed with EMA(`depth_ema_alpha`) and the actual target
/// is stepped toward the smoothed value at `expand_step_ms` or `shrink_step_ms`
/// per cycle (shrink only after `stable_cycles_before_shrink` stable cycles).
pub struct NetworkAwareJitterBuffer {
    config: NetworkAwareJitterConfig,
    buffer: BinaryHeap<JitterPacket>,
    target_depth_ms: u64,
    depth_ema_ms: f64,
    stable_cycles: u32,
    next_sequence: Option<u16>,
    stats: NetworkAwareJitterStats,
}

impl NetworkAwareJitterBuffer {
    /// Creates a new buffer with the given configuration.
    #[must_use]
    pub fn new(config: NetworkAwareJitterConfig) -> Self {
        let initial = config.initial_depth_ms;
        Self {
            buffer: BinaryHeap::new(),
            target_depth_ms: initial,
            depth_ema_ms: initial as f64,
            stable_cycles: 0,
            next_sequence: None,
            stats: NetworkAwareJitterStats {
                current_depth_ms: initial,
                ideal_depth_ms: initial as f64,
                ..Default::default()
            },
            config,
        }
    }

    /// Creates a buffer with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(NetworkAwareJitterConfig::default())
    }

    /// Adds a packet to the buffer.
    ///
    /// # Errors
    ///
    /// Returns [`VideoIpError::BufferOverflow`] when the buffer is full.
    pub fn add_packet(&mut self, packet: Packet) -> VideoIpResult<()> {
        if self
            .buffer
            .iter()
            .any(|jp| jp.packet.header.sequence == packet.header.sequence)
        {
            self.stats.packets_duplicate += 1;
            return Ok(());
        }
        if self.buffer.len() >= self.config.capacity {
            self.stats.packets_dropped += 1;
            return Err(VideoIpError::BufferOverflow);
        }
        if self.next_sequence.is_none() {
            self.next_sequence = Some(packet.header.sequence);
        }
        self.buffer.push(JitterPacket {
            packet,
            arrival_time: Instant::now(),
        });
        self.stats.packets_added += 1;
        Ok(())
    }

    /// Retrieves the next packet if its playout deadline has passed.
    ///
    /// Returns `None` if the buffer is empty or the oldest packet has not yet
    /// been held for `target_depth_ms` milliseconds.
    #[must_use]
    pub fn get_packet(&mut self) -> Option<Packet> {
        let oldest = self.buffer.peek()?;
        if oldest.arrival_time.elapsed() < Duration::from_millis(self.target_depth_ms) {
            return None;
        }
        let jp = self.buffer.pop()?;
        let seq = jp.packet.header.sequence;
        if let Some(expected) = self.next_sequence {
            self.next_sequence = Some(expected.wrapping_add(1));
            let _ = seq;
        } else {
            self.next_sequence = Some(seq.wrapping_add(1));
        }
        self.stats.packets_played += 1;
        Some(jp.packet)
    }

    /// Retrieves a packet immediately, bypassing the delay check.
    #[must_use]
    pub fn get_packet_immediate(&mut self) -> Option<Packet> {
        let jp = self.buffer.pop()?;
        self.stats.packets_played += 1;
        Some(jp.packet)
    }

    /// Runs one adaptation cycle.
    ///
    /// Should be called periodically (e.g. once per 50–200 ms).
    pub fn adapt(&mut self, cond: &NetworkCondition) {
        let loss_pct = cond.loss_rate * 100.0;
        let mut ideal = cond.rtt_variance_ms * self.config.variance_multiplier
            + loss_pct * self.config.loss_penalty_ms_per_pct;
        if cond.congested {
            ideal += self.config.congestion_penalty_ms;
        }
        ideal = ideal.max(self.config.min_depth_ms as f64);

        let alpha = self.config.depth_ema_alpha;
        self.depth_ema_ms = (1.0 - alpha) * self.depth_ema_ms + alpha * ideal;
        self.stats.ideal_depth_ms = self.depth_ema_ms;

        let target_f = self.target_depth_ms as f64;

        if self.depth_ema_ms > target_f + self.config.expand_step_ms as f64 {
            self.target_depth_ms =
                (self.target_depth_ms + self.config.expand_step_ms).min(self.config.max_depth_ms);
            self.stable_cycles = 0;
            self.stats.expand_steps += 1;
        } else if self.depth_ema_ms < target_f - self.config.shrink_step_ms as f64 {
            self.stable_cycles += 1;
            if self.stable_cycles >= self.config.stable_cycles_before_shrink {
                self.target_depth_ms = self
                    .target_depth_ms
                    .saturating_sub(self.config.shrink_step_ms)
                    .max(self.config.min_depth_ms);
                self.stable_cycles = 0;
                self.stats.shrink_steps += 1;
            }
        } else {
            self.stable_cycles = 0;
        }

        self.stats.current_depth_ms = self.target_depth_ms;
        self.stats.stable_cycles = self.stable_cycles;
    }

    /// Returns the current target playout depth in milliseconds.
    #[must_use]
    pub const fn target_depth_ms(&self) -> u64 {
        self.target_depth_ms
    }

    /// Returns the number of packets currently buffered.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buffer.len()
    }

    /// Returns `true` if no packets are buffered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    /// Returns a snapshot of buffer statistics.
    #[must_use]
    pub const fn stats(&self) -> &NetworkAwareJitterStats {
        &self.stats
    }

    /// Clears all buffered packets and resets the expected sequence tracker.
    pub fn clear(&mut self) {
        self.buffer.clear();
        self.next_sequence = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packet::PacketBuilder;
    use bytes::Bytes;
    use std::thread;

    #[test]
    fn test_jitter_buffer_creation() {
        let buffer = JitterBuffer::new(100, 20);
        assert_eq!(buffer.len(), 0);
        assert!(buffer.is_empty());
        assert_eq!(buffer.target_delay(), 20);
    }

    #[test]
    fn test_add_packet() {
        let mut buffer = JitterBuffer::new(100, 20);
        let packet = PacketBuilder::new(0)
            .video()
            .build(Bytes::from_static(b"test"))
            .expect("should succeed in test");

        buffer.add_packet(packet).expect("should succeed in test");
        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_buffer_overflow() {
        let mut buffer = JitterBuffer::new(2, 20);

        for i in 0..3 {
            let packet = PacketBuilder::new(i)
                .video()
                .build(Bytes::from_static(b"test"))
                .expect("should succeed in test");

            if i < 2 {
                buffer.add_packet(packet).expect("should succeed in test");
            } else {
                assert!(buffer.add_packet(packet).is_err());
            }
        }
    }

    #[test]
    fn test_duplicate_detection() {
        let mut buffer = JitterBuffer::new(100, 20);
        let packet = PacketBuilder::new(0)
            .video()
            .build(Bytes::from_static(b"test"))
            .expect("should succeed in test");

        buffer
            .add_packet(packet.clone())
            .expect("should succeed in test");
        buffer.add_packet(packet).expect("should succeed in test"); // Duplicate

        assert_eq!(buffer.stats().packets_duplicate, 1);
        assert_eq!(buffer.len(), 1); // Only one packet in buffer
    }

    #[test]
    fn test_get_packet_with_delay() {
        let mut buffer = JitterBuffer::new(100, 10);
        let packet = PacketBuilder::new(0)
            .video()
            .build(Bytes::from_static(b"test"))
            .expect("should succeed in test");

        buffer.add_packet(packet).expect("should succeed in test");

        // Should not be available immediately
        assert!(buffer.get_packet().is_none());

        // Wait for the delay
        thread::sleep(Duration::from_millis(15));

        // Should now be available
        assert!(buffer.get_packet().is_some());
    }

    #[test]
    fn test_get_packet_immediate() {
        let mut buffer = JitterBuffer::new(100, 100);
        let packet = PacketBuilder::new(0)
            .video()
            .build(Bytes::from_static(b"test"))
            .expect("should succeed in test");

        buffer.add_packet(packet).expect("should succeed in test");

        // Should be available immediately
        assert!(buffer.get_packet_immediate().is_some());
    }

    #[test]
    fn test_packet_ordering() {
        let mut buffer = JitterBuffer::new(100, 0);

        // Add packets out of order
        for seq in [2u16, 0, 1, 4, 3] {
            let packet = PacketBuilder::new(seq)
                .video()
                .build(Bytes::from_static(b"test"))
                .expect("should succeed in test");
            buffer.add_packet(packet).expect("should succeed in test");
        }

        // Should come out in order
        for expected in 0..5 {
            let packet = buffer
                .get_packet_immediate()
                .expect("should succeed in test");
            assert_eq!(packet.header.sequence, expected);
        }
    }

    #[test]
    fn test_statistics() {
        let mut buffer = JitterBuffer::new(100, 0);

        let packet = PacketBuilder::new(0)
            .video()
            .build(Bytes::from_static(b"test"))
            .expect("should succeed in test");
        buffer.add_packet(packet).expect("should succeed in test");

        assert_eq!(buffer.stats().packets_added, 1);
        assert_eq!(buffer.stats().buffer_occupancy, 1);

        let _ = buffer.get_packet_immediate();
        assert_eq!(buffer.stats().packets_played, 1);
    }

    #[test]
    fn test_clear() {
        let mut buffer = JitterBuffer::new(100, 20);
        let packet = PacketBuilder::new(0)
            .video()
            .build(Bytes::from_static(b"test"))
            .expect("should succeed in test");

        buffer.add_packet(packet).expect("should succeed in test");
        assert!(!buffer.is_empty());

        buffer.clear();
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_set_target_delay() {
        let mut buffer = JitterBuffer::new(100, 20);
        buffer.set_target_delay(50);
        assert_eq!(buffer.target_delay(), 50);
    }

    #[test]
    fn test_cleanup_old_packets() {
        let mut buffer = JitterBuffer::new(100, 0);

        let packet = PacketBuilder::new(0)
            .video()
            .build(Bytes::from_static(b"test"))
            .expect("should succeed in test");
        buffer.add_packet(packet).expect("should succeed in test");

        thread::sleep(Duration::from_millis(10));

        buffer.cleanup_old_packets(Duration::from_millis(5));
        assert_eq!(buffer.len(), 0);
        assert_eq!(buffer.stats().packets_dropped, 1);
    }

    // ── NetworkAwareJitterBuffer tests ─────────────────────────────────────────

    fn make_packet(seq: u16) -> Packet {
        PacketBuilder::new(seq)
            .video()
            .build(Bytes::from_static(b"netjitter"))
            .expect("packet build should succeed")
    }

    #[test]
    fn test_nab_creation_and_defaults() {
        let buf = NetworkAwareJitterBuffer::with_defaults();
        assert!(buf.is_empty());
        assert_eq!(buf.len(), 0);
        assert_eq!(buf.target_depth_ms(), 20);
    }

    #[test]
    fn test_nab_add_and_count() {
        let mut buf = NetworkAwareJitterBuffer::with_defaults();
        buf.add_packet(make_packet(0)).expect("add_packet ok");
        buf.add_packet(make_packet(1)).expect("add_packet ok");
        assert_eq!(buf.len(), 2);
        assert!(!buf.is_empty());
        assert_eq!(buf.stats().packets_added, 2);
    }

    #[test]
    fn test_nab_overflow_returns_error() {
        let config = NetworkAwareJitterConfig {
            capacity: 2,
            ..Default::default()
        };
        let mut buf = NetworkAwareJitterBuffer::new(config);
        buf.add_packet(make_packet(0)).expect("first add ok");
        buf.add_packet(make_packet(1)).expect("second add ok");
        let result = buf.add_packet(make_packet(2));
        assert!(result.is_err());
        assert_eq!(buf.stats().packets_dropped, 1);
    }

    #[test]
    fn test_nab_duplicate_not_double_counted() {
        let mut buf = NetworkAwareJitterBuffer::with_defaults();
        buf.add_packet(make_packet(5)).expect("add ok");
        buf.add_packet(make_packet(5)).expect("dup add ok");
        assert_eq!(buf.len(), 1);
        assert_eq!(buf.stats().packets_duplicate, 1);
    }

    #[test]
    fn test_nab_get_packet_before_deadline_returns_none() {
        let mut buf = NetworkAwareJitterBuffer::with_defaults();
        buf.add_packet(make_packet(0)).expect("add ok");
        assert!(buf.get_packet().is_none());
    }

    #[test]
    fn test_nab_get_packet_immediate_bypasses_delay() {
        let mut buf = NetworkAwareJitterBuffer::with_defaults();
        buf.add_packet(make_packet(0)).expect("add ok");
        let pkt = buf.get_packet_immediate();
        assert!(pkt.is_some());
        assert_eq!(buf.stats().packets_played, 1);
    }

    #[test]
    fn test_nab_clear_empties_buffer() {
        let mut buf = NetworkAwareJitterBuffer::with_defaults();
        for i in 0..5_u16 {
            buf.add_packet(make_packet(i)).expect("add ok");
        }
        buf.clear();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_nab_adapt_expands_on_congestion() {
        let config = NetworkAwareJitterConfig {
            initial_depth_ms: 20,
            expand_step_ms: 10,
            congestion_penalty_ms: 50.0,
            depth_ema_alpha: 1.0,
            min_depth_ms: 5,
            max_depth_ms: 200,
            ..Default::default()
        };
        let mut buf = NetworkAwareJitterBuffer::new(config);
        let initial = buf.target_depth_ms();
        let cond = NetworkCondition {
            rtt_ms: 30.0,
            rtt_variance_ms: 10.0,
            loss_rate: 0.05,
            congested: true,
        };
        buf.adapt(&cond);
        assert!(
            buf.target_depth_ms() >= initial,
            "depth should not decrease under congestion: {} >= {}",
            buf.target_depth_ms(),
            initial
        );
        assert!(buf.stats().expand_steps >= 1);
    }

    #[test]
    fn test_nab_adapt_shrinks_after_stable_cycles() {
        let config = NetworkAwareJitterConfig {
            initial_depth_ms: 100,
            shrink_step_ms: 5,
            stable_cycles_before_shrink: 3,
            expand_step_ms: 200,
            depth_ema_alpha: 1.0,
            variance_multiplier: 1.0,
            loss_penalty_ms_per_pct: 0.0,
            congestion_penalty_ms: 0.0,
            min_depth_ms: 5,
            max_depth_ms: 200,
            ..Default::default()
        };
        let mut buf = NetworkAwareJitterBuffer::new(config);
        let cond = NetworkCondition {
            rtt_ms: 5.0,
            rtt_variance_ms: 1.0,
            loss_rate: 0.0,
            congested: false,
        };
        for _ in 0..3 {
            buf.adapt(&cond);
        }
        assert!(
            buf.target_depth_ms() < 100,
            "depth should shrink after stable cycles, got {}",
            buf.target_depth_ms()
        );
        assert!(buf.stats().shrink_steps >= 1);
    }

    #[test]
    fn test_nab_depth_clamped_at_max() {
        let config = NetworkAwareJitterConfig {
            initial_depth_ms: 10,
            max_depth_ms: 50,
            expand_step_ms: 5,
            depth_ema_alpha: 1.0,
            congestion_penalty_ms: 1000.0,
            min_depth_ms: 5,
            ..Default::default()
        };
        let mut buf = NetworkAwareJitterBuffer::new(config);
        let cond = NetworkCondition {
            congested: true,
            rtt_variance_ms: 100.0,
            ..Default::default()
        };
        for _ in 0..50 {
            buf.adapt(&cond);
        }
        assert!(
            buf.target_depth_ms() <= 50,
            "depth must not exceed max_depth_ms, got {}",
            buf.target_depth_ms()
        );
    }

    #[test]
    fn test_nab_stats_track_expand_steps() {
        let config = NetworkAwareJitterConfig {
            initial_depth_ms: 5,
            max_depth_ms: 200,
            expand_step_ms: 5,
            depth_ema_alpha: 1.0,
            variance_multiplier: 3.0,
            congestion_penalty_ms: 50.0,
            min_depth_ms: 5,
            ..Default::default()
        };
        let mut buf = NetworkAwareJitterBuffer::new(config);
        let cond = NetworkCondition {
            rtt_variance_ms: 30.0,
            congested: true,
            ..Default::default()
        };
        buf.adapt(&cond);
        assert!(buf.stats().expand_steps >= 1);
    }
}
