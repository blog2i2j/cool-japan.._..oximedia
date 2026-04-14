//! Network simulation for testing congestion control, FEC recovery, and
//! jitter buffer behaviour under realistic network conditions.
//!
//! This module provides configurable network impairment models that can
//! inject latency, jitter, packet loss (random, burst, periodic), bandwidth
//! limits, and reordering into a packet stream.  It is designed for
//! deterministic, repeatable tests without requiring real sockets.
//!
//! # Architecture
//!
//! ```text
//! ┌──────────┐       ┌────────────┐       ┌──────────────┐
//! │  Sender  │──────>│ NetworkSim │──────>│  Receiver    │
//! └──────────┘       │  (impair)  │       └──────────────┘
//!                    └────────────┘
//! ```

#![allow(dead_code)]

use crate::error::{VideoIpError, VideoIpResult};
use bytes::Bytes;
use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Loss models
// ---------------------------------------------------------------------------

/// Packet loss model.
#[derive(Debug, Clone)]
pub enum LossModel {
    /// No loss.
    None,
    /// Uniform random loss at the given probability (0.0..1.0).
    Random {
        /// Probability of dropping a packet.
        probability: f64,
    },
    /// Gilbert-Elliott two-state burst model.
    Burst {
        /// Probability of transitioning from good to bad state.
        p_good_to_bad: f64,
        /// Probability of transitioning from bad to good state.
        p_bad_to_good: f64,
        /// Loss probability in good state.
        loss_in_good: f64,
        /// Loss probability in bad state.
        loss_in_bad: f64,
    },
    /// Periodic loss – drop every Nth packet.
    Periodic {
        /// Period.
        period: usize,
        /// Offset within the period at which to drop.
        offset: usize,
    },
}

impl Default for LossModel {
    fn default() -> Self {
        Self::None
    }
}

/// State for the Gilbert-Elliott burst loss model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GilbertState {
    Good,
    Bad,
}

// ---------------------------------------------------------------------------
// Impairment profile
// ---------------------------------------------------------------------------

/// Configurable network impairment profile.
#[derive(Debug, Clone)]
pub struct NetworkProfile {
    /// One-way latency in microseconds.
    pub latency_us: u64,
    /// Jitter amplitude in microseconds (uniform +/- jitter).
    pub jitter_us: u64,
    /// Packet loss model.
    pub loss_model: LossModel,
    /// Bandwidth cap in bytes per second (0 = unlimited).
    pub bandwidth_bps: u64,
    /// Reorder probability (probability a packet is delayed by one slot).
    pub reorder_probability: f64,
    /// Duplicate probability (probability a packet is sent twice).
    pub duplicate_probability: f64,
}

impl Default for NetworkProfile {
    fn default() -> Self {
        Self {
            latency_us: 0,
            jitter_us: 0,
            loss_model: LossModel::None,
            bandwidth_bps: 0,
            reorder_probability: 0.0,
            duplicate_probability: 0.0,
        }
    }
}

impl NetworkProfile {
    /// Perfect network (no impairments).
    #[must_use]
    pub fn perfect() -> Self {
        Self::default()
    }

    /// Simulates a LAN environment (< 1ms latency, ~0.01% loss).
    #[must_use]
    pub fn lan() -> Self {
        Self {
            latency_us: 200,
            jitter_us: 50,
            loss_model: LossModel::Random {
                probability: 0.0001,
            },
            bandwidth_bps: 1_000_000_000, // 1 Gbps
            reorder_probability: 0.0,
            duplicate_probability: 0.0,
        }
    }

    /// Simulates a WAN environment (~20ms, ~1% loss).
    #[must_use]
    pub fn wan() -> Self {
        Self {
            latency_us: 20_000,
            jitter_us: 5_000,
            loss_model: LossModel::Random {
                probability: 0.01,
            },
            bandwidth_bps: 100_000_000, // 100 Mbps
            reorder_probability: 0.005,
            duplicate_probability: 0.0,
        }
    }

    /// Simulates a lossy WiFi link (~5ms, ~3% loss with bursts).
    #[must_use]
    pub fn lossy_wifi() -> Self {
        Self {
            latency_us: 5_000,
            jitter_us: 10_000,
            loss_model: LossModel::Burst {
                p_good_to_bad: 0.02,
                p_bad_to_good: 0.3,
                loss_in_good: 0.001,
                loss_in_bad: 0.25,
            },
            bandwidth_bps: 50_000_000, // 50 Mbps
            reorder_probability: 0.01,
            duplicate_probability: 0.001,
        }
    }

    /// Validates the profile parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if any parameter is out of range.
    pub fn validate(&self) -> VideoIpResult<()> {
        if self.reorder_probability < 0.0 || self.reorder_probability > 1.0 {
            return Err(VideoIpError::InvalidVideoConfig(
                "reorder_probability must be 0.0..=1.0".into(),
            ));
        }
        if self.duplicate_probability < 0.0 || self.duplicate_probability > 1.0 {
            return Err(VideoIpError::InvalidVideoConfig(
                "duplicate_probability must be 0.0..=1.0".into(),
            ));
        }
        match &self.loss_model {
            LossModel::Random { probability } => {
                if *probability < 0.0 || *probability > 1.0 {
                    return Err(VideoIpError::InvalidVideoConfig(
                        "random loss probability must be 0.0..=1.0".into(),
                    ));
                }
            }
            LossModel::Burst {
                p_good_to_bad,
                p_bad_to_good,
                loss_in_good,
                loss_in_bad,
            } => {
                for (name, val) in [
                    ("p_good_to_bad", p_good_to_bad),
                    ("p_bad_to_good", p_bad_to_good),
                    ("loss_in_good", loss_in_good),
                    ("loss_in_bad", loss_in_bad),
                ] {
                    if *val < 0.0 || *val > 1.0 {
                        return Err(VideoIpError::InvalidVideoConfig(format!(
                            "{name} must be 0.0..=1.0"
                        )));
                    }
                }
            }
            LossModel::Periodic { period, .. } => {
                if *period == 0 {
                    return Err(VideoIpError::InvalidVideoConfig(
                        "periodic loss period must be > 0".into(),
                    ));
                }
            }
            LossModel::None => {}
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Simulated packet
// ---------------------------------------------------------------------------

/// A packet transiting through the simulated network.
#[derive(Debug, Clone)]
pub struct SimPacket {
    /// Sequence number assigned by the simulator.
    pub seq: u64,
    /// Payload data.
    pub data: Bytes,
    /// Delivery time in microseconds from simulation start.
    pub delivery_time_us: u64,
    /// Whether this is a duplicate injected by the simulator.
    pub is_duplicate: bool,
}

// ---------------------------------------------------------------------------
// Deterministic PRNG (xorshift64)
// ---------------------------------------------------------------------------

/// Simple xorshift64 PRNG for deterministic simulation.
#[derive(Debug, Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a value in `[0.0, 1.0)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Returns a value in `[low, high)`.
    fn next_range(&mut self, low: u64, high: u64) -> u64 {
        if high <= low {
            return low;
        }
        low + self.next_u64() % (high - low)
    }
}

// ---------------------------------------------------------------------------
// Network simulator
// ---------------------------------------------------------------------------

/// Deterministic network simulator.
///
/// Feed packets in with [`NetworkSimulator::send`] and drain delivered
/// packets with [`NetworkSimulator::receive`].
#[derive(Debug)]
pub struct NetworkSimulator {
    profile: NetworkProfile,
    rng: Xorshift64,
    /// In-flight packets ordered by delivery time.
    in_flight: VecDeque<SimPacket>,
    /// Current simulation time in microseconds.
    clock_us: u64,
    /// Next sequence number.
    next_seq: u64,
    /// Packet counter (for periodic loss).
    packet_counter: usize,
    /// Gilbert-Elliott state.
    gilbert_state: GilbertState,
    /// Stats.
    stats: SimStats,
    /// Reorder hold slot (holds one packet for potential swap).
    reorder_hold: Option<SimPacket>,
}

/// Cumulative simulation statistics.
#[derive(Debug, Clone, Default)]
pub struct SimStats {
    /// Total packets submitted.
    pub total_sent: u64,
    /// Packets dropped by loss model.
    pub total_dropped: u64,
    /// Packets delivered.
    pub total_delivered: u64,
    /// Packets duplicated.
    pub total_duplicated: u64,
    /// Packets reordered.
    pub total_reordered: u64,
}

impl SimStats {
    /// Returns the effective loss ratio.
    #[must_use]
    pub fn loss_ratio(&self) -> f64 {
        if self.total_sent == 0 {
            return 0.0;
        }
        self.total_dropped as f64 / self.total_sent as f64
    }
}

impl NetworkSimulator {
    /// Creates a new simulator with the given profile and PRNG seed.
    ///
    /// # Errors
    ///
    /// Returns an error if the profile is invalid.
    pub fn new(profile: NetworkProfile, seed: u64) -> VideoIpResult<Self> {
        profile.validate()?;
        Ok(Self {
            profile,
            rng: Xorshift64::new(seed),
            in_flight: VecDeque::new(),
            clock_us: 0,
            next_seq: 0,
            packet_counter: 0,
            gilbert_state: GilbertState::Good,
            stats: SimStats::default(),
            reorder_hold: None,
        })
    }

    /// Advances the simulation clock.
    pub fn advance_clock(&mut self, delta_us: u64) {
        self.clock_us = self.clock_us.saturating_add(delta_us);
    }

    /// Returns current clock value.
    #[must_use]
    pub fn clock_us(&self) -> u64 {
        self.clock_us
    }

    /// Returns simulation statistics.
    #[must_use]
    pub fn stats(&self) -> &SimStats {
        &self.stats
    }

    /// Submits a packet into the simulated network.
    ///
    /// The packet may be dropped, delayed, reordered, or duplicated
    /// according to the configured [`NetworkProfile`].
    pub fn send(&mut self, data: Bytes) {
        self.stats.total_sent += 1;
        self.packet_counter += 1;

        // --- loss decision ---
        if self.should_drop() {
            self.stats.total_dropped += 1;
            return;
        }

        // --- compute delivery time ---
        let base_latency = self.profile.latency_us;
        let jitter = if self.profile.jitter_us > 0 {
            self.rng.next_range(0, self.profile.jitter_us * 2)
        } else {
            0
        };
        let latency = base_latency.saturating_add(jitter).saturating_sub(self.profile.jitter_us);

        // --- bandwidth delay ---
        let bw_delay = {
            let bits = (data.len() as u64) * 8;
            (bits * 1_000_000).checked_div(self.profile.bandwidth_bps).unwrap_or(0)
        };

        let delivery = self.clock_us.saturating_add(latency).saturating_add(bw_delay);

        let seq = self.next_seq;
        self.next_seq += 1;

        let pkt = SimPacket {
            seq,
            data: data.clone(),
            delivery_time_us: delivery,
            is_duplicate: false,
        };

        // --- reorder ---
        if self.rng.next_f64() < self.profile.reorder_probability {
            self.stats.total_reordered += 1;
            if let Some(held) = self.reorder_hold.take() {
                // Deliver the held packet *after* this one by swapping delivery times.
                let mut new_pkt = pkt;
                let mut old_pkt = held;
                if new_pkt.delivery_time_us > old_pkt.delivery_time_us {
                    std::mem::swap(
                        &mut new_pkt.delivery_time_us,
                        &mut old_pkt.delivery_time_us,
                    );
                }
                self.insert_sorted(old_pkt);
                self.insert_sorted(new_pkt);
            } else {
                self.reorder_hold = Some(pkt);
            }
        } else {
            // Flush any held packet first.
            if let Some(held) = self.reorder_hold.take() {
                self.insert_sorted(held);
            }
            self.insert_sorted(pkt);
        }

        // --- duplicate ---
        if self.rng.next_f64() < self.profile.duplicate_probability {
            self.stats.total_duplicated += 1;
            let dup_delivery = delivery.saturating_add(
                self.rng.next_range(0, self.profile.latency_us.max(100)),
            );
            let dup = SimPacket {
                seq,
                data,
                delivery_time_us: dup_delivery,
                is_duplicate: true,
            };
            self.insert_sorted(dup);
        }
    }

    /// Drains all packets whose delivery time is <= current clock.
    pub fn receive(&mut self) -> Vec<SimPacket> {
        let mut delivered = Vec::new();
        while let Some(front) = self.in_flight.front() {
            if front.delivery_time_us <= self.clock_us {
                if let Some(pkt) = self.in_flight.pop_front() {
                    self.stats.total_delivered += 1;
                    delivered.push(pkt);
                }
            } else {
                break;
            }
        }
        delivered
    }

    /// Returns the number of packets currently in flight.
    #[must_use]
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Flushes any packet held in the reorder slot.
    pub fn flush_held(&mut self) {
        if let Some(held) = self.reorder_hold.take() {
            self.insert_sorted(held);
        }
    }

    // --- private helpers ---

    fn should_drop(&mut self) -> bool {
        match &self.profile.loss_model {
            LossModel::None => false,
            LossModel::Random { probability } => self.rng.next_f64() < *probability,
            LossModel::Burst {
                p_good_to_bad,
                p_bad_to_good,
                loss_in_good,
                loss_in_bad,
            } => {
                // State transition.
                let transition = self.rng.next_f64();
                match self.gilbert_state {
                    GilbertState::Good => {
                        if transition < *p_good_to_bad {
                            self.gilbert_state = GilbertState::Bad;
                        }
                    }
                    GilbertState::Bad => {
                        if transition < *p_bad_to_good {
                            self.gilbert_state = GilbertState::Good;
                        }
                    }
                }
                let loss_prob = match self.gilbert_state {
                    GilbertState::Good => *loss_in_good,
                    GilbertState::Bad => *loss_in_bad,
                };
                self.rng.next_f64() < loss_prob
            }
            LossModel::Periodic { period, offset } => {
                (self.packet_counter.wrapping_sub(1) % *period) == *offset
            }
        }
    }

    fn insert_sorted(&mut self, pkt: SimPacket) {
        // Most packets arrive in order, so search from the back.
        let pos = self
            .in_flight
            .iter()
            .rposition(|p| p.delivery_time_us <= pkt.delivery_time_us)
            .map_or(0, |i| i + 1);
        self.in_flight.insert(pos, pkt);
    }
}

// ---------------------------------------------------------------------------
// Bandwidth measurement helper
// ---------------------------------------------------------------------------

/// Measures throughput over a sliding window of delivered packets.
#[derive(Debug)]
pub struct ThroughputMeter {
    /// Window of (timestamp_us, bytes) samples.
    samples: VecDeque<(u64, usize)>,
    /// Window duration in microseconds.
    window_us: u64,
}

impl ThroughputMeter {
    /// Creates a new throughput meter with the given window size.
    #[must_use]
    pub fn new(window_us: u64) -> Self {
        Self {
            samples: VecDeque::new(),
            window_us,
        }
    }

    /// Records a delivered datagram.
    pub fn record(&mut self, timestamp_us: u64, bytes: usize) {
        self.samples.push_back((timestamp_us, bytes));
        // Evict old samples.
        let cutoff = timestamp_us.saturating_sub(self.window_us);
        while let Some(&(ts, _)) = self.samples.front() {
            if ts < cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Returns current throughput in bytes per second.
    #[must_use]
    pub fn throughput_bps(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let first_ts = self.samples.front().map_or(0, |s| s.0);
        let last_ts = self.samples.back().map_or(0, |s| s.0);
        let span = last_ts.saturating_sub(first_ts);
        if span == 0 {
            return 0.0;
        }
        let total_bytes: usize = self.samples.iter().map(|s| s.1).sum();
        (total_bytes as f64) * 1_000_000.0 / (span as f64)
    }

    /// Returns the number of samples in the window.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_network() {
        let profile = NetworkProfile::perfect();
        let mut sim = NetworkSimulator::new(profile, 42).expect("create sim");
        for i in 0..10 {
            sim.send(Bytes::from(vec![i; 100]));
        }
        // All should be deliverable immediately (0 latency).
        let delivered = sim.receive();
        assert_eq!(delivered.len(), 10);
        assert_eq!(sim.stats().total_dropped, 0);
    }

    #[test]
    fn test_latency_delivery() {
        let profile = NetworkProfile {
            latency_us: 1000,
            ..Default::default()
        };
        let mut sim = NetworkSimulator::new(profile, 1).expect("create sim");
        sim.send(Bytes::from_static(b"hello"));

        // Not yet delivered.
        let early = sim.receive();
        assert!(early.is_empty());

        // Advance past latency.
        sim.advance_clock(1001);
        let delivered = sim.receive();
        assert_eq!(delivered.len(), 1);
        assert_eq!(&delivered[0].data[..], b"hello");
    }

    #[test]
    fn test_periodic_loss() {
        let profile = NetworkProfile {
            loss_model: LossModel::Periodic {
                period: 5,
                offset: 0,
            },
            ..Default::default()
        };
        let mut sim = NetworkSimulator::new(profile, 1).expect("create sim");
        for i in 0u8..20 {
            sim.send(Bytes::from(vec![i; 50]));
        }
        // Every 5th packet is dropped (packets 1, 6, 11, 16 in 1-indexed).
        let delivered = sim.receive();
        assert_eq!(delivered.len(), 16);
        assert_eq!(sim.stats().total_dropped, 4);
    }

    #[test]
    fn test_periodic_loss_with_offset() {
        let profile = NetworkProfile {
            loss_model: LossModel::Periodic {
                period: 4,
                offset: 2,
            },
            ..Default::default()
        };
        let mut sim = NetworkSimulator::new(profile, 1).expect("create sim");
        for i in 0u8..12 {
            sim.send(Bytes::from(vec![i; 10]));
        }
        // Drops at counter positions 3, 7, 11 (0-indexed counter mod 4 == 2).
        let delivered = sim.receive();
        assert_eq!(delivered.len(), 9);
        assert_eq!(sim.stats().total_dropped, 3);
    }

    #[test]
    fn test_bandwidth_limiting() {
        let profile = NetworkProfile {
            bandwidth_bps: 8_000, // 8 kbps = 1 KBps
            ..Default::default()
        };
        let mut sim = NetworkSimulator::new(profile, 1).expect("create sim");
        // 1000 bytes at 1 KBps should take 1 second = 1_000_000 us.
        sim.send(Bytes::from(vec![0u8; 1000]));

        // Should not be delivered at t=0.
        let early = sim.receive();
        assert!(early.is_empty());

        // Should be delivered after 1 second.
        sim.advance_clock(1_000_001);
        let delivered = sim.receive();
        assert_eq!(delivered.len(), 1);
    }

    #[test]
    fn test_sim_stats_loss_ratio() {
        let stats = SimStats {
            total_sent: 100,
            total_dropped: 10,
            total_delivered: 90,
            total_duplicated: 0,
            total_reordered: 0,
        };
        let ratio = stats.loss_ratio();
        assert!((ratio - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_sim_stats_loss_ratio_zero_sent() {
        let stats = SimStats::default();
        assert_eq!(stats.loss_ratio(), 0.0);
    }

    #[test]
    fn test_network_profile_validate_ok() {
        assert!(NetworkProfile::lan().validate().is_ok());
        assert!(NetworkProfile::wan().validate().is_ok());
        assert!(NetworkProfile::lossy_wifi().validate().is_ok());
    }

    #[test]
    fn test_network_profile_validate_bad_reorder() {
        let mut p = NetworkProfile::default();
        p.reorder_probability = 1.5;
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_network_profile_validate_bad_periodic() {
        let p = NetworkProfile {
            loss_model: LossModel::Periodic {
                period: 0,
                offset: 0,
            },
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn test_throughput_meter() {
        let mut meter = ThroughputMeter::new(1_000_000); // 1 second window
        meter.record(0, 1000);
        meter.record(500_000, 1000);
        meter.record(1_000_000, 1000);
        // 3000 bytes over 1 second => 3000 bytes/sec.
        let bps = meter.throughput_bps();
        assert!((bps - 3000.0).abs() < 1.0);
    }

    #[test]
    fn test_throughput_meter_eviction() {
        let mut meter = ThroughputMeter::new(100);
        meter.record(0, 500);
        meter.record(50, 500);
        meter.record(120, 500);
        // cutoff = 120 - 100 = 20, evicts ts=0 (0 < 20), keeps ts=50 and ts=120
        assert_eq!(meter.sample_count(), 2);
    }

    #[test]
    fn test_throughput_meter_empty() {
        let meter = ThroughputMeter::new(1_000_000);
        assert_eq!(meter.throughput_bps(), 0.0);
    }

    #[test]
    fn test_xorshift_deterministic() {
        let mut rng1 = Xorshift64::new(12345);
        let mut rng2 = Xorshift64::new(12345);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_flush_held_packet() {
        let profile = NetworkProfile {
            reorder_probability: 1.0, // Always reorder.
            ..Default::default()
        };
        let mut sim = NetworkSimulator::new(profile, 42).expect("create sim");
        sim.send(Bytes::from_static(b"first"));
        // The first packet goes into reorder_hold.
        assert_eq!(sim.in_flight_count(), 0);

        sim.flush_held();
        assert_eq!(sim.in_flight_count(), 1);

        let delivered = sim.receive();
        assert_eq!(delivered.len(), 1);
    }
}
