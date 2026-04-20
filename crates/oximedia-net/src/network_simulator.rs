//! Deterministic network condition simulator for testing adaptive bitrate algorithms.
//!
//! This module provides a fully deterministic, pure-Rust network simulator that
//! models:
//!
//! - **Bandwidth** — configurable base bandwidth with optional sinusoidal
//!   fluctuation, stepped profiles, or trace-driven replay.
//! - **Latency** — round-trip time with configurable jitter (Gaussian noise).
//! - **Packet loss** — probabilistic independent loss and Gilbert-Elliott
//!   bursty-loss model.
//! - **Packet reordering** — percentage of packets that arrive out-of-order.
//! - **Transmission queue** — FIFO queue with configurable max depth;
//!   overflow drops packets.
//!
//! All randomness is produced by a deterministic LCG (linear congruential
//! generator) seeded at construction, so tests are fully reproducible without
//! any external crate.
//!
//! ## Architecture
//!
//! ```text
//!  Application
//!      │  send_packet(payload, size_bytes)
//!      ▼
//!  NetworkSimulator
//!      │  ┌─────────────────────────────────────────┐
//!      │  │ 1. Apply bandwidth → compute tx_time    │
//!      │  │ 2. Apply loss      → maybe drop         │
//!      │  │ 3. Apply latency   → schedule delivery  │
//!      │  │ 4. Apply reorder   → maybe swap last 2  │
//!      │  └─────────────────────────────────────────┘
//!      ▼
//!  receive_ready(now) → Vec<SimPacket>   // packets whose delivery_time ≤ now
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_net::network_simulator::{NetworkSimulator, SimConfig, BandwidthProfile};
//! use std::time::Duration;
//!
//! let config = SimConfig {
//!     bandwidth_profile: BandwidthProfile::Constant(2_000_000.0),
//!     base_latency: Duration::from_millis(50),
//!     ..SimConfig::default()
//! };
//! let mut sim = NetworkSimulator::new(config, 42);
//!
//! sim.send_packet(b"hello", 1400, Duration::from_millis(0)).expect("valid packet within limits");
//! let ready = sim.receive_ready(Duration::from_millis(200));
//! assert!(!ready.is_empty());
//! ```

#![allow(dead_code)]

use std::collections::VecDeque;
use std::time::Duration;

use crate::error::{NetError, NetResult};

// ─── LCG random number generator ─────────────────────────────────────────────

/// Minimal, deterministic linear congruential generator.
///
/// Uses the parameters from Numerical Recipes:
///   `x_{n+1} = 1664525 * x_n + 1013904223  (mod 2^32)`
#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Returns the next pseudo-random `u64`.
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Returns a uniform float in `[0, 1)`.
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Box-Muller transform: returns a standard-normal sample.
    fn next_normal(&mut self) -> f64 {
        let u1 = (self.next_f64() + 1e-300).max(1e-300);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    /// Returns `true` with probability `p`.
    fn bernoulli(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}

// ─── Bandwidth profiles ───────────────────────────────────────────────────────

/// Defines how available bandwidth changes over simulation time.
#[derive(Debug, Clone)]
pub enum BandwidthProfile {
    /// Fixed bandwidth (bits/s) for the entire simulation.
    Constant(f64),
    /// Sinusoidal fluctuation: `base + amplitude * sin(2π * t / period)`.
    Sinusoidal {
        /// Mean bandwidth in bits/s.
        base_bps: f64,
        /// Amplitude in bits/s (peak deviation from base).
        amplitude_bps: f64,
        /// Period in seconds.
        period_secs: f64,
    },
    /// Stepped profile: list of `(start_time, bandwidth_bps)` segments.
    /// The last entry applies for all subsequent time.
    Stepped(Vec<(Duration, f64)>),
    /// Trace-driven: bandwidth sampled at fixed intervals.
    Trace {
        /// Bandwidth samples in bits/s.
        samples: Vec<f64>,
        /// Duration between samples.
        interval: Duration,
    },
}

impl BandwidthProfile {
    /// Returns bandwidth (bits/s) at simulation time `t`.
    #[must_use]
    pub fn bandwidth_at(&self, t: Duration) -> f64 {
        match self {
            Self::Constant(bps) => *bps,
            Self::Sinusoidal {
                base_bps,
                amplitude_bps,
                period_secs,
            } => {
                if *period_secs <= 0.0 {
                    return *base_bps;
                }
                let t_secs = t.as_secs_f64();
                let sine = (2.0 * std::f64::consts::PI * t_secs / period_secs).sin();
                (base_bps + amplitude_bps * sine).max(0.0)
            }
            Self::Stepped(steps) => {
                let mut bw = 0.0f64;
                for (start, bps) in steps {
                    if t >= *start {
                        bw = *bps;
                    } else {
                        break;
                    }
                }
                bw
            }
            Self::Trace { samples, interval } => {
                if samples.is_empty() || interval.is_zero() {
                    return 0.0;
                }
                let idx = (t.as_secs_f64() / interval.as_secs_f64()) as usize;
                samples[idx.min(samples.len() - 1)]
            }
        }
    }
}

// ─── Loss model ───────────────────────────────────────────────────────────────

/// Packet loss model.
#[derive(Debug, Clone)]
pub enum LossModel {
    /// Independent Bernoulli loss: each packet is independently dropped with
    /// probability `loss_rate`.
    Independent {
        /// Loss probability per packet (0.0 = no loss, 1.0 = all packets lost).
        loss_rate: f64,
    },
    /// Gilbert-Elliott 2-state Markov model.
    ///
    /// States: **Good** (low loss) and **Bad** (high loss).
    GilbertElliott {
        /// Transition probability from Good → Bad per packet.
        p_good_to_bad: f64,
        /// Transition probability from Bad → Good per packet.
        p_bad_to_good: f64,
        /// Loss probability in Good state.
        loss_in_good: f64,
        /// Loss probability in Bad state.
        loss_in_bad: f64,
    },
    /// No packet loss.
    None,
}

impl LossModel {
    /// Predefined 1% independent loss.
    #[must_use]
    pub fn one_percent() -> Self {
        Self::Independent { loss_rate: 0.01 }
    }

    /// Predefined 5% bursty loss using Gilbert-Elliott.
    #[must_use]
    pub fn five_percent_bursty() -> Self {
        Self::GilbertElliott {
            p_good_to_bad: 0.02,
            p_bad_to_good: 0.3,
            loss_in_good: 0.0,
            loss_in_bad: 0.5,
        }
    }
}

// ─── Simulator configuration ──────────────────────────────────────────────────

/// Network simulator configuration.
#[derive(Debug, Clone)]
pub struct SimConfig {
    /// Bandwidth profile.
    pub bandwidth_profile: BandwidthProfile,
    /// One-way propagation latency (constant component).
    pub base_latency: Duration,
    /// Latency jitter standard deviation.  Each packet's delivery time is
    /// `base_latency + N(0, jitter_stddev)` (clamped to 0).
    pub jitter_stddev: Duration,
    /// Packet loss model.
    pub loss_model: LossModel,
    /// Fraction of packets that will be reordered (swapped with the previous
    /// packet in the delivery queue).
    pub reorder_rate: f64,
    /// Maximum number of packets in the transmission queue.  Additional
    /// packets are dropped (tail-drop).
    pub queue_max_packets: usize,
    /// If `true`, the simulator tracks per-packet delivery times precisely.
    /// If `false`, all packets in a burst are delivered at the same time.
    pub accurate_timing: bool,
}

impl Default for SimConfig {
    fn default() -> Self {
        Self {
            bandwidth_profile: BandwidthProfile::Constant(10_000_000.0),
            base_latency: Duration::from_millis(30),
            jitter_stddev: Duration::from_millis(5),
            loss_model: LossModel::None,
            reorder_rate: 0.0,
            queue_max_packets: 1000,
            accurate_timing: true,
        }
    }
}

// ─── Packet types ─────────────────────────────────────────────────────────────

/// A packet in transit inside the simulator.
#[derive(Debug, Clone)]
struct InFlightPacket {
    /// Unique packet identifier.
    id: u64,
    /// Payload bytes.
    payload: Vec<u8>,
    /// Nominal payload size (may differ from `payload.len()` if truncated).
    size_bytes: usize,
    /// Wall-clock time at which this packet was sent.
    sent_at: Duration,
    /// Scheduled delivery time.
    delivery_time: Duration,
}

/// A packet that has been received (delivered by the simulator).
#[derive(Debug, Clone)]
pub struct SimPacket {
    /// Unique packet identifier (monotonically increasing).
    pub id: u64,
    /// Payload bytes.
    pub payload: Vec<u8>,
    /// Nominal size in bytes.
    pub size_bytes: usize,
    /// Time at which the packet was originally sent.
    pub sent_at: Duration,
    /// Time at which the packet was delivered.
    pub delivered_at: Duration,
    /// One-way latency for this packet.
    pub latency: Duration,
}

// ─── Statistics ───────────────────────────────────────────────────────────────

/// Running statistics collected by the simulator.
#[derive(Debug, Clone, Default)]
pub struct SimStats {
    /// Total packets submitted.
    pub packets_sent: u64,
    /// Total packets that were dropped (loss + queue overflow).
    pub packets_dropped: u64,
    /// Total packets delivered.
    pub packets_delivered: u64,
    /// Total bytes transmitted (delivered).
    pub bytes_delivered: u64,
    /// Cumulative one-way latency of delivered packets (microseconds).
    pub cumulative_latency_us: u64,
    /// Queue overflow drops.
    pub queue_overflow_drops: u64,
    /// Loss model drops.
    pub loss_drops: u64,
}

impl SimStats {
    /// Mean one-way latency of delivered packets.
    #[must_use]
    pub fn mean_latency(&self) -> Option<Duration> {
        if self.packets_delivered == 0 {
            return None;
        }
        let mean_us = self.cumulative_latency_us / self.packets_delivered;
        Some(Duration::from_micros(mean_us))
    }

    /// Observed loss rate (fraction of sent packets that were dropped).
    #[must_use]
    pub fn loss_rate(&self) -> f64 {
        if self.packets_sent == 0 {
            return 0.0;
        }
        self.packets_dropped as f64 / self.packets_sent as f64
    }

    /// Effective goodput in bits/s over duration `elapsed`.
    #[must_use]
    pub fn goodput_bps(&self, elapsed: Duration) -> f64 {
        let secs = elapsed.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        (self.bytes_delivered * 8) as f64 / secs
    }
}

// ─── Main simulator ───────────────────────────────────────────────────────────

/// Deterministic network condition simulator.
pub struct NetworkSimulator {
    config: SimConfig,
    rng: Lcg,
    /// Packets in transit (ordered by `delivery_time`).
    in_flight: VecDeque<InFlightPacket>,
    /// Next packet ID.
    next_id: u64,
    /// Gilbert-Elliott state: `true` = Bad state.
    ge_bad_state: bool,
    /// Running statistics.
    stats: SimStats,
    /// Simulated "current time" — advances only when the user calls
    /// `advance_time` or when packets are sent.
    current_time: Duration,
}

impl NetworkSimulator {
    /// Creates a new simulator with the given configuration and RNG seed.
    ///
    /// Using the same `seed` always produces identical packet delivery
    /// sequences, enabling reproducible tests.
    #[must_use]
    pub fn new(config: SimConfig, seed: u64) -> Self {
        Self {
            config,
            rng: Lcg::new(seed),
            in_flight: VecDeque::new(),
            next_id: 0,
            ge_bad_state: false,
            stats: SimStats::default(),
            current_time: Duration::ZERO,
        }
    }

    /// Returns the current simulated time.
    #[must_use]
    pub fn current_time(&self) -> Duration {
        self.current_time
    }

    /// Advances the simulated clock to `new_time`.
    ///
    /// Returns any packets whose delivery time ≤ `new_time` (does NOT remove
    /// them from the queue — call [`Self::receive_ready`] for that).
    ///
    /// # Errors
    ///
    /// Returns [`NetError::InvalidState`] if `new_time` is before the current time.
    pub fn advance_time(&mut self, new_time: Duration) -> NetResult<()> {
        if new_time < self.current_time {
            return Err(NetError::invalid_state(format!(
                "advance_time: new_time {:?} < current_time {:?}",
                new_time, self.current_time
            )));
        }
        self.current_time = new_time;
        Ok(())
    }

    /// Submits a packet into the simulated network.
    ///
    /// `payload` — raw bytes to carry.
    /// `size_bytes` — nominal wire size (used for bandwidth consumption; can
    ///   differ from `payload.len()` to model headers/padding).
    /// `send_time` — the time at which the application sends this packet.
    ///   Must be ≥ `current_time`.
    ///
    /// Returns the assigned packet ID, or `None` if the packet was dropped
    /// (loss or queue overflow).
    ///
    /// # Errors
    ///
    /// Returns [`NetError::InvalidState`] if `send_time < current_time`.
    pub fn send_packet(
        &mut self,
        payload: &[u8],
        size_bytes: usize,
        send_time: Duration,
    ) -> NetResult<Option<u64>> {
        if send_time < self.current_time {
            return Err(NetError::invalid_state(format!(
                "send_packet: send_time {:?} < current_time {:?}",
                send_time, self.current_time
            )));
        }

        self.stats.packets_sent += 1;

        // Queue overflow check
        if self.in_flight.len() >= self.config.queue_max_packets {
            self.stats.packets_dropped += 1;
            self.stats.queue_overflow_drops += 1;
            return Ok(None);
        }

        // Loss decision
        if self.should_drop() {
            self.stats.packets_dropped += 1;
            self.stats.loss_drops += 1;
            return Ok(None);
        }

        // Bandwidth-based transmission time
        let bw = self
            .config
            .bandwidth_profile
            .bandwidth_at(send_time)
            .max(1.0);
        let tx_secs = (size_bytes as f64 * 8.0) / bw;
        let tx_duration = Duration::from_secs_f64(tx_secs);

        // Latency with jitter
        let base_us = self.config.base_latency.as_micros() as f64;
        let jitter_us = self.config.jitter_stddev.as_micros() as f64;
        let jitter_sample = if jitter_us > 0.0 {
            self.rng.next_normal() * jitter_us
        } else {
            0.0
        };
        let latency_us = (base_us + jitter_sample).max(0.0) as u64;
        let latency = Duration::from_micros(latency_us);

        let delivery_time = send_time + tx_duration + latency;

        let id = self.next_id;
        self.next_id += 1;

        let pkt = InFlightPacket {
            id,
            payload: payload.to_vec(),
            size_bytes,
            sent_at: send_time,
            delivery_time,
        };

        // Reordering: swap the last two packets in the queue occasionally
        self.in_flight.push_back(pkt);
        if self.in_flight.len() >= 2 && self.rng.bernoulli(self.config.reorder_rate) {
            let len = self.in_flight.len();
            self.in_flight.swap(len - 1, len - 2);
        }

        Ok(Some(id))
    }

    /// Returns and removes all packets whose delivery time ≤ `now`.
    ///
    /// Updates `current_time` to `now` if it is later.
    pub fn receive_ready(&mut self, now: Duration) -> Vec<SimPacket> {
        if now > self.current_time {
            self.current_time = now;
        }

        let mut ready = Vec::new();
        // Drain from front while delivery_time ≤ now
        while let Some(front) = self.in_flight.front() {
            if front.delivery_time <= now {
                // We just confirmed front exists via `while let Some(front)`.
                if let Some(pkt) = self.in_flight.pop_front() {
                    let latency = pkt.delivery_time.saturating_sub(pkt.sent_at);
                    self.stats.packets_delivered += 1;
                    self.stats.bytes_delivered += pkt.size_bytes as u64;
                    self.stats.cumulative_latency_us += latency.as_micros() as u64;
                    ready.push(SimPacket {
                        id: pkt.id,
                        payload: pkt.payload,
                        size_bytes: pkt.size_bytes,
                        sent_at: pkt.sent_at,
                        delivered_at: now,
                        latency,
                    });
                }
            } else {
                break;
            }
        }
        ready
    }

    /// Returns the number of packets currently in flight.
    #[must_use]
    pub fn in_flight_count(&self) -> usize {
        self.in_flight.len()
    }

    /// Returns a reference to the cumulative statistics.
    #[must_use]
    pub fn stats(&self) -> &SimStats {
        &self.stats
    }

    /// Resets the simulator to its initial state while keeping the configuration.
    pub fn reset(&mut self) {
        self.in_flight.clear();
        self.next_id = 0;
        self.ge_bad_state = false;
        self.stats = SimStats::default();
        self.current_time = Duration::ZERO;
    }

    /// Runs a simple segment download simulation, returning the effective
    /// download duration.
    ///
    /// Models downloading a media segment of `segment_bytes` bytes starting
    /// at `start_time` over the simulated network.  Returns the total wall
    /// time until all (non-dropped) packets are received.
    ///
    /// `mtu` — maximum transmission unit in bytes (packet payload size).
    pub fn simulate_segment_download(
        &mut self,
        segment_bytes: usize,
        mtu: usize,
        start_time: Duration,
    ) -> NetResult<SegmentDownloadResult> {
        if mtu == 0 {
            return Err(NetError::invalid_state("MTU must be > 0"));
        }

        let num_packets = segment_bytes.div_ceil(mtu);
        let bw = self
            .config
            .bandwidth_profile
            .bandwidth_at(start_time)
            .max(1.0);

        let mut t = start_time;
        let mut delivered = 0usize;
        let mut dropped = 0usize;
        let mut last_delivery = start_time;

        for i in 0..num_packets {
            let payload_size = if i + 1 == num_packets {
                segment_bytes - i * mtu
            } else {
                mtu
            };

            let result = self.send_packet(&[], payload_size, t)?;
            // Advance t by transmission time of this packet
            let tx_secs = (payload_size as f64 * 8.0) / bw;
            t += Duration::from_secs_f64(tx_secs);

            if result.is_some() {
                delivered += 1;
            } else {
                dropped += 1;
            }
        }

        // Collect all delivered packets
        let delivery_deadline = t + self.config.base_latency * 4;
        let pkts = self.receive_ready(delivery_deadline);
        if let Some(last) = pkts.last() {
            last_delivery = last.delivered_at;
        }

        let download_duration = last_delivery.saturating_sub(start_time);

        Ok(SegmentDownloadResult {
            segment_bytes,
            packets_sent: num_packets,
            packets_delivered: delivered,
            packets_dropped: dropped,
            download_duration,
            effective_throughput_bps: if download_duration.as_secs_f64() > 0.0 {
                (delivered * mtu * 8) as f64 / download_duration.as_secs_f64()
            } else {
                bw
            },
        })
    }

    /// Determines whether the current packet should be dropped.
    fn should_drop(&mut self) -> bool {
        match &self.config.loss_model.clone() {
            LossModel::None => false,
            LossModel::Independent { loss_rate } => self.rng.bernoulli(*loss_rate),
            LossModel::GilbertElliott {
                p_good_to_bad,
                p_bad_to_good,
                loss_in_good,
                loss_in_bad,
            } => {
                // State transition
                if self.ge_bad_state {
                    if self.rng.bernoulli(*p_bad_to_good) {
                        self.ge_bad_state = false;
                    }
                } else if self.rng.bernoulli(*p_good_to_bad) {
                    self.ge_bad_state = true;
                }
                // Loss in current state
                let loss_rate = if self.ge_bad_state {
                    *loss_in_bad
                } else {
                    *loss_in_good
                };
                self.rng.bernoulli(loss_rate)
            }
        }
    }
}

// ─── Segment download result ──────────────────────────────────────────────────

/// Result of a simulated segment download.
#[derive(Debug, Clone)]
pub struct SegmentDownloadResult {
    /// Total bytes in the segment.
    pub segment_bytes: usize,
    /// Number of packets sent.
    pub packets_sent: usize,
    /// Number of packets that were delivered.
    pub packets_delivered: usize,
    /// Number of packets that were dropped.
    pub packets_dropped: usize,
    /// Total time from first send to last receive.
    pub download_duration: Duration,
    /// Effective throughput (delivered bits / download_duration).
    pub effective_throughput_bps: f64,
}

impl SegmentDownloadResult {
    /// Packet loss rate for this segment download.
    #[must_use]
    pub fn loss_rate(&self) -> f64 {
        if self.packets_sent == 0 {
            return 0.0;
        }
        self.packets_dropped as f64 / self.packets_sent as f64
    }
}

// ─── Preset network profiles ──────────────────────────────────────────────────

/// Preset network profiles for common test scenarios.
pub struct NetworkPreset;

impl NetworkPreset {
    /// Ideal broadband: 100 Mbps, 5 ms RTT, no loss.
    #[must_use]
    pub fn broadband() -> SimConfig {
        SimConfig {
            bandwidth_profile: BandwidthProfile::Constant(100_000_000.0),
            base_latency: Duration::from_millis(5),
            jitter_stddev: Duration::from_millis(1),
            loss_model: LossModel::None,
            ..SimConfig::default()
        }
    }

    /// Mobile 4G: 10 Mbps average with sinusoidal fluctuation, 50 ms RTT.
    #[must_use]
    pub fn mobile_4g() -> SimConfig {
        SimConfig {
            bandwidth_profile: BandwidthProfile::Sinusoidal {
                base_bps: 10_000_000.0,
                amplitude_bps: 4_000_000.0,
                period_secs: 20.0,
            },
            base_latency: Duration::from_millis(50),
            jitter_stddev: Duration::from_millis(15),
            loss_model: LossModel::Independent { loss_rate: 0.01 },
            ..SimConfig::default()
        }
    }

    /// Congested WiFi: 5 Mbps, 80 ms RTT, bursty loss.
    #[must_use]
    pub fn congested_wifi() -> SimConfig {
        SimConfig {
            bandwidth_profile: BandwidthProfile::Constant(5_000_000.0),
            base_latency: Duration::from_millis(80),
            jitter_stddev: Duration::from_millis(30),
            loss_model: LossModel::five_percent_bursty(),
            ..SimConfig::default()
        }
    }

    /// Poor 3G / satellite: 1 Mbps, 400 ms RTT, 2% loss.
    #[must_use]
    pub fn satellite() -> SimConfig {
        SimConfig {
            bandwidth_profile: BandwidthProfile::Constant(1_000_000.0),
            base_latency: Duration::from_millis(400),
            jitter_stddev: Duration::from_millis(50),
            loss_model: LossModel::Independent { loss_rate: 0.02 },
            ..SimConfig::default()
        }
    }

    /// Stepped profile simulating a handoff: 10 Mbps for 30 s, then 2 Mbps.
    #[must_use]
    pub fn handoff() -> SimConfig {
        SimConfig {
            bandwidth_profile: BandwidthProfile::Stepped(vec![
                (Duration::ZERO, 10_000_000.0),
                (Duration::from_secs(30), 2_000_000.0),
            ]),
            base_latency: Duration::from_millis(40),
            jitter_stddev: Duration::from_millis(10),
            loss_model: LossModel::None,
            ..SimConfig::default()
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sim(bps: f64) -> NetworkSimulator {
        let cfg = SimConfig {
            bandwidth_profile: BandwidthProfile::Constant(bps),
            base_latency: Duration::from_millis(10),
            jitter_stddev: Duration::ZERO,
            loss_model: LossModel::None,
            ..SimConfig::default()
        };
        NetworkSimulator::new(cfg, 42)
    }

    // 1. Packet sent and received after latency
    #[test]
    fn test_send_and_receive() {
        let mut sim = make_sim(100_000_000.0);
        let id = sim
            .send_packet(b"test", 1400, Duration::ZERO)
            .expect("send ok")
            .expect("not dropped");
        // Nothing ready at t=0
        assert!(sim.receive_ready(Duration::ZERO).is_empty());
        // Ready after base_latency
        let pkts = sim.receive_ready(Duration::from_millis(200));
        assert_eq!(pkts.len(), 1);
        assert_eq!(pkts[0].id, id);
    }

    // 2. Bandwidth limits delivery rate
    #[test]
    fn test_bandwidth_limits_delivery() {
        // 1 Mbps, 1 MB packet → ~8 seconds tx time
        let mut sim = make_sim(1_000_000.0);
        sim.send_packet(b"", 125_000, Duration::ZERO)
            .expect("send ok");
        // Should not be ready at 1 second
        let at_1s = sim.receive_ready(Duration::from_secs(1));
        assert!(at_1s.is_empty(), "should not be delivered yet");
        // Should be ready by 10 seconds (8s tx + 10ms latency)
        let at_10s = sim.receive_ready(Duration::from_secs(10));
        assert_eq!(at_10s.len(), 1);
    }

    // 3. Packet loss model drops packets
    #[test]
    fn test_loss_drops_packets() {
        let cfg = SimConfig {
            bandwidth_profile: BandwidthProfile::Constant(100_000_000.0),
            base_latency: Duration::from_millis(1),
            jitter_stddev: Duration::ZERO,
            loss_model: LossModel::Independent { loss_rate: 1.0 }, // 100% loss
            ..SimConfig::default()
        };
        let mut sim = NetworkSimulator::new(cfg, 0);
        for _ in 0..10 {
            let r = sim.send_packet(b"data", 100, Duration::ZERO).expect("ok");
            assert!(r.is_none(), "all packets should be dropped at 100% loss");
        }
        assert_eq!(sim.stats().packets_delivered, 0);
        assert_eq!(sim.stats().loss_drops, 10);
    }

    // 4. Queue overflow drops excess packets
    #[test]
    fn test_queue_overflow() {
        let cfg = SimConfig {
            bandwidth_profile: BandwidthProfile::Constant(1.0), // very slow
            base_latency: Duration::from_secs(3600),
            jitter_stddev: Duration::ZERO,
            loss_model: LossModel::None,
            queue_max_packets: 3,
            ..SimConfig::default()
        };
        let mut sim = NetworkSimulator::new(cfg, 0);
        for _ in 0..5 {
            let _ = sim.send_packet(b"x", 1, Duration::ZERO);
        }
        assert_eq!(sim.stats().queue_overflow_drops, 2);
    }

    // 5. Deterministic with same seed
    #[test]
    fn test_deterministic_seed() {
        let cfg = SimConfig {
            bandwidth_profile: BandwidthProfile::Constant(10_000_000.0),
            base_latency: Duration::from_millis(20),
            jitter_stddev: Duration::from_millis(5),
            loss_model: LossModel::Independent { loss_rate: 0.1 },
            ..SimConfig::default()
        };

        let mut outcomes_a = Vec::new();
        let mut sim_a = NetworkSimulator::new(cfg.clone(), 12345);
        for _ in 0..20 {
            let r = sim_a.send_packet(b"x", 1400, Duration::ZERO).expect("ok");
            outcomes_a.push(r.is_some());
        }

        let mut outcomes_b = Vec::new();
        let mut sim_b = NetworkSimulator::new(cfg, 12345);
        for _ in 0..20 {
            let r = sim_b.send_packet(b"x", 1400, Duration::ZERO).expect("ok");
            outcomes_b.push(r.is_some());
        }

        assert_eq!(
            outcomes_a, outcomes_b,
            "same seed must produce same results"
        );
    }

    // 6. Statistics track delivered count and bytes
    #[test]
    fn test_stats_tracking() {
        let mut sim = make_sim(1_000_000_000.0);
        for _ in 0..5 {
            sim.send_packet(b"hello", 100, Duration::ZERO).expect("ok");
        }
        sim.receive_ready(Duration::from_millis(500));
        let s = sim.stats();
        assert_eq!(s.packets_sent, 5);
        assert_eq!(s.packets_delivered, 5);
        assert_eq!(s.bytes_delivered, 500);
    }

    // 7. Bandwidth profile: sinusoidal varies over time
    #[test]
    fn test_sinusoidal_bandwidth_profile() {
        let profile = BandwidthProfile::Sinusoidal {
            base_bps: 10_000_000.0,
            amplitude_bps: 5_000_000.0,
            period_secs: 10.0,
        };
        let bw_0 = profile.bandwidth_at(Duration::ZERO);
        let bw_half = profile.bandwidth_at(Duration::from_secs_f64(2.5)); // quarter period
                                                                          // At t=0 sin(0)=0 so bw=base; at t=2.5 sin(π/2)=1 so bw=15M
        assert!((bw_0 - 10_000_000.0).abs() < 1.0, "at t=0: {bw_0}");
        assert!(
            bw_half > 10_000_000.0,
            "at t=2.5s should be above base: {bw_half}"
        );
    }

    // 8. Segment download simulation returns valid result
    #[test]
    fn test_simulate_segment_download() {
        let mut sim = make_sim(10_000_000.0);
        let result = sim
            .simulate_segment_download(
                1_250_000, // 10 Mbit segment
                1400,
                Duration::ZERO,
            )
            .expect("download ok");
        assert_eq!(result.segment_bytes, 1_250_000);
        assert!(
            result.download_duration > Duration::ZERO,
            "should take non-zero time"
        );
        assert!(result.effective_throughput_bps > 0.0);
    }

    // 9. Preset network profiles can be constructed
    #[test]
    fn test_presets_construct() {
        let presets = [
            NetworkPreset::broadband(),
            NetworkPreset::mobile_4g(),
            NetworkPreset::congested_wifi(),
            NetworkPreset::satellite(),
            NetworkPreset::handoff(),
        ];
        for preset in &presets {
            // Just verify construction and bandwidth query work
            let bw = preset.bandwidth_profile.bandwidth_at(Duration::ZERO);
            assert!(bw > 0.0, "preset bandwidth must be positive");
        }
    }

    // 10. Reset clears state
    #[test]
    fn test_reset_clears_state() {
        let mut sim = make_sim(100_000_000.0);
        sim.send_packet(b"abc", 100, Duration::ZERO).expect("ok");
        assert_eq!(sim.in_flight_count(), 1);
        sim.reset();
        assert_eq!(sim.in_flight_count(), 0);
        assert_eq!(sim.stats().packets_sent, 0);
    }
}
