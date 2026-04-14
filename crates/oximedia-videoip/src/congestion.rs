#![allow(dead_code)]
//! Network congestion control for video-over-IP streams.
//!
//! This module implements congestion detection and rate adaptation for
//! professional video streaming. It monitors network conditions and adjusts
//! sending rates to prevent buffer overflow, packet loss, and latency spikes.
//!
//! # Algorithms
//!
//! - **AIMD (Additive Increase / Multiplicative Decrease)** - Classic TCP-like approach
//! - **Delay-based detection** - Uses RTT trends to detect congestion before loss
//! - **Loss-based detection** - Reacts to observed packet loss rates
//!
//! # Integration
//!
//! The congestion controller provides a target bitrate that the encoder or
//! packetizer should respect. It does not directly control the network stack.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Congestion state of the network path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CongestionState {
    /// No congestion detected, free to increase rate.
    SlowStart,
    /// Operating near capacity, probing cautiously.
    CongestionAvoidance,
    /// Active congestion detected, reducing rate.
    Recovery,
}

impl std::fmt::Display for CongestionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let label = match self {
            Self::SlowStart => "slow_start",
            Self::CongestionAvoidance => "congestion_avoidance",
            Self::Recovery => "recovery",
        };
        write!(f, "{label}")
    }
}

/// Configuration for the congestion controller.
#[derive(Debug, Clone)]
pub struct CongestionConfig {
    /// Minimum allowed bitrate in bits per second.
    pub min_bitrate_bps: u64,
    /// Maximum allowed bitrate in bits per second.
    pub max_bitrate_bps: u64,
    /// Initial bitrate in bits per second.
    pub initial_bitrate_bps: u64,
    /// Additive increase step in bits per second.
    pub additive_increase_bps: u64,
    /// Multiplicative decrease factor (0.0-1.0). Rate is multiplied by this on congestion.
    pub multiplicative_decrease: f64,
    /// Packet loss threshold to trigger congestion (0.0-1.0).
    pub loss_threshold: f64,
    /// RTT increase threshold (ratio over baseline) to trigger delay-based detection.
    pub rtt_increase_threshold: f64,
    /// Number of RTT samples to keep for baseline estimation.
    pub rtt_window_size: usize,
    /// Slow start threshold in bits per second.
    pub ssthresh_bps: u64,
}

impl Default for CongestionConfig {
    fn default() -> Self {
        Self {
            min_bitrate_bps: 1_000_000,      // 1 Mbps
            max_bitrate_bps: 100_000_000,    // 100 Mbps
            initial_bitrate_bps: 10_000_000, // 10 Mbps
            additive_increase_bps: 500_000,  // 500 kbps
            multiplicative_decrease: 0.5,
            loss_threshold: 0.02,
            rtt_increase_threshold: 1.5,
            rtt_window_size: 50,
            ssthresh_bps: 50_000_000, // 50 Mbps
        }
    }
}

/// RTT sample measurement.
#[derive(Debug, Clone, Copy)]
pub struct RttSample {
    /// Round-trip time measurement.
    pub rtt: Duration,
    /// Timestamp when sample was taken.
    pub timestamp: Instant,
}

/// Statistics from the congestion controller.
#[derive(Debug, Clone)]
pub struct CongestionStats {
    /// Current target bitrate in bps.
    pub current_bitrate_bps: u64,
    /// Current congestion state.
    pub state: CongestionState,
    /// Smoothed RTT estimate.
    pub smoothed_rtt: Duration,
    /// Minimum observed RTT (baseline).
    pub min_rtt: Duration,
    /// Current estimated packet loss rate (0.0-1.0).
    pub loss_rate: f64,
    /// Number of congestion events detected.
    pub congestion_events: u64,
    /// Total packets reported.
    pub total_packets: u64,
    /// Total lost packets reported.
    pub lost_packets: u64,
}

/// Congestion controller for video-over-IP streams.
pub struct CongestionController {
    /// Configuration.
    config: CongestionConfig,
    /// Current target bitrate.
    current_bitrate_bps: u64,
    /// Current congestion state.
    state: CongestionState,
    /// Slow-start threshold.
    ssthresh_bps: u64,
    /// RTT samples buffer.
    rtt_history: VecDeque<RttSample>,
    /// Smoothed RTT (exponential weighted moving average).
    smoothed_rtt: Duration,
    /// Minimum observed RTT.
    min_rtt: Duration,
    /// Recent loss rate.
    loss_rate: f64,
    /// Packets tracked in current window.
    window_packets: u64,
    /// Packets lost in current window.
    window_lost: u64,
    /// Total packets reported.
    total_packets: u64,
    /// Total lost packets.
    total_lost: u64,
    /// Number of congestion events.
    congestion_events: u64,
    /// Timestamp of last rate adjustment.
    last_adjust: Option<Instant>,
}

impl CongestionController {
    /// Create a new congestion controller with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(CongestionConfig::default())
    }

    /// Create a new congestion controller with custom configuration.
    #[must_use]
    pub fn with_config(config: CongestionConfig) -> Self {
        let initial = config.initial_bitrate_bps;
        let ssthresh = config.ssthresh_bps;
        Self {
            config,
            current_bitrate_bps: initial,
            state: CongestionState::SlowStart,
            ssthresh_bps: ssthresh,
            rtt_history: VecDeque::with_capacity(50),
            smoothed_rtt: Duration::from_millis(10),
            min_rtt: Duration::from_secs(1),
            loss_rate: 0.0,
            window_packets: 0,
            window_lost: 0,
            total_packets: 0,
            total_lost: 0,
            congestion_events: 0,
            last_adjust: None,
        }
    }

    /// Minimum number of RTT samples required before delay-based congestion
    /// detection kicks in.  This avoids false positives while the smoothed RTT
    /// is still converging toward the true baseline.
    const MIN_RTT_SAMPLES: usize = 4;

    /// Report an RTT measurement.
    pub fn report_rtt(&mut self, rtt: Duration) {
        let sample = RttSample {
            rtt,
            timestamp: Instant::now(),
        };

        if self.rtt_history.len() >= self.config.rtt_window_size {
            self.rtt_history.pop_front();
        }
        self.rtt_history.push_back(sample);

        // Update min RTT
        if rtt < self.min_rtt {
            self.min_rtt = rtt;
        }

        // Smoothed RTT: EWMA with alpha = 0.125
        let alpha = 0.125;
        let smoothed_us =
            self.smoothed_rtt.as_micros() as f64 * (1.0 - alpha) + rtt.as_micros() as f64 * alpha;
        self.smoothed_rtt = Duration::from_micros(smoothed_us as u64);

        // Check for delay-based congestion only after collecting enough samples
        // to establish a meaningful baseline.
        if self.rtt_history.len() >= Self::MIN_RTT_SAMPLES && self.min_rtt.as_micros() > 0 {
            let ratio = self.smoothed_rtt.as_micros() as f64 / self.min_rtt.as_micros() as f64;
            if ratio > self.config.rtt_increase_threshold {
                self.on_congestion_detected();
            }
        }
    }

    /// Report packet loss statistics.
    ///
    /// Call this periodically with the number of packets sent and lost in the
    /// reporting interval.
    #[allow(clippy::cast_precision_loss)]
    pub fn report_loss(&mut self, packets_sent: u64, packets_lost: u64) {
        self.window_packets += packets_sent;
        self.window_lost += packets_lost;
        self.total_packets += packets_sent;
        self.total_lost += packets_lost;

        if self.window_packets >= 100 {
            self.loss_rate = self.window_lost as f64 / self.window_packets as f64;
            self.window_packets = 0;
            self.window_lost = 0;

            if self.loss_rate > self.config.loss_threshold {
                self.on_congestion_detected();
            } else {
                self.on_no_congestion();
            }
        }
    }

    /// Called when congestion is detected.
    fn on_congestion_detected(&mut self) {
        match self.state {
            CongestionState::SlowStart | CongestionState::CongestionAvoidance => {
                self.ssthresh_bps =
                    (self.current_bitrate_bps as f64 * self.config.multiplicative_decrease) as u64;
                self.ssthresh_bps = self.ssthresh_bps.max(self.config.min_bitrate_bps);

                self.current_bitrate_bps = self.ssthresh_bps;
                self.current_bitrate_bps =
                    self.current_bitrate_bps.max(self.config.min_bitrate_bps);

                self.state = CongestionState::Recovery;
                self.congestion_events += 1;
            }
            CongestionState::Recovery => {
                // Already in recovery, don't reduce further immediately
            }
        }
    }

    /// Called when the network appears uncongested.
    fn on_no_congestion(&mut self) {
        match self.state {
            CongestionState::SlowStart => {
                // Double rate until ssthresh
                self.current_bitrate_bps = (self.current_bitrate_bps * 2)
                    .min(self.ssthresh_bps)
                    .min(self.config.max_bitrate_bps);
                if self.current_bitrate_bps >= self.ssthresh_bps {
                    self.state = CongestionState::CongestionAvoidance;
                }
            }
            CongestionState::CongestionAvoidance => {
                // Additive increase
                self.current_bitrate_bps = (self.current_bitrate_bps
                    + self.config.additive_increase_bps)
                    .min(self.config.max_bitrate_bps);
            }
            CongestionState::Recovery => {
                // Exit recovery, enter congestion avoidance
                self.state = CongestionState::CongestionAvoidance;
            }
        }
    }

    /// Get the current recommended target bitrate.
    #[must_use]
    pub fn target_bitrate_bps(&self) -> u64 {
        self.current_bitrate_bps
    }

    /// Get the current congestion state.
    #[must_use]
    pub fn state(&self) -> CongestionState {
        self.state
    }

    /// Get full congestion statistics.
    #[must_use]
    pub fn stats(&self) -> CongestionStats {
        CongestionStats {
            current_bitrate_bps: self.current_bitrate_bps,
            state: self.state,
            smoothed_rtt: self.smoothed_rtt,
            min_rtt: self.min_rtt,
            loss_rate: self.loss_rate,
            congestion_events: self.congestion_events,
            total_packets: self.total_packets,
            lost_packets: self.total_lost,
        }
    }

    /// Manually set the target bitrate, clamped to configured bounds.
    pub fn set_bitrate(&mut self, bitrate_bps: u64) {
        self.current_bitrate_bps = bitrate_bps
            .max(self.config.min_bitrate_bps)
            .min(self.config.max_bitrate_bps);
    }

    /// Reset the controller to initial state.
    pub fn reset(&mut self) {
        self.current_bitrate_bps = self.config.initial_bitrate_bps;
        self.state = CongestionState::SlowStart;
        self.ssthresh_bps = self.config.ssthresh_bps;
        self.rtt_history.clear();
        self.smoothed_rtt = Duration::from_millis(10);
        self.min_rtt = Duration::from_secs(1);
        self.loss_rate = 0.0;
        self.window_packets = 0;
        self.window_lost = 0;
        self.total_packets = 0;
        self.total_lost = 0;
        self.congestion_events = 0;
        self.last_adjust = None;
    }
}

/// Estimate available bandwidth from a series of packet arrival times.
///
/// Uses packet dispersion to estimate the bottleneck link capacity.
/// Returns bandwidth in bits per second.
#[allow(clippy::cast_precision_loss)]
#[must_use]
pub fn estimate_bandwidth_from_dispersion(
    packet_sizes_bytes: &[usize],
    inter_arrival_us: &[u64],
) -> f64 {
    if packet_sizes_bytes.len() < 2 || inter_arrival_us.is_empty() {
        return 0.0;
    }

    let pairs = packet_sizes_bytes.len().min(inter_arrival_us.len() + 1) - 1;
    if pairs == 0 {
        return 0.0;
    }

    let mut total_bits = 0u64;
    let mut total_us = 0u64;

    for i in 0..pairs {
        total_bits += (packet_sizes_bytes[i + 1] * 8) as u64;
        total_us += inter_arrival_us[i];
    }

    if total_us == 0 {
        return 0.0;
    }

    total_bits as f64 / (total_us as f64 / 1_000_000.0)
}

// ── BBR-based Congestion Controller ──────────────────────────────────────────

use crate::bbr::{AckSample, BbrConfig, BbrController, BbrState};

/// A congestion controller that uses BBR (Bottleneck Bandwidth and Round-trip
/// propagation time) instead of AIMD.
///
/// BBR models the network path using two measurements:
///
/// - **BtlBw** — bottleneck bandwidth (max delivery rate over a sliding window)
/// - **RTprop** — round-trip propagation delay (min RTT over ≥10 seconds)
///
/// It uses these to set a *pacing rate* (`BtlBw × pacing_gain`) and a
/// *congestion window* (`BDP × cwnd_gain`), never inducing queue build-up
/// during steady-state operation.
///
/// # Integration
///
/// Call [`report_ack`](Self::report_ack) for every acknowledged packet group,
/// then read [`target_bitrate_bps`](Self::target_bitrate_bps) to obtain the
/// recommended sending rate in bits per second.
pub struct BbrCongestionController {
    /// Core BBR algorithm.
    bbr: BbrController,
    /// Minimum bitrate floor in bits per second.
    min_bitrate_bps: u64,
    /// Maximum bitrate ceiling in bits per second.
    max_bitrate_bps: u64,
    /// Total ACK events processed.
    ack_count: u64,
    /// Last reported pacing rate in bytes/second (cached).
    last_pacing_rate: f64,
}

impl BbrCongestionController {
    /// Creates a new BBR congestion controller with explicit bounds.
    ///
    /// # Arguments
    ///
    /// * `min_bitrate_bps` — floor in bits per second (e.g. `1_000_000` = 1 Mbps)
    /// * `max_bitrate_bps` — ceiling in bits per second (e.g. `100_000_000` = 100 Mbps)
    #[must_use]
    pub fn new(min_bitrate_bps: u64, max_bitrate_bps: u64) -> Self {
        Self {
            bbr: BbrController::new(BbrConfig::default()),
            min_bitrate_bps,
            max_bitrate_bps,
            ack_count: 0,
            last_pacing_rate: 0.0,
        }
    }

    /// Creates a controller with the given custom BBR configuration and bounds.
    #[must_use]
    pub fn with_config(bbr_config: BbrConfig, min_bitrate_bps: u64, max_bitrate_bps: u64) -> Self {
        Self {
            bbr: BbrController::new(bbr_config),
            min_bitrate_bps,
            max_bitrate_bps,
            ack_count: 0,
            last_pacing_rate: 0.0,
        }
    }

    /// Processes an acknowledgement event and updates the BBR model.
    ///
    /// # Arguments
    ///
    /// * `delivered_bytes` — bytes confirmed delivered since the last ACK.
    /// * `elapsed_secs` — wall-clock time elapsed since the last ACK (seconds).
    /// * `rtt_secs` — round-trip time for this ACK event (seconds).
    /// * `is_app_limited` — `true` when the sender was application-limited
    ///   (not constrained by cwnd) when the acknowledged packet was sent.
    pub fn report_ack(
        &mut self,
        delivered_bytes: u64,
        elapsed_secs: f64,
        rtt_secs: f64,
        is_app_limited: bool,
    ) {
        let sample = AckSample {
            delivered: delivered_bytes,
            elapsed_secs,
            rtt_secs,
            is_app_limited,
        };
        self.bbr.on_ack(sample);
        self.last_pacing_rate = self.bbr.pacing_rate();
        self.ack_count += 1;
    }

    /// Returns the recommended sending rate in bits per second, clamped to
    /// `[min_bitrate_bps, max_bitrate_bps]`.
    #[must_use]
    pub fn target_bitrate_bps(&self) -> u64 {
        // BBR's pacing_rate is in bytes/sec; convert to bits/sec.
        let bps = (self.last_pacing_rate * 8.0) as u64;
        bps.clamp(self.min_bitrate_bps, self.max_bitrate_bps)
    }

    /// Returns the current congestion window in bytes.
    #[must_use]
    pub fn cwnd_bytes(&self) -> u64 {
        self.bbr.cwnd()
    }

    /// Returns the current BBR state machine phase.
    #[must_use]
    pub fn bbr_state(&self) -> BbrState {
        *self.bbr.state()
    }

    /// Returns the bottleneck bandwidth estimate in bytes per second.
    #[must_use]
    pub fn btlbw_bytes_per_sec(&self) -> f64 {
        self.bbr.btlbw()
    }

    /// Returns the round-trip propagation time estimate in seconds.
    #[must_use]
    pub fn rtprop_secs(&self) -> f64 {
        self.bbr.rtprop()
    }

    /// Returns the bandwidth-delay product in bytes.
    #[must_use]
    pub fn bdp_bytes(&self) -> u64 {
        (self.bbr.btlbw() * self.bbr.rtprop()) as u64
    }

    /// Returns the total number of ACK events processed.
    #[must_use]
    pub const fn ack_count(&self) -> u64 {
        self.ack_count
    }

    /// Returns `true` if the controller has processed enough ACKs for the BBR
    /// model to have meaningful bandwidth and RTT estimates.
    ///
    /// BBR requires at least a few round trips to leave Startup and settle into
    /// ProbeBw.  This heuristic considers the model "warmed up" after 10 ACKs.
    #[must_use]
    pub fn is_warmed_up(&self) -> bool {
        self.ack_count >= 10
    }

    /// Resets the BBR controller to its initial state, discarding all
    /// bandwidth and RTT estimates.
    pub fn reset(&mut self) {
        self.bbr = BbrController::new(BbrConfig::default());
        self.ack_count = 0;
        self.last_pacing_rate = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_initial_state() {
        let cc = CongestionController::new();
        assert_eq!(cc.state(), CongestionState::SlowStart);
        assert_eq!(cc.target_bitrate_bps(), 10_000_000);
    }

    #[test]
    fn test_slow_start_doubles_rate() {
        let mut cc = CongestionController::new();
        let initial = cc.target_bitrate_bps();
        cc.report_loss(200, 0); // no loss, 200 packets triggers window eval
        let after = cc.target_bitrate_bps();
        assert!(
            after > initial,
            "rate should increase in slow start: {} -> {}",
            initial,
            after
        );
    }

    #[test]
    fn test_loss_triggers_reduction() {
        let mut cc = CongestionController::new();
        let before = cc.target_bitrate_bps();
        // Heavy loss
        cc.report_loss(100, 50);
        let after = cc.target_bitrate_bps();
        assert!(
            after < before,
            "rate should decrease on loss: {} -> {}",
            before,
            after
        );
        assert_eq!(cc.state(), CongestionState::Recovery);
    }

    #[test]
    fn test_rtt_based_congestion() {
        let config = CongestionConfig {
            rtt_increase_threshold: 1.5,
            ..Default::default()
        };
        let mut cc = CongestionController::with_config(config);
        // Establish baseline
        cc.report_rtt(Duration::from_millis(5));
        cc.report_rtt(Duration::from_millis(5));
        let before = cc.target_bitrate_bps();

        // Spike RTT well above threshold
        for _ in 0..20 {
            cc.report_rtt(Duration::from_millis(50));
        }
        let after = cc.target_bitrate_bps();
        assert!(
            after < before,
            "high RTT should trigger congestion: {} -> {}",
            before,
            after
        );
    }

    #[test]
    fn test_rate_clamped_to_min() {
        let config = CongestionConfig {
            min_bitrate_bps: 5_000_000,
            initial_bitrate_bps: 6_000_000,
            multiplicative_decrease: 0.1,
            ..Default::default()
        };
        let mut cc = CongestionController::with_config(config);
        cc.report_loss(100, 50);
        assert!(
            cc.target_bitrate_bps() >= 5_000_000,
            "rate should not go below min: {}",
            cc.target_bitrate_bps()
        );
    }

    #[test]
    fn test_rate_clamped_to_max() {
        let config = CongestionConfig {
            max_bitrate_bps: 20_000_000,
            initial_bitrate_bps: 18_000_000,
            ssthresh_bps: 100_000_000,
            ..Default::default()
        };
        let mut cc = CongestionController::with_config(config);
        // No loss, slow start should double but clamp
        cc.report_loss(200, 0);
        assert!(
            cc.target_bitrate_bps() <= 20_000_000,
            "rate should not exceed max: {}",
            cc.target_bitrate_bps()
        );
    }

    #[test]
    fn test_congestion_event_count() {
        let mut cc = CongestionController::new();
        cc.report_loss(100, 50);
        assert_eq!(cc.stats().congestion_events, 1);
        // In recovery, additional loss should not increment again
        cc.report_loss(100, 50);
        assert_eq!(cc.stats().congestion_events, 1);
    }

    #[test]
    fn test_recovery_to_avoidance() {
        let mut cc = CongestionController::new();
        // Trigger congestion
        cc.report_loss(100, 50);
        assert_eq!(cc.state(), CongestionState::Recovery);
        // No loss -> exit recovery
        cc.report_loss(200, 0);
        assert_eq!(cc.state(), CongestionState::CongestionAvoidance);
    }

    #[test]
    fn test_set_bitrate_manual() {
        let mut cc = CongestionController::new();
        cc.set_bitrate(50_000_000);
        assert_eq!(cc.target_bitrate_bps(), 50_000_000);
        // Beyond max should be clamped
        cc.set_bitrate(200_000_000);
        assert_eq!(cc.target_bitrate_bps(), 100_000_000);
    }

    #[test]
    fn test_reset() {
        let mut cc = CongestionController::new();
        cc.report_loss(100, 50);
        cc.reset();
        assert_eq!(cc.state(), CongestionState::SlowStart);
        assert_eq!(cc.target_bitrate_bps(), 10_000_000);
        assert_eq!(cc.stats().congestion_events, 0);
    }

    #[test]
    fn test_congestion_state_display() {
        assert_eq!(format!("{}", CongestionState::SlowStart), "slow_start");
        assert_eq!(
            format!("{}", CongestionState::CongestionAvoidance),
            "congestion_avoidance"
        );
        assert_eq!(format!("{}", CongestionState::Recovery), "recovery");
    }

    #[test]
    fn test_estimate_bandwidth_from_dispersion() {
        // 1000 byte packets, 1ms apart => 8_000_000 bps = 8 Mbps
        let sizes = vec![1000, 1000, 1000, 1000];
        let arrivals = vec![1000, 1000, 1000]; // microseconds
        let bw = estimate_bandwidth_from_dispersion(&sizes, &arrivals);
        assert!(
            (bw - 8_000_000.0).abs() < 100_000.0,
            "bandwidth estimate should be ~8 Mbps: {}",
            bw
        );
    }

    #[test]
    fn test_estimate_bandwidth_empty() {
        assert_eq!(estimate_bandwidth_from_dispersion(&[], &[]), 0.0);
        assert_eq!(estimate_bandwidth_from_dispersion(&[100], &[]), 0.0);
    }

    // ── BbrCongestionController tests ──────────────────────────────────────────

    fn make_ack(delivered: u64, elapsed_secs: f64, rtt_secs: f64) -> (u64, f64, f64, bool) {
        (delivered, elapsed_secs, rtt_secs, false)
    }

    #[test]
    fn test_bbr_initial_state_is_startup() {
        let ctrl = BbrCongestionController::new(1_000_000, 100_000_000);
        assert_eq!(ctrl.bbr_state(), BbrState::Startup);
        assert_eq!(ctrl.ack_count(), 0);
    }

    #[test]
    fn test_bbr_target_bitrate_clamped_at_min() {
        // Zero ACKs → pacing_rate is still the initial estimate.
        let ctrl = BbrCongestionController::new(10_000_000, 100_000_000);
        // Rate should be ≥ min even if internal estimate is low.
        assert!(ctrl.target_bitrate_bps() >= 10_000_000);
    }

    #[test]
    fn test_bbr_target_bitrate_clamped_at_max() {
        let mut ctrl = BbrCongestionController::new(1_000_000, 5_000_000);
        // Feed many large ACKs to drive BtlBw high.
        for _ in 0..20 {
            let (d, e, r, a) = make_ack(100_000, 0.01, 0.005);
            ctrl.report_ack(d, e, r, a);
        }
        assert!(
            ctrl.target_bitrate_bps() <= 5_000_000,
            "target must be clamped to max: {}",
            ctrl.target_bitrate_bps()
        );
    }

    #[test]
    fn test_bbr_ack_count_increments() {
        let mut ctrl = BbrCongestionController::new(1_000_000, 100_000_000);
        assert_eq!(ctrl.ack_count(), 0);
        for i in 1..=5 {
            ctrl.report_ack(1_000, 0.001, 0.010, false);
            assert_eq!(ctrl.ack_count(), i);
        }
    }

    #[test]
    fn test_bbr_not_warmed_up_initially() {
        let ctrl = BbrCongestionController::new(1_000_000, 100_000_000);
        assert!(!ctrl.is_warmed_up());
    }

    #[test]
    fn test_bbr_warmed_up_after_ten_acks() {
        let mut ctrl = BbrCongestionController::new(1_000_000, 100_000_000);
        for _ in 0..10 {
            ctrl.report_ack(1_000, 0.001, 0.010, false);
        }
        assert!(ctrl.is_warmed_up());
    }

    #[test]
    fn test_bbr_cwnd_nonzero() {
        let ctrl = BbrCongestionController::new(1_000_000, 100_000_000);
        assert!(ctrl.cwnd_bytes() > 0);
    }

    #[test]
    fn test_bbr_rtprop_positive() {
        let ctrl = BbrCongestionController::new(1_000_000, 100_000_000);
        assert!(ctrl.rtprop_secs() > 0.0);
    }

    #[test]
    fn test_bbr_btlbw_updates_on_acks() {
        let mut ctrl = BbrCongestionController::new(1_000_000, 100_000_000);
        let initial_bw = ctrl.btlbw_bytes_per_sec();
        // Large ACK: 1 MB delivered in 10 ms = 100 MB/s delivery rate.
        ctrl.report_ack(1_000_000, 0.01, 0.005, false);
        let after_bw = ctrl.btlbw_bytes_per_sec();
        // BtlBw should be non-zero and ≥ initial.
        assert!(after_bw >= initial_bw);
    }

    #[test]
    fn test_bbr_bdp_calculation() {
        let ctrl = BbrCongestionController::new(1_000_000, 100_000_000);
        let bdp = ctrl.bdp_bytes();
        let expected = (ctrl.btlbw_bytes_per_sec() * ctrl.rtprop_secs()) as u64;
        assert_eq!(bdp, expected);
    }

    #[test]
    fn test_bbr_with_custom_config() {
        let bbr_cfg = BbrConfig::default();
        let ctrl = BbrCongestionController::with_config(bbr_cfg, 2_000_000, 50_000_000);
        assert_eq!(ctrl.bbr_state(), BbrState::Startup);
        assert!(ctrl.target_bitrate_bps() >= 2_000_000);
        assert!(ctrl.target_bitrate_bps() <= 50_000_000);
    }

    #[test]
    fn test_bbr_reset_clears_ack_count() {
        let mut ctrl = BbrCongestionController::new(1_000_000, 100_000_000);
        for _ in 0..5 {
            ctrl.report_ack(1_000, 0.001, 0.010, false);
        }
        assert_eq!(ctrl.ack_count(), 5);
        ctrl.reset();
        assert_eq!(ctrl.ack_count(), 0);
        assert_eq!(ctrl.bbr_state(), BbrState::Startup);
    }
}
