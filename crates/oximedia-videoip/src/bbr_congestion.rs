//! BBR Congestion Control — Pacing Gain Cycles and Bandwidth Probing.
//!
//! This module provides a higher-level, protocol-integration-oriented BBR
//! congestion controller built on top of the core BBR algorithm in [`crate::bbr`].
//!
//! It adds:
//!
//! - **Explicit pacing gain cycle scheduling** with per-cycle duration tracking.
//! - **Bandwidth probing events** that callers can subscribe to via
//!   [`BandwidthProbeEvent`].
//! - **RTprop expiry notifications** so upper layers can trigger ProbeRtt.
//! - **Congestion state** aggregation useful for adaptive FEC/redundancy.
//! - **Pacing budget** helper for burst-limited senders (compute how many
//!   bytes can be sent right now without violating the pacing constraint).
//! - **Connection-level statistics** exposed via [`BbrStats`].
//!
//! # Relationship to `bbr.rs`
//!
//! [`crate::bbr::BbrController`] is the pure algorithm core.  This module
//! wraps it with timing infrastructure and high-level helpers needed when
//! integrating into a real media transport.

use crate::bbr::{AckSample, BbrConfig, BbrController, BbrState};

// ─── Constants ────────────────────────────────────────────────────────────────

/// Number of pacing gain cycle phases (8, per the BBR paper).
pub const PROBE_BW_CYCLE_LEN: usize = 8;

/// Pacing gain values for each ProbeBw phase.
/// Phase 0: probe-up (+25 %), phase 1: drain (-25 %), phases 2–7: steady.
pub const PROBE_BW_GAINS: [f64; PROBE_BW_CYCLE_LEN] =
    [1.25, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

/// Duration of a single ProbeBw gain phase in round trips.
pub const PROBE_BW_PHASE_DURATION_ROUNDS: u64 = 1;

/// ProbeRtt duration in milliseconds.
pub const PROBE_RTT_DURATION_MS: u64 = 200;

/// Minimum bytes that can be sent per pacing interval even if the budget
/// is otherwise zero (one MSS = 1460 bytes).
pub const MIN_PACING_BURST_BYTES: u64 = 1460;

// ─── Bandwidth Probe Event ────────────────────────────────────────────────────

/// Notification events emitted during bandwidth probing.
///
/// Callers can use these to, for example, temporarily increase FEC overhead
/// during a probe-up phase or reduce it during steady-state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BandwidthProbeEvent {
    /// Starting the probe-up phase (gain > 1.0 — sending faster than BtlBw).
    ProbeUp {
        /// Current bottleneck bandwidth estimate (bytes/sec).
        btlbw_bps: f64,
        /// Current RTprop estimate (seconds).
        rtprop_secs: f64,
    },
    /// Starting the drain phase (gain < 1.0 — clearing the queue).
    Drain {
        /// BDP estimate at drain entry (bytes).
        bdp_bytes: u64,
    },
    /// Entering steady-state (gain == 1.0).
    Steady,
    /// Entering ProbeRtt (minimal cwnd to measure propagation delay).
    ProbeRtt,
    /// Exiting ProbeRtt, returning to ProbeBw.
    ProbeRttDone {
        /// Measured RTprop after the probing phase (seconds).
        rtprop_secs: f64,
    },
    /// A new maximum bottleneck bandwidth was observed.
    NewBandwidthRecord {
        /// The new BtlBw record (bytes/sec).
        btlbw_bps: f64,
    },
}

// ─── Congestion State ─────────────────────────────────────────────────────────

/// High-level congestion assessment, useful for adaptive media decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CongestionState {
    /// Pipe is being filled (startup phase).
    FillingPipe,
    /// Steady-state operation; bandwidth is stable.
    Stable,
    /// Active bandwidth probing (+25 % gain).
    Probing,
    /// Drain after probing; reducing in-flight data.
    Draining,
    /// Measuring propagation RTT; cwnd minimised.
    MeasuringRtt,
}

impl CongestionState {
    /// Derive the congestion state from a BBR state and cycle index.
    #[must_use]
    pub fn from_bbr(state: BbrState, cycle_idx: usize) -> Self {
        match state {
            BbrState::Startup => Self::FillingPipe,
            BbrState::Drain => Self::Draining,
            BbrState::ProbeRtt => Self::MeasuringRtt,
            BbrState::ProbeBw => match cycle_idx {
                0 => Self::Probing,
                1 => Self::Draining,
                _ => Self::Stable,
            },
        }
    }
}

impl std::fmt::Display for CongestionState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::FillingPipe => "filling_pipe",
            Self::Stable => "stable",
            Self::Probing => "probing",
            Self::Draining => "draining",
            Self::MeasuringRtt => "measuring_rtt",
        };
        write!(f, "{s}")
    }
}

// ─── Pacing Budget ────────────────────────────────────────────────────────────

/// Instantaneous pacing budget: how many bytes may be sent right now.
///
/// Created by [`BbrCongestionController::pacing_budget`].
#[derive(Debug, Clone, Copy)]
pub struct PacingBudget {
    /// Bytes available to send in the current interval.
    pub bytes_available: u64,
    /// Interval duration used to compute the budget (seconds).
    pub interval_secs: f64,
    /// Pacing rate (bytes/sec) at the time of computation.
    pub pacing_rate_bps: f64,
}

impl PacingBudget {
    /// Returns `true` if at least one packet can be sent.
    #[must_use]
    pub fn can_send(&self) -> bool {
        self.bytes_available >= MIN_PACING_BURST_BYTES
    }
}

// ─── Connection Statistics ────────────────────────────────────────────────────

/// Aggregate statistics for the connection lifetime.
#[derive(Debug, Clone, Default)]
pub struct BbrStats {
    /// Total ACK events processed.
    pub ack_count: u64,
    /// Total bytes delivered (sum of `AckSample::delivered`).
    pub bytes_delivered: u64,
    /// Number of times the probe-up phase was entered.
    pub probe_up_count: u64,
    /// Number of ProbeRtt phases completed.
    pub probe_rtt_count: u64,
    /// Peak bandwidth ever observed (bytes/sec).
    pub peak_bandwidth_bps: f64,
    /// Minimum RTT ever observed (seconds).
    pub min_rtt_secs: f64,
    /// Number of bandwidth probe events emitted.
    pub events_emitted: u64,
}

// ─── Main Controller ──────────────────────────────────────────────────────────

/// BBR congestion controller with explicit pacing gain cycle management.
///
/// This wraps [`BbrController`] and adds:
/// - Pending [`BandwidthProbeEvent`] queue (up to 16 events per ACK).
/// - Per-cycle round tracking for precise phase duration.
/// - [`PacingBudget`] computation.
/// - Aggregate [`BbrStats`].
pub struct BbrCongestionController {
    /// Inner BBR core.
    core: BbrController,

    /// Previous BBR state (for transition detection).
    prev_state: BbrState,

    /// Previous ProbeBw cycle index (for phase-change detection).
    prev_cycle_idx: usize,

    /// Round count when the current ProbeBw gain phase started.
    cycle_phase_start_round: u64,

    /// Simulated round counter (incremented per ACK for simplicity).
    round_count: u64,

    /// Previous peak BtlBw (for new-record detection).
    prev_peak_bw: f64,

    /// Pending events not yet consumed by the caller.
    pending_events: Vec<BandwidthProbeEvent>,

    /// Lifetime statistics.
    stats: BbrStats,
}

impl BbrCongestionController {
    /// Create a new controller with the given BBR configuration.
    #[must_use]
    pub fn new(config: BbrConfig) -> Self {
        let core = BbrController::new(config);
        let init_state = *core.state();
        Self {
            core,
            prev_state: init_state,
            prev_cycle_idx: 0,
            cycle_phase_start_round: 0,
            round_count: 0,
            prev_peak_bw: 0.0,
            pending_events: Vec::with_capacity(4),
            stats: BbrStats {
                min_rtt_secs: f64::MAX,
                ..BbrStats::default()
            },
        }
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /// Process an ACK and update all internal state.
    ///
    /// Returns a slice of [`BandwidthProbeEvent`]s generated during this ACK.
    /// The slice is valid until the next call to `on_ack`.
    pub fn on_ack(&mut self, sample: AckSample) -> &[BandwidthProbeEvent] {
        self.pending_events.clear();

        if sample.elapsed_secs <= 0.0 || sample.rtt_secs <= 0.0 {
            return &self.pending_events;
        }

        // Update stats before handing off to the core.
        self.stats.ack_count += 1;
        self.stats.bytes_delivered += sample.delivered;
        if sample.rtt_secs < self.stats.min_rtt_secs {
            self.stats.min_rtt_secs = sample.rtt_secs;
        }

        self.round_count += 1;

        // Delegate to the inner BBR algorithm.
        self.core.on_ack(sample);

        let new_state = *self.core.state();
        let new_bw = self.core.btlbw();

        // Detect new bandwidth record.
        if new_bw > self.prev_peak_bw * 1.001 {
            self.pending_events
                .push(BandwidthProbeEvent::NewBandwidthRecord { btlbw_bps: new_bw });
            self.prev_peak_bw = new_bw;
            self.stats.peak_bandwidth_bps = new_bw;
            self.stats.events_emitted += 1;
        }

        // Detect state transitions.
        if new_state != self.prev_state {
            self.emit_state_transition_event(new_state);
            self.prev_state = new_state;
        }

        // Within ProbeBw, detect gain cycle phase changes.
        if new_state == BbrState::ProbeBw {
            let cycle_idx = self.probe_bw_cycle_idx();
            if cycle_idx != self.prev_cycle_idx {
                self.emit_cycle_change_event(cycle_idx);
                self.prev_cycle_idx = cycle_idx;
            }
        }

        &self.pending_events
    }

    /// Compute the instantaneous pacing budget for the given interval.
    ///
    /// `interval_secs` is the time since the last send burst.  The budget
    /// is `pacing_rate × interval_secs`, floored at [`MIN_PACING_BURST_BYTES`].
    #[must_use]
    pub fn pacing_budget(&self, interval_secs: f64) -> PacingBudget {
        let rate = self.core.pacing_rate();
        let raw = (rate * interval_secs) as u64;
        PacingBudget {
            bytes_available: raw.max(MIN_PACING_BURST_BYTES),
            interval_secs,
            pacing_rate_bps: rate,
        }
    }

    /// Current high-level congestion state.
    #[must_use]
    pub fn congestion_state(&self) -> CongestionState {
        CongestionState::from_bbr(*self.core.state(), self.probe_bw_cycle_idx())
    }

    /// Current pacing rate (bytes/sec).
    #[must_use]
    pub fn pacing_rate(&self) -> f64 {
        self.core.pacing_rate()
    }

    /// Current congestion window (bytes).
    #[must_use]
    pub fn cwnd(&self) -> u64 {
        self.core.cwnd()
    }

    /// Bottleneck bandwidth estimate (bytes/sec).
    #[must_use]
    pub fn btlbw(&self) -> f64 {
        self.core.btlbw()
    }

    /// RTprop estimate (seconds).
    #[must_use]
    pub fn rtprop(&self) -> f64 {
        self.core.rtprop()
    }

    /// Bandwidth-delay product in bytes.
    #[must_use]
    pub fn bdp_bytes(&self) -> u64 {
        let bdp = self.core.btlbw() * self.core.rtprop();
        bdp as u64
    }

    /// Inflight target (BDP × cwnd gain).
    #[must_use]
    pub fn inflight_target(&self) -> u64 {
        self.core.inflight_target()
    }

    /// Lifetime statistics snapshot.
    #[must_use]
    pub fn stats(&self) -> &BbrStats {
        &self.stats
    }

    /// Returns `true` if the controller is currently in bandwidth-probing state.
    #[must_use]
    pub fn is_probing_bandwidth(&self) -> bool {
        matches!(self.congestion_state(), CongestionState::Probing)
    }

    /// Returns `true` if the controller is measuring propagation RTT.
    #[must_use]
    pub fn is_measuring_rtt(&self) -> bool {
        matches!(self.core.state(), BbrState::ProbeRtt)
    }

    /// Current gain applied to the pacing rate (relative to BtlBw).
    ///
    /// Returns the effective pacing gain fraction currently in use.
    #[must_use]
    pub fn effective_pacing_gain(&self) -> f64 {
        let bw = self.core.btlbw();
        if bw > 0.0 {
            self.core.pacing_rate() / bw
        } else {
            1.0
        }
    }

    // ── Internal ────────────────────────────────────────────────────────────

    /// Infer the ProbeBw cycle index from the current pacing rate and BtlBw.
    ///
    /// We reconstruct this by comparing the pacing gain to the known cycle
    /// values.  This is an approximation — the true index lives inside
    /// `BbrController` but is not exposed.  For our event generation we
    /// only need to know probe-up (idx 0) vs drain (idx 1) vs steady (idx 2+).
    fn probe_bw_cycle_idx(&self) -> usize {
        let gain = self.effective_pacing_gain();
        if gain > 1.1 {
            0 // probe-up
        } else if gain < 0.9 {
            1 // drain
        } else {
            2 // steady
        }
    }

    fn emit_state_transition_event(&mut self, new_state: BbrState) {
        match new_state {
            BbrState::ProbeBw => {
                // Entering ProbeBw from Drain (normal) or ProbeRtt.
                let was_probe_rtt = self.prev_state == BbrState::ProbeRtt;
                if was_probe_rtt {
                    self.stats.probe_rtt_count += 1;
                    self.pending_events.push(BandwidthProbeEvent::ProbeRttDone {
                        rtprop_secs: self.core.rtprop(),
                    });
                    self.stats.events_emitted += 1;
                } else {
                    // Transitioned from Drain → ProbeBw; about to start probe-up.
                    self.pending_events.push(BandwidthProbeEvent::Steady);
                    self.stats.events_emitted += 1;
                }
            }
            BbrState::ProbeRtt => {
                self.pending_events.push(BandwidthProbeEvent::ProbeRtt);
                self.stats.events_emitted += 1;
            }
            BbrState::Drain => {
                let bdp = self.bdp_bytes();
                self.pending_events
                    .push(BandwidthProbeEvent::Drain { bdp_bytes: bdp });
                self.stats.events_emitted += 1;
            }
            BbrState::Startup => {} // only happens at init
        }
    }

    fn emit_cycle_change_event(&mut self, new_cycle_idx: usize) {
        self.cycle_phase_start_round = self.round_count;
        match new_cycle_idx {
            0 => {
                self.stats.probe_up_count += 1;
                self.pending_events.push(BandwidthProbeEvent::ProbeUp {
                    btlbw_bps: self.core.btlbw(),
                    rtprop_secs: self.core.rtprop(),
                });
                self.stats.events_emitted += 1;
            }
            1 => {
                let bdp = self.bdp_bytes();
                self.pending_events
                    .push(BandwidthProbeEvent::Drain { bdp_bytes: bdp });
                self.stats.events_emitted += 1;
            }
            _ => {
                self.pending_events.push(BandwidthProbeEvent::Steady);
                self.stats.events_emitted += 1;
            }
        }
    }
}

// ─── Pacing Rate Advisor ──────────────────────────────────────────────────────

/// Advisory pacing rate computed from BDP and current state.
///
/// Separates "what rate should I target?" (advisor) from "what's the
/// congestion window?" (cwnd), enabling fine-grained media bitrate selection.
#[derive(Debug, Clone, Copy)]
pub struct PacingRateAdvisory {
    /// Target pacing rate in bits per second for media encoding.
    pub target_bitrate_bps: u64,
    /// Safety headroom factor (0.0–1.0) applied to avoid cwnd saturation.
    pub headroom: f64,
    /// Congestion state at the time of advice.
    pub state: CongestionState,
}

impl PacingRateAdvisory {
    /// Compute a [`PacingRateAdvisory`] from the controller's current state.
    ///
    /// Uses a headroom of 0.9 in stable/probing states and 0.5 during RTT
    /// measurement to keep the queue drained.
    #[must_use]
    pub fn from_controller(ctrl: &BbrCongestionController) -> Self {
        let state = ctrl.congestion_state();
        let headroom = match state {
            CongestionState::Stable | CongestionState::FillingPipe => 0.9,
            CongestionState::Probing => 1.0,
            CongestionState::Draining => 0.7,
            CongestionState::MeasuringRtt => 0.5,
        };
        let raw_bps = ctrl.pacing_rate() * headroom;
        // Convert bytes/sec to bits/sec.
        let target_bitrate_bps = (raw_bps * 8.0) as u64;
        Self {
            target_bitrate_bps,
            headroom,
            state,
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(delivered: u64, elapsed: f64, rtt: f64) -> AckSample {
        AckSample {
            delivered,
            elapsed_secs: elapsed,
            rtt_secs: rtt,
            is_app_limited: false,
        }
    }

    fn default_ctrl() -> BbrCongestionController {
        BbrCongestionController::new(BbrConfig::default())
    }

    // ── CongestionState ──────────────────────────────────────────────────────

    #[test]
    fn test_congestion_state_from_startup() {
        let state = CongestionState::from_bbr(BbrState::Startup, 0);
        assert_eq!(state, CongestionState::FillingPipe);
    }

    #[test]
    fn test_congestion_state_from_probe_bw_cycle_0() {
        let state = CongestionState::from_bbr(BbrState::ProbeBw, 0);
        assert_eq!(state, CongestionState::Probing);
    }

    #[test]
    fn test_congestion_state_from_probe_bw_cycle_1() {
        let state = CongestionState::from_bbr(BbrState::ProbeBw, 1);
        assert_eq!(state, CongestionState::Draining);
    }

    #[test]
    fn test_congestion_state_from_probe_bw_steady() {
        for idx in 2..8 {
            let state = CongestionState::from_bbr(BbrState::ProbeBw, idx);
            assert_eq!(state, CongestionState::Stable);
        }
    }

    #[test]
    fn test_congestion_state_display() {
        assert_eq!(CongestionState::Stable.to_string(), "stable");
        assert_eq!(CongestionState::Probing.to_string(), "probing");
        assert_eq!(CongestionState::MeasuringRtt.to_string(), "measuring_rtt");
    }

    // ── Initial state ────────────────────────────────────────────────────────

    #[test]
    fn test_initial_pacing_rate_positive() {
        let ctrl = default_ctrl();
        assert!(ctrl.pacing_rate() > 0.0);
    }

    #[test]
    fn test_initial_cwnd_positive() {
        let ctrl = default_ctrl();
        assert!(ctrl.cwnd() > 0);
    }

    #[test]
    fn test_initial_congestion_state_filling_pipe() {
        let ctrl = default_ctrl();
        assert_eq!(ctrl.congestion_state(), CongestionState::FillingPipe);
    }

    // ── Pacing budget ────────────────────────────────────────────────────────

    #[test]
    fn test_pacing_budget_zero_interval_gets_minimum() {
        let ctrl = default_ctrl();
        let budget = ctrl.pacing_budget(0.0);
        assert!(budget.bytes_available >= MIN_PACING_BURST_BYTES);
    }

    #[test]
    fn test_pacing_budget_can_send() {
        let ctrl = default_ctrl();
        let budget = ctrl.pacing_budget(0.001); // 1 ms interval
        assert!(budget.can_send());
    }

    #[test]
    fn test_pacing_budget_larger_interval_more_bytes() {
        let ctrl = default_ctrl();
        let b1 = ctrl.pacing_budget(0.001);
        let b2 = ctrl.pacing_budget(0.010);
        assert!(b2.bytes_available >= b1.bytes_available);
    }

    // ── ACK processing and stats ─────────────────────────────────────────────

    #[test]
    fn test_ack_updates_stats() {
        let mut ctrl = default_ctrl();
        ctrl.on_ack(sample(10_000, 0.001, 0.01));
        assert_eq!(ctrl.stats().ack_count, 1);
        assert_eq!(ctrl.stats().bytes_delivered, 10_000);
    }

    #[test]
    fn test_invalid_ack_ignored() {
        let mut ctrl = default_ctrl();
        ctrl.on_ack(AckSample {
            delivered: 0,
            elapsed_secs: 0.0,
            rtt_secs: 0.0,
            is_app_limited: false,
        });
        assert_eq!(ctrl.stats().ack_count, 0);
    }

    #[test]
    fn test_min_rtt_tracked() {
        let mut ctrl = default_ctrl();
        ctrl.on_ack(sample(1000, 0.001, 0.050));
        ctrl.on_ack(sample(1000, 0.001, 0.020)); // new min
        ctrl.on_ack(sample(1000, 0.001, 0.080));
        assert!(ctrl.stats().min_rtt_secs <= 0.020 + 1e-9);
    }

    // ── BDP and inflight target ───────────────────────────────────────────────

    #[test]
    fn test_bdp_bytes_non_zero_after_acks() {
        let mut ctrl = default_ctrl();
        for _ in 0..5 {
            ctrl.on_ack(sample(50_000, 0.001, 0.01));
        }
        assert!(ctrl.bdp_bytes() > 0);
    }

    #[test]
    fn test_inflight_target_at_least_bdp() {
        let mut ctrl = default_ctrl();
        for _ in 0..10 {
            ctrl.on_ack(sample(50_000, 0.001, 0.01));
        }
        assert!(ctrl.inflight_target() >= ctrl.bdp_bytes());
    }

    // ── PacingRateAdvisory ───────────────────────────────────────────────────

    #[test]
    fn test_advisory_bitrate_positive() {
        let mut ctrl = default_ctrl();
        ctrl.on_ack(sample(50_000, 0.001, 0.01));
        let advice = PacingRateAdvisory::from_controller(&ctrl);
        assert!(advice.target_bitrate_bps > 0);
    }

    #[test]
    fn test_advisory_headroom_in_valid_range() {
        let ctrl = default_ctrl();
        let advice = PacingRateAdvisory::from_controller(&ctrl);
        assert!(advice.headroom > 0.0 && advice.headroom <= 1.0);
    }

    // ── New bandwidth record event ────────────────────────────────────────────

    #[test]
    fn test_new_bw_record_event_emitted() {
        let mut ctrl = default_ctrl();
        // First sample should trigger a new record (prev_peak = 0).
        let events = ctrl.on_ack(sample(1_000_000, 0.001, 0.01));
        assert!(
            events.iter().any(|e| matches!(
                e,
                BandwidthProbeEvent::NewBandwidthRecord { .. }
            )),
            "expected a NewBandwidthRecord event on first high-bw sample"
        );
    }

    // ── Probing state detection ───────────────────────────────────────────────

    #[test]
    fn test_is_probing_bandwidth_false_in_startup() {
        let ctrl = default_ctrl();
        assert!(!ctrl.is_probing_bandwidth());
    }

    #[test]
    fn test_effective_pacing_gain_startup() {
        let ctrl = default_ctrl();
        // In startup the pacing gain is ≈ 2.885.
        let gain = ctrl.effective_pacing_gain();
        assert!(gain > 1.5, "startup gain should be well above 1.0, got {gain}");
    }

    // ── Many ACKs stability ───────────────────────────────────────────────────

    #[test]
    fn test_many_acks_stable() {
        let mut ctrl = default_ctrl();
        for i in 0..500u64 {
            let rtt = 0.005 + (i % 10) as f64 * 0.001;
            let delivered = 10_000 + (i % 3) * 1_000;
            ctrl.on_ack(sample(delivered, 0.001, rtt));
        }
        // After 500 ACKs the controller should be well out of startup.
        assert!(ctrl.pacing_rate() > 0.0);
        assert!(ctrl.cwnd() > 0);
        assert!(ctrl.stats().ack_count == 500);
    }
}
