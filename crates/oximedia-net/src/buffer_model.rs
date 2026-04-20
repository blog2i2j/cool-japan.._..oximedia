//! Client buffer model for adaptive streaming.
//!
//! This module models the player's media buffer as a fluid queue where:
//!
//! - **Fill rate** (`r` in seconds of media / second of wall-clock) depends on
//!   the available download bandwidth and the current segment bitrate.
//! - **Drain rate** is 1.0 s/s during normal playback, 0.0 during stall.
//! - **Rebuffer probability** is estimated analytically from a Gaussian
//!   approximation of the buffer trajectory.
//!
//! ## Key types
//!
//! | Type | Role |
//! |------|------|
//! | [`BufferConfig`] | Static configuration (target level, safety margins) |
//! | [`BufferState`] | Mutable runtime state (current fill, stall counters) |
//! | [`BufferModel`] | High-level controller combining config + state |
//! | [`RebufferEstimator`] | Probabilistic rebuffer-risk calculator |
//! | [`FillDrainBalance`] | Fill / drain rates and their ratio |
//!
//! ## Algorithm
//!
//! The rebuffer probability model follows the approach used in BOLA and
//! similar systems: given a buffer level `b` seconds and a fill rate `r`,
//! we compute the expected time to depletion under a worst-case bandwidth
//! drop and model it as a Gaussian random variable whose standard deviation
//! grows with jitter.  The CDF gives the probability that the buffer reaches
//! zero before recovering.

#![allow(dead_code)]

use std::collections::VecDeque;

use crate::error::{NetError, NetResult};

// ─── Configuration ────────────────────────────────────────────────────────────

/// Static configuration for the buffer model.
#[derive(Debug, Clone)]
pub struct BufferConfig {
    /// Target buffer level in seconds — the model aims to keep the buffer here.
    pub target_level_secs: f64,
    /// Minimum safe buffer level in seconds — below this the model enters
    /// emergency mode.
    pub min_safe_secs: f64,
    /// Maximum buffer capacity in seconds — segments above this are dropped.
    pub max_capacity_secs: f64,
    /// Drain rate during normal playback (usually 1.0 s/s).
    pub playback_drain_rate: f64,
    /// Number of recent bandwidth samples to keep for jitter estimation.
    pub bandwidth_history_len: usize,
}

impl Default for BufferConfig {
    fn default() -> Self {
        Self {
            target_level_secs: 15.0,
            min_safe_secs: 3.0,
            max_capacity_secs: 60.0,
            playback_drain_rate: 1.0,
            bandwidth_history_len: 8,
        }
    }
}

// ─── Fill / drain balance ─────────────────────────────────────────────────────

/// The instantaneous fill-rate/drain-rate balance.
#[derive(Debug, Clone, Copy)]
pub struct FillDrainBalance {
    /// Download fill rate: seconds of media added per second of wall-clock.
    ///
    /// `fill_rate = available_bandwidth_bps / segment_bitrate_bps`.
    pub fill_rate: f64,
    /// Playback drain rate (1.0 during normal playback, 0.0 when stalled).
    pub drain_rate: f64,
    /// Net rate: positive = buffer growing, negative = buffer shrinking.
    pub net_rate: f64,
}

impl FillDrainBalance {
    /// Computes the balance for the given bandwidth and segment bitrate.
    ///
    /// `bandwidth_bps` — available download bandwidth in bits/s.
    /// `segment_bitrate_bps` — bitrate of the segment currently being
    ///   downloaded in bits/s.
    /// `is_playing` — `true` if the player is actively draining the buffer.
    ///
    /// # Errors
    ///
    /// Returns [`NetError::InvalidState`] if `segment_bitrate_bps` is zero.
    pub fn compute(
        bandwidth_bps: f64,
        segment_bitrate_bps: f64,
        is_playing: bool,
    ) -> NetResult<Self> {
        if segment_bitrate_bps <= 0.0 {
            return Err(NetError::invalid_state(
                "segment_bitrate_bps must be positive",
            ));
        }
        let fill_rate = bandwidth_bps / segment_bitrate_bps;
        let drain_rate = if is_playing { 1.0 } else { 0.0 };
        Ok(Self {
            fill_rate,
            drain_rate,
            net_rate: fill_rate - drain_rate,
        })
    }

    /// Returns `true` if the buffer is growing.
    #[must_use]
    pub fn is_growing(&self) -> bool {
        self.net_rate > 0.0
    }

    /// Returns `true` if the buffer is shrinking faster than it is filling.
    #[must_use]
    pub fn is_draining(&self) -> bool {
        self.net_rate < 0.0
    }
}

// ─── Buffer state ─────────────────────────────────────────────────────────────

/// The runtime state of the player buffer.
#[derive(Debug, Clone)]
pub struct BufferState {
    /// Current buffer fill level in seconds.
    level_secs: f64,
    /// Whether the player is currently stalled (buffer empty during playback).
    is_stalled: bool,
    /// Total stall duration accumulated since last reset.
    total_stall_secs: f64,
    /// Number of stall events since last reset.
    stall_count: u32,
    /// Recent bandwidth measurements (bits/s) for jitter computation.
    bandwidth_history: VecDeque<f64>,
    /// Configuration (kept for capacity checks).
    max_capacity_secs: f64,
    /// Maximum history length.
    history_len: usize,
}

impl BufferState {
    /// Creates a new buffer state with the given initial fill level and config.
    #[must_use]
    pub fn new(config: &BufferConfig) -> Self {
        Self {
            level_secs: 0.0,
            is_stalled: false,
            total_stall_secs: 0.0,
            stall_count: 0,
            bandwidth_history: VecDeque::with_capacity(config.bandwidth_history_len),
            max_capacity_secs: config.max_capacity_secs,
            history_len: config.bandwidth_history_len,
        }
    }

    /// Current buffer fill level in seconds.
    #[must_use]
    pub fn level(&self) -> f64 {
        self.level_secs
    }

    /// Whether the player is stalled.
    #[must_use]
    pub fn is_stalled(&self) -> bool {
        self.is_stalled
    }

    /// Total stall time in seconds since last reset.
    #[must_use]
    pub fn total_stall_secs(&self) -> f64 {
        self.total_stall_secs
    }

    /// Number of stall events since last reset.
    #[must_use]
    pub fn stall_count(&self) -> u32 {
        self.stall_count
    }

    /// Adds `segment_secs` seconds of media to the buffer.
    ///
    /// The level is clamped to `max_capacity_secs`.
    pub fn add_segment(&mut self, segment_secs: f64) {
        self.level_secs = (self.level_secs + segment_secs).min(self.max_capacity_secs);
        if self.is_stalled && self.level_secs > 0.0 {
            // Buffer recovered — exit stall
            self.is_stalled = false;
        }
    }

    /// Advances the simulation by `elapsed` seconds of wall-clock time.
    ///
    /// This drains the buffer at `drain_rate` (typically 1.0 s/s).  If the
    /// buffer reaches zero during playback a stall is recorded.
    ///
    /// Returns the stall duration that occurred during this step (0.0 if none).
    pub fn advance(&mut self, elapsed_secs: f64, drain_rate: f64) -> f64 {
        if elapsed_secs <= 0.0 {
            return 0.0;
        }

        let to_drain = elapsed_secs * drain_rate;
        if to_drain <= self.level_secs {
            self.level_secs -= to_drain;
            0.0
        } else {
            // Buffer depleted — compute stall duration
            let stall_secs = if drain_rate > 0.0 {
                elapsed_secs - self.level_secs / drain_rate
            } else {
                0.0
            };
            self.level_secs = 0.0;
            if !self.is_stalled && drain_rate > 0.0 {
                self.is_stalled = true;
                self.stall_count += 1;
            }
            self.total_stall_secs += stall_secs;
            stall_secs
        }
    }

    /// Records a bandwidth sample (bits/s) used for jitter estimation.
    pub fn record_bandwidth(&mut self, bps: f64) {
        if self.bandwidth_history.len() >= self.history_len {
            self.bandwidth_history.pop_front();
        }
        self.bandwidth_history.push_back(bps);
    }

    /// Returns the mean of recent bandwidth samples.
    #[must_use]
    pub fn mean_bandwidth(&self) -> Option<f64> {
        if self.bandwidth_history.is_empty() {
            return None;
        }
        let sum: f64 = self.bandwidth_history.iter().sum();
        Some(sum / self.bandwidth_history.len() as f64)
    }

    /// Returns the standard deviation of recent bandwidth samples (jitter).
    #[must_use]
    pub fn bandwidth_stddev(&self) -> Option<f64> {
        if self.bandwidth_history.len() < 2 {
            return None;
        }
        let mean = self.mean_bandwidth()?;
        let variance = self
            .bandwidth_history
            .iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>()
            / (self.bandwidth_history.len() - 1) as f64;
        Some(variance.sqrt())
    }

    /// Resets stall counters without changing buffer level.
    pub fn reset_stall_counters(&mut self) {
        self.total_stall_secs = 0.0;
        self.stall_count = 0;
        self.is_stalled = false;
    }
}

// ─── Rebuffer probability estimator ──────────────────────────────────────────

/// Estimates the probability that the buffer will run empty before the next
/// segment can be downloaded and appended.
///
/// ## Model
///
/// Given:
/// - buffer level `b` (seconds)
/// - fill rate `r` (s/s)
/// - bandwidth coefficient of variation `cv = σ / μ`
///
/// Time to buffer empty (when draining at 1 s/s):
///   `T_empty = b / (1 - r)`   when `r < 1.0`
///
/// Time to download next segment of duration `d` seconds at fill rate `r`:
///   `T_download = d / r`
///
/// Rebuffer happens when `T_download > T_empty`:
///   `d/r > b/(1 - r)` ⟺ `d*(1-r) > b*r`
///
/// Under bandwidth jitter (`cv`) the fill rate is modelled as a Gaussian
/// r.v. with mean `r` and std `r * cv`.  The rebuffer probability is the
/// fraction of the distribution that falls below `b / (b + d)`.
pub struct RebufferEstimator {
    /// Coefficient of variation of bandwidth (σ/μ).
    cv: f64,
}

impl RebufferEstimator {
    /// Creates a new estimator.
    ///
    /// `cv` — coefficient of variation of bandwidth, e.g. `0.2` for
    ///   moderately variable networks.  Clamped to `[0, 2]`.
    #[must_use]
    pub fn new(cv: f64) -> Self {
        Self {
            cv: cv.clamp(0.0, 2.0),
        }
    }

    /// Creates an estimator from a [`BufferState`]'s bandwidth history.
    ///
    /// If insufficient history is available, uses `fallback_cv`.
    #[must_use]
    pub fn from_state(state: &BufferState, fallback_cv: f64) -> Self {
        let cv = match (state.mean_bandwidth(), state.bandwidth_stddev()) {
            (Some(mean), Some(std)) if mean > 0.0 => std / mean,
            _ => fallback_cv,
        };
        Self::new(cv)
    }

    /// Estimates the probability that downloading a segment of `segment_secs`
    /// duration at mean fill rate `fill_rate` will cause a rebuffer given the
    /// current buffer level `buffer_secs`.
    ///
    /// Returns a value in `[0.0, 1.0]`.
    #[must_use]
    pub fn rebuffer_probability(&self, buffer_secs: f64, fill_rate: f64, segment_secs: f64) -> f64 {
        if buffer_secs <= 0.0 {
            return 1.0; // already stalled
        }
        if fill_rate <= 0.0 {
            return 1.0; // no bandwidth
        }
        if segment_secs <= 0.0 {
            return 0.0;
        }

        // Critical fill rate below which rebuffer occurs:
        //   r_critical = b / (b + d)
        let r_critical = buffer_secs / (buffer_secs + segment_secs);

        if fill_rate >= 1.0 {
            // We're downloading faster than playing — rebuffer only under
            // severe bandwidth drop.
            if r_critical >= 1.0 {
                return 0.0;
            }
        }

        // Model fill rate as Gaussian(fill_rate, fill_rate * cv)
        let sigma = fill_rate * self.cv;
        if sigma <= 0.0 {
            // Deterministic — either definitely rebuffers or doesn't
            return if fill_rate < r_critical { 1.0 } else { 0.0 };
        }

        // P(r < r_critical) = Φ((r_critical - fill_rate) / sigma)
        let z = (r_critical - fill_rate) / sigma;
        gaussian_cdf(z)
    }

    /// Estimates the expected stall duration (seconds) per segment under
    /// current conditions.
    #[must_use]
    pub fn expected_stall_secs(&self, buffer_secs: f64, fill_rate: f64, segment_secs: f64) -> f64 {
        let prob = self.rebuffer_probability(buffer_secs, fill_rate, segment_secs);
        if prob <= 0.0 {
            return 0.0;
        }
        // Expected shortfall: how long does a rebuffer event last?
        // Approximation: stall_secs ≈ download_time - time_to_empty when
        // fill_rate is drawn from the left tail.
        if fill_rate <= 0.0 {
            return segment_secs; // worst case
        }
        let download_time = segment_secs / fill_rate;
        let time_to_empty = if fill_rate < 1.0 {
            buffer_secs / (1.0 - fill_rate)
        } else {
            f64::INFINITY
        };
        let stall_if_rebuffers = (download_time - time_to_empty).max(0.0);
        prob * stall_if_rebuffers
    }
}

/// Approximates Φ(z) (standard normal CDF) using the Abramowitz & Stegun
/// rational approximation (error < 7.5e-8).
fn gaussian_cdf(z: f64) -> f64 {
    const P: f64 = 0.231_641_9;
    const A: [f64; 5] = [
        0.319_381_53,
        -0.356_563_782,
        1.781_477_937,
        -1.821_255_978,
        1.330_274_429,
    ];

    let t = 1.0 / (1.0 + P * z.abs());
    let poly = t * (A[0] + t * (A[1] + t * (A[2] + t * (A[3] + t * A[4]))));
    let pdf = (-z * z / 2.0).exp() / (2.0 * std::f64::consts::PI).sqrt();
    let tail = pdf * poly;

    if z >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

// ─── High-level buffer model ──────────────────────────────────────────────────

/// Buffer model phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferPhase {
    /// Buffer is critically low — enter emergency quality mode.
    Critical,
    /// Buffer is building towards target level.
    Building,
    /// Buffer is at or above target level — steady state.
    Steady,
    /// Buffer is overflowing (above max capacity).
    Full,
}

/// High-level controller combining configuration, state, and policy.
pub struct BufferModel {
    config: BufferConfig,
    state: BufferState,
}

impl BufferModel {
    /// Creates a new buffer model.
    #[must_use]
    pub fn new(config: BufferConfig) -> Self {
        let state = BufferState::new(&config);
        Self { config, state }
    }

    /// Returns the current buffer phase.
    #[must_use]
    pub fn phase(&self) -> BufferPhase {
        let level = self.state.level_secs;
        if level >= self.config.max_capacity_secs {
            BufferPhase::Full
        } else if level >= self.config.target_level_secs {
            BufferPhase::Steady
        } else if level >= self.config.min_safe_secs {
            BufferPhase::Building
        } else {
            BufferPhase::Critical
        }
    }

    /// Current buffer level in seconds.
    #[must_use]
    pub fn level(&self) -> f64 {
        self.state.level_secs
    }

    /// Reference to configuration.
    #[must_use]
    pub fn config(&self) -> &BufferConfig {
        &self.config
    }

    /// Mutable reference to buffer state.
    #[must_use]
    pub fn state(&self) -> &BufferState {
        &self.state
    }

    /// Mutable reference to buffer state.
    pub fn state_mut(&mut self) -> &mut BufferState {
        &mut self.state
    }

    /// Advances the buffer by `elapsed` seconds and records bandwidth.
    ///
    /// Returns any stall duration that occurred.
    pub fn tick(&mut self, elapsed_secs: f64, bandwidth_bps: Option<f64>) -> f64 {
        let drain = if self.state.is_stalled {
            0.0
        } else {
            self.config.playback_drain_rate
        };
        let stall = self.state.advance(elapsed_secs, drain);
        if let Some(bw) = bandwidth_bps {
            self.state.record_bandwidth(bw);
        }
        stall
    }

    /// Adds a downloaded segment of `segment_secs` duration.
    pub fn add_segment(&mut self, segment_secs: f64) {
        self.state.add_segment(segment_secs);
    }

    /// Returns a rebuffer probability estimate for the given conditions.
    ///
    /// `fill_rate` — current download fill rate (s of media / s of wall time).
    /// `segment_secs` — duration of the next segment to download.
    #[must_use]
    pub fn rebuffer_probability(&self, fill_rate: f64, segment_secs: f64) -> f64 {
        let estimator = RebufferEstimator::from_state(&self.state, 0.2);
        estimator.rebuffer_probability(self.state.level_secs, fill_rate, segment_secs)
    }

    /// Computes the recommended quality cap based on buffer health.
    ///
    /// Returns a multiplier in `[0.0, 1.0]` relative to the maximum available
    /// bitrate: 1.0 = no cap, 0.0 = use minimum quality.
    #[must_use]
    pub fn quality_cap(&self) -> f64 {
        match self.phase() {
            BufferPhase::Critical => 0.0,
            BufferPhase::Building => {
                // Linear ramp from 0 at min_safe to 0.5 at target
                let range = self.config.target_level_secs - self.config.min_safe_secs;
                if range <= 0.0 {
                    return 0.25;
                }
                let t = (self.state.level_secs - self.config.min_safe_secs) / range;
                (t * 0.5).clamp(0.0, 0.5)
            }
            BufferPhase::Steady => 1.0,
            BufferPhase::Full => 1.0,
        }
    }

    /// Resets the model to its initial state.
    pub fn reset(&mut self) {
        self.state = BufferState::new(&self.config);
    }
}

// ─── Playback quality of experience metrics ───────────────────────────────────

/// Aggregate playback quality-of-experience (QoE) metrics.
#[derive(Debug, Clone, Default)]
pub struct PlaybackQoE {
    /// Total playback duration (seconds) including stall time.
    pub total_secs: f64,
    /// Total stall time (seconds).
    pub stall_secs: f64,
    /// Number of stall events.
    pub stall_count: u32,
    /// Number of quality switch events.
    pub quality_switches: u32,
    /// Mean segment quality (e.g. average bitrate / max bitrate).
    pub mean_quality_score: f64,
}

impl PlaybackQoE {
    /// Adds a playback step observation.
    ///
    /// `elapsed` — wall-clock duration of this step.
    /// `stall_secs` — stall that occurred during this step (0 if none).
    /// `quality_score` — normalised quality in `[0, 1]` for this step.
    pub fn observe(&mut self, elapsed: f64, stall_secs: f64, quality_score: f64, switched: bool) {
        self.total_secs += elapsed;
        self.stall_secs += stall_secs;
        if stall_secs > 0.0 {
            self.stall_count += 1;
        }
        if switched {
            self.quality_switches += 1;
        }
        // Running mean
        if self.total_secs > 0.0 {
            let w = elapsed / self.total_secs;
            self.mean_quality_score = (1.0 - w) * self.mean_quality_score + w * quality_score;
        }
    }

    /// Stall ratio: fraction of total time spent stalled.
    #[must_use]
    pub fn stall_ratio(&self) -> f64 {
        if self.total_secs <= 0.0 {
            0.0
        } else {
            (self.stall_secs / self.total_secs).clamp(0.0, 1.0)
        }
    }

    /// Composite QoE score in `[0, 1]`.
    ///
    /// Weights: quality (50%), stall penalty (40%), switch penalty (10%).
    #[must_use]
    pub fn composite_score(&self) -> f64 {
        let stall_penalty = (self.stall_ratio() * 5.0).clamp(0.0, 1.0);
        let switch_penalty = (self.quality_switches as f64 * 0.01).clamp(0.0, 0.2);
        let q = self.mean_quality_score.clamp(0.0, 1.0);
        let score = 0.5 * q - 0.4 * stall_penalty - 0.1 * switch_penalty;
        score.clamp(0.0, 1.0)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_model() -> BufferModel {
        BufferModel::new(BufferConfig::default())
    }

    // 1. Buffer starts empty
    #[test]
    fn test_initial_buffer_empty() {
        let m = default_model();
        assert_eq!(m.level(), 0.0);
        assert_eq!(m.phase(), BufferPhase::Critical);
    }

    // 2. Adding segments increases buffer level
    #[test]
    fn test_add_segment_increases_level() {
        let mut m = default_model();
        m.add_segment(6.0);
        assert!((m.level() - 6.0).abs() < 1e-9);
        assert_eq!(m.phase(), BufferPhase::Building);
    }

    // 3. Buffer drains during tick
    #[test]
    fn test_tick_drains_buffer() {
        let mut m = default_model();
        m.add_segment(10.0);
        let stall = m.tick(3.0, None);
        assert_eq!(stall, 0.0);
        assert!((m.level() - 7.0).abs() < 1e-9);
    }

    // 4. Buffer stall detection
    #[test]
    fn test_stall_detected_when_buffer_empty() {
        let mut m = default_model();
        m.add_segment(2.0);
        let stall = m.tick(5.0, None); // drain 5s but only 2s available
        assert!(stall > 0.0, "should stall");
        assert_eq!(m.state().stall_count(), 1);
    }

    // 5. Buffer recovers from stall after segment added
    #[test]
    fn test_stall_recovery_after_segment_add() {
        let mut m = default_model();
        // Drain to empty
        m.tick(1.0, None);
        assert!(m.state().is_stalled());
        // Add a segment — should exit stall
        m.add_segment(5.0);
        assert!(!m.state().is_stalled());
    }

    // 6. Fill/drain balance computation
    #[test]
    fn test_fill_drain_balance_growing() {
        // 4 Mbps bandwidth, 2 Mbps segment → fill rate 2.0, net +1.0
        let bal = FillDrainBalance::compute(4_000_000.0, 2_000_000.0, true)
            .expect("bandwidth and segment size are valid positive values");
        assert!((bal.fill_rate - 2.0).abs() < 1e-9);
        assert!(bal.is_growing());
        assert!(!bal.is_draining());
    }

    // 7. Rebuffer probability rises as buffer drops and fill rate is marginal
    #[test]
    fn test_rebuffer_probability_increases_low_buffer() {
        let estimator = RebufferEstimator::new(0.3);
        // Safe scenario: 20s buffer, fill rate 2.0 (clearly filling faster)
        // r_critical = 20/(20+6) ≈ 0.77; fill_rate=2.0 >> critical → very low prob
        let prob_safe = estimator.rebuffer_probability(20.0, 2.0, 6.0);
        // Risky scenario: 1s buffer, fill rate 0.8 (barely above critical=1/7≈0.14
        // but with 30% CV there is meaningful probability of going below critical)
        // Under zero buffer → prob = 1.0
        let prob_empty = estimator.rebuffer_probability(0.0, 0.8, 6.0);
        assert_eq!(prob_empty, 1.0, "zero buffer should give probability 1.0");
        assert!(
            prob_safe < 0.5,
            "prob at safe buffer + high fill rate should be low: {prob_safe}"
        );
    }

    // 8. Quality cap reflects buffer phase
    #[test]
    fn test_quality_cap_by_phase() {
        let mut m = default_model();
        // Critical phase: cap = 0
        assert_eq!(m.quality_cap(), 0.0);

        // Building phase
        m.add_segment(8.0); // above min_safe (3) but below target (15)
        assert!(m.quality_cap() > 0.0 && m.quality_cap() < 1.0);

        // Steady phase
        m.add_segment(20.0); // above target (15)
        assert_eq!(m.quality_cap(), 1.0);
    }

    // 9. QoE metrics accumulate correctly
    #[test]
    fn test_qoe_stall_ratio() {
        let mut qoe = PlaybackQoE::default();
        qoe.observe(10.0, 0.0, 1.0, false);
        qoe.observe(10.0, 2.0, 0.5, true);
        assert!((qoe.stall_ratio() - 0.1).abs() < 0.01);
        assert_eq!(qoe.quality_switches, 1);
    }

    // 10. Bandwidth jitter from history
    #[test]
    fn test_bandwidth_stddev_from_history() {
        let config = BufferConfig::default();
        let mut state = BufferState::new(&config);
        for &bw in &[1_000_000.0f64, 2_000_000.0, 1_500_000.0, 1_800_000.0] {
            state.record_bandwidth(bw);
        }
        let mean = state.mean_bandwidth().expect("mean should be available");
        assert!(mean > 0.0);
        let std = state
            .bandwidth_stddev()
            .expect("stddev should be available");
        assert!(std > 0.0);
    }
}
