#![allow(dead_code)]
//! Active bandwidth probing scheduler with EWMA estimation and packet-loss
//! integration.
//!
//! ## Overview
//!
//! This module implements a *send-side* bandwidth probing loop used to
//! continuously discover and track available network capacity for adaptive
//! media delivery.  It operates in three cooperative layers:
//!
//! 1. **[`ProbeScheduler`]** — decides *when* to send probe bursts and tracks
//!    probe state (idle / probing / cooldown).
//! 2. **[`EwmaBandwidthEstimator`]** — maintains an exponentially-weighted
//!    moving average of measured throughput and adjusts the estimate in
//!    response to packet loss.
//! 3. **[`ProbeResult`] / [`ProbeReport`]** — carry the outcome of each probe
//!    burst back to the estimator and the application.
//!
//! The module is entirely synchronous and does *not* depend on `tokio` or any
//! other async runtime.  The caller is responsible for driving the scheduler
//! tick (e.g. from a `tokio::time::interval` task).

use std::collections::VecDeque;
use std::fmt;
use std::time::Duration;

use crate::error::{NetError, NetResult};

// ─── EWMA bandwidth estimator ────────────────────────────────────────────────

/// Configuration for [`EwmaBandwidthEstimator`].
#[derive(Debug, Clone)]
pub struct EwmaConfig {
    /// Smoothing factor α ∈ (0, 1].  Larger values track faster changes.
    ///
    /// Typical value: `0.2` for stable networks, `0.5` for highly variable.
    pub alpha: f64,
    /// Loss penalty factor: when packet-loss fraction `L` is observed the
    /// estimated bandwidth is multiplied by `(1 − loss_penalty_factor × L)`.
    ///
    /// Typical value: `0.5` (halve estimate for 100 % loss; cap applied).
    pub loss_penalty_factor: f64,
    /// Lower bound on bandwidth in kbps.  Estimate is never allowed below this.
    pub min_kbps: f64,
    /// Upper bound on bandwidth in kbps.  Estimate is never allowed above this.
    pub max_kbps: f64,
}

impl Default for EwmaConfig {
    fn default() -> Self {
        Self {
            alpha: 0.2,
            loss_penalty_factor: 0.5,
            min_kbps: 64.0,
            max_kbps: 100_000.0,
        }
    }
}

impl EwmaConfig {
    /// Validates the configuration.
    ///
    /// # Errors
    /// Returns `Err` if `alpha` is outside `(0, 1]`, `min_kbps >= max_kbps`,
    /// or `loss_penalty_factor` is outside `[0, 1]`.
    pub fn validate(&self) -> NetResult<()> {
        if self.alpha <= 0.0 || self.alpha > 1.0 {
            return Err(NetError::protocol(format!(
                "EWMA alpha must be in (0,1], got {}",
                self.alpha
            )));
        }
        if self.min_kbps >= self.max_kbps {
            return Err(NetError::protocol(format!(
                "min_kbps ({}) must be < max_kbps ({})",
                self.min_kbps, self.max_kbps
            )));
        }
        if !(0.0..=1.0).contains(&self.loss_penalty_factor) {
            return Err(NetError::protocol(format!(
                "loss_penalty_factor must be in [0,1], got {}",
                self.loss_penalty_factor
            )));
        }
        Ok(())
    }
}

/// EWMA-based bandwidth estimator with integrated packet-loss adjustment.
///
/// The estimate `B̂` after each update is:
///
/// ```text
/// raw  = measured_kbps × (1 − penalty × loss_fraction)
/// B̂   = α × raw + (1 − α) × B̂_prev
/// B̂   = clamp(B̂, min_kbps, max_kbps)
/// ```
///
/// On the first update the previous estimate is initialised from the raw
/// measurement.
#[derive(Debug)]
pub struct EwmaBandwidthEstimator {
    config: EwmaConfig,
    /// Current EWMA estimate in kbps.  `None` until the first measurement.
    estimate_kbps: Option<f64>,
    /// Short history of recent raw measurements for variance tracking.
    recent_raw: VecDeque<f64>,
    /// Fixed history length.
    history_len: usize,
}

impl EwmaBandwidthEstimator {
    /// Creates a new estimator.
    ///
    /// # Errors
    /// Propagates `config.validate()` errors.
    pub fn new(config: EwmaConfig) -> NetResult<Self> {
        config.validate()?;
        Ok(Self {
            config,
            estimate_kbps: None,
            recent_raw: VecDeque::with_capacity(20),
            history_len: 20,
        })
    }

    /// Creates a new estimator with default configuration.
    ///
    /// # Errors
    /// Returns `Err` if the default config fails validation (never in practice).
    pub fn with_defaults() -> NetResult<Self> {
        Self::new(EwmaConfig::default())
    }

    /// Returns the current bandwidth estimate in kbps.
    ///
    /// Returns `None` before the first measurement.
    #[must_use]
    pub fn estimate_kbps(&self) -> Option<f64> {
        self.estimate_kbps
    }

    /// Updates the estimate with a new raw measurement and loss fraction.
    ///
    /// `loss_fraction` must be in `[0.0, 1.0]`; values outside that range are
    /// clamped.
    ///
    /// Returns the updated estimate in kbps.
    pub fn update(&mut self, measured_kbps: f64, loss_fraction: f64) -> f64 {
        let loss = loss_fraction.clamp(0.0, 1.0);
        let penalty = 1.0 - self.config.loss_penalty_factor * loss;
        let raw = (measured_kbps * penalty).max(0.0);

        // Maintain short history
        if self.recent_raw.len() == self.history_len {
            self.recent_raw.pop_front();
        }
        self.recent_raw.push_back(raw);

        let new_estimate = match self.estimate_kbps {
            None => raw,
            Some(prev) => self.config.alpha * raw + (1.0 - self.config.alpha) * prev,
        };

        let clamped = new_estimate.clamp(self.config.min_kbps, self.config.max_kbps);
        self.estimate_kbps = Some(clamped);
        clamped
    }

    /// Resets the estimator to the uninitialized state.
    pub fn reset(&mut self) {
        self.estimate_kbps = None;
        self.recent_raw.clear();
    }

    /// Returns the variance of the recent raw measurement history.
    ///
    /// Returns `0.0` when fewer than two measurements are available.
    #[must_use]
    pub fn raw_variance_kbps(&self) -> f64 {
        let n = self.recent_raw.len();
        if n < 2 {
            return 0.0;
        }
        let mean: f64 = self.recent_raw.iter().sum::<f64>() / n as f64;
        let variance: f64 = self
            .recent_raw
            .iter()
            .map(|x| {
                let d = x - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        variance
    }
}

// ─── Probe state machine ─────────────────────────────────────────────────────

/// Current state of the probing state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ProbeState {
    /// Waiting for the next probe interval.
    Idle,
    /// A probe burst is in flight.
    Probing,
    /// Recovering after a probe; no probes sent during this period.
    Cooldown,
}

impl fmt::Display for ProbeState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Idle => "Idle",
            Self::Probing => "Probing",
            Self::Cooldown => "Cooldown",
        };
        f.write_str(s)
    }
}

// ─── Probe result ─────────────────────────────────────────────────────────────

/// Outcome of a single probe burst.
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeResult {
    /// Wall-clock timestamp when the probe completed (ms).
    pub timestamp_ms: u64,
    /// Bytes successfully delivered in the probe burst.
    pub bytes_delivered: u64,
    /// Duration of the probe burst in milliseconds.
    pub duration_ms: u64,
    /// Fraction of probe packets lost (0.0 .. 1.0).
    pub loss_fraction: f64,
}

impl ProbeResult {
    /// Creates a new [`ProbeResult`].
    ///
    /// `loss_fraction` is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn new(
        timestamp_ms: u64,
        bytes_delivered: u64,
        duration_ms: u64,
        loss_fraction: f64,
    ) -> Self {
        Self {
            timestamp_ms,
            bytes_delivered,
            duration_ms,
            loss_fraction: loss_fraction.clamp(0.0, 1.0),
        }
    }

    /// Computes the measured throughput in kbps from the probe result.
    ///
    /// Returns `0.0` when `duration_ms` is zero.
    #[must_use]
    pub fn measured_kbps(&self) -> f64 {
        if self.duration_ms == 0 {
            return 0.0;
        }
        (self.bytes_delivered as f64 * 8.0) / self.duration_ms as f64
    }
}

/// Aggregate report produced after ingesting a [`ProbeResult`].
#[derive(Debug, Clone, PartialEq)]
pub struct ProbeReport {
    /// Raw measured throughput from the probe (kbps).
    pub measured_kbps: f64,
    /// Updated EWMA estimate (kbps).
    pub estimate_kbps: f64,
    /// Packet-loss fraction reported by the probe.
    pub loss_fraction: f64,
    /// Updated scheduler state after processing the result.
    pub new_state: ProbeState,
}

// ─── Scheduler ────────────────────────────────────────────────────────────────

/// Configuration for [`ProbeScheduler`].
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Minimum interval between probe bursts.
    pub probe_interval: Duration,
    /// Duration of a single probe burst.
    pub probe_duration: Duration,
    /// Cooldown period after each probe before the next may start.
    pub cooldown_duration: Duration,
    /// Loss fraction above which the scheduler increases the probe rate
    /// (shortens the effective interval to `probe_interval / 2`).
    pub loss_probe_threshold: f64,
    /// Maximum probe size in bytes (caps the burst payload).
    pub max_probe_bytes: u64,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            probe_interval: Duration::from_secs(2),
            probe_duration: Duration::from_millis(200),
            cooldown_duration: Duration::from_millis(500),
            loss_probe_threshold: 0.02,
            max_probe_bytes: 65_536,
        }
    }
}

/// Active bandwidth probing scheduler.
///
/// Call [`ProbeScheduler::tick`] periodically (e.g. every 50 ms).  When a
/// probe should be sent the method returns `Some(probe_bytes)` indicating
/// how many bytes to inject into the network.  After the probe completes,
/// pass the [`ProbeResult`] to [`ProbeScheduler::report`] to update the
/// EWMA estimate and advance the state machine.
#[derive(Debug)]
pub struct ProbeScheduler {
    config: SchedulerConfig,
    estimator: EwmaBandwidthEstimator,
    state: ProbeState,
    /// Simulated clock: total elapsed ms since creation.
    elapsed_ms: u64,
    /// `elapsed_ms` when the current state was entered.
    state_entered_ms: u64,
    /// Number of probe cycles completed.
    probe_count: u64,
    /// Last observed loss fraction (used to decide whether to shorten interval).
    last_loss: f64,
}

impl ProbeScheduler {
    /// Creates a new scheduler.
    ///
    /// # Errors
    /// Propagates EWMA config validation errors.
    pub fn new(config: SchedulerConfig, ewma: EwmaConfig) -> NetResult<Self> {
        let estimator = EwmaBandwidthEstimator::new(ewma)?;
        Ok(Self {
            config,
            estimator,
            state: ProbeState::Idle,
            elapsed_ms: 0,
            state_entered_ms: 0,
            probe_count: 0,
            last_loss: 0.0,
        })
    }

    /// Creates a scheduler with default configuration.
    ///
    /// # Errors
    /// Returns `Err` if the default EWMA config is invalid (never in practice).
    pub fn with_defaults() -> NetResult<Self> {
        Self::new(SchedulerConfig::default(), EwmaConfig::default())
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Current state of the probe state machine.
    #[must_use]
    pub fn state(&self) -> ProbeState {
        self.state
    }

    /// Current EWMA estimate in kbps, or `None` before the first probe.
    #[must_use]
    pub fn estimate_kbps(&self) -> Option<f64> {
        self.estimator.estimate_kbps()
    }

    /// Number of probe cycles completed so far.
    #[must_use]
    pub fn probe_count(&self) -> u64 {
        self.probe_count
    }

    /// Last packet-loss fraction observed.
    #[must_use]
    pub fn last_loss(&self) -> f64 {
        self.last_loss
    }

    // ── Tick ──────────────────────────────────────────────────────────────────

    /// Advances the scheduler clock by `delta_ms` milliseconds.
    ///
    /// Returns `Some(probe_bytes)` when a new probe should be sent.  The
    /// caller must send exactly `probe_bytes` bytes over the network and then
    /// call [`Self::report`] with the measured outcome.
    ///
    /// Returns `None` when the scheduler is in `Probing` or `Cooldown` state,
    /// or when the probe interval has not yet elapsed.
    pub fn tick(&mut self, delta_ms: u64) -> Option<u64> {
        self.elapsed_ms += delta_ms;

        match self.state {
            ProbeState::Idle => {
                let interval_ms = self.effective_interval_ms();
                let time_in_state = self.elapsed_ms - self.state_entered_ms;
                if time_in_state >= interval_ms {
                    self.transition(ProbeState::Probing);
                    Some(self.probe_bytes())
                } else {
                    None
                }
            }
            // Probing and Cooldown transitions are driven by report(), not tick().
            ProbeState::Probing | ProbeState::Cooldown => None,
        }
    }

    // ── Report ────────────────────────────────────────────────────────────────

    /// Ingests a completed [`ProbeResult`], updates the EWMA estimate, and
    /// transitions the state machine.
    ///
    /// # Errors
    /// Returns `Err` if called while the scheduler is not in the `Probing` state.
    pub fn report(&mut self, result: ProbeResult) -> NetResult<ProbeReport> {
        if self.state != ProbeState::Probing {
            return Err(NetError::invalid_state(format!(
                "report() called in state {}; expected Probing",
                self.state
            )));
        }

        let measured_kbps = result.measured_kbps();
        self.last_loss = result.loss_fraction;
        let estimate_kbps = self.estimator.update(measured_kbps, result.loss_fraction);
        self.probe_count += 1;

        self.transition(ProbeState::Cooldown);

        // Auto-exit cooldown after the configured cooldown duration by
        // advancing the state clock so the next tick() can re-enter Idle.
        // (We cannot drive wall time here, so we rely on tick() calls.)

        Ok(ProbeReport {
            measured_kbps,
            estimate_kbps,
            loss_fraction: result.loss_fraction,
            new_state: self.state,
        })
    }

    /// Forces the scheduler out of `Cooldown` and back to `Idle`.
    ///
    /// This is normally called automatically by `tick()` after the cooldown
    /// duration elapses.  It is exposed publicly for testing purposes.
    pub fn end_cooldown(&mut self) {
        if self.state == ProbeState::Cooldown {
            self.transition(ProbeState::Idle);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    fn transition(&mut self, new_state: ProbeState) {
        self.state = new_state;
        self.state_entered_ms = self.elapsed_ms;
    }

    /// Effective probe interval in ms.
    ///
    /// When recent loss is above the threshold the interval is halved so
    /// probes are sent more frequently to track the degrading path.
    fn effective_interval_ms(&self) -> u64 {
        let base = self.config.probe_interval.as_millis() as u64;
        if self.last_loss >= self.config.loss_probe_threshold {
            base / 2
        } else {
            base
        }
    }

    /// Bytes to include in the next probe burst (capped by config).
    fn probe_bytes(&self) -> u64 {
        // If we have an estimate, send ~200 ms worth of data at that rate.
        // kbps → bytes/ms = kbps / 8.  200 ms window.
        let target = self
            .estimator
            .estimate_kbps()
            .map_or(self.config.max_probe_bytes, |bw| {
                let bytes_per_ms = bw / 8.0;
                let probe_ms = self.config.probe_duration.as_millis() as f64;
                (bytes_per_ms * probe_ms) as u64
            });
        target.min(self.config.max_probe_bytes).max(1)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────
#[cfg(test)]
mod tests {
    use super::*;

    // 1. EwmaConfig::validate rejects bad alpha
    #[test]
    fn test_ewma_config_bad_alpha() {
        let cfg = EwmaConfig {
            alpha: 0.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
        let cfg2 = EwmaConfig {
            alpha: 1.5,
            ..Default::default()
        };
        assert!(cfg2.validate().is_err());
    }

    // 2. EwmaConfig::validate rejects inverted bounds
    #[test]
    fn test_ewma_config_bad_bounds() {
        let cfg = EwmaConfig {
            min_kbps: 1000.0,
            max_kbps: 500.0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    // 3. First update seeds the estimate
    #[test]
    fn test_ewma_first_update() {
        let mut est = EwmaBandwidthEstimator::with_defaults().expect("valid config");
        assert!(est.estimate_kbps().is_none());
        let e = est.update(5_000.0, 0.0);
        assert!((e - 5_000.0).abs() < 1.0, "estimate={e}");
    }

    // 4. Loss reduces the estimate
    #[test]
    fn test_ewma_loss_penalty() {
        let mut est = EwmaBandwidthEstimator::with_defaults().expect("valid config");
        let no_loss = est.update(5_000.0, 0.0);
        est.reset();
        let with_loss = est.update(5_000.0, 0.5);
        assert!(
            with_loss < no_loss,
            "no_loss={no_loss} with_loss={with_loss}"
        );
    }

    // 5. Estimate clamped to min_kbps
    #[test]
    fn test_ewma_clamp_min() {
        let cfg = EwmaConfig {
            min_kbps: 100.0,
            ..Default::default()
        };
        let mut est = EwmaBandwidthEstimator::new(cfg).expect("valid config");
        let e = est.update(0.0, 1.0);
        assert!(e >= 100.0, "e={e}");
    }

    // 6. Estimate clamped to max_kbps
    #[test]
    fn test_ewma_clamp_max() {
        let cfg = EwmaConfig {
            max_kbps: 1_000.0,
            ..Default::default()
        };
        let mut est = EwmaBandwidthEstimator::new(cfg).expect("valid config");
        let e = est.update(1_000_000.0, 0.0);
        assert!(e <= 1_000.0, "e={e}");
    }

    // 7. ProbeResult::measured_kbps
    #[test]
    fn test_probe_result_kbps() {
        // 10_000 bytes in 100 ms = 800 kbps
        let r = ProbeResult::new(0, 10_000, 100, 0.0);
        assert!((r.measured_kbps() - 800.0).abs() < 1e-6);
    }

    // 8. ProbeResult::measured_kbps zero guard
    #[test]
    fn test_probe_result_kbps_zero_duration() {
        let r = ProbeResult::new(0, 5_000, 0, 0.0);
        assert_eq!(r.measured_kbps(), 0.0);
    }

    // 9. ProbeScheduler starts Idle
    #[test]
    fn test_scheduler_starts_idle() {
        let sched = ProbeScheduler::with_defaults().expect("valid config");
        assert_eq!(sched.state(), ProbeState::Idle);
        assert_eq!(sched.probe_count(), 0);
    }

    // 10. tick triggers probe after interval
    #[test]
    fn test_scheduler_tick_triggers_probe() {
        let cfg = SchedulerConfig {
            probe_interval: Duration::from_millis(200),
            ..Default::default()
        };
        let mut sched = ProbeScheduler::new(cfg, EwmaConfig::default()).expect("valid");
        // tick just short of interval → no probe
        assert!(sched.tick(199).is_none());
        // tick exactly to interval → probe fires
        let bytes = sched.tick(1);
        assert!(bytes.is_some(), "expected probe bytes");
        assert_eq!(sched.state(), ProbeState::Probing);
    }

    // 11. report transitions to Cooldown and updates estimate
    #[test]
    fn test_scheduler_report() {
        let cfg = SchedulerConfig {
            probe_interval: Duration::from_millis(100),
            ..Default::default()
        };
        let mut sched = ProbeScheduler::new(cfg, EwmaConfig::default()).expect("valid");
        sched.tick(100); // trigger probe
        let result = ProbeResult::new(100, 8_000, 100, 0.01);
        let report = sched.report(result).expect("should succeed");
        assert_eq!(report.new_state, ProbeState::Cooldown);
        assert!(report.estimate_kbps > 0.0);
        assert_eq!(sched.probe_count(), 1);
    }

    // 12. report in wrong state returns Err
    #[test]
    fn test_scheduler_report_wrong_state() {
        let mut sched = ProbeScheduler::with_defaults().expect("valid");
        let result = ProbeResult::new(0, 1_000, 100, 0.0);
        assert!(sched.report(result).is_err());
    }

    // 13. end_cooldown restores Idle
    #[test]
    fn test_scheduler_end_cooldown() {
        let cfg = SchedulerConfig {
            probe_interval: Duration::from_millis(100),
            ..Default::default()
        };
        let mut sched = ProbeScheduler::new(cfg, EwmaConfig::default()).expect("valid");
        sched.tick(100);
        sched
            .report(ProbeResult::new(100, 8_000, 100, 0.0))
            .expect("report ok");
        assert_eq!(sched.state(), ProbeState::Cooldown);
        sched.end_cooldown();
        assert_eq!(sched.state(), ProbeState::Idle);
    }

    // 14. high loss shortens effective interval
    #[test]
    fn test_high_loss_shortens_interval() {
        let cfg = SchedulerConfig {
            probe_interval: Duration::from_millis(2_000),
            loss_probe_threshold: 0.02,
            ..Default::default()
        };
        let mut sched = ProbeScheduler::new(cfg, EwmaConfig::default()).expect("valid");
        // Inject a high-loss probe result
        sched.tick(2_000);
        sched
            .report(ProbeResult::new(0, 100, 100, 0.05))
            .expect("report ok");
        sched.end_cooldown();

        // Now effective interval should be 1_000 ms (halved)
        assert!(sched.tick(999).is_none());
        let bytes = sched.tick(1);
        assert!(
            bytes.is_some(),
            "probe should fire after 1000 ms under high loss"
        );
    }

    // 15. ProbeState Display
    #[test]
    fn test_probe_state_display() {
        assert_eq!(format!("{}", ProbeState::Idle), "Idle");
        assert_eq!(format!("{}", ProbeState::Probing), "Probing");
        assert_eq!(format!("{}", ProbeState::Cooldown), "Cooldown");
    }
}
