//! Media clock — monotonic media time, clock drift compensation, and PTS/DTS
//! relationship tracking.
//!
//! A [`MediaClock`] tracks the relationship between a local wall-clock and a
//! remote media timeline.  It models:
//!
//! - **Nominal rate**: the declared tick rate of the source (e.g. 90 000 Hz).
//! - **Observed drift**: the per-sample offset between the nominal and the
//!   actually received timestamps, smoothed with an exponential moving average.
//! - **PTS→DTS mapping**: the constant B-frame offset recorded when the first
//!   pair of presentation / decode timestamps is observed.
//!
//! # Example
//!
//! ```
//! use oximedia_core::media_clock::{MediaClock, ClockConfig};
//!
//! let cfg = ClockConfig { nominal_rate: 90_000, ema_alpha: 0.1 };
//! let mut clock = MediaClock::new(cfg);
//!
//! // Feed a few (wall_ns, pts) observations.
//! clock.observe(0, 0);
//! clock.observe(1_000_000_000, 90_000); // 1 second later — no drift
//!
//! let drift = clock.drift_ppm();
//! assert!(drift.abs() < 1.0, "drift should be ~0 ppm");
//! ```

#![allow(dead_code)]

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// ClockConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`MediaClock`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClockConfig {
    /// Nominal clock rate in ticks per second (e.g. 90 000 for MPEG-TS).
    pub nominal_rate: u64,
    /// Exponential moving-average smoothing factor ∈ (0, 1].
    ///
    /// A value close to `1.0` weights new samples heavily (fast response,
    /// more noise); a value close to `0.0` smooths heavily (slow response,
    /// less noise).  Typical values are in `[0.05, 0.3]`.
    pub ema_alpha: f64,
}

impl Default for ClockConfig {
    fn default() -> Self {
        Self {
            nominal_rate: 90_000,
            ema_alpha: 0.1,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClockObservation
// ─────────────────────────────────────────────────────────────────────────────

/// A single clock observation: a wall-clock nanosecond timestamp paired with a
/// media PTS tick value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ClockObservation {
    /// Wall-clock time in nanoseconds (monotonic).
    pub wall_ns: u64,
    /// Media presentation timestamp at this wall-clock instant (ticks).
    pub pts: i64,
}

// ─────────────────────────────────────────────────────────────────────────────
// DriftEstimate
// ─────────────────────────────────────────────────────────────────────────────

/// The current drift estimate produced by a [`MediaClock`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftEstimate {
    /// Estimated clock rate in ticks per second (observed).
    pub observed_rate: f64,
    /// Drift relative to nominal rate in **parts per million** (ppm).
    ///
    /// Positive values mean the media clock is running *faster* than nominal;
    /// negative values mean it is running *slower*.
    pub drift_ppm: f64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PtsDtsRelation
// ─────────────────────────────────────────────────────────────────────────────

/// Observed relationship between presentation and decode timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtsDtsRelation {
    /// The constant `pts - dts` offset in ticks (B-frame delay).
    ///
    /// For streams without B-frames this will be `0`.
    pub pts_dts_offset: i64,
}

impl PtsDtsRelation {
    /// Derives the DTS from a PTS using the stored offset.
    #[must_use]
    pub fn dts_from_pts(&self, pts: i64) -> i64 {
        pts - self.pts_dts_offset
    }

    /// Derives the PTS from a DTS using the stored offset.
    #[must_use]
    pub fn pts_from_dts(&self, dts: i64) -> i64 {
        dts + self.pts_dts_offset
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ClockState (internal)
// ─────────────────────────────────────────────────────────────────────────────

/// Internal state accumulated from observations.
struct ClockState {
    /// The very first observation (anchor).
    anchor: ClockObservation,
    /// Most recent observation.
    latest: ClockObservation,
    /// EMA of the observed tick rate.
    ema_rate: f64,
    /// Total number of observations received (≥ 1 when this struct exists).
    obs_count: u64,
}

impl ClockState {
    fn new(first: ClockObservation, nominal_rate: f64) -> Self {
        Self {
            anchor: first,
            latest: first,
            ema_rate: nominal_rate,
            obs_count: 1,
        }
    }

    /// Update with a new observation and return the instant observed tick rate.
    fn update(&mut self, obs: ClockObservation, alpha: f64) -> f64 {
        let delta_ns = obs.wall_ns.saturating_sub(self.latest.wall_ns);
        let delta_pts = obs.pts.wrapping_sub(self.latest.pts);

        let instant_rate = if delta_ns > 0 && delta_pts > 0 {
            // ticks / second = (delta_pts) / (delta_ns / 1e9)
            delta_pts as f64 / (delta_ns as f64 / 1_000_000_000.0)
        } else {
            self.ema_rate
        };

        self.ema_rate = alpha * instant_rate + (1.0 - alpha) * self.ema_rate;
        self.latest = obs;
        self.obs_count += 1;
        instant_rate
    }

    /// Returns the EMA-smoothed drift in ppm relative to `nominal_rate`.
    fn drift_ppm(&self, nominal_rate: f64) -> f64 {
        (self.ema_rate - nominal_rate) / nominal_rate * 1_000_000.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MediaClock
// ─────────────────────────────────────────────────────────────────────────────

/// A media clock that tracks wall-clock to PTS correspondence and estimates
/// clock drift.
///
/// # Usage
///
/// 1. Create a [`MediaClock`] with a [`ClockConfig`].
/// 2. Call [`MediaClock::observe`] each time you receive a frame; provide the
///    wall-clock nanoseconds and the frame's PTS.
/// 3. Query [`MediaClock::drift_ppm`], [`MediaClock::estimate`], or
///    [`MediaClock::predict_pts`] as needed.
pub struct MediaClock {
    config: ClockConfig,
    state: Option<ClockState>,
    pts_dts_relation: Option<PtsDtsRelation>,
}

impl MediaClock {
    /// Creates a new `MediaClock` with the given configuration.
    ///
    /// No observations have been recorded yet; the clock is in an
    /// uninitialized state until the first call to [`MediaClock::observe`].
    #[must_use]
    pub fn new(config: ClockConfig) -> Self {
        Self {
            config,
            state: None,
            pts_dts_relation: None,
        }
    }

    /// Creates a `MediaClock` with default configuration (90 kHz, α = 0.1).
    #[must_use]
    pub fn default_90k() -> Self {
        Self::new(ClockConfig::default())
    }

    /// Returns the nominal clock rate (ticks / second) as configured.
    #[must_use]
    pub fn nominal_rate(&self) -> u64 {
        self.config.nominal_rate
    }

    /// Returns the number of observations recorded so far.
    #[must_use]
    pub fn observation_count(&self) -> u64 {
        self.state.as_ref().map_or(0, |s| s.obs_count)
    }

    /// Records a new wall-clock / PTS observation.
    ///
    /// # Arguments
    ///
    /// * `wall_ns` — monotonic wall-clock time in nanoseconds.
    /// * `pts`     — the media PTS at that instant (ticks).
    ///
    /// Observations with a wall-clock time ≤ the previous observation's
    /// wall-clock time are silently ignored (clock cannot go backwards).
    pub fn observe(&mut self, wall_ns: u64, pts: i64) {
        let obs = ClockObservation { wall_ns, pts };
        match &mut self.state {
            None => {
                self.state = Some(ClockState::new(obs, self.config.nominal_rate as f64));
            }
            Some(state) => {
                if obs.wall_ns <= state.latest.wall_ns {
                    // Non-monotonic or duplicate observation — ignore.
                    return;
                }
                state.update(obs, self.config.ema_alpha);
            }
        }
    }

    /// Records a PTS/DTS pair, calibrating the stored PTS→DTS relationship.
    ///
    /// The relationship is only recorded from the **first** call; subsequent
    /// calls with a different offset are ignored (the offset should be
    /// constant for a given stream).
    pub fn observe_pts_dts(&mut self, pts: i64, dts: i64) {
        if self.pts_dts_relation.is_none() {
            self.pts_dts_relation = Some(PtsDtsRelation {
                pts_dts_offset: pts - dts,
            });
        }
    }

    /// Returns the current drift estimate, or `None` if fewer than two
    /// observations have been recorded.
    #[must_use]
    pub fn estimate(&self) -> Option<DriftEstimate> {
        let state = self.state.as_ref()?;
        if state.obs_count < 2 {
            return None;
        }
        let nominal = self.config.nominal_rate as f64;
        Some(DriftEstimate {
            observed_rate: state.ema_rate,
            drift_ppm: state.drift_ppm(nominal),
        })
    }

    /// Returns the current drift in parts per million, or `0.0` if fewer than
    /// two observations have been recorded.
    ///
    /// Positive values indicate the media clock runs *faster* than nominal.
    #[must_use]
    pub fn drift_ppm(&self) -> f64 {
        self.estimate().map_or(0.0, |e| e.drift_ppm)
    }

    /// Predicts the PTS for a future wall-clock time `wall_ns`, based on the
    /// current drift estimate.
    ///
    /// Returns `None` if the clock has not yet been calibrated (< 2
    /// observations).
    #[must_use]
    pub fn predict_pts(&self, wall_ns: u64) -> Option<i64> {
        let state = self.state.as_ref()?;
        if state.obs_count < 2 {
            return None;
        }
        // Use the anchor to get a stable long-range prediction.
        let delta_ns = (wall_ns as i64).wrapping_sub(state.anchor.wall_ns as i64);
        let delta_secs = delta_ns as f64 / 1_000_000_000.0;
        let delta_pts = (delta_secs * state.ema_rate).round() as i64;
        Some(state.anchor.pts.wrapping_add(delta_pts))
    }

    /// Returns the stored PTS/DTS offset, if any has been observed via
    /// [`MediaClock::observe_pts_dts`].
    #[must_use]
    pub fn pts_dts_relation(&self) -> Option<PtsDtsRelation> {
        self.pts_dts_relation
    }

    /// Converts a PTS to a DTS using the stored PTS/DTS relation.
    ///
    /// Returns `None` if no PTS/DTS observation has been recorded yet.
    #[must_use]
    pub fn pts_to_dts(&self, pts: i64) -> Option<i64> {
        self.pts_dts_relation.map(|r| r.dts_from_pts(pts))
    }

    /// Converts a DTS to a PTS using the stored PTS/DTS relation.
    ///
    /// Returns `None` if no PTS/DTS observation has been recorded yet.
    #[must_use]
    pub fn dts_to_pts(&self, dts: i64) -> Option<i64> {
        self.pts_dts_relation.map(|r| r.pts_from_dts(dts))
    }

    /// Resets the clock state, discarding all observations and drift estimates.
    pub fn reset(&mut self) {
        self.state = None;
        self.pts_dts_relation = None;
    }
}

impl fmt::Debug for MediaClock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let obs = self.observation_count();
        let drift = self.drift_ppm();
        write!(
            f,
            "MediaClock {{ nominal_rate: {}, obs: {obs}, drift_ppm: {drift:.2} }}",
            self.config.nominal_rate
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_clock() -> MediaClock {
        MediaClock::new(ClockConfig {
            nominal_rate: 90_000,
            ema_alpha: 0.5,
        })
    }

    // 1. A brand-new clock has zero observations.
    #[test]
    fn test_initial_state() {
        let clock = make_clock();
        assert_eq!(clock.observation_count(), 0);
        assert!(clock.estimate().is_none());
        assert_eq!(clock.drift_ppm(), 0.0);
    }

    // 2. After one observation drift is still None (need ≥ 2).
    #[test]
    fn test_single_observation_no_estimate() {
        let mut clock = make_clock();
        clock.observe(0, 0);
        assert_eq!(clock.observation_count(), 1);
        assert!(clock.estimate().is_none());
    }

    // 3. Perfect-rate stream → drift ~0 ppm.
    #[test]
    fn test_perfect_rate_drift_near_zero() {
        let mut clock = make_clock();
        clock.observe(0, 0);
        clock.observe(1_000_000_000, 90_000); // exactly 90 000 ticks / second
        let drift = clock.drift_ppm();
        assert!(drift.abs() < 1.0, "drift = {drift} ppm, expected ~0");
    }

    // 4. Fast source (+100 ppm) produces positive drift.
    #[test]
    fn test_fast_source_positive_drift() {
        let mut clock = make_clock();
        // +100 ppm → source delivers 90_009 ticks per second
        let pts_per_sec: i64 = 90_009;
        clock.observe(0, 0);
        clock.observe(1_000_000_000, pts_per_sec);
        let drift = clock.drift_ppm();
        assert!(drift > 0.0, "fast source must yield positive drift");
    }

    // 5. Slow source (−100 ppm) produces negative drift.
    #[test]
    fn test_slow_source_negative_drift() {
        let mut clock = make_clock();
        let pts_per_sec: i64 = 89_991;
        clock.observe(0, 0);
        clock.observe(1_000_000_000, pts_per_sec);
        let drift = clock.drift_ppm();
        assert!(drift < 0.0, "slow source must yield negative drift");
    }

    // 6. Non-monotonic wall-clock observation is silently ignored.
    #[test]
    fn test_non_monotonic_observation_ignored() {
        let mut clock = make_clock();
        clock.observe(1_000_000_000, 90_000);
        clock.observe(500_000_000, 45_000); // earlier wall_ns → ignored
        assert_eq!(clock.observation_count(), 1);
    }

    // 7. predict_pts returns None before calibration.
    #[test]
    fn test_predict_pts_uncalibrated() {
        let mut clock = make_clock();
        clock.observe(0, 0);
        // Only 1 observation — prediction unavailable.
        assert!(clock.predict_pts(1_000_000_000).is_none());
    }

    // 8. predict_pts is correct for a nominal-rate stream.
    #[test]
    fn test_predict_pts_nominal_rate() {
        let mut clock = MediaClock::new(ClockConfig {
            nominal_rate: 90_000,
            ema_alpha: 1.0, // take new sample directly (no smoothing)
        });
        clock.observe(0, 0);
        clock.observe(1_000_000_000, 90_000);
        // Predict at t=2s
        let predicted = clock.predict_pts(2_000_000_000).expect("should predict");
        let error = (predicted - 180_000i64).abs();
        assert!(error <= 2, "predicted PTS = {predicted}, error = {error}");
    }

    // 9. PTS/DTS relationship is stored on first observe_pts_dts call.
    #[test]
    fn test_pts_dts_relation_stored() {
        let mut clock = make_clock();
        clock.observe_pts_dts(100, 90); // pts_dts_offset = 10
        let rel = clock.pts_dts_relation().expect("relation stored");
        assert_eq!(rel.pts_dts_offset, 10);
    }

    // 10. Second observe_pts_dts call with different offset is ignored.
    #[test]
    fn test_pts_dts_relation_immutable_after_first() {
        let mut clock = make_clock();
        clock.observe_pts_dts(100, 90);
        clock.observe_pts_dts(200, 150); // different offset (50) — ignored
        let rel = clock.pts_dts_relation().expect("relation stored");
        assert_eq!(rel.pts_dts_offset, 10);
    }

    // 11. pts_to_dts / dts_to_pts round-trip.
    #[test]
    fn test_pts_dts_roundtrip() {
        let mut clock = make_clock();
        clock.observe_pts_dts(180_000, 162_000); // offset = 18000
        let dts = clock.pts_to_dts(270_000).expect("relation set");
        assert_eq!(dts, 252_000);
        let pts_back = clock.dts_to_pts(dts).expect("relation set");
        assert_eq!(pts_back, 270_000);
    }

    // 12. reset() clears all state.
    #[test]
    fn test_reset_clears_state() {
        let mut clock = make_clock();
        clock.observe(0, 0);
        clock.observe(1_000_000_000, 90_000);
        clock.observe_pts_dts(90_000, 81_000);
        clock.reset();
        assert_eq!(clock.observation_count(), 0);
        assert!(clock.estimate().is_none());
        assert!(clock.pts_dts_relation().is_none());
    }

    // 13. nominal_rate() matches config.
    #[test]
    fn test_nominal_rate() {
        let clock = MediaClock::new(ClockConfig {
            nominal_rate: 48_000,
            ema_alpha: 0.1,
        });
        assert_eq!(clock.nominal_rate(), 48_000);
    }

    // 14. DriftEstimate observed_rate is close to nominal for perfect-rate stream.
    #[test]
    fn test_drift_estimate_observed_rate() {
        let mut clock = MediaClock::new(ClockConfig {
            nominal_rate: 90_000,
            ema_alpha: 1.0,
        });
        clock.observe(0, 0);
        clock.observe(1_000_000_000, 90_000);
        let est = clock.estimate().expect("estimate available");
        let diff = (est.observed_rate - 90_000.0).abs();
        assert!(diff < 1.0, "observed_rate = {}", est.observed_rate);
    }

    // 15. PtsDtsRelation helper methods work correctly.
    #[test]
    fn test_pts_dts_relation_helpers() {
        let rel = PtsDtsRelation {
            pts_dts_offset: 9000,
        };
        assert_eq!(rel.dts_from_pts(180_000), 171_000);
        assert_eq!(rel.pts_from_dts(171_000), 180_000);
    }

    // 16. Debug impl does not panic.
    #[test]
    fn test_debug_impl() {
        let mut clock = make_clock();
        clock.observe(0, 0);
        clock.observe(1_000_000_000, 90_000);
        let s = format!("{clock:?}");
        assert!(s.contains("MediaClock"));
    }

    // 17. Multiple observations converge the EMA.
    #[test]
    fn test_multiple_observations_ema_converges() {
        let mut clock = MediaClock::new(ClockConfig {
            nominal_rate: 90_000,
            ema_alpha: 0.3,
        });
        let ns_per_sec: u64 = 1_000_000_000;
        for i in 0..10u64 {
            clock.observe(i * ns_per_sec, (i * 90_000) as i64);
        }
        let drift = clock.drift_ppm();
        assert!(drift.abs() < 5.0, "EMA drift = {drift} ppm after 10 obs");
    }
}
