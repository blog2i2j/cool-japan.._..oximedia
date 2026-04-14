#![allow(dead_code)]
//! Stereo phase correlation meter for audio monitoring.
//!
//! This module provides [`PhaseCorrelationMeter`] which computes the normalized
//! cross-correlation coefficient between left and right audio channels. The output
//! ranges from -1.0 (fully out of phase / inverted) through 0.0 (uncorrelated)
//! to +1.0 (mono-identical / perfectly in phase).
//!
//! [`PhaseAlert`] monitors the correlation over time and fires when the coefficient
//! drops below a configurable threshold, indicating potential phase cancellation
//! problems.
//!
//! # Example
//!
//! ```
//! use oximedia_scopes::audio_phase_meter::{PhaseCorrelationMeter, PhaseAlert};
//!
//! let mut meter = PhaseCorrelationMeter::new(48000, 0.3);
//! // Feed stereo interleaved samples.
//! let left  = [0.5_f64, 0.3, -0.2, 0.1, 0.0, -0.4, 0.6, -0.1];
//! let right = [0.5, 0.3, -0.2, 0.1, 0.0, -0.4, 0.6, -0.1]; // identical = +1.0
//! meter.push_interleaved(&left.iter().zip(right.iter())
//!     .flat_map(|(&l, &r)| [l, r])
//!     .collect::<Vec<_>>());
//! assert!((meter.correlation() - 1.0).abs() < 0.01);
//! ```

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// PhaseCorrelationMeter
// ---------------------------------------------------------------------------

/// Stereo phase correlation meter using normalized cross-correlation.
///
/// The correlation is computed over a sliding window whose duration is controlled
/// by `integration_seconds`. New samples pushed into the meter replace the oldest
/// samples once the window is full.
#[derive(Debug, Clone)]
pub struct PhaseCorrelationMeter {
    /// Ring buffer for left channel samples.
    left_buf: VecDeque<f64>,
    /// Ring buffer for right channel samples.
    right_buf: VecDeque<f64>,
    /// Maximum number of samples in the window (sample_rate * integration_seconds).
    window_size: usize,
    /// Sample rate in Hz.
    sample_rate: u32,
    /// Integration time in seconds.
    integration_seconds: f64,
    /// Cached correlation value (updated on each push).
    cached_correlation: f64,
    /// Running sums for incremental computation.
    sum_lr: f64,
    sum_ll: f64,
    sum_rr: f64,
}

impl PhaseCorrelationMeter {
    /// Create a new phase correlation meter.
    ///
    /// * `sample_rate` — audio sample rate in Hz.
    /// * `integration_seconds` — duration of the sliding analysis window.
    ///   Clamped to \[0.01, 10.0\] seconds.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn new(sample_rate: u32, integration_seconds: f64) -> Self {
        let int_sec = integration_seconds.clamp(0.01, 10.0);
        let window_size = ((f64::from(sample_rate)) * int_sec).ceil() as usize;
        let window_size = window_size.max(1);

        Self {
            left_buf: VecDeque::with_capacity(window_size),
            right_buf: VecDeque::with_capacity(window_size),
            window_size,
            sample_rate,
            integration_seconds: int_sec,
            cached_correlation: 0.0,
            sum_lr: 0.0,
            sum_ll: 0.0,
            sum_rr: 0.0,
        }
    }

    /// Push a single L/R sample pair and update the running correlation.
    pub fn push_sample(&mut self, left: f64, right: f64) {
        // Remove oldest if buffer is full.
        if self.left_buf.len() >= self.window_size {
            if let (Some(old_l), Some(old_r)) =
                (self.left_buf.pop_front(), self.right_buf.pop_front())
            {
                self.sum_lr -= old_l * old_r;
                self.sum_ll -= old_l * old_l;
                self.sum_rr -= old_r * old_r;
            }
        }

        self.left_buf.push_back(left);
        self.right_buf.push_back(right);
        self.sum_lr += left * right;
        self.sum_ll += left * left;
        self.sum_rr += right * right;

        self.recompute_correlation();
    }

    /// Push interleaved stereo samples: `[L0, R0, L1, R1, ...]`.
    ///
    /// If the slice length is odd, the last sample is ignored.
    pub fn push_interleaved(&mut self, samples: &[f64]) {
        let pairs = samples.len() / 2;
        for i in 0..pairs {
            self.push_sample(samples[i * 2], samples[i * 2 + 1]);
        }
    }

    /// Push separate left and right channel buffers. If lengths differ, the shorter
    /// length is used.
    pub fn push_channels(&mut self, left: &[f64], right: &[f64]) {
        let n = left.len().min(right.len());
        for i in 0..n {
            self.push_sample(left[i], right[i]);
        }
    }

    /// Current correlation coefficient in the range \[-1.0, +1.0\].
    ///
    /// * `+1.0` — perfectly in phase (mono).
    /// * `0.0` — uncorrelated.
    /// * `-1.0` — perfectly out of phase (inverted).
    ///
    /// Returns 0.0 if no samples have been pushed.
    #[must_use]
    pub fn correlation(&self) -> f64 {
        self.cached_correlation
    }

    /// Reset the meter, discarding all buffered samples.
    pub fn reset(&mut self) {
        self.left_buf.clear();
        self.right_buf.clear();
        self.sum_lr = 0.0;
        self.sum_ll = 0.0;
        self.sum_rr = 0.0;
        self.cached_correlation = 0.0;
    }

    /// Number of samples currently in the analysis window.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.left_buf.len()
    }

    /// The configured window size in samples.
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.window_size
    }

    /// The configured sample rate.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The configured integration time in seconds.
    #[must_use]
    pub fn integration_seconds(&self) -> f64 {
        self.integration_seconds
    }

    /// Recompute the cached correlation from running sums.
    fn recompute_correlation(&mut self) {
        let denom = (self.sum_ll * self.sum_rr).sqrt();
        if denom < 1e-30 {
            self.cached_correlation = 0.0;
        } else {
            self.cached_correlation = (self.sum_lr / denom).clamp(-1.0, 1.0);
        }
    }

    /// Full recompute from buffers (used after drift accumulation to correct float errors).
    /// Call periodically (e.g. every N seconds) for long-running meters.
    pub fn recalibrate(&mut self) {
        self.sum_lr = 0.0;
        self.sum_ll = 0.0;
        self.sum_rr = 0.0;
        for (l, r) in self.left_buf.iter().zip(self.right_buf.iter()) {
            self.sum_lr += l * r;
            self.sum_ll += l * l;
            self.sum_rr += r * r;
        }
        self.recompute_correlation();
    }
}

// ---------------------------------------------------------------------------
// PhaseAlert
// ---------------------------------------------------------------------------

/// Alert severity levels for phase problems.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhaseAlertSeverity {
    /// No problem detected.
    Ok,
    /// Correlation is low — stereo image may be very wide or partially cancelling.
    Warning,
    /// Correlation is strongly negative — significant phase cancellation.
    Critical,
}

/// Phase cancellation alert monitor.
///
/// Wraps a [`PhaseCorrelationMeter`] and checks the correlation against configurable
/// thresholds to detect phase cancellation issues.
#[derive(Debug, Clone)]
pub struct PhaseAlert {
    meter: PhaseCorrelationMeter,
    /// Correlation threshold below which a warning is issued.
    warning_threshold: f64,
    /// Correlation threshold below which a critical alert is issued.
    critical_threshold: f64,
    /// Number of consecutive frames where severity >= Warning.
    consecutive_alert_frames: u64,
    /// Minimum consecutive alert frames before actually reporting.
    min_consecutive_frames: u64,
}

impl PhaseAlert {
    /// Create a new phase alert with default thresholds.
    ///
    /// * `sample_rate` — audio sample rate in Hz.
    /// * `integration_seconds` — sliding window duration.
    ///
    /// Defaults: warning at correlation < 0.0, critical at < -0.5,
    /// minimum 3 consecutive alert frames before reporting.
    #[must_use]
    pub fn new(sample_rate: u32, integration_seconds: f64) -> Self {
        Self {
            meter: PhaseCorrelationMeter::new(sample_rate, integration_seconds),
            warning_threshold: 0.0,
            critical_threshold: -0.5,
            consecutive_alert_frames: 0,
            min_consecutive_frames: 3,
        }
    }

    /// Set the warning threshold. Correlation below this triggers a warning.
    #[must_use]
    pub fn with_warning_threshold(mut self, threshold: f64) -> Self {
        self.warning_threshold = threshold.clamp(-1.0, 1.0);
        self
    }

    /// Set the critical threshold. Correlation below this triggers a critical alert.
    #[must_use]
    pub fn with_critical_threshold(mut self, threshold: f64) -> Self {
        self.critical_threshold = threshold.clamp(-1.0, 1.0);
        self
    }

    /// Set the minimum number of consecutive alert frames before reporting.
    #[must_use]
    pub fn with_min_consecutive_frames(mut self, n: u64) -> Self {
        self.min_consecutive_frames = n.max(1);
        self
    }

    /// Push interleaved stereo samples and evaluate the alert state.
    pub fn push_interleaved(&mut self, samples: &[f64]) {
        self.meter.push_interleaved(samples);
        self.update_alert_state();
    }

    /// Push separate left and right channel buffers.
    pub fn push_channels(&mut self, left: &[f64], right: &[f64]) {
        self.meter.push_channels(left, right);
        self.update_alert_state();
    }

    /// Current correlation value.
    #[must_use]
    pub fn correlation(&self) -> f64 {
        self.meter.correlation()
    }

    /// Current severity (considering consecutive-frame debouncing).
    #[must_use]
    pub fn severity(&self) -> PhaseAlertSeverity {
        if self.consecutive_alert_frames < self.min_consecutive_frames {
            return PhaseAlertSeverity::Ok;
        }
        let corr = self.meter.correlation();
        if corr < self.critical_threshold {
            PhaseAlertSeverity::Critical
        } else if corr < self.warning_threshold {
            PhaseAlertSeverity::Warning
        } else {
            PhaseAlertSeverity::Ok
        }
    }

    /// Whether an alert is currently active (severity is Warning or Critical).
    #[must_use]
    pub fn is_alerting(&self) -> bool {
        self.severity() != PhaseAlertSeverity::Ok
    }

    /// Number of consecutive frames where the correlation was below warning threshold.
    #[must_use]
    pub fn consecutive_alert_frames(&self) -> u64 {
        self.consecutive_alert_frames
    }

    /// Access the underlying meter.
    #[must_use]
    pub fn meter(&self) -> &PhaseCorrelationMeter {
        &self.meter
    }

    /// Reset both the meter and the alert counter.
    pub fn reset(&mut self) {
        self.meter.reset();
        self.consecutive_alert_frames = 0;
    }

    fn update_alert_state(&mut self) {
        let corr = self.meter.correlation();
        if corr < self.warning_threshold {
            self.consecutive_alert_frames += 1;
        } else {
            self.consecutive_alert_frames = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sine(freq: f64, sample_rate: u32, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| {
                let t = i as f64 / f64::from(sample_rate);
                (2.0 * std::f64::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_identical_channels_correlation_is_one() {
        let mut meter = PhaseCorrelationMeter::new(48000, 0.1);
        let sig = make_sine(440.0, 48000, 4800);
        meter.push_channels(&sig, &sig);
        assert!(
            (meter.correlation() - 1.0).abs() < 0.01,
            "corr = {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_inverted_channels_correlation_is_neg_one() {
        let mut meter = PhaseCorrelationMeter::new(48000, 0.1);
        let sig = make_sine(440.0, 48000, 4800);
        let inv: Vec<f64> = sig.iter().map(|&s| -s).collect();
        meter.push_channels(&sig, &inv);
        assert!(
            (meter.correlation() + 1.0).abs() < 0.01,
            "corr = {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_uncorrelated_channels_near_zero() {
        let mut meter = PhaseCorrelationMeter::new(48000, 0.5);
        // Two sines at non-harmonic frequencies over many cycles
        let l = make_sine(440.0, 48000, 24000);
        let r = make_sine(571.0, 48000, 24000);
        meter.push_channels(&l, &r);
        assert!(
            meter.correlation().abs() < 0.15,
            "corr = {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_silence_correlation_is_zero() {
        let mut meter = PhaseCorrelationMeter::new(48000, 0.1);
        let silence = vec![0.0_f64; 4800];
        meter.push_channels(&silence, &silence);
        assert!(
            meter.correlation().abs() < 1e-6,
            "corr = {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_push_interleaved() {
        let mut meter = PhaseCorrelationMeter::new(48000, 0.1);
        let sig = make_sine(1000.0, 48000, 4800);
        let interleaved: Vec<f64> = sig.iter().flat_map(|&s| [s, s]).collect();
        meter.push_interleaved(&interleaved);
        assert!(
            (meter.correlation() - 1.0).abs() < 0.01,
            "corr = {}",
            meter.correlation()
        );
    }

    #[test]
    fn test_window_slides() {
        let mut meter = PhaseCorrelationMeter::new(48000, 0.01); // 480 samples window
        // Push correlated data
        let sig = make_sine(440.0, 48000, 480);
        meter.push_channels(&sig, &sig);
        assert!(meter.correlation() > 0.9);
        // Now push anti-correlated data to replace the window
        let inv: Vec<f64> = sig.iter().map(|&s| -s).collect();
        meter.push_channels(&sig, &inv);
        // Should be near -1 after the window is fully replaced
        assert!(meter.correlation() < -0.9, "corr = {}", meter.correlation());
    }

    #[test]
    fn test_reset_clears_state() {
        let mut meter = PhaseCorrelationMeter::new(48000, 0.1);
        let sig = make_sine(440.0, 48000, 4800);
        meter.push_channels(&sig, &sig);
        assert!(meter.sample_count() > 0);
        meter.reset();
        assert_eq!(meter.sample_count(), 0);
        assert!(meter.correlation().abs() < 1e-10);
    }

    #[test]
    fn test_recalibrate_matches() {
        let mut meter = PhaseCorrelationMeter::new(48000, 0.1);
        let sig = make_sine(440.0, 48000, 4800);
        meter.push_channels(&sig, &sig);
        let before = meter.correlation();
        meter.recalibrate();
        let after = meter.correlation();
        assert!(
            (before - after).abs() < 1e-6,
            "before={before}, after={after}"
        );
    }

    #[test]
    fn test_phase_alert_ok_when_correlated() {
        let mut alert = PhaseAlert::new(48000, 0.1);
        let sig = make_sine(440.0, 48000, 4800);
        let interleaved: Vec<f64> = sig.iter().flat_map(|&s| [s, s]).collect();
        alert.push_interleaved(&interleaved);
        assert_eq!(alert.severity(), PhaseAlertSeverity::Ok);
        assert!(!alert.is_alerting());
    }

    #[test]
    fn test_phase_alert_warning_on_inversion() {
        let mut alert = PhaseAlert::new(48000, 0.1).with_min_consecutive_frames(1);
        let sig = make_sine(440.0, 48000, 4800);
        let inv: Vec<f64> = sig.iter().map(|&s| -s).collect();
        // Push multiple frames to satisfy consecutive-frame requirement.
        for _ in 0..3 {
            alert.push_channels(&sig, &inv);
        }
        assert!(alert.is_alerting());
        assert_eq!(alert.severity(), PhaseAlertSeverity::Critical);
    }

    #[test]
    fn test_phase_alert_debounce() {
        let mut alert = PhaseAlert::new(48000, 0.01).with_min_consecutive_frames(5);
        let sig = make_sine(440.0, 48000, 480);
        let inv: Vec<f64> = sig.iter().map(|&s| -s).collect();
        // Push only 2 anti-correlated frames — should not alert yet.
        alert.push_channels(&sig, &inv);
        alert.push_channels(&sig, &inv);
        assert_eq!(alert.severity(), PhaseAlertSeverity::Ok);
    }

    #[test]
    fn test_phase_alert_reset() {
        let mut alert = PhaseAlert::new(48000, 0.1).with_min_consecutive_frames(1);
        let sig = make_sine(440.0, 48000, 4800);
        let inv: Vec<f64> = sig.iter().map(|&s| -s).collect();
        alert.push_channels(&sig, &inv);
        alert.push_channels(&sig, &inv);
        assert!(alert.consecutive_alert_frames() > 0);
        alert.reset();
        assert_eq!(alert.consecutive_alert_frames(), 0);
        assert_eq!(alert.severity(), PhaseAlertSeverity::Ok);
    }

    #[test]
    fn test_meter_sample_rate_and_window() {
        let meter = PhaseCorrelationMeter::new(44100, 0.5);
        assert_eq!(meter.sample_rate(), 44100);
        // 44100 * 0.5 = 22050
        assert_eq!(meter.window_size(), 22050);
    }

    #[test]
    fn test_integration_clamped() {
        let meter = PhaseCorrelationMeter::new(48000, 100.0);
        assert!((meter.integration_seconds() - 10.0).abs() < 1e-6);
        let meter2 = PhaseCorrelationMeter::new(48000, 0.001);
        assert!((meter2.integration_seconds() - 0.01).abs() < 1e-6);
    }
}
