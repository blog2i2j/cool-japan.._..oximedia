#![allow(dead_code)]
//! Timecode drift detection and correction.
//!
//! Detects clock drift between timecode sources and a reference clock,
//! and provides correction strategies for maintaining synchronization
//! in long-form recordings.

use crate::{FrameRate, Timecode, TimecodeError};

/// A drift measurement sample: the observed timecode vs expected timecode at a given wall time.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DriftSample {
    /// Wall-clock time in seconds since the measurement started.
    pub wall_time_secs: f64,
    /// Observed frame count from the timecode source.
    pub observed_frames: u64,
    /// Expected frame count from the reference clock.
    pub expected_frames: u64,
}

impl DriftSample {
    /// Creates a new drift sample.
    pub fn new(wall_time_secs: f64, observed_frames: u64, expected_frames: u64) -> Self {
        Self {
            wall_time_secs,
            observed_frames,
            expected_frames,
        }
    }

    /// Returns the drift in frames (observed - expected). Positive means ahead.
    #[allow(clippy::cast_precision_loss)]
    pub fn drift_frames(&self) -> i64 {
        self.observed_frames as i64 - self.expected_frames as i64
    }

    /// Returns the drift as a fraction of the expected frames.
    #[allow(clippy::cast_precision_loss)]
    pub fn drift_ratio(&self) -> f64 {
        if self.expected_frames == 0 {
            return 0.0;
        }
        (self.observed_frames as f64 - self.expected_frames as f64) / self.expected_frames as f64
    }
}

/// Drift correction strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CorrectionStrategy {
    /// No correction applied.
    None,
    /// Drop or repeat frames to re-sync.
    FrameDropRepeat,
    /// Adjust the effective frame rate by a small PPM offset.
    RateAdjust,
    /// Phase-shift: apply a one-time frame offset.
    PhaseShift,
}

/// Drift analysis result.
#[derive(Debug, Clone)]
pub struct DriftAnalysis {
    /// Average drift in frames per hour.
    pub drift_frames_per_hour: f64,
    /// Drift rate in parts per million (PPM).
    pub drift_ppm: f64,
    /// Maximum absolute drift observed in frames.
    pub max_drift_frames: i64,
    /// Whether drift is within acceptable tolerance.
    pub within_tolerance: bool,
    /// Number of samples analyzed.
    pub sample_count: usize,
    /// Recommended correction strategy.
    pub recommended_strategy: CorrectionStrategy,
}

/// Configuration for drift detection.
#[derive(Debug, Clone)]
pub struct DriftConfig {
    /// Frame rate of the timecode source.
    pub frame_rate: FrameRate,
    /// Maximum acceptable drift in frames before correction is recommended.
    pub tolerance_frames: u32,
    /// Minimum number of samples before analysis is valid.
    pub min_samples: usize,
    /// PPM threshold above which rate-adjust is recommended.
    pub ppm_threshold: f64,
}

impl DriftConfig {
    /// Creates a default configuration for the given frame rate.
    pub fn new(frame_rate: FrameRate) -> Self {
        Self {
            frame_rate,
            tolerance_frames: 2,
            min_samples: 3,
            ppm_threshold: 100.0,
        }
    }

    /// Sets the tolerance in frames.
    pub fn with_tolerance(mut self, frames: u32) -> Self {
        self.tolerance_frames = frames;
        self
    }

    /// Sets the minimum sample count.
    pub fn with_min_samples(mut self, n: usize) -> Self {
        self.min_samples = n;
        self
    }

    /// Sets the PPM threshold.
    pub fn with_ppm_threshold(mut self, ppm: f64) -> Self {
        self.ppm_threshold = ppm;
        self
    }
}

/// Timecode drift detector and corrector.
#[derive(Debug, Clone)]
pub struct DriftDetector {
    /// Configuration.
    config: DriftConfig,
    /// Collected samples.
    samples: Vec<DriftSample>,
}

impl DriftDetector {
    /// Creates a new drift detector with the given configuration.
    pub fn new(config: DriftConfig) -> Self {
        Self {
            config,
            samples: Vec::new(),
        }
    }

    /// Adds a drift sample.
    pub fn add_sample(&mut self, sample: DriftSample) {
        self.samples.push(sample);
    }

    /// Returns the number of collected samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Clears all collected samples.
    pub fn clear_samples(&mut self) {
        self.samples.clear();
    }

    /// Returns the latest drift in frames, or None if no samples.
    pub fn latest_drift(&self) -> Option<i64> {
        self.samples.last().map(DriftSample::drift_frames)
    }

    /// Analyzes collected drift data and returns a summary.
    #[allow(clippy::cast_precision_loss)]
    pub fn analyze(&self) -> Option<DriftAnalysis> {
        if self.samples.len() < self.config.min_samples {
            return None;
        }

        let n = self.samples.len() as f64;

        // Calculate drift rate via linear regression (drift vs wall time)
        let sum_t: f64 = self.samples.iter().map(|s| s.wall_time_secs).sum();
        let sum_d: f64 = self.samples.iter().map(|s| s.drift_frames() as f64).sum();
        let sum_td: f64 = self
            .samples
            .iter()
            .map(|s| s.wall_time_secs * s.drift_frames() as f64)
            .sum();
        let sum_t2: f64 = self
            .samples
            .iter()
            .map(|s| s.wall_time_secs * s.wall_time_secs)
            .sum();

        let denom = n * sum_t2 - sum_t * sum_t;
        let slope = if denom.abs() > 1e-12 {
            (n * sum_td - sum_t * sum_d) / denom
        } else {
            0.0
        };

        // slope is frames drift per second
        let drift_frames_per_hour = slope * 3600.0;
        let fps = self.config.frame_rate.as_float();
        let drift_ppm = if fps > 0.0 {
            (slope / fps) * 1_000_000.0
        } else {
            0.0
        };

        let max_drift_frames = self
            .samples
            .iter()
            .map(|s| s.drift_frames().unsigned_abs() as i64)
            .max()
            .unwrap_or(0);

        let within_tolerance = max_drift_frames <= self.config.tolerance_frames as i64;

        let recommended_strategy = if within_tolerance {
            CorrectionStrategy::None
        } else if drift_ppm.abs() > self.config.ppm_threshold {
            CorrectionStrategy::RateAdjust
        } else if max_drift_frames <= 5 {
            CorrectionStrategy::PhaseShift
        } else {
            CorrectionStrategy::FrameDropRepeat
        };

        Some(DriftAnalysis {
            drift_frames_per_hour,
            drift_ppm,
            max_drift_frames,
            within_tolerance,
            sample_count: self.samples.len(),
            recommended_strategy,
        })
    }

    /// Computes a corrected timecode by applying a frame offset to an observed timecode.
    pub fn correct_timecode(
        &self,
        observed: &Timecode,
        correction_frames: i64,
    ) -> Result<Timecode, TimecodeError> {
        let frame_rate = self.config.frame_rate;
        let current = observed.to_frames();
        let corrected = if correction_frames >= 0 {
            current + correction_frames as u64
        } else {
            current.saturating_sub(correction_frames.unsigned_abs())
        };
        Timecode::from_frames(corrected, frame_rate)
    }

    /// Returns the frame rate from the configuration.
    pub fn frame_rate(&self) -> FrameRate {
        self.config.frame_rate
    }
}

/// Computes effective PPM offset from two clock measurements.
#[allow(clippy::cast_precision_loss)]
pub fn compute_ppm(reference_frames: u64, observed_frames: u64) -> f64 {
    if reference_frames == 0 {
        return 0.0;
    }
    let diff = observed_frames as f64 - reference_frames as f64;
    (diff / reference_frames as f64) * 1_000_000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_drift_sample_creation() {
        let s = DriftSample::new(1.0, 25, 25);
        assert_eq!(s.wall_time_secs, 1.0);
        assert_eq!(s.observed_frames, 25);
        assert_eq!(s.expected_frames, 25);
    }

    #[test]
    fn test_drift_sample_zero_drift() {
        let s = DriftSample::new(1.0, 100, 100);
        assert_eq!(s.drift_frames(), 0);
        assert!((s.drift_ratio()).abs() < 1e-10);
    }

    #[test]
    fn test_drift_sample_positive() {
        let s = DriftSample::new(1.0, 102, 100);
        assert_eq!(s.drift_frames(), 2);
        assert!((s.drift_ratio() - 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_drift_sample_negative() {
        let s = DriftSample::new(1.0, 98, 100);
        assert_eq!(s.drift_frames(), -2);
        assert!((s.drift_ratio() + 0.02).abs() < 1e-10);
    }

    #[test]
    fn test_drift_sample_zero_expected() {
        let s = DriftSample::new(0.0, 0, 0);
        assert!((s.drift_ratio()).abs() < 1e-10);
    }

    #[test]
    fn test_config_defaults() {
        let c = DriftConfig::new(FrameRate::Fps25);
        assert_eq!(c.tolerance_frames, 2);
        assert_eq!(c.min_samples, 3);
        assert!((c.ppm_threshold - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_config_builder() {
        let c = DriftConfig::new(FrameRate::Fps25)
            .with_tolerance(5)
            .with_min_samples(10)
            .with_ppm_threshold(50.0);
        assert_eq!(c.tolerance_frames, 5);
        assert_eq!(c.min_samples, 10);
        assert!((c.ppm_threshold - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_detector_no_samples() {
        let det = DriftDetector::new(DriftConfig::new(FrameRate::Fps25));
        assert_eq!(det.sample_count(), 0);
        assert!(det.latest_drift().is_none());
        assert!(det.analyze().is_none());
    }

    #[test]
    fn test_detector_add_sample() {
        let mut det = DriftDetector::new(DriftConfig::new(FrameRate::Fps25));
        det.add_sample(DriftSample::new(0.0, 0, 0));
        det.add_sample(DriftSample::new(1.0, 25, 25));
        assert_eq!(det.sample_count(), 2);
        assert_eq!(det.latest_drift(), Some(0));
    }

    #[test]
    fn test_analyze_no_drift() {
        let mut det = DriftDetector::new(DriftConfig::new(FrameRate::Fps25));
        det.add_sample(DriftSample::new(0.0, 0, 0));
        det.add_sample(DriftSample::new(1.0, 25, 25));
        det.add_sample(DriftSample::new(2.0, 50, 50));
        let analysis = det.analyze().expect("analysis should succeed");
        assert!(analysis.within_tolerance);
        assert!((analysis.drift_ppm).abs() < 1.0);
        assert_eq!(analysis.recommended_strategy, CorrectionStrategy::None);
    }

    #[test]
    fn test_analyze_with_drift() {
        let config = DriftConfig::new(FrameRate::Fps25).with_tolerance(1);
        let mut det = DriftDetector::new(config);
        det.add_sample(DriftSample::new(0.0, 0, 0));
        det.add_sample(DriftSample::new(1.0, 26, 25));
        det.add_sample(DriftSample::new(2.0, 52, 50));
        let analysis = det.analyze().expect("analysis should succeed");
        assert!(!analysis.within_tolerance);
        assert!(analysis.max_drift_frames >= 1);
    }

    #[test]
    fn test_correct_timecode_forward() {
        let det = DriftDetector::new(DriftConfig::new(FrameRate::Fps25));
        let tc = Timecode::new(0, 0, 1, 0, FrameRate::Fps25).expect("valid timecode");
        let corrected = det
            .correct_timecode(&tc, 5)
            .expect("correction should succeed");
        assert_eq!(corrected.seconds, 1);
        assert_eq!(corrected.frames, 5);
    }

    #[test]
    fn test_correct_timecode_backward() {
        let det = DriftDetector::new(DriftConfig::new(FrameRate::Fps25));
        let tc = Timecode::new(0, 0, 1, 5, FrameRate::Fps25).expect("valid timecode");
        let corrected = det
            .correct_timecode(&tc, -5)
            .expect("correction should succeed");
        assert_eq!(corrected.seconds, 1);
        assert_eq!(corrected.frames, 0);
    }

    #[test]
    fn test_compute_ppm_zero() {
        assert!((compute_ppm(1000, 1000)).abs() < 1e-10);
    }

    #[test]
    fn test_compute_ppm_positive() {
        // 1001 vs 1000 => 1000 PPM
        let ppm = compute_ppm(1000, 1001);
        assert!((ppm - 1000.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_ppm_zero_reference() {
        assert!((compute_ppm(0, 100)).abs() < 1e-10);
    }

    #[test]
    fn test_clear_samples() {
        let mut det = DriftDetector::new(DriftConfig::new(FrameRate::Fps25));
        det.add_sample(DriftSample::new(0.0, 0, 0));
        assert_eq!(det.sample_count(), 1);
        det.clear_samples();
        assert_eq!(det.sample_count(), 0);
    }
}
