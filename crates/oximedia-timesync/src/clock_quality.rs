#![allow(dead_code)]

//! Clock quality metrics and classification for time synchronization.
//!
//! This module provides types and algorithms for assessing the quality of
//! clock sources based on metrics such as accuracy, stability, jitter,
//! and holdover performance.

/// Clock accuracy class as defined by ITU-T and IEEE standards.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClockAccuracyClass {
    /// Stratum 1: primary reference (atomic clock, GPS receiver).
    Stratum1,
    /// Stratum 2: synchronized to a Stratum 1 source.
    Stratum2,
    /// Stratum 3: synchronized to a Stratum 2 source.
    Stratum3,
    /// Stratum 4: synchronized to a Stratum 3 source.
    Stratum4,
    /// Free-running: no external reference.
    FreeRunning,
}

impl ClockAccuracyClass {
    /// Returns the typical accuracy in nanoseconds for this class.
    #[must_use]
    pub const fn typical_accuracy_ns(&self) -> u64 {
        match self {
            Self::Stratum1 => 100,            // 100 ns
            Self::Stratum2 => 1_000,          // 1 us
            Self::Stratum3 => 10_000,         // 10 us
            Self::Stratum4 => 1_000_000,      // 1 ms
            Self::FreeRunning => 100_000_000, // 100 ms
        }
    }

    /// Returns a human-readable label for the class.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Stratum1 => "Stratum 1 (Primary Reference)",
            Self::Stratum2 => "Stratum 2",
            Self::Stratum3 => "Stratum 3",
            Self::Stratum4 => "Stratum 4",
            Self::FreeRunning => "Free-Running",
        }
    }

    /// Returns the numeric stratum number (1-5, 5 = free-running).
    #[must_use]
    pub const fn stratum_number(&self) -> u8 {
        match self {
            Self::Stratum1 => 1,
            Self::Stratum2 => 2,
            Self::Stratum3 => 3,
            Self::Stratum4 => 4,
            Self::FreeRunning => 5,
        }
    }
}

/// Measured quality metrics for a clock source.
#[derive(Debug, Clone)]
pub struct ClockQualityMetrics {
    /// Mean offset from reference in nanoseconds.
    pub mean_offset_ns: f64,
    /// Standard deviation of offset in nanoseconds.
    pub offset_stddev_ns: f64,
    /// Maximum observed jitter in nanoseconds.
    pub max_jitter_ns: f64,
    /// Allan deviation at 1-second averaging time.
    pub allan_deviation_1s: f64,
    /// Drift rate in parts per billion (ppb).
    pub drift_rate_ppb: f64,
    /// Number of measurements taken.
    pub sample_count: u64,
    /// Holdover time before exceeding threshold (in seconds).
    pub holdover_seconds: f64,
}

impl ClockQualityMetrics {
    /// Creates metrics with all zeros.
    #[must_use]
    pub fn zero() -> Self {
        Self {
            mean_offset_ns: 0.0,
            offset_stddev_ns: 0.0,
            max_jitter_ns: 0.0,
            allan_deviation_1s: 0.0,
            drift_rate_ppb: 0.0,
            sample_count: 0,
            holdover_seconds: 0.0,
        }
    }

    /// Computes an overall quality score from 0.0 (worst) to 1.0 (best).
    ///
    /// Uses a weighted combination of offset stability, jitter, and drift.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn quality_score(&self) -> f64 {
        if self.sample_count == 0 {
            return 0.0;
        }

        // Offset component: lower stddev is better, reference = 100 ns
        let offset_score = (100.0 / (self.offset_stddev_ns.abs() + 100.0)).min(1.0);

        // Jitter component: lower jitter is better, reference = 1000 ns
        let jitter_score = (1000.0 / (self.max_jitter_ns.abs() + 1000.0)).min(1.0);

        // Drift component: lower drift is better, reference = 10 ppb
        let drift_score = (10.0 / (self.drift_rate_ppb.abs() + 10.0)).min(1.0);

        // Weighted average
        offset_score * 0.4 + jitter_score * 0.3 + drift_score * 0.3
    }

    /// Classifies the clock accuracy based on measured metrics.
    #[must_use]
    pub fn classify(&self) -> ClockAccuracyClass {
        let stddev = self.offset_stddev_ns.abs();
        if stddev < 500.0 && self.drift_rate_ppb.abs() < 1.0 {
            ClockAccuracyClass::Stratum1
        } else if stddev < 5_000.0 && self.drift_rate_ppb.abs() < 10.0 {
            ClockAccuracyClass::Stratum2
        } else if stddev < 50_000.0 && self.drift_rate_ppb.abs() < 100.0 {
            ClockAccuracyClass::Stratum3
        } else if stddev < 500_000.0 {
            ClockAccuracyClass::Stratum4
        } else {
            ClockAccuracyClass::FreeRunning
        }
    }

    /// Returns whether the metrics indicate an acceptable clock for broadcast use.
    ///
    /// Broadcast requires offset stddev < 500 ns and max jitter < 1 us.
    #[must_use]
    pub fn is_broadcast_grade(&self) -> bool {
        self.offset_stddev_ns.abs() < 500.0 && self.max_jitter_ns.abs() < 1_000.0
    }
}

/// Accumulator for computing clock quality metrics from raw offset samples.
#[derive(Debug, Clone)]
pub struct QualityAccumulator {
    /// Collected offset samples in nanoseconds.
    samples: Vec<f64>,
    /// Maximum number of samples to retain.
    max_samples: usize,
}

impl QualityAccumulator {
    /// Creates a new accumulator with the given maximum sample count.
    #[must_use]
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples.min(10_000)),
            max_samples,
        }
    }

    /// Adds an offset measurement in nanoseconds.
    pub fn add_sample(&mut self, offset_ns: f64) {
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push(offset_ns);
    }

    /// Returns the current number of samples.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Computes the mean of collected samples.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.samples.iter().sum();
        sum / self.samples.len() as f64
    }

    /// Computes the standard deviation of collected samples.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn stddev(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        let mean = self.mean();
        let variance: f64 = self.samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>()
            / (self.samples.len() - 1) as f64;
        variance.sqrt()
    }

    /// Returns the maximum absolute jitter (difference between consecutive samples).
    #[must_use]
    pub fn max_jitter(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        self.samples
            .windows(2)
            .map(|w| (w[1] - w[0]).abs())
            .fold(0.0_f64, f64::max)
    }

    /// Computes drift rate in ppb from first and last samples, given sample interval.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_drift_ppb(&self, sample_interval_s: f64) -> f64 {
        if self.samples.len() < 2 || sample_interval_s <= 0.0 {
            return 0.0;
        }
        let first = self.samples[0];
        let last = self.samples[self.samples.len() - 1];
        let elapsed_s = (self.samples.len() - 1) as f64 * sample_interval_s;
        if elapsed_s <= 0.0 {
            return 0.0;
        }
        // Drift in ns/s, converted to ppb (1 ns/s = 1 ppb)
        (last - first) / elapsed_s
    }

    /// Builds a `ClockQualityMetrics` from the accumulated samples.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_metrics(&self, sample_interval_s: f64) -> ClockQualityMetrics {
        ClockQualityMetrics {
            mean_offset_ns: self.mean(),
            offset_stddev_ns: self.stddev(),
            max_jitter_ns: self.max_jitter(),
            allan_deviation_1s: self.stddev() * 0.707, // approximate for white noise
            drift_rate_ppb: self.estimate_drift_ppb(sample_interval_s),
            sample_count: self.samples.len() as u64,
            holdover_seconds: 0.0,
        }
    }

    /// Clears all samples.
    pub fn clear(&mut self) {
        self.samples.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accuracy_class_ordering() {
        assert!(ClockAccuracyClass::Stratum1 < ClockAccuracyClass::Stratum2);
        assert!(ClockAccuracyClass::Stratum3 < ClockAccuracyClass::FreeRunning);
    }

    #[test]
    fn test_accuracy_class_labels() {
        assert!(ClockAccuracyClass::Stratum1.label().contains("Primary"));
        assert!(ClockAccuracyClass::FreeRunning.label().contains("Free"));
    }

    #[test]
    fn test_stratum_numbers() {
        assert_eq!(ClockAccuracyClass::Stratum1.stratum_number(), 1);
        assert_eq!(ClockAccuracyClass::FreeRunning.stratum_number(), 5);
    }

    #[test]
    fn test_typical_accuracy() {
        assert!(
            ClockAccuracyClass::Stratum1.typical_accuracy_ns()
                < ClockAccuracyClass::Stratum2.typical_accuracy_ns()
        );
    }

    #[test]
    fn test_quality_score_zero_samples() {
        let m = ClockQualityMetrics::zero();
        assert!((m.quality_score() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_quality_score_perfect() {
        let m = ClockQualityMetrics {
            mean_offset_ns: 0.0,
            offset_stddev_ns: 0.0,
            max_jitter_ns: 0.0,
            allan_deviation_1s: 0.0,
            drift_rate_ppb: 0.0,
            sample_count: 100,
            holdover_seconds: 3600.0,
        };
        let score = m.quality_score();
        assert!((score - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_classify_stratum1() {
        let m = ClockQualityMetrics {
            offset_stddev_ns: 50.0,
            drift_rate_ppb: 0.5,
            ..ClockQualityMetrics::zero()
        };
        assert_eq!(m.classify(), ClockAccuracyClass::Stratum1);
    }

    #[test]
    fn test_classify_free_running() {
        let m = ClockQualityMetrics {
            offset_stddev_ns: 1_000_000.0,
            drift_rate_ppb: 500.0,
            ..ClockQualityMetrics::zero()
        };
        assert_eq!(m.classify(), ClockAccuracyClass::FreeRunning);
    }

    #[test]
    fn test_broadcast_grade() {
        let good = ClockQualityMetrics {
            offset_stddev_ns: 100.0,
            max_jitter_ns: 500.0,
            ..ClockQualityMetrics::zero()
        };
        assert!(good.is_broadcast_grade());

        let bad = ClockQualityMetrics {
            offset_stddev_ns: 1000.0,
            max_jitter_ns: 5000.0,
            ..ClockQualityMetrics::zero()
        };
        assert!(!bad.is_broadcast_grade());
    }

    #[test]
    fn test_accumulator_mean_and_stddev() {
        let mut acc = QualityAccumulator::new(100);
        for v in [10.0, 20.0, 30.0, 40.0, 50.0] {
            acc.add_sample(v);
        }
        assert!((acc.mean() - 30.0).abs() < 1e-9);
        assert!(acc.stddev() > 0.0);
        assert_eq!(acc.sample_count(), 5);
    }

    #[test]
    fn test_accumulator_max_jitter() {
        let mut acc = QualityAccumulator::new(100);
        acc.add_sample(0.0);
        acc.add_sample(100.0);
        acc.add_sample(50.0);
        // Max jitter between consecutive: |100-0|=100, |50-100|=50 => 100
        assert!((acc.max_jitter() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_accumulator_drift_estimate() {
        let mut acc = QualityAccumulator::new(100);
        // Linear drift: 10 ns per sample, 1s interval => 10 ppb
        for i in 0..10 {
            acc.add_sample(i as f64 * 10.0);
        }
        let drift = acc.estimate_drift_ppb(1.0);
        assert!((drift - 10.0).abs() < 1e-9);
    }

    #[test]
    fn test_accumulator_max_samples_eviction() {
        let mut acc = QualityAccumulator::new(3);
        acc.add_sample(1.0);
        acc.add_sample(2.0);
        acc.add_sample(3.0);
        acc.add_sample(4.0);
        assert_eq!(acc.sample_count(), 3);
        // First sample (1.0) should have been evicted
        assert!((acc.mean() - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_compute_metrics() {
        let mut acc = QualityAccumulator::new(100);
        acc.add_sample(100.0);
        acc.add_sample(200.0);
        acc.add_sample(300.0);
        let metrics = acc.compute_metrics(1.0);
        assert_eq!(metrics.sample_count, 3);
        assert!((metrics.mean_offset_ns - 200.0).abs() < 1e-9);
        assert!(metrics.offset_stddev_ns > 0.0);
    }

    #[test]
    fn test_accumulator_clear() {
        let mut acc = QualityAccumulator::new(100);
        acc.add_sample(1.0);
        acc.add_sample(2.0);
        acc.clear();
        assert_eq!(acc.sample_count(), 0);
        assert!((acc.mean() - 0.0).abs() < f64::EPSILON);
    }
}
