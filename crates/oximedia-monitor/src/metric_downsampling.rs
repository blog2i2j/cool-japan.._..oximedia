//! High-frequency time series metric downsampling for long-term storage.
//!
//! Implements several downsampling strategies for reducing the number of
//! data points in a time series while preserving its visual shape:
//!
//! - **LTTB** (Largest Triangle Three Buckets): perceptually accurate shape
//!   preservation — the gold standard for time-series downsampling.
//! - **Average**: arithmetic mean per bucket.
//! - **MaxMin**: include both maximum and minimum per bucket.
//! - **LastValue**: last observed value per bucket.
//!
//! # Example
//!
//! ```rust
//! use oximedia_monitor::metric_downsampling::{
//!     Sample, DownsampleMethod, MetricDownsampler,
//! };
//!
//! let samples: Vec<Sample> = (0..1000)
//!     .map(|i| Sample { timestamp_secs: i as u64, value: (i as f64 * 0.1).sin() })
//!     .collect();
//!
//! let result = MetricDownsampler::downsample(&samples, 50, DownsampleMethod::Lttb)
//!     .expect("downsample ok");
//! assert_eq!(result.original_count, 1000);
//! assert_eq!(result.samples.len(), 50);
//! ```

#![allow(dead_code)]

use thiserror::Error;

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single time-stamped metric sample.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sample {
    /// Unix timestamp in seconds.
    pub timestamp_secs: u64,
    /// Metric value.
    pub value: f64,
}

impl Sample {
    /// Create a new sample.
    #[must_use]
    pub fn new(timestamp_secs: u64, value: f64) -> Self {
        Self {
            timestamp_secs,
            value,
        }
    }
}

/// Downsampling method to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DownsampleMethod {
    /// Largest Triangle Three Buckets — visually faithful shape preservation.
    Lttb,
    /// Arithmetic average per bucket.
    Average,
    /// Include both the maximum and minimum per bucket (may produce 2× points).
    MaxMin,
    /// Last observed value per bucket.
    LastValue,
}

/// The result of a downsampling operation.
#[derive(Debug, Clone)]
pub struct DownsampledSeries {
    /// Downsampled samples (sorted by timestamp).
    pub samples: Vec<Sample>,
    /// Number of samples in the original series.
    pub original_count: usize,
    /// Method used to produce this series.
    pub method: DownsampleMethod,
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors that can occur during downsampling.
#[derive(Error, Debug, PartialEq, Eq)]
pub enum DownsampleError {
    /// The input series is empty.
    #[error("Empty series: cannot downsample an empty input")]
    EmptySeries,

    /// `target_points` was set to zero.
    #[error("Zero target points: target_points must be >= 1")]
    ZeroTargetPoints,

    /// The series has fewer than the minimum required points for the algorithm.
    #[error("Too few points: series has {found} points but at least {required} are required")]
    TooFewPoints {
        /// Points found in the series.
        found: usize,
        /// Minimum points required.
        required: usize,
    },
}

// ---------------------------------------------------------------------------
// MetricDownsampler
// ---------------------------------------------------------------------------

/// Stateless downsampler — all methods are pure functions.
pub struct MetricDownsampler;

impl MetricDownsampler {
    /// Downsample `samples` to (approximately) `target_points` using the
    /// given method.
    ///
    /// # Behaviour
    ///
    /// - If `samples.len() <= target_points`, the original series is returned
    ///   unchanged (no data is lost).
    /// - `MaxMin` may produce up to `2 * target_points` points when every
    ///   bucket's maximum differs from its minimum.
    ///
    /// # Errors
    ///
    /// - [`DownsampleError::EmptySeries`] when `samples` is empty.
    /// - [`DownsampleError::ZeroTargetPoints`] when `target_points == 0`.
    pub fn downsample(
        samples: &[Sample],
        target_points: usize,
        method: DownsampleMethod,
    ) -> Result<DownsampledSeries, DownsampleError> {
        if samples.is_empty() {
            return Err(DownsampleError::EmptySeries);
        }
        if target_points == 0 {
            return Err(DownsampleError::ZeroTargetPoints);
        }

        let original_count = samples.len();

        // If target >= source there is nothing to drop.
        if target_points >= original_count {
            return Ok(DownsampledSeries {
                samples: samples.to_vec(),
                original_count,
                method,
            });
        }

        let downsampled = match method {
            DownsampleMethod::Lttb => Self::lttb(samples, target_points),
            DownsampleMethod::Average => Self::average(samples, target_points),
            DownsampleMethod::MaxMin => Self::max_min(samples, target_points),
            DownsampleMethod::LastValue => Self::last_value(samples, target_points),
        };

        Ok(DownsampledSeries {
            samples: downsampled,
            original_count,
            method,
        })
    }

    // -----------------------------------------------------------------------
    // LTTB — Largest Triangle Three Buckets
    // -----------------------------------------------------------------------

    /// LTTB algorithm: preserves the visual shape of a time series.
    ///
    /// Reference: Sveinn Steinarsson, "Downsampling Time Series for Visual
    /// Representation", MSc thesis, University of Iceland, 2013.
    ///
    /// Steps
    /// 1. Always keep the first and last points.
    /// 2. Split the middle of the series into `(target - 2)` equal-width buckets.
    /// 3. For each bucket, pick the point that forms the largest triangle area
    ///    with the previously selected point and the average of the next bucket.
    fn lttb(samples: &[Sample], target_points: usize) -> Vec<Sample> {
        let n = samples.len();

        // Edge case: target == 1 → just return the first sample.
        if target_points == 1 {
            return vec![samples[0]];
        }
        // Edge case: target == 2 → first and last.
        if target_points == 2 {
            return vec![samples[0], samples[n - 1]];
        }

        let mut result: Vec<Sample> = Vec::with_capacity(target_points);

        // Always include the first point.
        result.push(samples[0]);

        // Number of middle points to select.
        let middle_count = target_points - 2;

        // Bucket size for the middle portion (indices 1 .. n-1).
        let middle_len = n - 2; // number of middle samples
        let bucket_size = middle_len as f64 / middle_count as f64;

        let mut prev_selected_idx: usize = 0;

        for i in 0..middle_count {
            // Current bucket bounds (within middle samples, 0-indexed into
            // samples[1..n-1]).
            let bucket_start =
                (i as f64 * bucket_size).floor() as usize;
            let bucket_end =
                ((i + 1) as f64 * bucket_size).floor() as usize;
            let bucket_end = bucket_end.min(middle_len - 1);

            // Next bucket (used for look-ahead average).
            let next_bucket_start = bucket_end + 1;
            let next_bucket_end =
                (((i + 2) as f64) * bucket_size).floor() as usize;
            let next_bucket_end = next_bucket_end.min(middle_len);

            // Compute average of the next bucket (centroid).
            let (avg_ts, avg_val) = if next_bucket_start < middle_len {
                let slice_end = next_bucket_end.min(middle_len - 1);
                let count = (slice_end - next_bucket_start + 1) as f64;
                let sum_ts: f64 = (next_bucket_start..=slice_end)
                    .map(|k| samples[k + 1].timestamp_secs as f64)
                    .sum();
                let sum_val: f64 = (next_bucket_start..=slice_end)
                    .map(|k| samples[k + 1].value)
                    .sum();
                (sum_ts / count, sum_val / count)
            } else {
                // Last bucket: use the final sample as centroid.
                (
                    samples[n - 1].timestamp_secs as f64,
                    samples[n - 1].value,
                )
            };

            // Previously selected point.
            let prev = samples[prev_selected_idx];

            // Find the point in the current bucket that maximises triangle area.
            let mut max_area = -1.0_f64;
            let mut best_idx = bucket_start + 1; // samples index (offset +1 for middle)

            for k in bucket_start..=bucket_end {
                let s = samples[k + 1]; // +1 because bucket indices start at samples[1]
                let area = triangle_area(
                    prev.timestamp_secs as f64,
                    prev.value,
                    s.timestamp_secs as f64,
                    s.value,
                    avg_ts,
                    avg_val,
                );
                if area > max_area {
                    max_area = area;
                    best_idx = k + 1;
                }
            }

            result.push(samples[best_idx]);
            prev_selected_idx = best_idx;
        }

        // Always include the last point.
        result.push(samples[n - 1]);
        result
    }

    // -----------------------------------------------------------------------
    // Average
    // -----------------------------------------------------------------------

    /// Bucket average: divide the series into `target_points` equal buckets
    /// and represent each bucket by its arithmetic mean (timestamp is the
    /// midpoint of the bucket's time range).
    fn average(samples: &[Sample], target_points: usize) -> Vec<Sample> {
        let n = samples.len();
        let bucket_size = n as f64 / target_points as f64;
        let mut result = Vec::with_capacity(target_points);

        for i in 0..target_points {
            let start = (i as f64 * bucket_size).floor() as usize;
            let end = ((i + 1) as f64 * bucket_size).floor() as usize;
            let end = end.min(n).max(start + 1);
            let bucket = &samples[start..end];

            let count = bucket.len() as f64;
            let avg_ts = bucket.iter().map(|s| s.timestamp_secs as f64).sum::<f64>() / count;
            let avg_val = bucket.iter().map(|s| s.value).sum::<f64>() / count;

            result.push(Sample {
                timestamp_secs: avg_ts.round() as u64,
                value: avg_val,
            });
        }

        result
    }

    // -----------------------------------------------------------------------
    // MaxMin
    // -----------------------------------------------------------------------

    /// MaxMin: for each bucket, emit the sample with the minimum value
    /// followed by the sample with the maximum value.  This preserves the
    /// full value range and is useful for oscillating signals.
    ///
    /// The returned list may contain up to `2 * target_points` samples.
    fn max_min(samples: &[Sample], target_points: usize) -> Vec<Sample> {
        let n = samples.len();
        let bucket_size = n as f64 / target_points as f64;
        let mut result = Vec::with_capacity(2 * target_points);

        for i in 0..target_points {
            let start = (i as f64 * bucket_size).floor() as usize;
            let end = ((i + 1) as f64 * bucket_size).floor() as usize;
            let end = end.min(n).max(start + 1);
            let bucket = &samples[start..end];

            // Find min and max by value.
            let min_sample = bucket
                .iter()
                .copied()
                .min_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal));
            let max_sample = bucket
                .iter()
                .copied()
                .max_by(|a, b| a.value.partial_cmp(&b.value).unwrap_or(std::cmp::Ordering::Equal));

            if let (Some(mn), Some(mx)) = (min_sample, max_sample) {
                // Emit in chronological order.
                if mn.timestamp_secs <= mx.timestamp_secs {
                    result.push(mn);
                    if mn != mx {
                        result.push(mx);
                    }
                } else {
                    result.push(mx);
                    if mn != mx {
                        result.push(mn);
                    }
                }
            }
        }

        result
    }

    // -----------------------------------------------------------------------
    // LastValue
    // -----------------------------------------------------------------------

    /// LastValue: take the last (most recent) sample in each bucket.
    fn last_value(samples: &[Sample], target_points: usize) -> Vec<Sample> {
        let n = samples.len();
        let bucket_size = n as f64 / target_points as f64;
        let mut result = Vec::with_capacity(target_points);

        for i in 0..target_points {
            let start = (i as f64 * bucket_size).floor() as usize;
            let end = ((i + 1) as f64 * bucket_size).floor() as usize;
            let end = end.min(n).max(start + 1);

            // Last sample in bucket.
            if let Some(&s) = samples[start..end].last() {
                result.push(s);
            }
        }

        result
    }
}

// ---------------------------------------------------------------------------
// Geometry helper
// ---------------------------------------------------------------------------

/// Compute the area of a triangle given three points.
///
/// Uses the shoelace formula: |½ · ((x₁(y₂−y₃) + x₂(y₃−y₁) + x₃(y₁−y₂))|
#[inline]
fn triangle_area(x1: f64, y1: f64, x2: f64, y2: f64, x3: f64, y3: f64) -> f64 {
    ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)).abs()) * 0.5
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_samples(n: usize) -> Vec<Sample> {
        (0..n)
            .map(|i| Sample::new(i as u64, i as f64))
            .collect()
    }

    fn sine_samples(n: usize) -> Vec<Sample> {
        use std::f64::consts::PI;
        (0..n)
            .map(|i| {
                let t = i as f64;
                Sample::new(i as u64, (2.0 * PI * t / n as f64).sin())
            })
            .collect()
    }

    // ---- Error cases -------------------------------------------------------

    #[test]
    fn test_empty_series_error() {
        let result = MetricDownsampler::downsample(&[], 10, DownsampleMethod::Lttb);
        assert_eq!(result.err(), Some(DownsampleError::EmptySeries));
    }

    #[test]
    fn test_zero_target_points_error() {
        let samples = make_samples(100);
        let result = MetricDownsampler::downsample(&samples, 0, DownsampleMethod::Average);
        assert_eq!(result.err(), Some(DownsampleError::ZeroTargetPoints));
    }

    // ---- Pass-through when target >= source --------------------------------

    #[test]
    fn test_target_equals_source_returns_all() {
        let samples = make_samples(10);
        let result =
            MetricDownsampler::downsample(&samples, 10, DownsampleMethod::Lttb).expect("ok");
        assert_eq!(result.samples.len(), 10);
        assert_eq!(result.original_count, 10);
    }

    #[test]
    fn test_target_exceeds_source_returns_all() {
        let samples = make_samples(5);
        let result =
            MetricDownsampler::downsample(&samples, 100, DownsampleMethod::Average).expect("ok");
        assert_eq!(result.samples.len(), 5);
        assert_eq!(result.original_count, 5);
    }

    // ---- LTTB properties ---------------------------------------------------

    #[test]
    fn test_lttb_output_count() {
        let samples = make_samples(1000);
        let result =
            MetricDownsampler::downsample(&samples, 50, DownsampleMethod::Lttb).expect("ok");
        assert_eq!(result.samples.len(), 50);
        assert_eq!(result.original_count, 1000);
    }

    #[test]
    fn test_lttb_preserves_endpoints() {
        let samples = make_samples(500);
        let result =
            MetricDownsampler::downsample(&samples, 20, DownsampleMethod::Lttb).expect("ok");
        let first = result.samples.first().expect("first");
        let last = result.samples.last().expect("last");
        assert_eq!(first.timestamp_secs, 0);
        assert_eq!(last.timestamp_secs, 499);
    }

    #[test]
    fn test_lttb_monotone_timestamps() {
        let samples = make_samples(200);
        let result =
            MetricDownsampler::downsample(&samples, 30, DownsampleMethod::Lttb).expect("ok");
        let timestamps: Vec<u64> = result.samples.iter().map(|s| s.timestamp_secs).collect();
        let sorted = {
            let mut t = timestamps.clone();
            t.sort_unstable();
            t
        };
        assert_eq!(timestamps, sorted, "LTTB output must be in timestamp order");
    }

    #[test]
    fn test_lttb_sine_preserves_shape() {
        // The downsampled sine should still alternate sign (crosses zero).
        let samples = sine_samples(1000);
        let result =
            MetricDownsampler::downsample(&samples, 50, DownsampleMethod::Lttb).expect("ok");
        let has_positive = result.samples.iter().any(|s| s.value > 0.1);
        let has_negative = result.samples.iter().any(|s| s.value < -0.1);
        assert!(has_positive, "Downsampled sine should include positive values");
        assert!(has_negative, "Downsampled sine should include negative values");
    }

    #[test]
    fn test_lttb_target_one() {
        let samples = make_samples(100);
        let result =
            MetricDownsampler::downsample(&samples, 1, DownsampleMethod::Lttb).expect("ok");
        assert_eq!(result.samples.len(), 1);
    }

    #[test]
    fn test_lttb_target_two() {
        let samples = make_samples(100);
        let result =
            MetricDownsampler::downsample(&samples, 2, DownsampleMethod::Lttb).expect("ok");
        assert_eq!(result.samples.len(), 2);
        assert_eq!(result.samples[0].timestamp_secs, 0);
        assert_eq!(result.samples[1].timestamp_secs, 99);
    }

    // ---- Average -----------------------------------------------------------

    #[test]
    fn test_average_output_count() {
        let samples = make_samples(1000);
        let result =
            MetricDownsampler::downsample(&samples, 100, DownsampleMethod::Average).expect("ok");
        assert_eq!(result.samples.len(), 100);
    }

    #[test]
    fn test_average_correctness() {
        // 10 samples with values 0..9; target 5 → 2 samples per bucket.
        // Bucket 0: [0, 1] → avg 0.5; Bucket 1: [2, 3] → avg 2.5; etc.
        let samples = make_samples(10);
        let result =
            MetricDownsampler::downsample(&samples, 5, DownsampleMethod::Average).expect("ok");
        assert_eq!(result.samples.len(), 5);
        // All averages should be between 0 and 9.
        for s in &result.samples {
            assert!(s.value >= 0.0 && s.value <= 9.0);
        }
        // First bucket mean should be less than last bucket mean (monotone series).
        assert!(
            result.samples[0].value < result.samples[4].value,
            "average of first bucket < average of last bucket"
        );
    }

    #[test]
    fn test_average_single_target_is_global_mean() {
        let samples: Vec<Sample> = (0..5)
            .map(|i| Sample::new(i as u64, i as f64))
            .collect(); // values 0,1,2,3,4 → mean = 2.0
        let result =
            MetricDownsampler::downsample(&samples, 1, DownsampleMethod::Average).expect("ok");
        assert_eq!(result.samples.len(), 1);
        assert!((result.samples[0].value - 2.0).abs() < 1e-9);
    }

    // ---- MaxMin ------------------------------------------------------------

    #[test]
    fn test_max_min_output_at_most_double() {
        let samples = make_samples(100);
        let result =
            MetricDownsampler::downsample(&samples, 10, DownsampleMethod::MaxMin).expect("ok");
        assert!(
            result.samples.len() <= 20,
            "MaxMin should produce at most 2× target points"
        );
    }

    #[test]
    fn test_max_min_preserves_extremes() {
        // Create a step function: first half = 0.0, second half = 100.0.
        let mut samples: Vec<Sample> = (0..50).map(|i| Sample::new(i, 0.0)).collect();
        samples.extend((50..100).map(|i| Sample::new(i, 100.0)));
        let result =
            MetricDownsampler::downsample(&samples, 5, DownsampleMethod::MaxMin).expect("ok");
        let has_zero = result.samples.iter().any(|s| s.value == 0.0);
        let has_hundred = result.samples.iter().any(|s| s.value == 100.0);
        assert!(has_zero, "MaxMin should preserve minimum value (0.0)");
        assert!(has_hundred, "MaxMin should preserve maximum value (100.0)");
    }

    // ---- LastValue ---------------------------------------------------------

    #[test]
    fn test_last_value_output_count() {
        let samples = make_samples(100);
        let result =
            MetricDownsampler::downsample(&samples, 10, DownsampleMethod::LastValue).expect("ok");
        assert_eq!(result.samples.len(), 10);
    }

    #[test]
    fn test_last_value_is_last_in_bucket() {
        // 10 samples 0..9, target 5 → 2 per bucket: last of each = 1, 3, 5, 7, 9.
        let samples = make_samples(10);
        let result =
            MetricDownsampler::downsample(&samples, 5, DownsampleMethod::LastValue).expect("ok");
        assert_eq!(result.samples.len(), 5);
        // Last bucket's last value should be 9.
        assert_eq!(
            result.samples.last().expect("last").value,
            9.0,
            "Last bucket last value should be 9"
        );
    }

    // ---- Metadata ----------------------------------------------------------

    #[test]
    fn test_original_count_stored() {
        let samples = make_samples(500);
        let result =
            MetricDownsampler::downsample(&samples, 50, DownsampleMethod::LastValue).expect("ok");
        assert_eq!(result.original_count, 500);
    }

    #[test]
    fn test_method_stored() {
        let samples = make_samples(100);
        let result =
            MetricDownsampler::downsample(&samples, 10, DownsampleMethod::Average).expect("ok");
        assert_eq!(result.method, DownsampleMethod::Average);
    }

    // ---- Triangle area helper ----------------------------------------------

    #[test]
    fn test_triangle_area_right_triangle() {
        // Right triangle with legs 3 and 4 → area = 6.
        let area = triangle_area(0.0, 0.0, 3.0, 0.0, 0.0, 4.0);
        assert!((area - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_triangle_area_collinear_is_zero() {
        let area = triangle_area(0.0, 0.0, 1.0, 1.0, 2.0, 2.0);
        assert!(area.abs() < 1e-9, "Collinear points have zero area");
    }
}
