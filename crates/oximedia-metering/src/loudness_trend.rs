#![allow(dead_code)]
//! Loudness trending over time windows.
//!
//! Tracks loudness measurements over time to detect trends such as rising
//! or falling loudness, compute moving averages, and identify segments that
//! exceed compliance thresholds. Useful for long-form content monitoring.

/// A timestamped loudness measurement.
#[derive(Clone, Copy, Debug)]
pub struct LoudnessSample {
    /// Timestamp in seconds from start.
    pub time_seconds: f64,
    /// Momentary loudness in LUFS.
    pub momentary_lufs: f64,
    /// Short-term loudness in LUFS.
    pub short_term_lufs: f64,
}

impl LoudnessSample {
    /// Create a new sample.
    pub fn new(time_seconds: f64, momentary_lufs: f64, short_term_lufs: f64) -> Self {
        Self {
            time_seconds,
            momentary_lufs,
            short_term_lufs,
        }
    }
}

/// Trend direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrendDirection {
    /// Loudness is rising over time.
    Rising,
    /// Loudness is falling over time.
    Falling,
    /// Loudness is relatively stable.
    Stable,
    /// Not enough data to determine trend.
    Unknown,
}

/// Statistics for a segment of loudness data.
#[derive(Clone, Debug)]
pub struct LoudnessStats {
    /// Mean momentary loudness in LUFS.
    pub mean_momentary: f64,
    /// Mean short-term loudness in LUFS.
    pub mean_short_term: f64,
    /// Max momentary loudness in LUFS.
    pub max_momentary: f64,
    /// Min momentary loudness in LUFS.
    pub min_momentary: f64,
    /// Standard deviation of momentary loudness.
    pub std_dev_momentary: f64,
    /// Detected trend direction.
    pub trend: TrendDirection,
    /// Number of samples in this segment.
    pub sample_count: usize,
}

/// A time segment that exceeds a loudness threshold.
#[derive(Clone, Debug)]
pub struct ExceedanceSegment {
    /// Start time in seconds.
    pub start_time: f64,
    /// End time in seconds.
    pub end_time: f64,
    /// Maximum loudness during exceedance in LUFS.
    pub max_lufs: f64,
    /// Duration of the exceedance in seconds.
    pub duration: f64,
}

/// Loudness trend tracker.
#[derive(Clone, Debug)]
pub struct LoudnessTrend {
    /// All collected samples.
    samples: Vec<LoudnessSample>,
    /// Target loudness for exceedance detection.
    target_lufs: f64,
    /// Tolerance for exceedance detection.
    tolerance_lu: f64,
}

impl LoudnessTrend {
    /// Create a new trend tracker.
    pub fn new(target_lufs: f64, tolerance_lu: f64) -> Self {
        Self {
            samples: Vec::new(),
            target_lufs,
            tolerance_lu,
        }
    }

    /// Create with EBU R128 defaults (-23 LUFS, 1 LU tolerance).
    pub fn ebu_r128() -> Self {
        Self::new(-23.0, 1.0)
    }

    /// Add a loudness sample.
    pub fn add_sample(&mut self, sample: LoudnessSample) {
        self.samples.push(sample);
    }

    /// Add a sample from values.
    pub fn add(&mut self, time_seconds: f64, momentary_lufs: f64, short_term_lufs: f64) {
        self.samples.push(LoudnessSample::new(
            time_seconds,
            momentary_lufs,
            short_term_lufs,
        ));
    }

    /// Get the total number of samples.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Get the duration covered in seconds.
    pub fn duration(&self) -> f64 {
        if self.samples.len() < 2 {
            return 0.0;
        }
        // SAFETY: len >= 2 is checked above so both indices are in-bounds
        self.samples[self.samples.len() - 1].time_seconds - self.samples[0].time_seconds
    }

    /// Compute overall statistics.
    pub fn overall_stats(&self) -> LoudnessStats {
        compute_stats(&self.samples)
    }

    /// Compute a moving average of momentary loudness with the given window in seconds.
    pub fn moving_average(&self, window_seconds: f64) -> Vec<(f64, f64)> {
        if self.samples.is_empty() {
            return Vec::new();
        }

        let mut result = Vec::new();
        for sample in &self.samples {
            let t = sample.time_seconds;
            let window_start = t - window_seconds;
            let window_samples: Vec<f64> = self
                .samples
                .iter()
                .filter(|s| s.time_seconds >= window_start && s.time_seconds <= t)
                .filter(|s| s.momentary_lufs.is_finite())
                .map(|s| s.momentary_lufs)
                .collect();

            if window_samples.is_empty() {
                result.push((t, f64::NEG_INFINITY));
            } else {
                let avg = window_samples.iter().sum::<f64>() / window_samples.len() as f64;
                result.push((t, avg));
            }
        }
        result
    }

    /// Detect segments that exceed the target loudness plus tolerance.
    pub fn exceedance_segments(&self) -> Vec<ExceedanceSegment> {
        let threshold = self.target_lufs + self.tolerance_lu;
        let mut segments = Vec::new();
        let mut in_exceed = false;
        let mut start_time = 0.0;
        let mut max_lufs = f64::NEG_INFINITY;

        for sample in &self.samples {
            let above = sample.short_term_lufs.is_finite() && sample.short_term_lufs > threshold;
            if above {
                if !in_exceed {
                    in_exceed = true;
                    start_time = sample.time_seconds;
                    max_lufs = sample.short_term_lufs;
                } else if sample.short_term_lufs > max_lufs {
                    max_lufs = sample.short_term_lufs;
                }
            } else if in_exceed {
                let end = sample.time_seconds;
                segments.push(ExceedanceSegment {
                    start_time,
                    end_time: end,
                    max_lufs,
                    duration: end - start_time,
                });
                in_exceed = false;
                max_lufs = f64::NEG_INFINITY;
            }
        }
        // Close any open segment
        if in_exceed {
            // SAFETY: we entered `in_exceed` only via iterating self.samples, so it is non-empty
            // SAFETY: in_exceed is set to true only after iterating at least one sample,
            // so self.samples is guaranteed non-empty here
            let end = self.samples[self.samples.len() - 1].time_seconds;
            segments.push(ExceedanceSegment {
                start_time,
                end_time: end,
                max_lufs,
                duration: end - start_time,
            });
        }
        segments
    }

    /// Segment the timeline into equal-duration windows and compute stats for each.
    pub fn segmented_stats(&self, segment_seconds: f64) -> Vec<LoudnessStats> {
        if self.samples.is_empty() || segment_seconds <= 0.0 {
            return Vec::new();
        }
        let total_dur = self.duration();
        let num_segments = ((total_dur / segment_seconds).ceil() as usize).max(1);
        // SAFETY: is_empty() is checked at the top of this function
        let start = self.samples[0].time_seconds;

        let mut results = Vec::new();
        for i in 0..num_segments {
            let seg_start = start + i as f64 * segment_seconds;
            let seg_end = seg_start + segment_seconds;
            let seg_samples: Vec<LoudnessSample> = self
                .samples
                .iter()
                .filter(|s| s.time_seconds >= seg_start && s.time_seconds < seg_end)
                .copied()
                .collect();
            results.push(compute_stats(&seg_samples));
        }
        results
    }

    /// Reset the tracker, clearing all samples.
    pub fn reset(&mut self) {
        self.samples.clear();
    }

    /// Get a reference to all samples.
    pub fn samples(&self) -> &[LoudnessSample] {
        &self.samples
    }
}

/// Compute statistics for a slice of samples.
fn compute_stats(samples: &[LoudnessSample]) -> LoudnessStats {
    if samples.is_empty() {
        return LoudnessStats {
            mean_momentary: f64::NEG_INFINITY,
            mean_short_term: f64::NEG_INFINITY,
            max_momentary: f64::NEG_INFINITY,
            min_momentary: f64::INFINITY,
            std_dev_momentary: 0.0,
            trend: TrendDirection::Unknown,
            sample_count: 0,
        };
    }

    let finite_momentary: Vec<f64> = samples
        .iter()
        .map(|s| s.momentary_lufs)
        .filter(|v| v.is_finite())
        .collect();

    let finite_short_term: Vec<f64> = samples
        .iter()
        .map(|s| s.short_term_lufs)
        .filter(|v| v.is_finite())
        .collect();

    let mean_m = if finite_momentary.is_empty() {
        f64::NEG_INFINITY
    } else {
        finite_momentary.iter().sum::<f64>() / finite_momentary.len() as f64
    };

    let mean_st = if finite_short_term.is_empty() {
        f64::NEG_INFINITY
    } else {
        finite_short_term.iter().sum::<f64>() / finite_short_term.len() as f64
    };

    let max_m = finite_momentary
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let min_m = finite_momentary
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);

    let std_dev = if finite_momentary.len() > 1 && mean_m.is_finite() {
        let var = finite_momentary
            .iter()
            .map(|v| (v - mean_m).powi(2))
            .sum::<f64>()
            / (finite_momentary.len() - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    // Determine trend via simple linear regression on momentary values
    let trend = if finite_momentary.len() >= 3 {
        let n = finite_momentary.len() as f64;
        let mut sum_x = 0.0;
        let mut sum_y = 0.0;
        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        for (i, &y) in finite_momentary.iter().enumerate() {
            let x = i as f64;
            sum_x += x;
            sum_y += y;
            sum_xy += x * y;
            sum_xx += x * x;
        }
        let denom = n * sum_xx - sum_x * sum_x;
        if denom.abs() > 1e-12 {
            let slope = (n * sum_xy - sum_x * sum_y) / denom;
            if slope > 0.1 {
                TrendDirection::Rising
            } else if slope < -0.1 {
                TrendDirection::Falling
            } else {
                TrendDirection::Stable
            }
        } else {
            TrendDirection::Stable
        }
    } else {
        TrendDirection::Unknown
    };

    LoudnessStats {
        mean_momentary: mean_m,
        mean_short_term: mean_st,
        max_momentary: max_m,
        min_momentary: min_m,
        std_dev_momentary: std_dev,
        trend,
        sample_count: samples.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_create() {
        let s = LoudnessSample::new(1.0, -20.0, -21.0);
        assert!((s.time_seconds - 1.0).abs() < f64::EPSILON);
        assert!((s.momentary_lufs - (-20.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_empty_trend() {
        let t = LoudnessTrend::ebu_r128();
        assert_eq!(t.sample_count(), 0);
        assert!((t.duration()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_add_samples() {
        let mut t = LoudnessTrend::ebu_r128();
        t.add(0.0, -23.0, -23.0);
        t.add(1.0, -22.0, -22.5);
        assert_eq!(t.sample_count(), 2);
        assert!((t.duration() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_overall_stats_stable() {
        let mut t = LoudnessTrend::ebu_r128();
        for i in 0..20 {
            t.add(i as f64 * 0.4, -23.0, -23.0);
        }
        let s = t.overall_stats();
        assert!((s.mean_momentary - (-23.0)).abs() < 0.1);
        assert_eq!(s.trend, TrendDirection::Stable);
    }

    #[test]
    fn test_overall_stats_rising() {
        let mut t = LoudnessTrend::ebu_r128();
        for i in 0..20 {
            let lufs = -30.0 + i as f64 * 1.0;
            t.add(i as f64, lufs, lufs);
        }
        let s = t.overall_stats();
        assert_eq!(s.trend, TrendDirection::Rising);
    }

    #[test]
    fn test_overall_stats_falling() {
        let mut t = LoudnessTrend::ebu_r128();
        for i in 0..20 {
            let lufs = -10.0 - i as f64 * 1.0;
            t.add(i as f64, lufs, lufs);
        }
        let s = t.overall_stats();
        assert_eq!(s.trend, TrendDirection::Falling);
    }

    #[test]
    fn test_moving_average() {
        let mut t = LoudnessTrend::ebu_r128();
        for i in 0..10 {
            t.add(i as f64, -20.0, -20.0);
        }
        let ma = t.moving_average(3.0);
        assert_eq!(ma.len(), 10);
        // All values should be -20.0
        for (_, avg) in &ma {
            assert!((avg - (-20.0)).abs() < 0.01);
        }
    }

    #[test]
    fn test_exceedance_segments() {
        let mut t = LoudnessTrend::new(-23.0, 1.0);
        // Normal range
        t.add(0.0, -23.0, -23.0);
        t.add(1.0, -23.0, -23.0);
        // Exceed threshold (-22.0)
        t.add(2.0, -20.0, -20.0);
        t.add(3.0, -19.0, -19.0);
        // Back to normal
        t.add(4.0, -23.0, -23.0);
        let segs = t.exceedance_segments();
        assert_eq!(segs.len(), 1);
        assert!((segs[0].start_time - 2.0).abs() < f64::EPSILON);
        assert!((segs[0].end_time - 4.0).abs() < f64::EPSILON);
        assert!((segs[0].max_lufs - (-19.0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_no_exceedance() {
        let mut t = LoudnessTrend::ebu_r128();
        for i in 0..10 {
            t.add(i as f64, -25.0, -25.0);
        }
        let segs = t.exceedance_segments();
        assert!(segs.is_empty());
    }

    #[test]
    fn test_segmented_stats() {
        let mut t = LoudnessTrend::ebu_r128();
        for i in 0..20 {
            t.add(i as f64, -23.0, -23.0);
        }
        let segs = t.segmented_stats(5.0);
        assert!(segs.len() >= 3);
        for s in &segs {
            assert!((s.mean_momentary - (-23.0)).abs() < 0.1);
        }
    }

    #[test]
    fn test_reset() {
        let mut t = LoudnessTrend::ebu_r128();
        t.add(0.0, -23.0, -23.0);
        t.reset();
        assert_eq!(t.sample_count(), 0);
    }

    #[test]
    fn test_empty_stats() {
        let s = compute_stats(&[]);
        assert!(s.mean_momentary.is_infinite());
        assert_eq!(s.sample_count, 0);
        assert_eq!(s.trend, TrendDirection::Unknown);
    }

    #[test]
    fn test_std_dev() {
        let mut t = LoudnessTrend::ebu_r128();
        t.add(0.0, -20.0, -20.0);
        t.add(1.0, -26.0, -26.0);
        t.add(2.0, -20.0, -20.0);
        t.add(3.0, -26.0, -26.0);
        let s = t.overall_stats();
        assert!(s.std_dev_momentary > 2.0);
    }

    #[test]
    fn test_exceedance_at_end() {
        let mut t = LoudnessTrend::new(-23.0, 1.0);
        t.add(0.0, -23.0, -23.0);
        t.add(1.0, -18.0, -18.0);
        t.add(2.0, -17.0, -17.0);
        let segs = t.exceedance_segments();
        assert_eq!(segs.len(), 1);
        assert!((segs[0].end_time - 2.0).abs() < f64::EPSILON);
    }
}
