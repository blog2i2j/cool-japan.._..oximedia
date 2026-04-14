#![allow(dead_code)]

//! Shot tempo and editing pace analysis.
//!
//! This module measures the tempo of editing — the rhythmic pattern formed
//! by shot durations over time. It detects tempo changes, classifies
//! segments as slow/medium/fast, and computes autocorrelation of shot
//! durations to find periodic editing patterns.

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// Classification of editing pace.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaceClass {
    /// Long shots, slow pacing.
    Slow,
    /// Moderate shot lengths.
    Medium,
    /// Quick cuts, fast pacing.
    Fast,
}

/// A temporal segment with a classified pace.
#[derive(Debug, Clone)]
pub struct TempoSegment {
    /// Start index (shot index) of this segment.
    pub start_shot: usize,
    /// End index (exclusive) of this segment.
    pub end_shot: usize,
    /// Classified pace for this segment.
    pub pace: PaceClass,
    /// Mean shot duration in seconds within this segment.
    pub mean_duration: f64,
}

/// A detected tempo change point.
#[derive(Debug, Clone)]
pub struct TempoChange {
    /// Shot index where the change occurs.
    pub shot_index: usize,
    /// Pace before the change.
    pub before: PaceClass,
    /// Pace after the change.
    pub after: PaceClass,
    /// Magnitude of change (ratio of mean durations).
    pub magnitude: f64,
}

/// Autocorrelation result for shot durations.
#[derive(Debug, Clone)]
pub struct Autocorrelation {
    /// Lag in number of shots.
    pub lag: usize,
    /// Correlation coefficient at this lag (-1.0 to 1.0).
    pub coefficient: f64,
}

/// Summary of tempo analysis.
#[derive(Debug, Clone)]
pub struct TempoSummary {
    /// Total shots analysed.
    pub total_shots: usize,
    /// Global mean shot duration (seconds).
    pub global_mean: f64,
    /// Global median shot duration (seconds).
    pub global_median: f64,
    /// Detected segments.
    pub segments: Vec<TempoSegment>,
    /// Detected tempo changes.
    pub changes: Vec<TempoChange>,
    /// Dominant periodicity lag (0 if none found).
    pub dominant_period: usize,
}

/// Tempo analyser.
#[derive(Debug, Clone)]
pub struct TempoAnalyzer {
    /// Window size (number of shots) for segment classification.
    segment_window: usize,
    /// Threshold (seconds) below which pace is classified as Fast.
    fast_threshold: f64,
    /// Threshold (seconds) above which pace is classified as Slow.
    slow_threshold: f64,
    /// Maximum autocorrelation lag to compute.
    max_lag: usize,
}

// ---------------------------------------------------------------------------
// Implementation
// ---------------------------------------------------------------------------

impl Default for TempoAnalyzer {
    fn default() -> Self {
        Self {
            segment_window: 5,
            fast_threshold: 2.0,
            slow_threshold: 6.0,
            max_lag: 20,
        }
    }
}

impl TempoAnalyzer {
    /// Create a new analyser with custom parameters.
    pub fn new(
        segment_window: usize,
        fast_threshold: f64,
        slow_threshold: f64,
        max_lag: usize,
    ) -> Self {
        Self {
            segment_window,
            fast_threshold,
            slow_threshold,
            max_lag,
        }
    }

    /// Classify a mean duration into a pace class.
    pub fn classify_pace(&self, mean_duration: f64) -> PaceClass {
        if mean_duration <= self.fast_threshold {
            PaceClass::Fast
        } else if mean_duration >= self.slow_threshold {
            PaceClass::Slow
        } else {
            PaceClass::Medium
        }
    }

    /// Segment shot durations into contiguous pace regions.
    #[allow(clippy::cast_precision_loss)]
    pub fn segment(&self, durations: &[f64]) -> Vec<TempoSegment> {
        if durations.is_empty() {
            return Vec::new();
        }

        let mut segments = Vec::new();
        let window = self.segment_window.max(1);
        let mut i = 0;

        while i < durations.len() {
            let end = (i + window).min(durations.len());
            let slice = &durations[i..end];
            let mean = slice.iter().sum::<f64>() / slice.len() as f64;
            let pace = self.classify_pace(mean);

            // Extend the segment while pace remains the same.
            let mut j = end;
            while j < durations.len() {
                let ext_end = (j + window).min(durations.len());
                let ext_slice = &durations[j..ext_end];
                let ext_mean = ext_slice.iter().sum::<f64>() / ext_slice.len() as f64;
                if self.classify_pace(ext_mean) != pace {
                    break;
                }
                j = ext_end;
            }

            let seg_slice = &durations[i..j];
            let seg_mean = seg_slice.iter().sum::<f64>() / seg_slice.len() as f64;

            segments.push(TempoSegment {
                start_shot: i,
                end_shot: j,
                pace,
                mean_duration: seg_mean,
            });
            i = j;
        }
        segments
    }

    /// Detect tempo change points between consecutive segments.
    pub fn detect_changes(&self, segments: &[TempoSegment]) -> Vec<TempoChange> {
        let mut changes = Vec::new();
        for pair in segments.windows(2) {
            let before = &pair[0];
            let after = &pair[1];
            if before.pace != after.pace {
                let magnitude = if before.mean_duration > 0.0 {
                    after.mean_duration / before.mean_duration
                } else {
                    0.0
                };
                changes.push(TempoChange {
                    shot_index: after.start_shot,
                    before: before.pace,
                    after: after.pace,
                    magnitude,
                });
            }
        }
        changes
    }

    /// Compute autocorrelation of shot durations for lags 1..max_lag.
    #[allow(clippy::cast_precision_loss)]
    pub fn autocorrelation(&self, durations: &[f64]) -> Vec<Autocorrelation> {
        let n = durations.len();
        if n < 2 {
            return Vec::new();
        }

        let mean = durations.iter().sum::<f64>() / n as f64;
        let variance: f64 = durations.iter().map(|d| (d - mean).powi(2)).sum();

        if variance.abs() < f64::EPSILON {
            return Vec::new();
        }

        let mut results = Vec::new();
        let max = self.max_lag.min(n - 1);

        for lag in 1..=max {
            let mut cov = 0.0_f64;
            for i in 0..(n - lag) {
                cov += (durations[i] - mean) * (durations[i + lag] - mean);
            }
            let coeff = cov / variance;
            results.push(Autocorrelation {
                lag,
                coefficient: coeff,
            });
        }
        results
    }

    /// Compute the median of a slice of f64 values (non-destructive).
    fn median(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }

    /// Produce a full tempo summary.
    #[allow(clippy::cast_precision_loss)]
    pub fn summarize(&self, durations: &[f64]) -> TempoSummary {
        if durations.is_empty() {
            return TempoSummary {
                total_shots: 0,
                global_mean: 0.0,
                global_median: 0.0,
                segments: Vec::new(),
                changes: Vec::new(),
                dominant_period: 0,
            };
        }

        let n = durations.len();
        let global_mean = durations.iter().sum::<f64>() / n as f64;
        let global_median = Self::median(durations);
        let segments = self.segment(durations);
        let changes = self.detect_changes(&segments);
        let ac = self.autocorrelation(durations);

        let dominant_period = ac
            .iter()
            .max_by(|a, b| {
                a.coefficient
                    .partial_cmp(&b.coefficient)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map_or(0, |a| a.lag);

        TempoSummary {
            total_shots: n,
            global_mean,
            global_median,
            segments,
            changes,
            dominant_period,
        }
    }

    /// Configured segment window size.
    pub fn segment_window(&self) -> usize {
        self.segment_window
    }

    /// Configured fast threshold.
    pub fn fast_threshold(&self) -> f64 {
        self.fast_threshold
    }

    /// Configured slow threshold.
    pub fn slow_threshold(&self) -> f64 {
        self.slow_threshold
    }

    /// Configured max autocorrelation lag.
    pub fn max_lag(&self) -> usize {
        self.max_lag
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_fast() {
        let a = TempoAnalyzer::default();
        assert_eq!(a.classify_pace(1.0), PaceClass::Fast);
    }

    #[test]
    fn test_classify_medium() {
        let a = TempoAnalyzer::default();
        assert_eq!(a.classify_pace(4.0), PaceClass::Medium);
    }

    #[test]
    fn test_classify_slow() {
        let a = TempoAnalyzer::default();
        assert_eq!(a.classify_pace(8.0), PaceClass::Slow);
    }

    #[test]
    fn test_segment_empty() {
        let a = TempoAnalyzer::default();
        let s = a.segment(&[]);
        assert!(s.is_empty());
    }

    #[test]
    fn test_segment_uniform_fast() {
        let a = TempoAnalyzer::new(3, 2.0, 6.0, 10);
        let durations = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let s = a.segment(&durations);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].pace, PaceClass::Fast);
    }

    #[test]
    fn test_segment_mixed() {
        let a = TempoAnalyzer::new(2, 2.0, 6.0, 10);
        let durations = vec![1.0, 1.0, 8.0, 8.0];
        let s = a.segment(&durations);
        assert!(s.len() >= 2);
    }

    #[test]
    fn test_detect_changes_none() {
        let segments = vec![TempoSegment {
            start_shot: 0,
            end_shot: 5,
            pace: PaceClass::Medium,
            mean_duration: 4.0,
        }];
        let a = TempoAnalyzer::default();
        let c = a.detect_changes(&segments);
        assert!(c.is_empty());
    }

    #[test]
    fn test_detect_changes_present() {
        let segments = vec![
            TempoSegment {
                start_shot: 0,
                end_shot: 3,
                pace: PaceClass::Fast,
                mean_duration: 1.0,
            },
            TempoSegment {
                start_shot: 3,
                end_shot: 6,
                pace: PaceClass::Slow,
                mean_duration: 8.0,
            },
        ];
        let a = TempoAnalyzer::default();
        let c = a.detect_changes(&segments);
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].before, PaceClass::Fast);
        assert_eq!(c[0].after, PaceClass::Slow);
    }

    #[test]
    fn test_autocorrelation_too_short() {
        let a = TempoAnalyzer::default();
        let ac = a.autocorrelation(&[1.0]);
        assert!(ac.is_empty());
    }

    #[test]
    fn test_autocorrelation_constant() {
        let a = TempoAnalyzer::new(3, 2.0, 6.0, 5);
        let durations = vec![3.0; 10];
        let ac = a.autocorrelation(&durations);
        // Constant signal → zero variance → empty
        assert!(ac.is_empty());
    }

    #[test]
    fn test_autocorrelation_periodic() {
        let a = TempoAnalyzer::new(3, 2.0, 6.0, 10);
        let durations: Vec<f64> = (0..20)
            .map(|i| if i % 2 == 0 { 1.0 } else { 5.0 })
            .collect();
        let ac = a.autocorrelation(&durations);
        assert!(!ac.is_empty());
        // Lag 2 should have high positive correlation for alternating signal
        let lag2 = ac
            .iter()
            .find(|a| a.lag == 2)
            .expect("should succeed in test");
        assert!(lag2.coefficient > 0.5);
    }

    #[test]
    fn test_summarize_empty() {
        let a = TempoAnalyzer::default();
        let s = a.summarize(&[]);
        assert_eq!(s.total_shots, 0);
        assert!((s.global_mean - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_summarize_basic() {
        let a = TempoAnalyzer::default();
        let durations = vec![2.0, 3.0, 4.0, 5.0];
        let s = a.summarize(&durations);
        assert_eq!(s.total_shots, 4);
        assert!((s.global_mean - 3.5).abs() < f64::EPSILON);
        assert!((s.global_median - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_analyzer_accessors() {
        let a = TempoAnalyzer::new(7, 1.5, 5.5, 30);
        assert_eq!(a.segment_window(), 7);
        assert!((a.fast_threshold() - 1.5).abs() < f64::EPSILON);
        assert!((a.slow_threshold() - 5.5).abs() < f64::EPSILON);
        assert_eq!(a.max_lag(), 30);
    }
}
